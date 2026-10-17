#!/usr/bin/env python
"""Assemble results.md and results.json from eval outputs.

Inputs (all under /data/models/harrier-quant):
  eval/cosine/<v>_vs_bf16-vllm.json                 H100: per-item cosine vs BF16 via vLLM
  eval/cosine/bf16-{vllm,st}.npy                    H100: engine-drift check (computed here)
  eval/blackwell/cosine/bw-<v>_vs_bw-bf16.json       RTX PRO 6000 Blackwell: cosine vs BF16 on the same GPU
  eval/blackwell/logs/ctx*.log                      24 GB-card context probes (RESULT lines)
  eval/bf16-st/summary.json                         mteb, BF16 via sentence-transformers (reference implementation)
  eval/vllm-<v>/summary.json                        mteb via vLLM (BF16 + variants)
Usage: python make_results.py
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/data/models/harrier-quant")
EV = ROOT / "eval"
BUCKETS = ["short_en", "long", "multilingual", "code"]
TIE_TASKS = {"CodeTransOceanDL", "SyntheticText2SQL", "DS1000Retrieval", "CosQA"}  # see eval/tie_affected_tasks.json
TASK_ORDER = ["SciFact", "NFCorpus", "FiQA2018", "ArguAna", "SCIDOCS", "LEMBSummScreenFDRetrieval",
              "BelebeleRetrieval", "MLQARetrieval"]
# key -> (round, short name, description)
VARIANTS = {
    "w4a16-gptq": ("R1", "W4A16 g128 · FP8 edges", "GPTQ INT4 g128; FP8 layers 0-5, 56-61"),
    "nvfp4a16": ("R1", "NVFP4A16 · FP8 edges", "NVFP4 weights (data-free); FP8 layers 0-5, 56-61"),
    "nvfp4": ("R1", "NVFP4 W4A4 · FP8 edges", "NVFP4 weights + activations; FP8 layers 0-5, 56-61"),
    "w4a16-gptq-late12-attn": ("R2", "W4A16 g128 · FP8 late12+attn", "GPTQ INT4 g128; FP8 layers 50-61 + q/k/v/o everywhere"),
    "w4a16-gptq-late12-attn-g64": ("R3", "W4A16 g64 · FP8 late12+attn", "GPTQ INT4 g64; FP8 layers 50-61 + q/k/v/o everywhere"),
    "w4a16-gptq-late12-attn-down": ("R3", "W4A16 g128 · FP8 late12+attn+down", "GPTQ INT4 g128; FP8 layers 50-61 + q/k/v/o/down everywhere"),
}


def size_gb(path):
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1e9


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def cos_stats(a, b, items):
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    bk = np.array([i["bucket"] for i in items])
    return {k: dict(n=int((bk == k).sum()), median=float(np.median(cos[bk == k])),
                    p5=float(np.percentile(cos[bk == k], 5)), p1=float(np.percentile(cos[bk == k], 1)),
                    min=float(cos[bk == k].min())) for k in BUCKETS}


def context_results():
    """Parse every RESULT line from the Blackwell context probes (latest run per config wins)."""
    out = {}
    for f in sorted((EV / "blackwell" / "logs").glob("ctx*.log")):
        if "warmup" in f.name:
            continue
        txt = f.read_text(errors="ignore")
        m = re.findall(r"^RESULT (\{.*\})$", txt, re.M)
        if not m:
            continue
        r = json.loads(m[-1])
        kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt)
        r["kv_tokens"] = int(kv[-1].replace(",", "")) if kv else None
        r["model"] = Path(r["model"]).name
        key = (r["model"], r.get("chunked", 0), r.get("eager", False), r["budget_gib"], r["max_len"])
        out[key] = r
    return list(out.values())


def max_ctx(ctx, model, budget, eager=True, chunked=2048):
    ok = [r["max_len"] for r in ctx if r["model"] == model and r["budget_gib"] == budget
          and r.get("eager", False) == eager and r.get("chunked", 0) == chunked and r["ok"]]
    tried = [r["max_len"] for r in ctx if r["model"] == model and r["budget_gib"] == budget
             and r.get("eager", False) == eager and r.get("chunked", 0) == chunked]
    if not tried:
        return None
    return max(ok) if ok else -min(tried)  # negative = failed even at the shortest length tried


def f4(x):
    return "–" if x is None else f"{x:.4f}"


def fctx(x):
    if x is None:
        return "not tested"
    return f"<{-x // 1024}k" if x < 0 else f"{x // 1024}k"


def main():
    items = [json.loads(l) for l in open(EV / "cosine" / "items.jsonl")]
    ctx = context_results()
    res = {"variants": {}, "context_probes": ctx}

    drift = cos_stats(np.load(EV / "cosine" / "bf16-vllm.npy"), np.load(EV / "cosine" / "bf16-st.npy"), items)
    hw = load(EV / "blackwell" / "cosine" / "bw-bf16_vs_bf16-vllm.json")
    res["engine_drift_vllm_vs_st_bf16"] = drift
    res["hardware_drift_blackwell_vs_h100_bf16"] = hw["buckets"] if hw else None
    st, vb = load(EV / "bf16-st" / "summary.json"), load(EV / "vllm-bf16" / "summary.json")
    res["mteb_bf16_st"], res["mteb_bf16_vllm"] = st, vb

    for v, (rnd, short, desc) in VARIANTS.items():
        h100 = load(EV / "cosine" / f"{v}_vs_bf16-vllm.json")
        bw = load(EV / "blackwell" / "cosine" / f"bw-{v}_vs_bw-bf16.json")
        gate = bw if v == "nvfp4" else h100  # W4A4 can only be measured with native FP4 on Blackwell
        mteb = load(EV / f"vllm-{v}" / "summary.json")
        res["variants"][v] = dict(round=rnd, name=short, desc=desc, size_gb=size_gb(ROOT / v), cos_h100=h100,
                                  cos_blackwell=bw, gate=gate, gate_pass=bool(gate and gate["pass"]), mteb=mteb,
                                  ctx_090=max_ctx(ctx, v, 21.5), ctx_095=max_ctx(ctx, v, 22.7))

    def mean(m, k):
        if not m:
            return None
        v = [m["tasks"][t][k] for t in TASK_ORDER if t in m["tasks"]]
        return float(np.mean(v)) if len(v) == len(TASK_ORDER) else None

    base_nd = mean(vb, "ndcg_at_10")
    header = ROOT / "results_header.md"  # hand-written findings / serve commands, kept in sync with these tables
    L = ([header.read_text().rstrip(), "", "# Detailed results", ""] if header.exists() else ["# harrier-oss-v1-27b quantization results", ""]) + [
         "Base model: [`microsoft/harrier-oss-v1-27b`](https://huggingface.co/microsoft/harrier-oss-v1-27b) @ "
         "`0c0fc62f6d8af9e8604cb818c412301b103a0093`. Target: 24 GB RTX Pro 4000 Blackwell SFF.", "",
         "- **Cosine gate**: per-item cosine vs BF16 on 2,000 eval-derived items (500 each: short English, long docs, "
         "multilingual, code); a variant passes if p5 >= 0.98 in every bucket. Reference: BF16 served by vLLM.",
         "- **Retrieval**: mteb 2.24.9 with Harrier's official mteb prompts and instruction template; inputs truncated to "
         "8,192 tokens for every model; all runs via vLLM 0.31.0 except the sentence-transformers reference.",
         "- **24 GB fit**: vLLM capped to 21.5 GiB (gpu-memory-utilization 0.90 of a 24 GB card) or 22.7 GiB (0.95) on an "
         "RTX PRO 6000 Blackwell; longest single input that loads and embeds, eager mode, chunked prefill 2048.", "",
         "## Summary", "",
         "| Variant | Round | Size on disk | Cosine p5 (worst bucket) | Cosine min | Mean nDCG@10 | Δ vs BF16 | Mean R@10 | Max ctx @0.90 / @0.95 | Status |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    L.append(f"| BF16 · sentence-transformers (reference) | – | {size_gb('/data/models/harrier-oss-v1-27b'):.1f} GB | – | – | "
             f"{f4(mean(st, 'ndcg_at_10'))} | {mean(st, 'ndcg_at_10') - base_nd:+.4f} | {f4(mean(st, 'recall_at_10'))} | does not fit | reference |")
    L.append(f"| BF16 · vLLM | – | {size_gb('/data/models/harrier-oss-v1-27b'):.1f} GB | – | – | {f4(base_nd)} | – | "
             f"{f4(mean(vb, 'recall_at_10'))} | does not fit | reference |")
    for v, m in res["variants"].items():
        g = m["gate"]
        p5 = min(g["buckets"][b]["p5"] for b in BUCKETS) if g else None
        mn = min(g["buckets"][b]["min"] for b in BUCKETS) if g else None
        nd = mean(m["mteb"], "ndcg_at_10")
        fits = m["ctx_095"] is not None and m["ctx_095"] > 0
        if not g:
            status = "not evaluated"
        elif not m["gate_pass"]:
            status = "FAIL (cosine p5 < 0.98)"
        elif not fits:
            status = "PASS quality · does not fit 24 GB"
        else:
            status = "**PASS**"
        L.append(f"| {m['name']} | {m['round']} | {m['size_gb']:.2f} GB | {f4(p5)} | {f4(mn)} | {f4(nd)} | "
                 f"{'–' if nd is None else f'{nd - base_nd:+.4f}'} | {f4(mean(m['mteb'], 'recall_at_10'))} | "
                 f"{fctx(m['ctx_090'])} / {fctx(m['ctx_095'])} | {status} |")
    L += ["", "NVFP4 W4A4 cosine is measured on Blackwell (native FP4 kernels); on Hopper vLLM runs it weight-only, "
          "which does not exercise FP4 activations.", ""]

    L += ["## Per-item cosine vs BF16 (2,000 items)", "",
          "| Comparison | Hardware | short_en p5 | long p5 | multilingual p5 | code p5 | worst min |", "|---|---|---|---|---|---|---|"]
    L.append(f"| BF16 vLLM vs BF16 sentence-transformers (engine drift) | H100 | " +
             " | ".join(f"{drift[b]['p5']:.5f}" for b in BUCKETS) + f" | {min(drift[b]['min'] for b in BUCKETS):.5f} |")
    if hw:
        L.append("| BF16 Blackwell vs BF16 H100 (hardware drift) | both | " +
                 " | ".join(f"{hw['buckets'][b]['p5']:.5f}" for b in BUCKETS) + f" | {min(hw['buckets'][b]['min'] for b in BUCKETS):.5f} |")
    for v, m in res["variants"].items():
        for hwname, c in [("H100", m["cos_h100"]), ("RTX PRO 6000", m["cos_blackwell"])]:
            if c:
                L.append(f"| {m['name']} | {hwname} | " + " | ".join(f"{c['buckets'][b]['p5']:.5f}" for b in BUCKETS) +
                         f" | {min(c['buckets'][b]['min'] for b in BUCKETS):.5f} |")
    L += ["", "Full median / p5 / p1 / min per bucket are in `results.json` and `eval/cosine/*.json`.", ""]

    L += ["## Retrieval (mteb) per task, nDCG@10 / R@10", "",
          "Seen in training = listed in Harrier's `training_datasets` in mteb's ModelMeta (not zero-shot).", ""]
    cols = [("BF16 ST", st), ("BF16 vLLM", vb)] + [(m["name"], m["mteb"]) for m in res["variants"].values() if m["mteb"]]
    L.append("| Task | Seen in training | " + " | ".join(c for c, _ in cols) + " |")
    L.append("|---|---|" + "---|" * len(cols))
    for t in TASK_ORDER:
        seen = (st["tasks"][t]["seen_in_training"]) if st and t in st["tasks"] else None
        cells = [(f"{m['tasks'][t]['ndcg_at_10']:.4f} / {m['tasks'][t]['recall_at_10']:.4f}" if m and t in m["tasks"] else "–")
                 for _, m in cols]
        L.append(f"| {t} | {'yes' if seen else 'no'} | " + " | ".join(cells) + " |")
    L.append("| **Mean** | | " + " | ".join(f"**{f4(mean(m, 'ndcg_at_10'))}** / {f4(mean(m, 'recall_at_10'))}" for _, m in cols) + " |")
    L += [""]

    # code retrieval (MTEB(Code, v1) + RTEB(Code, beta) public), H100 vLLM, same harness
    code = {k: load(EV / f"code-vllm-{k}" / "summary.json") for k in ["bf16", "w4a16-gptq-late12-attn-g64", "w4a16-gptq"]}
    res["code_retrieval"] = code
    if all(code.values()):
        cb = code["bf16"]["tasks"]
        L += ["## Code retrieval: MTEB(Code, v1) + RTEB(Code, beta) public datasets (18 tasks), nDCG@10", "",
              "Same vLLM harness and H100s as the BF16 reference; Harrier's official mteb prompts (for these tasks mteb 2.24 falls back to "
              "the generic instruction \"Retrieve text based on user query.\"), inputs truncated to 8,192 tokens. "
              "RTEB's private Code1Retrieval and JapaneseCode1Retrieval cannot be run locally.", "",
              "| Task | Seen in training | BF16 | g64 | Δ | W4A16 r1 | Δ |", "|---|---|---|---|---|---|---|"]
        for t in sorted(cb):
            g, w = code["w4a16-gptq-late12-attn-g64"]["tasks"][t]["ndcg_at_10"], code["w4a16-gptq"]["tasks"][t]["ndcg_at_10"]
            bb = cb[t]["ndcg_at_10"]
            L.append(f"| {t}{' \\*' if t in TIE_TASKS else ''} | {'yes' if cb[t]['seen_in_training'] else 'no'} | {bb:.4f} | {g:.4f} | {g - bb:+.4f} | {w:.4f} | {w - bb:+.4f} |")
        mean = lambda d, ts: float(np.mean([d[t]["ndcg_at_10"] for t in ts]))
        allt = sorted(cb); rob = [t for t in allt if t not in TIE_TASKS]
        cg, cw = code["w4a16-gptq-late12-attn-g64"]["tasks"], code["w4a16-gptq"]["tasks"]
        for label, ts in [("Mean, all 18", allt), (f"Mean, {len(rob)} tie-robust tasks (excl. *)", rob)]:
            mb, mg, mw = mean(cb, ts), mean(cg, ts), mean(cw, ts)
            L.append(f"| **{label}** | | **{mb:.4f}** | **{mg:.4f}** | **{mg - mb:+.4f}** | **{mw:.4f}** | **{mw - mb:+.4f}** |")
        L += ["", "\\* Tie-dominated tasks: most relevant documents have an exact-text duplicate in the corpus that is not relevant "
              "(CodeTransOceanDL 100%, SyntheticText2SQL 94%, DS1000 93%, CosQA 85%; `eval/tie_affected_tasks.json`). Identical texts get "
              "identical embeddings, so the relevant document's rank among its twins is decided by the search implementation's tie-breaking, "
              "not the model; nDCG@10 on these tasks can move by ~0.15 between engines with identical recall@10. Deltas on them are not "
              "meaningful.", ""]
    L += ["## 24 GB card fit (RTX PRO 6000 Blackwell, vLLM memory capped)", "",
          "| Model | Prefill | CUDA graphs | Budget GiB | max_len | Loads + embeds | KV cache tokens | Time for one max_len input (s) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(ctx, key=lambda r: (r["model"], r.get("chunked", 0), r.get("eager", False), r["budget_gib"], r["max_len"])):
        L.append(f"| {r['model']} | {'chunked 2048' if r.get('chunked') else 'single-shot'} | {'no (eager)' if r.get('eager') else 'yes'} | "
                 f"{r['budget_gib']} | {r['max_len']} | {'yes' if r['ok'] else 'no'} | {r.get('kv_tokens') or '–'} | {r.get('long_s', '–')} |")
    L += [""]

    (ROOT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
