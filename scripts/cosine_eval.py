#!/usr/bin/env python
"""Per-item cosine similarity of quantized variants vs the BF16 reference.

  build   : sample ~2,000 items (seed 42) from the eval tasks into eval/cosine/items.jsonl,
            500 per bucket: short_en, long, multilingual, code. Queries carry Harrier's
            mteb prompt; documents carry none (same formatting as run_mteb.py).
  embed   : embed items.jsonl with one engine/model -> eval/cosine/<label>.npy (float32)
  compare : per-bucket median / p5 / p1 / min cosine of <label> vs <ref>, plus pass/fail
            (fail if p5 < 0.98 in any bucket) -> eval/cosine/<label>_vs_<ref>.json

Usage:
  python cosine_eval.py build
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python cosine_eval.py embed --engine vllm --model <path> --label bf16-vllm
  python cosine_eval.py compare --label w4a16-gptq --ref bf16-vllm
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

ROOT = Path(os.environ.get("HQ_ROOT", "/data/models/harrier-quant"))  # override on other hosts
OUT = ROOT / "eval" / "cosine"
SEED = 42
PER_BUCKET = 500
MAX_LEN = 8192
P5_THRESHOLD = 0.98
UPSTREAM = "microsoft/harrier-oss-v1-27b"


def doc_text(row):
    title = (row.get("title") or "").strip()
    return f"{title} {row['text']}".strip() if title else row["text"]


def build(args):
    import mteb
    from mteb.types import PromptType
    from run_mteb import TASKS, VLLMHarrier
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/data/models/harrier-oss-v1-27b")
    meta = mteb.get_model_meta(UPSTREAM)
    # Borrow the instruction logic without loading a model.
    prompter = VLLMHarrier.__new__(VLLMHarrier)
    kw = meta.loader_kwargs
    prompter.instruction_template = kw["instruction_template"]
    prompter.prompts_dict = kw.get("prompts_dict")
    prompter.apply_instruction_to_passages = False

    rng = random.Random(SEED)
    ntok = lambda s: len(tok(s, add_special_tokens=True)["input_ids"])

    def load(name, **kwargs):
        t = mteb.get_task(name, **kwargs)
        t.load_data()
        qp = prompter.get_task_instruction(t.metadata, PromptType.query) or ""
        if t.dataset is None:
            # Legacy layout (e.g. LEMB): task.corpus[split] = {id: {text, title?}}, task.queries[split] = {id: text}
            # Multilingual legacy tasks nest one more level: task.corpus[subset][split].
            def is_split_level(m):
                v = next(iter(m.values()))
                return isinstance(v, (str, dict)) and (isinstance(v, str) or "text" in v)

            nested = {"default": t.corpus} if is_split_level(next(iter(t.corpus.values()))) else t.corpus
            qnested = {"default": t.queries} if "default" in nested else t.queries
            for subset in nested:
                for split in nested[subset]:
                    corpus = nested[subset][split]
                    d = dict(corpus=[dict(id=k, **(v if isinstance(v, dict) else {"text": v})) for k, v in corpus.items()],
                             queries=[dict(id=k, text=v) for k, v in qnested[subset][split].items()])
                    yield name, subset, d, qp
            return
        for subset, splits in t.dataset.items():
            for split, d in splits.items():
                yield name, subset, d, qp

    pools = {b: [] for b in ["short_en", "long", "multilingual", "code"]}

    def add(bucket, task, subset, role, text, src_id):
        pools[bucket].append(dict(bucket=bucket, task=task, subset=subset, role=role, src_id=src_id, text=text))

    english = ["SciFact", "NFCorpus", "FiQA2018", "ArguAna", "SCIDOCS"]
    for name in english:
        for task, subset, d, qp in load(name, **TASKS[name][0]):
            for r in d["queries"]:
                add("short_en", task, subset, "query", qp + r["text"], r["id"])
            for r in d["corpus"]:
                add("short_en", task, subset, "doc", doc_text(r), r["id"])
    for task, subset, d, qp in load("LEMBSummScreenFDRetrieval"):
        for r in d["corpus"]:
            add("long", task, subset, "doc", doc_text(r), r["id"])
    for name in ["BelebeleRetrieval", "MLQARetrieval"]:
        for task, subset, d, qp in load(name, **TASKS[name][0]):
            q_lang, d_lang = subset.split("-")
            if not q_lang.startswith("eng"):
                for r in d["queries"]:
                    add("multilingual", task, subset, "query", qp + r["text"], r["id"])
            if not d_lang.startswith("eng"):
                for r in d["corpus"]:
                    add("multilingual", task, subset, "doc", doc_text(r), r["id"])
    for task, subset, d, qp in load("CodeSearchNetRetrieval"):
        for r in d["queries"]:
            add("code", task, subset, "query", qp + r["text"], r["id"])
        for r in d["corpus"]:
            add("code", task, subset, "doc", doc_text(r), r["id"])

    items = []
    for bucket, pool in pools.items():
        # de-duplicate identical texts (multilingual subsets share passages), stable order
        seen, uniq = set(), []
        for it in pool:
            if it["text"] not in seen:
                seen.add(it["text"]); uniq.append(it)
        rng.shuffle(uniq)
        if bucket == "short_en":
            # balance queries and docs; "short" = under 512 tokens
            picked = []
            for role in ["query", "doc"]:
                cand = [it for it in uniq if it["role"] == role]
                for it in cand:
                    if len([p for p in picked if p["role"] == role]) >= PER_BUCKET // 2:
                        break
                    if ntok(it["text"]) < 512:
                        picked.append(it)
        elif bucket == "long":
            # long = at least 1,024 tokens (inputs are truncated to MAX_LEN like the mteb runs)
            picked = [it for it in uniq if ntok(it["text"]) >= 1024][:PER_BUCKET]
            if len(picked) < PER_BUCKET:
                extra = [it for it in pools["short_en"] if it["role"] == "doc" and ntok(it["text"]) >= 1024]
                picked += [dict(it, bucket="long") for it in extra[: PER_BUCKET - len(picked)]]
        else:
            q = [it for it in uniq if it["role"] == "query"][: PER_BUCKET // 2]
            d = [it for it in uniq if it["role"] == "doc"][: PER_BUCKET - len(q)]
            picked = q + d
        for it in picked:
            it["n_tokens"] = min(ntok(it["text"]), MAX_LEN)
        items += picked
        print(bucket, len(picked), "queries", sum(i["role"] == "query" for i in picked),
              "median tokens", int(np.median([i["n_tokens"] for i in picked])))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "items.jsonl", "w") as f:
        for i, it in enumerate(items):
            it["idx"] = i
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("total", len(items))


def embed(args):
    items = [json.loads(l) for l in open(OUT / "items.jsonl")]
    texts = [it["text"] for it in items]
    if args.engine == "vllm":
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.inputs import TokensPrompt
        tok = AutoTokenizer.from_pretrained(args.model)
        ids = tok(texts, truncation=True, max_length=MAX_LEN)["input_ids"]
        llm = LLM(model=args.model, runner="pooling", max_model_len=MAX_LEN,
                  pooler_config={"seq_pooling_type": "LAST", "use_activation": True},
                  gpu_memory_utilization=0.85, seed=0,
                  # see run_mteb.py: chunked prefill stalls on max_model_len-length pooling requests
                  enable_chunked_prefill=False, max_num_batched_tokens=16384)
        outs = llm.embed([TokensPrompt(prompt_token_ids=i) for i in ids])
        emb = np.asarray([o.outputs.embedding for o in outs], dtype=np.float32)
    else:
        import st_compat  # noqa: F401
        import torch
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(args.model, device="cuda", model_kwargs={"dtype": torch.bfloat16})
        st.max_seq_length = MAX_LEN
        emb = st.encode(texts, batch_size=8, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    np.save(OUT / f"{args.label}.npy", emb)
    print("saved", emb.shape)
    sys.stdout.flush()
    from hard_exit import hard_exit
    hard_exit(0)  # vLLM teardown can hang; also kills the EngineCore child


def compare(args):
    items = [json.loads(l) for l in open(OUT / "items.jsonl")]
    a, b = np.load(OUT / f"{args.label}.npy"), np.load(OUT / f"{args.ref}.npy")
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    buckets = np.array([it["bucket"] for it in items])
    res = {"label": args.label, "ref": args.ref, "threshold_p5": P5_THRESHOLD, "buckets": {}}
    for bk in ["short_en", "long", "multilingual", "code", "all"]:
        c = cos if bk == "all" else cos[buckets == bk]
        res["buckets"][bk] = dict(n=int(len(c)), median=float(np.median(c)), p5=float(np.percentile(c, 5)),
                                  p1=float(np.percentile(c, 1)), min=float(c.min()),
                                  argmin=int(np.flatnonzero(cos == c.min())[0]))
    res["pass"] = all(v["p5"] >= P5_THRESHOLD for k, v in res["buckets"].items() if k != "all")
    (OUT / f"{args.label}_vs_{args.ref}.json").write_text(json.dumps(res, indent=2))
    print(f"{args.label} vs {args.ref}: {'PASS' if res['pass'] else 'FAIL (p5 < 0.98)'}")
    for k, v in res["buckets"].items():
        print(f"  {k:13s} n={v['n']:4d} median={v['median']:.5f} p5={v['p5']:.5f} p1={v['p1']:.5f} min={v['min']:.5f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    e = sub.add_parser("embed")
    e.add_argument("--engine", choices=["vllm", "st"], required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--label", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--label", required=True)
    c.add_argument("--ref", required=True)
    args = ap.parse_args()
    {"build": build, "embed": embed, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
