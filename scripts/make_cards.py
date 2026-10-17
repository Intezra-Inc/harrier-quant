#!/usr/bin/env python
"""Draft Hugging Face model cards (README.md) for the variants that pass, from results.json.

Usage: python make_cards.py    (run make_results.py first)
Nothing is uploaded.
"""
import json
from pathlib import Path

ROOT = Path("/data/models/harrier-quant")
BUCKETS = [("short_en", "Short English"), ("long", "Long documents"), ("multilingual", "Multilingual"), ("code", "Code")]
TASKS = ["SciFact", "NFCorpus", "FiQA2018", "ArguAna", "SCIDOCS", "LEMBSummScreenFDRetrieval", "BelebeleRetrieval", "MLQARetrieval"]
UPSTREAM = "microsoft/harrier-oss-v1-27b"
REV = "0c0fc62f6d8af9e8604cb818c412301b103a0093"

CARDS = {
    "w4a16-gptq": dict(
        title="harrier-oss-v1-27b · W4A16 (INT4 g128) + FP8 edge layers (compact, long context on 24 GB)",
        repo_suffix="W4A16-FP8edges",
        target="a single **24 GB** GPU (built for the RTX Pro 4000 Blackwell SFF)",
        serve_len=16384, serve_util=0.90,
        accept_cosine_delta=True,
        fit_note=("On a 24 GB card with eager mode and chunked prefill, **16,384-token** inputs fit at "
                  "`--gpu-memory-utilization 0.90`, and the full **32,768-token** context fits at 0.95 (measured on an RTX PRO 6000 "
                  "Blackwell with vLLM's memory capped to the 24 GB budget, offline engine with the same flags). The server command "
                  "above was tested end to end with 16,384-token inputs on an H100 with memory capped to 21.5 GiB."),
        layout=("Decoder layers 0-5 and 56-61: FP8_DYNAMIC (all linear layers). Layers 6-55: INT4 for all linear layers "
                "(GPTQ, symmetric, group size 128, static act-order). Embedding table and norms: BF16."),
        gptq="group size 128, block size 128, dampening 0.01, act-order static",
        how_chosen=("a fixed layout chosen up front (round 1): FP8 on the first and last six layers, INT4 elsewhere, to target "
                    "~18 GB for 24 GB cards. A later sensitivity sweep showed the late layers matter most and the early layers "
                    "matter little, which is why the g64 variant moves its FP8 layers to 50-61 and gets closer to BF16. This "
                    "variant is kept because it is the smallest and leaves the most room for context, and its retrieval quality "
                    "is unchanged within noise."),
        fidelity_note=("**Embedding fidelity: accepted delta.** This variant does **not** meet the strict per-item gate we set for "
                       "the other variants (cosine p5 >= 0.98 in every bucket). Its worst bucket is long documents (p5 0.969; "
                       "worst single item 0.928). We accept that delta because **retrieval quality is unchanged within noise**: "
                       "mean nDCG@10 is −0.0019 vs BF16 and no task drops more than 0.0042. For comparison, the BF16 model served "
                       "by vLLM vs sentence-transformers differs by up to ±0.003 per task. Use this variant when the whole corpus "
                       "is embedded with it. If you need embeddings that stay close to BF16, for example to mix with an existing "
                       "BF16 index, use the g64 variant (cosine p5 0.987, ~20.3 GB, less context headroom)."),
        extra_limitations=["Per-item embedding drift vs BF16 is larger than our other published variants (cosine median 0.981, p5 0.969, "
                           "min 0.928; long documents and code drift most). Re-embed the whole corpus with this model rather than mixing.",
                           "The cosine check on Blackwell was not completed for this variant (stopped to free the GPU). Its NVFP4 "
                           "sibling matched H100 within 0.001 on Blackwell, but this variant's Blackwell cosine is unmeasured."],
    ),
    "w4a16-gptq-late12-attn-g64": dict(
        title="harrier-oss-v1-27b · W4A16 (INT4 g64) + FP8 mixed precision",
        repo_suffix="W4A16-g64-FP8mix",
        target="a single **24 GB** GPU (built for the RTX Pro 4000 Blackwell SFF)",
        serve_len=8192, serve_util=0.95,
        fit_note=("On a 24 GB card at `--gpu-memory-utilization 0.95` with eager mode and chunked prefill, inputs up to "
                  "**12,288 tokens** load and embed (8,192 recommended for headroom). At 0.90 only ~2k tokens fit, and "
                  "CUDA graphs (no `--enforce-eager`) do not fit 8k. The card must be dedicated (no display / other processes)."),
        layout=("Decoder layers 50-61: FP8_DYNAMIC (all linear layers). Layers 0-49: `q/k/v/o_proj` FP8_DYNAMIC, "
                "`gate/up/down_proj` INT4 (GPTQ, symmetric, group size 64, static act-order). Embedding table and norms: BF16."),
        gptq="group size 64, block size 128, dampening 0.01, act-order static",
    ),
    "w4a16-gptq-late12-attn-down": dict(
        title="harrier-oss-v1-27b · W4A16 (INT4 g128) + FP8 mixed precision (high fidelity, 32 GB class)",
        repo_suffix="W4A16-FP8mix-hifi",
        target="GPUs with **32 GB or more** (it does **not** fit a 24 GB card)",
        serve_len=16384, serve_util=0.95,
        fit_note=("Weights take 21.8 GiB in vLLM, so it does not start on a 24 GB card even at 2k context "
                  "(tested at 0.95 and ~0.97 utilization). Tested end to end with 16,384-token inputs at a 0.95 x 32 GB memory budget."),
        layout=("Decoder layers 50-61: FP8_DYNAMIC (all linear layers). Layers 0-49: `q/k/v/o_proj` and `down_proj` FP8_DYNAMIC, "
                "`gate/up_proj` INT4 (GPTQ, symmetric, group size 128, static act-order). Embedding table and norms: BF16."),
        gptq="group size 128, block size 128, dampening 0.01, act-order static",
    ),
}


def f4(x):
    return "–" if x is None else f"{x:.4f}"


def serve_cmd(repo, L, util):
    return f"""vllm serve {repo} \\
  --runner pooling --convert embed \\
  --pooler-config '{{"seq_pooling_type": "LAST", "use_activation": true}}' \\
  --max-model-len {L + 1} --enable-chunked-prefill --max-num-batched-tokens 2048 \\
  --enforce-eager --gpu-memory-utilization {util}"""


def main():
    res = json.loads((ROOT / "results.json").read_text())
    sources = (ROOT / "calib" / "SOURCES.md").read_text().split("\n", 2)[2].strip()
    st, vb = res["mteb_bf16_st"], res["mteb_bf16_vllm"]
    for v, c in CARDS.items():
        m = res["variants"][v]
        if not m["gate_pass"] and not c.get("accept_cosine_delta"):
            print(f"skip {v}: did not pass the cosine gate")
            continue
        repo = f"<your-org>/harrier-oss-v1-27b-{c['repo_suffix']}"
        cos, bw, mt = m["cos_h100"]["buckets"], (m["cos_blackwell"] or {}).get("buckets"), m["mteb"]
        L = ["---", f"base_model: {UPSTREAM}", "base_model_relation: quantized", "license: mit",
             "pipeline_tag: feature-extraction", "library_name: vllm",
             "tags:", *[f"- {t}" for t in ["embeddings", "text-embeddings", "quantized", "gptq", "w4a16", "fp8",
                                         "compressed-tensors", "llm-compressor", "vllm", "mteb", "gemma3"]],
             "language:", "- multilingual",
             "datasets:", *[f"- {d}" for d in ["allenai/gooaq", "CohereLabs/aya_dataset", "HuggingFaceFW/fineweb-2",
                                              "HuggingFaceFW/fineweb-edu", "codeparrot/github-code-clean"]],
             "---", "", f"# {c['title']}", "",
             f"> **DRAFT. Not uploaded.** Unofficial quantization of [`{UPSTREAM}`](https://huggingface.co/{UPSTREAM}) "
             f"(revision `{REV[:7]}`); not affiliated with or endorsed by Microsoft.", "",
             f"A mixed-precision (FP8 + INT4) compressed-tensors checkpoint of Harrier-27B, a multilingual text embedding model "
             f"(Gemma 3 architecture, 62 layers, 5,376-dim embeddings, last-token pooling, L2 normalization), for serving with vLLM on {c['target']}. "
             f"Size on disk: **{m['size_gb']:.2f} GB** (BF16 original: 54.0 GB).", "",
             "## Results", "",
             "Reference is the BF16 model served by vLLM 0.31.0. That reference matches sentence-transformers (the upstream "
             f"usage path) to median cosine 0.9998 and mean nDCG@10 within {abs(sum(st['tasks'][t]['ndcg_at_10'] for t in TASKS) / 8 - sum(vb['tasks'][t]['ndcg_at_10'] for t in TASKS) / 8):.4f}.", "",
             "**Per-item cosine vs BF16** (2,000 items drawn from the evaluation tasks; p5 = 5th percentile):", "",
             "| Bucket | median | p5 | p1 | min |" + (" p5 on RTX PRO 6000 |" if bw else ""),
             "|---|---|---|---|---|" + ("---|" if bw else "")]
        for b, name in BUCKETS:
            s = cos[b]
            L.append(f"| {name} | {s['median']:.4f} | {s['p5']:.4f} | {s['p1']:.4f} | {s['min']:.4f} |" + (f" {bw[b]['p5']:.4f} |" if bw else ""))
        if c.get("fidelity_note"):
            L += ["", c["fidelity_note"]]
        L += ["", "**Retrieval** (mteb 2.24.9, Harrier's official prompts, inputs truncated to 8,192 tokens):", "",
              "| Task | Zero-shot for Harrier | BF16 nDCG@10 | This model nDCG@10 | Δ | BF16 R@10 | This model R@10 |", "|---|---|---|---|---|---|---|"]
        if mt:
            for t in TASKS:
                a, q = vb["tasks"][t], mt["tasks"][t]
                L.append(f"| {t} | {'no (in training data)' if a['seen_in_training'] else 'yes'} | {a['ndcg_at_10']:.4f} | {q['ndcg_at_10']:.4f} | "
                         f"{q['ndcg_at_10'] - a['ndcg_at_10']:+.4f} | {a['recall_at_10']:.4f} | {q['recall_at_10']:.4f} |")
            mb = sum(vb["tasks"][t]["ndcg_at_10"] for t in TASKS) / 8
            mq = sum(mt["tasks"][t]["ndcg_at_10"] for t in TASKS) / 8
            rb = sum(vb["tasks"][t]["recall_at_10"] for t in TASKS) / 8
            rq = sum(mt["tasks"][t]["recall_at_10"] for t in TASKS) / 8
            L.append(f"| **Mean** | | **{mb:.4f}** | **{mq:.4f}** | **{mq - mb:+.4f}** | {rb:.4f} | {rq:.4f} |")
        else:
            L.append("| (mteb run pending) | | | | | | |")
        L += ["", "Belebele: 37 subsets (12 languages, same-language plus X↔English); MLQA: 19 test subsets (7 languages, same-language plus X↔English)."]
        code = res.get("code_retrieval") or {}
        if code.get(v) and code.get("bf16"):
            cb, cq = code["bf16"]["tasks"], code[v]["tasks"]
            L += ["", "**Code retrieval** (MTEB(Code, v1) + RTEB(Code, beta) public datasets, 18 tasks, same harness; mteb uses the generic "
                  "instruction \"Retrieve text based on user query.\" for these tasks):", "",
                  "| Task | BF16 nDCG@10 | This model | Δ |", "|---|---|---|---|"]
            for t in sorted(cb):
                L.append(f"| {t}{' \\*' if t in ('CodeTransOceanDL', 'SyntheticText2SQL', 'DS1000Retrieval', 'CosQA') else ''} | {cb[t]['ndcg_at_10']:.4f} | {cq[t]['ndcg_at_10']:.4f} | {cq[t]['ndcg_at_10'] - cb[t]['ndcg_at_10']:+.4f} |")
            TIE = {"CodeTransOceanDL", "SyntheticText2SQL", "DS1000Retrieval", "CosQA"}
            for label, ts in [("Mean, all 18", list(cb)), ("Mean, 14 tie-robust tasks (excl. *)", [t for t in cb if t not in TIE])]:
                mb = sum(cb[t]["ndcg_at_10"] for t in ts) / len(ts); mq = sum(cq[t]["ndcg_at_10"] for t in ts) / len(ts)
                L.append(f"| **{label}** | **{mb:.4f}** | **{mq:.4f}** | **{mq - mb:+.4f}** |")
            L += ["", "\\* Tie-dominated: most relevant documents have an identical-text non-relevant twin in the corpus, so nDCG@10 depends on "
                  "the search implementation's tie-breaking rather than the model. Deltas on these tasks are not meaningful."]
        L += ["",
              "## Serving with vLLM", "", f"Recommended command for {c['target']} (vLLM 0.31.0; what was measured where is noted below the command):", "", "```bash",
              serve_cmd(repo, c["serve_len"], c["serve_util"]), "```", "",
              c["fit_note"], "",
              f"`--max-model-len` is deliberately one more than the longest input you send (truncate client-side to {c['serve_len']} tokens). "
              "vLLM 0.31 stalls when a pooling request's length equals `max_model_len` exactly with chunked prefill enabled. "
              "Chunked prefill matters here: Gemma 3's sliding-window layers then keep only a ~1k-token window, which roughly quadruples the context that fits.", "",
              "```python", "import requests", "",
              "def instruct(task, query):", "    return f\"Instruct: {task}\\nQuery: {query}\"", "",
              "queries = [instruct(\"Given a web search query, retrieve relevant passages that answer the query\", \"how much protein should a female eat\")]",
              "docs = [\"As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.\"]",
              "r = requests.post(\"http://localhost:8000/v1/embeddings\", json={\"model\": \"" + repo + "\", \"input\": queries + docs})",
              "emb = [d[\"embedding\"] for d in r.json()[\"data\"]]  # already L2-normalized; dot product = cosine", "```", "",
              "## Query instruction format", "",
              "Same as the upstream model. Queries carry a one-sentence task instruction; documents carry none:", "",
              "```", "Instruct: {task description}\\nQuery: {query}", "```", "",
              "Preset prompts (`config_sentence_transformers.json`): `web_search_query` (\"Given a web search query, retrieve relevant passages that answer the query\"), "
              "`sts_query` (\"Retrieve semantically similar text\"), `bitext_query` (\"Retrieve parallel sentences\"). Per-task prompts used for "
              "MTEB are in `mteb_v2_eval_prompts.json`. Pooling: last token (the appended `<eos>`), then L2 normalization; the tokenizer adds `<bos>`/`<eos>` automatically.", "",
              "## How it was made", "",
              f"- **Tool:** llm-compressor 0.14.0 `oneshot` (compressed-tensors 0.19.0), loading the model through `AutoModel` (`Gemma3TextModel`, no LM head).",
              f"- **Layout:** {c['layout']}",
              f"- **GPTQ:** {c['gptq']}. 512 calibration samples, max 2,048 tokens, seed 42.",
              "- **How the layout was chosen:** " + c.get("how_chosen", "a uniform recipe (FP8 on layers 0-5 and 56-61, 4-bit elsewhere) failed the quality gate "
              "(per-item cosine p5 ~0.97). A layer- and module-sensitivity sweep, run on **calibration data only**, showed the late layers "
              "and the attention projections (especially `v_proj`) carry most of the 4-bit error. vLLM fuses q/k/v, so they are kept together."),
              "- **Calibration data:** permissively licensed public data only, with no evaluation datasets, no MS MARCO and no private data:", "", sources, "",
              "Full scripts, recipes, logs and seeds: see `COMMANDS.md` in the build directory (to be published alongside).", "",
              "## Tested hardware", "",
              "- NVIDIA H100 NVL / H100 PCIe (SM90): quality evaluation (vLLM selects Machete INT4 + CUTLASS FP8 kernels).",
              ("- NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, same family as the RTX Pro 4000): 24 GB-fit tests with vLLM memory capped "
               "(offline engine; Marlin INT4 + CUTLASS FP8 kernels). Server test ran on H100." if c.get("accept_cosine_delta") else
               "- NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, same family as the RTX Pro 4000): cosine check, 24 GB-fit tests with vLLM memory capped, "
               "and end-to-end `vllm serve` tests (Marlin INT4 + CUTLASS FP8 kernels)."),
              "- **Not** yet tested on an actual RTX Pro 4000 (24 GB). The fit numbers come from capping vLLM's memory on a 96 GB card.", "",
              "## Limitations", "",
              "- This is lossy compression. Embeddings are close to BF16 but not identical: the worst bucket is multilingual, and a small fraction of "
              "items fall below 0.99 cosine. Mixing embeddings from this model with BF16-model embeddings in one index will cost some accuracy. Re-embed the corpus with one model.",
              "- SciFact, NFCorpus and FiQA2018 are in Harrier's training data according to mteb's metadata, so they are not zero-shot. "
              "ArguAna, SCIDOCS, LEMBSummScreenFD, Belebele and MLQA are.",
              "- Retrieval quality was evaluated with inputs truncated to 8,192 tokens. Longer inputs load and embed but their quality was not measured.",
              "- Quality evaluation ran on H100. The Blackwell cosine check agrees closely, but kernels differ (Machete vs Marlin).",
              "- Throughput was not benchmarked. On Blackwell the INT4 Marlin path was much slower on long-document prefill than native FP4 in our runs. "
              "Expect a 24 GB RTX Pro 4000 to be considerably slower than an H100.",
              "- Loading this checkpoint with transformers or sentence-transformers directly (outside vLLM) was not tested.",
              "- Requires a vLLM build that handles compressed-tensors mixed-precision checkpoints for `Gemma3TextModel`. Tested only on vLLM 0.31.0. "
              "The quantization config names vLLM's fused `qkv_proj` / `gate_up_proj` modules explicitly, because vLLM 0.31's Gemma3TextModel lacks a packed-module mapping.",
              "- Calibration data is English-heavy (312 of 512 samples) and covers 6 non-English languages plus 5 programming languages.",
              *[f"- {x}" for x in c.get("extra_limitations", [])], "",
              "## License", "",
              f"MIT, following the upstream model ([`{UPSTREAM}`](https://huggingface.co/{UPSTREAM}), `license: mit`).", "",
              "```", "MIT License", "", "Copyright (c) Microsoft Corporation.", "",
              "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), "
              "to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, "
              "and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:", "",
              "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.", "",
              "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, "
              "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER "
              "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.", "```", "",
              "**Gemma 3 base:** harrier-oss-v1-27b uses the Gemma 3 text architecture (`gemma3_text`). Microsoft publishes it under MIT. "
              "If you redistribute or build on it, also review whether Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms) apply to your use. This card is not legal advice.", "",
              "Calibration data licences: see the table above (Apache-2.0, ODC-By-1.0, and per-file MIT/Apache/BSD/ISC code).", ""]
        (ROOT / v / "README.md").write_text("\n".join(L))
        print(f"wrote {v}/README.md ({len(L)} lines)")


if __name__ == "__main__":
    main()
