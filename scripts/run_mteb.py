#!/usr/bin/env python
"""Run the Harrier retrieval eval suite with either sentence-transformers (reference) or vLLM.

Both engines use mteb's official Harrier ModelMeta (microsoft/harrier-oss-v1-27b @ 0c0fc62):
same instruction template ("Instruct: {task}\\nQuery:"), same per-task prompts, no
instruction on documents, prompt concatenated directly onto the text (as sentence-transformers
does), same HF tokenizer call with truncation to --max-len. The vLLM engine receives token ids,
so both engines see byte-identical inputs; only the forward pass differs.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run_mteb.py --engine st \
      --model /data/models/harrier-oss-v1-27b --label bf16-st
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run_mteb.py --engine vllm \
      --model /data/models/harrier-oss-v1-27b --label bf16-vllm
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import st_compat  # noqa: F401,E402  (see st_compat.py)

import mteb  # noqa: E402
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel  # noqa: E402
from mteb.types import PromptType  # noqa: E402

UPSTREAM = "microsoft/harrier-oss-v1-27b"
ROOT = Path(os.environ.get("HQ_ROOT", "/data/models/harrier-quant"))  # override on other hosts

# Languages: same-language plus X->eng and eng->X pairs.
BELEBELE_LANGS = ["arb_Arab", "deu_Latn", "spa_Latn", "jpn_Jpan", "hin_Deva", "zho_Hans",
                  "rus_Cyrl", "kor_Hang", "fra_Latn", "por_Latn", "swh_Latn", "tha_Thai"]
MLQA_LANGS = ["ara", "deu", "spa", "hin", "vie", "zho"]


def belebele_subsets():
    s = ["eng_Latn-eng_Latn"]
    for l in BELEBELE_LANGS:
        s += [f"{l}-{l}", f"{l}-eng_Latn", f"eng_Latn-{l}"]
    return s


def mlqa_subsets():
    s = ["eng-eng"]
    for l in MLQA_LANGS:
        s += [f"{l}-{l}", f"{l}-eng", f"eng-{l}"]
    return s


# name -> (kwargs for mteb.get_task, seen_in_harrier_training per mteb ModelMeta)
TASKS = {
    "SciFact": ({}, True),
    "NFCorpus": ({}, True),
    "FiQA2018": ({}, True),
    "ArguAna": ({}, False),
    "SCIDOCS": ({}, False),
    "LEMBSummScreenFDRetrieval": ({}, False),
    "BelebeleRetrieval": ({"hf_subsets": belebele_subsets()}, False),
    "MLQARetrieval": ({"hf_subsets": mlqa_subsets(), "eval_splits": ["test"]}, False),
}


def code_suite():
    """MTEB(Code, v1) + RTEB(Code, beta) public datasets: the same 18 tasks as the Blackwell embed-eval run."""
    code = [t.metadata.name for t in mteb.get_benchmark("MTEB(Code, v1)").tasks]
    rteb = [t.metadata.name for t in mteb.get_benchmark("RTEB(Code, beta)").tasks if t.metadata.is_public]
    return list(dict.fromkeys(code + rteb))


def get_tasks(names):
    return [mteb.get_task(n, **TASKS[n][0]) if n in TASKS else mteb.get_task(n) for n in names]


def seen_in_training(name):
    td = mteb.get_model_meta(UPSTREAM).training_datasets or set()
    return name in td


class VLLMHarrier(InstructSentenceTransformerModel):
    """mteb encoder backed by vLLM. Reuses the parent's instruction resolution unchanged."""

    def __init__(self, model_path, max_len, loader_kwargs, gpu_mem=0.85):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.instruction_template = loader_kwargs["instruction_template"]
        self.apply_instruction_to_passages = loader_kwargs.get("apply_instruction_to_passages", False)
        self.prompts_dict = loader_kwargs.get("prompts_dict")
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_path)
        # Chunked prefill disabled: vLLM 0.31 stalls on pooling requests whose length equals
        # max_model_len when chunked prefill is on (see logs/diag_long_*.log).
        self.llm = LLM(model=model_path, runner="pooling", max_model_len=max_len,
                       pooler_config={"seq_pooling_type": "LAST", "use_activation": True},
                       gpu_memory_utilization=gpu_mem, seed=0,
                       enable_chunked_prefill=False, max_num_batched_tokens=max(16384, max_len))

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        from vllm.inputs import TokensPrompt
        instruction = self.get_task_instruction(task_metadata, prompt_type)
        if not self.apply_instruction_to_passages and prompt_type == PromptType.document:
            instruction = None
        sentences = [t for batch in inputs for t in batch["text"]]
        prefix = instruction or ""
        ids = self.tok([prefix + s for s in sentences], truncation=True, max_length=self.max_len)["input_ids"]
        outs = self.llm.embed([TokensPrompt(prompt_token_ids=i) for i in ids], use_tqdm=False)
        return np.asarray([o.outputs.embedding for o in outs], dtype=np.float32)


def build_encoder(engine, model_path, label, max_len, batch_size):
    meta = mteb.get_model_meta(UPSTREAM)
    kw = dict(meta.loader_kwargs)
    if engine == "st":
        enc = InstructSentenceTransformerModel(
            model_name=model_path, revision=None, device="cuda", max_seq_length=max_len,
            model_kwargs={"dtype": torch.bfloat16}, **kw)
    else:
        enc = VLLMHarrier(model_path, max_len, kw)
    # Distinct name so each run gets its own results folder; everything else from upstream meta.
    enc.mteb_model_meta = meta.model_copy(update={"name": f"harrier-quant/{label}", "revision": label})
    return enc


def summarize(results, names):
    rows = {}
    for tr in results.task_results:
        scores = [s for split in tr.scores.values() for s in split]
        agg = lambda k: float(np.mean([s[k] for s in scores if k in s]))
        rows[tr.task_name] = dict(ndcg_at_10=agg("ndcg_at_10"), recall_at_10=agg("recall_at_10"),
                                  n_subsets=len(scores), seen_in_training=seen_in_training(tr.task_name),
                                  per_subset={s.get("hf_subset", "default"): dict(ndcg_at_10=s.get("ndcg_at_10"),
                                                                                    recall_at_10=s.get("recall_at_10"))
                                              for s in scores})
    return {n: rows[n] for n in names if n in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["st", "vllm"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", nargs="*", default=None, help="default: the --suite task list")
    ap.add_argument("--suite", choices=["quant", "code"], default="quant",
                    help="quant: the 8-task quantization eval; code: MTEB(Code, v1) + RTEB(Code, beta) public")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    out = ROOT / "eval" / args.label
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    enc = build_encoder(args.engine, args.model, args.label, args.max_len, args.batch_size)
    args.tasks = args.tasks or (code_suite() if args.suite == "code" else list(TASKS))
    tasks = get_tasks(args.tasks)
    res = mteb.evaluate(enc, tasks, cache=mteb.ResultCache(cache_path=ROOT / "eval" / "mteb_cache"),
                        encode_kwargs={"batch_size": args.batch_size}, overwrite_strategy="only-missing",
                        co2_tracker=False)
    summary = dict(label=args.label, engine=args.engine, model=args.model, max_len=args.max_len,
                   minutes=round((time.time() - t0) / 60, 1), tasks=summarize(res, args.tasks))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for n, r in summary["tasks"].items():
        print(f"{n:28s} nDCG@10={r['ndcg_at_10']:.4f} R@10={r['recall_at_10']:.4f} subsets={r['n_subsets']}")
    sys.stdout.flush()
    # vLLM 0.31 engine teardown can hang after all work is done; results are already on disk.
    from hard_exit import hard_exit
    hard_exit(0)  # vLLM teardown can hang; also kills the EngineCore child


if __name__ == "__main__":
    main()
