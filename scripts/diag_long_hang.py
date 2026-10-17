#!/usr/bin/env python
"""Diagnose vLLM stalling on long inputs: embed the cosine 'long' bucket under one config.

Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 timeout 1200 python diag_long_hang.py [--no-prefix-cache]
"""
import argparse
import json
import time

ap = argparse.ArgumentParser()
ap.add_argument("--no-prefix-cache", action="store_true")
ap.add_argument("--model", default="/data/models/harrier-oss-v1-27b")
ap.add_argument("--trunc", type=int, default=8192)
ap.add_argument("--no-chunked-prefill", action="store_true")
args = ap.parse_args()

from transformers import AutoTokenizer
from vllm import LLM
from vllm.inputs import TokensPrompt

items = [json.loads(l) for l in open("/data/models/harrier-quant/eval/cosine/items.jsonl")]
items = [i for i in items if i["bucket"] == "long"]
tok = AutoTokenizer.from_pretrained(args.model)
ids = tok([i["text"] for i in items], truncation=True, max_length=args.trunc)["input_ids"]
lens = [len(x) for x in ids]
print("n", len(ids), "len==8192:", sum(l == 8192 for l in lens), "max", max(lens), flush=True)
dup = len(ids) - len({tuple(x) for x in ids})
print("exact duplicate token sequences:", dup, flush=True)

llm = LLM(model=args.model, runner="pooling", max_model_len=8192,
          pooler_config={"seq_pooling_type": "LAST", "use_activation": True},
          gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=not args.no_prefix_cache,
          **({"enable_chunked_prefill": False, "max_num_batched_tokens": 16384} if args.no_chunked_prefill else {}))
t = time.time()
outs = llm.embed([TokensPrompt(prompt_token_ids=x) for x in ids])
print(f"DONE {len(outs)} in {time.time() - t:.1f}s prefix_cache={not args.no_prefix_cache} trunc={args.trunc} chunked={not args.no_chunked_prefill}", flush=True)
