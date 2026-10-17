#!/usr/bin/env python
"""Emulate a 24 GB card: cap vLLM to --budget-gib and test whether --max-len works end to end.

Starts vLLM with gpu_memory_utilization = budget / total_memory (same serving flags as the
recommended serve command: pooling runner, LAST pooling, chunked prefill off), embeds one input
of exactly --max-len tokens plus a batch of 32 short inputs, and prints one RESULT JSON line.
CUDA is not touched in this process before vLLM starts (total memory comes from nvidia-smi),
so vLLM keeps its default worker start method.

Usage: CUDA_VISIBLE_DEVICES=0 python context_probe.py --model <path> --max-len 16384 --budget-gib 21.5
"""
import argparse
import json
import os
import subprocess
import sys
import time


def total_gib():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.check_output(["nvidia-smi", "-i", dev, "--query-gpu=memory.total",
                                   "--format=csv,noheader,nounits"], text=True)
    return int(out.strip()) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--budget-gib", type=float, default=21.5)
    ap.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs (saves their memory)")
    ap.add_argument("--chunked", type=int, default=0,
                    help="enable chunked prefill with this max_num_batched_tokens; max_model_len is set to "
                         "max_len+1 so a max_len-token input never equals max_model_len (vLLM 0.31 stall bug)")
    args = ap.parse_args()

    util = args.budget_gib / total_gib()
    res = dict(model=args.model, max_len=args.max_len, budget_gib=args.budget_gib, eager=args.enforce_eager, chunked=args.chunked, util=round(util, 4), ok=False)
    try:
        from vllm import LLM
        from vllm.inputs import TokensPrompt
        if args.chunked:
            sched = dict(max_model_len=args.max_len + 1, enable_chunked_prefill=True,
                         max_num_batched_tokens=args.chunked)
        else:
            sched = dict(max_model_len=args.max_len, enable_chunked_prefill=False,
                         max_num_batched_tokens=args.max_len)
        llm = LLM(model=args.model, runner="pooling",
                  pooler_config={"seq_pooling_type": "LAST", "use_activation": True},
                  gpu_memory_utilization=util, seed=0, enforce_eager=args.enforce_eager, **sched)
        long = [TokensPrompt(prompt_token_ids=[2] + [1000 + (i % 5000) for i in range(args.max_len - 2)] + [1])]
        t = time.time()
        out = llm.embed(long, use_tqdm=False)
        res["long_s"] = round(time.time() - t, 2)
        short = [TokensPrompt(prompt_token_ids=[2] + [1000 + j] * 64 + [1]) for j in range(32)]
        t = time.time()
        llm.embed(short, use_tqdm=False)
        res["short32_s"] = round(time.time() - t, 2)
        res["dim"] = len(out[0].outputs.embedding)
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print("RESULT " + json.dumps(res), flush=True)
    sys.path.insert(0, os.path.dirname(__file__))
    from hard_exit import hard_exit
    hard_exit(0)


if __name__ == "__main__":
    main()
