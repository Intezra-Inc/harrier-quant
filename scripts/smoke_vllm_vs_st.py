#!/usr/bin/env python
"""Smoke test: vLLM pooling vs sentence-transformers on identical token ids.

Usage: CUDA_VISIBLE_DEVICES=1 python smoke_vllm_vs_st.py --model <path> [--st]
"""
import argparse
import gc
import json

import numpy as np
import torch

TEXTS = [
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:how much protein should a female eat",
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:summit define",
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a + b\n",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--st", action="store_true", help="also run sentence-transformers reference")
    ap.add_argument("--max-len", type=int, default=8192)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = [tok(t, truncation=True, max_length=args.max_len)["input_ids"] for t in TEXTS]
    print("first/last tokens:", [(tok.convert_ids_to_tokens(i[0]), tok.convert_ids_to_tokens(i[-1])) for i in ids])

    from vllm import LLM
    from vllm.inputs import TokensPrompt
    llm = LLM(model=args.model, runner="pooling", max_model_len=args.max_len,
              pooler_config={"seq_pooling_type": "LAST", "use_activation": True},
              gpu_memory_utilization=0.85, seed=0)
    outs = llm.embed([TokensPrompt(prompt_token_ids=i) for i in ids])
    v = np.array([o.outputs.embedding for o in outs], dtype=np.float32)
    print("vllm shape", v.shape, "norms", np.linalg.norm(v, axis=1).round(4))
    print("vllm query-doc scores\n", (v[:2] @ v[2:4].T * 100).round(2))
    res = {"vllm_scores": (v[:2] @ v[2:4].T * 100).round(3).tolist()}

    if args.st:
        del llm; gc.collect(); torch.cuda.empty_cache()
        import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); import st_compat  # noqa: F401
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(args.model, device="cuda", model_kwargs={"dtype": torch.bfloat16})
        st.max_seq_length = args.max_len
        s = st.encode(TEXTS, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        cos = (s * v).sum(1) / (np.linalg.norm(s, axis=1) * np.linalg.norm(v, axis=1))
        print("st norms", np.linalg.norm(s, axis=1).round(4))
        print("st query-doc scores\n", (s[:2] @ s[2:4].T * 100).round(2))
        print("cosine vLLM vs ST per text:", cos.round(5))
        res.update(st_scores=(s[:2] @ s[2:4].T * 100).round(3).tolist(), cos_vllm_st=cos.round(6).tolist())
    print(json.dumps(res))


if __name__ == "__main__":
    main()
