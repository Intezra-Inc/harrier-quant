# harrier-oss-v1-27b quantization: results

## Bottom line

- **Gate result:** one variant passes the agreed gate (per-item cosine p5 >= 0.98 in every bucket) **and** fits a 24 GB card:
  **`w4a16-gptq-late12-attn-g64`** (20.3 GB). Retrieval matches BF16 (mean nDCG@10 +0.0001; worst task −0.0019).
  Context on a 24 GB card is tight: 12k tokens at `--gpu-memory-utilization 0.95` (eager), ~2k at 0.90.
- **Retrieval barely moves for any 4-bit variant.** Every variant that was retrieval-tested is within 0.002 of BF16
  on mean nDCG@10. That includes the round-1 variants that *fail* the cosine gate (W4A16 −0.0019, NVFP4A16 −0.0006),
  and it is the same size as the vLLM vs sentence-transformers difference on identical BF16 weights (−0.0001 on the
  mean, up to ±0.003 per task). The p5 >= 0.98 cosine gate is much stricter than what retrieval quality needs.
- **Code retrieval confirms it.** On MTEB(Code, v1) + RTEB(Code) public (18 tasks), g64 is −0.0015 and round-1 W4A16
  −0.0020 mean nDCG@10 vs BF16. On the 14 tasks that are not dominated by duplicate-document ties: g64 −0.0018,
  round-1 W4A16 −0.0035 (worst single task −0.016 / −0.020). Four code tasks (CodeTransOceanDL, SyntheticText2SQL, DS1000,
  CosQA) are tie-dominated and their per-task deltas are not meaningful (see the code-retrieval section).
- **Practical alternative:** `w4a16-gptq` (round 1, 18.4 GB) fails the cosine gate but has retrieval within 0.002 of
  BF16 and far more context headroom on 24 GB (16k at 0.90, 32k at 0.95). If identical-to-BF16 embeddings are not a
  requirement (for example, the whole corpus is embedded with the quantized model), it is the better deployment choice.
- **Does not fit 24 GB:** `w4a16-gptq-late12-attn-down` (22.8 GB) has the highest fidelity (cosine p5 0.991), but the
  weights alone take 21.8 GiB. It is for 32 GB-class cards (tested end to end at 0.95 x 32 GB with 16k inputs).
- **NVFP4 W4A4 is dropped.** With native FP4 activations on Blackwell, multilingual p5 is 0.924. It is the fastest
  path by far, but the quality cost is clear. NVFP4A16 (FP4 weights only) behaves like W4A16.

## How the passing layout was found

1. Round 1 (FP8 on layers 0-5 and 56-61, 4-bit elsewhere) failed the gate for all three schemes, with nearly
   identical profiles for INT4 GPTQ and data-free NVFP4. This was 4-bit loss, not a recipe or kernel bug.
2. A sensitivity sweep on **calibration data only** (`scripts/sensitivity.py`, `eval/sensitivity/`) showed:
   - the late layers (47-61) matter most and the early layers matter little;
   - `v_proj` and `down_proj` carry the most 4-bit error;
   - FP8 everywhere is near lossless (proxy median 0.9985).
3. Round 2 moved the 12 FP8 layers to 50-61 and put all attention projections in FP8 (vLLM fuses q/k/v, so they stay
   together). Short English, long documents and code passed; multilingual missed (p5 0.977).
4. Round 3: INT4 group size 64 instead of 128 lifted multilingual p5 to 0.987 for +0.27 GB. Adding `down_proj` in
   FP8 also passes, but at 22.8 GB it does not fit a 24 GB card.

## vLLM serve commands

vLLM 0.31.0. The g64 (8k and 12k at 0.95 x 24 GB) and late12-attn-down (16k at 0.95 x 32 GB) commands were tested end to end through the OpenAI-compatible server (`scripts/serve_test.sh`). The `w4a16-gptq` configuration was verified with the offline engine and the same flags (`scripts/context_probe.py`) but not through `vllm serve`. All tests ran on a Blackwell RTX PRO 6000 with memory capped to the target card.
Clients should truncate inputs to the stated length; `--max-model-len` is set one higher on purpose (vLLM 0.31
stalls when a pooling request's length equals `max_model_len` with chunked prefill on).

```bash
# Recommended for 24 GB (passes the gate): 8k inputs, dedicated card
vllm serve w4a16-gptq-late12-attn-g64 --runner pooling --convert embed \
  --pooler-config '{"seq_pooling_type": "LAST", "use_activation": true}' \
  --max-model-len 8193 --enable-chunked-prefill --max-num-batched-tokens 2048 \
  --enforce-eager --gpu-memory-utilization 0.95

# Practical 24 GB alternative (fails the cosine gate; retrieval within 0.002 of BF16): 16k inputs at 0.90
vllm serve w4a16-gptq --runner pooling --convert embed \
  --pooler-config '{"seq_pooling_type": "LAST", "use_activation": true}' \
  --max-model-len 16385 --enable-chunked-prefill --max-num-batched-tokens 2048 \
  --enforce-eager --gpu-memory-utilization 0.90

# 32 GB-class cards only (passes the gate, highest fidelity)
vllm serve w4a16-gptq-late12-attn-down --runner pooling --convert embed \
  --pooler-config '{"seq_pooling_type": "LAST", "use_activation": true}' \
  --max-model-len 16385 --enable-chunked-prefill --max-num-batched-tokens 2048 \
  --enforce-eager --gpu-memory-utilization 0.95

# Others (failed the gate; listed for completeness): nvfp4a16 and w4a16-gptq-late12-attn use the same flags with
# --max-model-len 16385 at --gpu-memory-utilization 0.95 (neither fits 8k / 4k at 0.90); nvfp4 (W4A4) needs Blackwell for FP4 activations.
```

Query format: `Instruct: {one-sentence task}\nQuery: {query}`; documents get no instruction. Pooling: last token
(the tokenizer appends `<eos>`), then L2 normalization. Draft Hugging Face cards: `w4a16-gptq-late12-attn-g64/README.md`
and `w4a16-gptq-late12-attn-down/README.md` (not uploaded).

## Caveats

- 24 GB fit was measured on a 96 GB RTX PRO 6000 Max-Q with vLLM's memory capped, not on a real RTX Pro 4000.
- SciFact, NFCorpus and FiQA2018 are in Harrier's training data (mteb metadata). The others are zero-shot.
- Retrieval inputs were truncated to 8,192 tokens for every model.
- vLLM 0.31 workarounds are documented in `COMMANDS.md`: fused-module targets, the stall at `len == max_model_len`,
  the teardown hang, and transformers' gemma3_text processor mapping.

---

