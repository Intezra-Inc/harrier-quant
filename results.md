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

# Detailed results

Base model: [`microsoft/harrier-oss-v1-27b`](https://huggingface.co/microsoft/harrier-oss-v1-27b) @ `0c0fc62f6d8af9e8604cb818c412301b103a0093`. Target: 24 GB RTX Pro 4000 Blackwell SFF.

- **Cosine gate**: per-item cosine vs BF16 on 2,000 eval-derived items (500 each: short English, long docs, multilingual, code); a variant passes if p5 >= 0.98 in every bucket. Reference: BF16 served by vLLM.
- **Retrieval**: mteb 2.24.9 with Harrier's official mteb prompts and instruction template; inputs truncated to 8,192 tokens for every model; all runs via vLLM 0.31.0 except the sentence-transformers reference.
- **24 GB fit**: vLLM capped to 21.5 GiB (gpu-memory-utilization 0.90 of a 24 GB card) or 22.7 GiB (0.95) on an RTX PRO 6000 Blackwell; longest single input that loads and embeds, eager mode, chunked prefill 2048.

## Summary

| Variant | Round | Size on disk | Cosine p5 (worst bucket) | Cosine min | Mean nDCG@10 | Δ vs BF16 | Mean R@10 | Max ctx @0.90 / @0.95 | Status |
|---|---|---|---|---|---|---|---|---|---|
| BF16 · sentence-transformers (reference) | – | 54.0 GB | – | – | 0.6850 | -0.0001 | 0.7395 | does not fit | reference |
| BF16 · vLLM | – | 54.0 GB | – | – | 0.6851 | – | 0.7396 | does not fit | reference |
| W4A16 g128 · FP8 edges | R1 | 18.42 GB | 0.9689 | 0.9280 | 0.6832 | -0.0019 | 0.7386 | 16k / 32k | FAIL (cosine p5 < 0.98) |
| NVFP4A16 · FP8 edges | R1 | 19.39 GB | 0.9675 | 0.9573 | 0.6845 | -0.0006 | 0.7395 | <8k / 16k | FAIL (cosine p5 < 0.98) |
| NVFP4 W4A4 · FP8 edges | R1 | 19.39 GB | 0.9236 | 0.9015 | – | – | – | not tested / not tested | FAIL (cosine p5 < 0.98) |
| W4A16 g128 · FP8 late12+attn | R2 | 20.02 GB | 0.9771 | 0.9704 | – | – | – | <4k / 16k | FAIL (cosine p5 < 0.98) |
| W4A16 g64 · FP8 late12+attn | R3 | 20.29 GB | 0.9867 | 0.9818 | 0.6852 | +0.0001 | 0.7390 | 2k / 12k | **PASS** |
| W4A16 g128 · FP8 late12+attn+down | R3 | 22.82 GB | 0.9913 | 0.9856 | 0.6831 | -0.0020 | 0.7393 | not tested / <2k | PASS quality · does not fit 24 GB |

NVFP4 W4A4 cosine is measured on Blackwell (native FP4 kernels); on Hopper vLLM runs it weight-only, which does not exercise FP4 activations.

## Per-item cosine vs BF16 (2,000 items)

| Comparison | Hardware | short_en p5 | long p5 | multilingual p5 | code p5 | worst min |
|---|---|---|---|---|---|---|
| BF16 vLLM vs BF16 sentence-transformers (engine drift) | H100 | 0.99982 | 0.99922 | 0.99978 | 0.99971 | 0.99835 |
| BF16 Blackwell vs BF16 H100 (hardware drift) | both | 0.99991 | 0.99958 | 0.99987 | 0.99985 | 0.99840 |
| W4A16 g128 · FP8 edges | H100 | 0.98184 | 0.96886 | 0.97387 | 0.97116 | 0.92802 |
| NVFP4A16 · FP8 edges | H100 | 0.98200 | 0.97589 | 0.96754 | 0.96930 | 0.95731 |
| NVFP4A16 · FP8 edges | RTX PRO 6000 | 0.98183 | 0.97602 | 0.96777 | 0.96933 | 0.95662 |
| NVFP4 W4A4 · FP8 edges | RTX PRO 6000 | 0.97088 | 0.96764 | 0.92363 | 0.95638 | 0.90153 |
| W4A16 g128 · FP8 late12+attn | H100 | 0.99251 | 0.99219 | 0.97706 | 0.98854 | 0.97035 |
| W4A16 g64 · FP8 late12+attn | H100 | 0.99377 | 0.99287 | 0.98669 | 0.98981 | 0.98176 |
| W4A16 g64 · FP8 late12+attn | RTX PRO 6000 | 0.99409 | 0.99338 | 0.98771 | 0.99023 | 0.98355 |
| W4A16 g128 · FP8 late12+attn+down | H100 | 0.99555 | 0.99437 | 0.99126 | 0.99219 | 0.98562 |
| W4A16 g128 · FP8 late12+attn+down | RTX PRO 6000 | 0.99633 | 0.99510 | 0.99288 | 0.99334 | 0.98953 |

Full median / p5 / p1 / min per bucket are in `results.json` and `eval/cosine/*.json`.

## Retrieval (mteb) per task, nDCG@10 / R@10

Seen in training = listed in Harrier's `training_datasets` in mteb's ModelMeta (not zero-shot).

| Task | Seen in training | BF16 ST | BF16 vLLM | W4A16 g128 · FP8 edges | NVFP4A16 · FP8 edges | W4A16 g64 · FP8 late12+attn | W4A16 g128 · FP8 late12+attn+down |
|---|---|---|---|---|---|---|---|
| SciFact | yes | 0.7966 / 0.9406 | 0.7995 / 0.9406 | 0.7956 / 0.9383 | 0.7977 / 0.9372 | 0.7982 / 0.9372 | 0.7968 / 0.9433 |
| NFCorpus | yes | 0.4201 / 0.2056 | 0.4192 / 0.2056 | 0.4192 / 0.2063 | 0.4154 / 0.2066 | 0.4208 / 0.2060 | 0.4190 / 0.2055 |
| FiQA2018 | yes | 0.6223 / 0.6976 | 0.6221 / 0.6978 | 0.6187 / 0.6965 | 0.6134 / 0.6907 | 0.6204 / 0.6923 | 0.6203 / 0.6964 |
| ArguAna | no | 0.6820 / 0.9346 | 0.6814 / 0.9360 | 0.6804 / 0.9317 | 0.6899 / 0.9403 | 0.6831 / 0.9367 | 0.6812 / 0.9374 |
| SCIDOCS | no | 0.2918 / 0.3139 | 0.2907 / 0.3127 | 0.2898 / 0.3117 | 0.2936 / 0.3157 | 0.2894 / 0.3108 | 0.2905 / 0.3129 |
| LEMBSummScreenFDRetrieval | no | 0.9858 / 0.9970 | 0.9858 / 0.9970 | 0.9816 / 0.9970 | 0.9814 / 0.9970 | 0.9838 / 0.9970 | 0.9818 / 0.9970 |
| BelebeleRetrieval | no | 0.9493 / 0.9839 | 0.9495 / 0.9839 | 0.9492 / 0.9843 | 0.9515 / 0.9848 | 0.9492 / 0.9837 | 0.9482 / 0.9838 |
| MLQARetrieval | no | 0.7325 / 0.8432 | 0.7329 / 0.8434 | 0.7315 / 0.8433 | 0.7330 / 0.8437 | 0.7363 / 0.8479 | 0.7270 / 0.8383 |
| **Mean** | | **0.6850** / 0.7395 | **0.6851** / 0.7396 | **0.6832** / 0.7386 | **0.6845** / 0.7395 | **0.6852** / 0.7390 | **0.6831** / 0.7393 |

## Code retrieval: MTEB(Code, v1) + RTEB(Code, beta) public datasets (18 tasks), nDCG@10

Same vLLM harness and H100s as the BF16 reference; Harrier's official mteb prompts (for these tasks mteb 2.24 falls back to the generic instruction "Retrieve text based on user query."), inputs truncated to 8,192 tokens. RTEB's private Code1Retrieval and JapaneseCode1Retrieval cannot be run locally.

| Task | Seen in training | BF16 | g64 | Δ | W4A16 r1 | Δ |
|---|---|---|---|---|---|---|
| AppsRetrieval | yes | 0.7242 | 0.7144 | -0.0098 | 0.7095 | -0.0147 |
| COIRCodeSearchNetRetrieval | yes | 0.8621 | 0.8588 | -0.0033 | 0.8577 | -0.0044 |
| CodeEditSearchRetrieval | no | 0.7590 | 0.7653 | +0.0063 | 0.7626 | +0.0036 |
| CodeFeedbackMT | yes | 0.9518 | 0.9503 | -0.0015 | 0.9480 | -0.0038 |
| CodeFeedbackST | yes | 0.8928 | 0.8920 | -0.0009 | 0.8906 | -0.0022 |
| CodeSearchNetCCRetrieval | yes | 0.7902 | 0.7978 | +0.0077 | 0.8045 | +0.0144 |
| CodeSearchNetRetrieval | yes | 0.8920 | 0.8923 | +0.0004 | 0.8918 | -0.0002 |
| CodeTransOceanContest | no | 0.9362 | 0.9362 | +0.0000 | 0.9368 | +0.0006 |
| CodeTransOceanDL \* | no | 0.3469 | 0.3405 | -0.0064 | 0.3426 | -0.0043 |
| CosQA \* | no | 0.3615 | 0.3356 | -0.0259 | 0.3647 | +0.0031 |
| DS1000Retrieval \* | no | 0.6717 | 0.6733 | +0.0016 | 0.6721 | +0.0004 |
| FreshStackRetrieval | no | 0.4444 | 0.4414 | -0.0031 | 0.4342 | -0.0102 |
| HumanEvalRetrieval | no | 0.9724 | 0.9743 | +0.0019 | 0.9657 | -0.0068 |
| MBPPRetrieval | no | 0.9225 | 0.9212 | -0.0012 | 0.9222 | -0.0003 |
| SWEbenchCodeRetrieval | no | 0.0672 | 0.0512 | -0.0160 | 0.0473 | -0.0199 |
| StackOverflowQA | no | 0.9700 | 0.9683 | -0.0017 | 0.9685 | -0.0015 |
| SyntheticText2SQL \* | no | 0.6078 | 0.6373 | +0.0296 | 0.6226 | +0.0148 |
| WikiSQLRetrieval | no | 0.8973 | 0.8926 | -0.0047 | 0.8933 | -0.0041 |
| **Mean, all 18** | | **0.7261** | **0.7246** | **-0.0015** | **0.7241** | **-0.0020** |
| **Mean, 14 tie-robust tasks (excl. *)** | | **0.7916** | **0.7897** | **-0.0018** | **0.7880** | **-0.0035** |

\* Tie-dominated tasks: most relevant documents have an exact-text duplicate in the corpus that is not relevant (CodeTransOceanDL 100%, SyntheticText2SQL 94%, DS1000 93%, CosQA 85%; `eval/tie_affected_tasks.json`). Identical texts get identical embeddings, so the relevant document's rank among its twins is decided by the search implementation's tie-breaking, not the model; nDCG@10 on these tasks can move by ~0.15 between engines with identical recall@10. Deltas on them are not meaningful.

## 24 GB card fit (RTX PRO 6000 Blackwell, vLLM memory capped)

| Model | Prefill | CUDA graphs | Budget GiB | max_len | Loads + embeds | KV cache tokens | Time for one max_len input (s) |
|---|---|---|---|---|---|---|---|
| nvfp4 | single-shot | yes | 21.5 | 2048 | no | – | – |
| nvfp4 | single-shot | yes | 22.7 | 2048 | yes | 5169 | 0.23 |
| nvfp4 | single-shot | yes | 22.7 | 4096 | yes | 4588 | 0.4 |
| nvfp4 | single-shot | yes | 22.7 | 8192 | no | – | – |
| nvfp4 | single-shot | no (eager) | 21.5 | 2048 | yes | 3880 | 0.83 |
| nvfp4 | single-shot | no (eager) | 21.5 | 4096 | no | – | – |
| nvfp4 | single-shot | no (eager) | 22.7 | 2048 | yes | 6100 | 0.65 |
| nvfp4 | single-shot | no (eager) | 22.7 | 4096 | yes | 5440 | 1.08 |
| nvfp4 | single-shot | no (eager) | 22.7 | 8192 | no | – | – |
| nvfp4a16 | chunked 2048 | yes | 21.5 | 8192 | no | – | – |
| nvfp4a16 | chunked 2048 | yes | 22.7 | 8192 | yes | 11022 | 1.82 |
| nvfp4a16 | chunked 2048 | yes | 22.7 | 16384 | yes | 16872 | 3.65 |
| nvfp4a16 | chunked 2048 | yes | 22.7 | 32768 | no | – | – |
| nvfp4a16 | chunked 2048 | no (eager) | 21.5 | 8192 | no | – | – |
| nvfp4a16 | chunked 2048 | no (eager) | 22.7 | 8192 | yes | 12811 | 2.31 |
| nvfp4a16 | chunked 2048 | no (eager) | 22.7 | 16384 | yes | 19612 | 4.33 |
| nvfp4a16 | chunked 2048 | no (eager) | 22.7 | 32768 | no | – | – |
| w4a16-gptq | single-shot | yes | 21.5 | 2048 | no | – | – |
| w4a16-gptq | single-shot | yes | 22.7 | 2048 | yes | 6856 | 0.49 |
| w4a16-gptq | single-shot | yes | 22.7 | 4096 | no | – | – |
| w4a16-gptq | single-shot | no (eager) | 21.5 | 2048 | yes | 5465 | 1.52 |
| w4a16-gptq | single-shot | no (eager) | 21.5 | 4096 | yes | 4838 | 1.37 |
| w4a16-gptq | single-shot | no (eager) | 21.5 | 8192 | no | – | – |
| w4a16-gptq | single-shot | no (eager) | 22.7 | 2048 | yes | 7683 | 0.87 |
| w4a16-gptq | single-shot | no (eager) | 22.7 | 4096 | yes | 7064 | 1.55 |
| w4a16-gptq | single-shot | no (eager) | 22.7 | 8192 | no | – | – |
| w4a16-gptq | chunked 2048 | yes | 21.5 | 8192 | yes | 10012 | 1.83 |
| w4a16-gptq | chunked 2048 | yes | 21.5 | 16384 | no | – | – |
| w4a16-gptq | chunked 2048 | yes | 22.7 | 8192 | yes | 14807 | 1.8 |
| w4a16-gptq | chunked 2048 | yes | 22.7 | 16384 | yes | 22667 | 3.61 |
| w4a16-gptq | chunked 2048 | yes | 22.7 | 32768 | no | – | – |
| w4a16-gptq | chunked 2048 | no (eager) | 21.5 | 8192 | yes | 11801 | 2.3 |
| w4a16-gptq | chunked 2048 | no (eager) | 21.5 | 16384 | yes | 18066 | 4.42 |
| w4a16-gptq | chunked 2048 | no (eager) | 21.5 | 32768 | no | – | – |
| w4a16-gptq | chunked 2048 | no (eager) | 22.7 | 8192 | yes | 16591 | 2.42 |
| w4a16-gptq | chunked 2048 | no (eager) | 22.7 | 16384 | yes | 25399 | 4.37 |
| w4a16-gptq | chunked 2048 | no (eager) | 22.7 | 32768 | yes | 34577 | 8.34 |
| w4a16-gptq-late12-attn | chunked 2048 | no (eager) | 21.5 | 4096 | no | – | – |
| w4a16-gptq-late12-attn | chunked 2048 | no (eager) | 22.7 | 8192 | yes | 10928 | 2.38 |
| w4a16-gptq-late12-attn | chunked 2048 | no (eager) | 22.7 | 16384 | yes | 16730 | 4.17 |
| w4a16-gptq-late12-attn | chunked 2048 | no (eager) | 22.7 | 32768 | no | – | – |
| w4a16-gptq-late12-attn-down | chunked 2048 | no (eager) | 22.7 | 2048 | no | – | – |
| w4a16-gptq-late12-attn-down | chunked 2048 | no (eager) | 23.2 | 2048 | no | – | – |
| w4a16-gptq-late12-attn-g64 | chunked 2048 | yes | 22.7 | 8192 | no | – | – |
| w4a16-gptq-late12-attn-g64 | chunked 2048 | no (eager) | 21.5 | 2048 | yes | 2245 | 0.85 |
| w4a16-gptq-late12-attn-g64 | chunked 2048 | no (eager) | 21.5 | 4096 | no | – | – |
| w4a16-gptq-late12-attn-g64 | chunked 2048 | no (eager) | 22.7 | 8192 | yes | 9673 | 2.16 |
| w4a16-gptq-late12-attn-g64 | chunked 2048 | no (eager) | 22.7 | 12288 | yes | 12582 | 3.19 |
| w4a16-gptq-late12-attn-g64 | chunked 2048 | no (eager) | 22.7 | 16384 | no | – | – |

