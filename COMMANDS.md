# Reproducing the harrier-oss-v1-27b quantization

Everything here was run on 2026-10-11/12. Paths are as used on the build host; set `HQ_ROOT` to run the eval
scripts from another directory.

## Inputs, versions, seeds

| Item | Value |
|---|---|
| Base model | `microsoft/harrier-oss-v1-27b` @ `0c0fc62f6d8af9e8604cb818c412301b103a0093` (SHA-256 of every file: `env/bf16_sha256.txt`, verified against HF LFS hashes) |
| Calibration set | `calib/calib.jsonl`, 512 samples, SHA-256 `0cdc0751f76ea0543132b3d23ce3552a51a7a9af4563507e84a484c019ba8762` (rebuilt byte-identically twice). Sources and licences: `calib/SOURCES.md` |
| Seeds | 42 everywhere (calibration sampling, quantization, cosine item sampling, sensitivity probe) |
| Quantization venv | Python 3.12.11, torch 2.14.0, transformers 5.17.0, llmcompressor 0.14.0, compressed-tensors 0.19.0 (`env/venv-quant_freeze.txt`) |
| Serving / eval venv | Python 3.12.11, vLLM 0.31.0, torch 2.13.0, transformers 5.17.0, compressed-tensors 0.17.0, mteb 2.24.9, sentence-transformers 6.1.0 (`env/venv-serve_freeze.txt`) |
| Build / eval hardware | 2x H100 NVL (builds, cosine) + 2x H100 PCIe (mteb), driver 615.71.09, CUDA 13 wheels; 4x RTX PRO 6000 Blackwell Max-Q (SM120) for Blackwell cosine, FP4, 24 GB-fit and serve tests (same venv from the same freeze file) |
| GPU pinning | always `CUDA_DEVICE_ORDER=PCI_BUS_ID` (mixed GPU models; CUDA's default order does not match `nvidia-smi`) |

## 1. Environments

```bash
cd /data/models/harrier-quant
uv venv --python 3.12.11 venv-quant && VIRTUAL_ENV=$PWD/venv-quant uv pip install -r env/venv-quant_freeze.txt
uv venv --python 3.12.11 venv-serve && VIRTUAL_ENV=$PWD/venv-serve uv pip install -r env/venv-serve_freeze.txt
```

## 2. Base model

```bash
hf download microsoft/harrier-oss-v1-27b --revision 0c0fc62f6d8af9e8604cb818c412301b103a0093 \
  --local-dir /data/models/harrier-oss-v1-27b
```

## 3. Calibration set

```bash
venv-quant/bin/python calib/build_calib.py --model /data/models/harrier-oss-v1-27b --out calib
```

## 4. Quantization (one GPU, ~15 min each for GPTQ)

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
Q="venv-quant/bin/python scripts/quantize.py --model /data/models/harrier-oss-v1-27b --calib calib/calib.jsonl"
# Round 1 (layout "edges": FP8 layers 0-5 and 56-61)
$Q --variant w4a16-gptq --out w4a16-gptq      # then: venv-quant/bin/python scripts/patch_vllm_targets.py w4a16-gptq
$Q --variant nvfp4a16  --out nvfp4a16
$Q --variant nvfp4     --out nvfp4
# Round 2 / 3 (sensitivity-guided layouts)
$Q --variant w4a16-gptq --layout late12-attn                     --out w4a16-gptq-late12-attn
$Q --variant w4a16-gptq --layout late12-attn --group-size 64     --out w4a16-gptq-late12-attn-g64
$Q --variant w4a16-gptq --layout late12-attn-down                --out w4a16-gptq-late12-attn-down
```

Each output folder contains `recipe.yaml` (written by llm-compressor), `quantization_meta.json` (variant, layout,
group size, regex targets, seed, minutes), the tokenizer, and Harrier's sentence-transformers files
(`modules.json`, `1_Pooling/`, `config_sentence_transformers.json`, `mteb_v2_eval_prompts.json`).

`w4a16-gptq` was built before vLLM's fused module names were added to the targets; `patch_vllm_targets.py` rewrites
only the regex strings in its `config.json` (original kept as `config.json.orig`). The weights are unaffected, and
every later build includes the fused names directly.

## 5. Layer sensitivity (guides rounds 2-3; calibration data only, never eval data)

```bash
CUDA_VISIBLE_DEVICES=0 venv-quant/bin/python scripts/sensitivity.py            # per layer / per module type
CUDA_VISIBLE_DEVICES=0 venv-quant/bin/python scripts/sensitivity.py --combos   # candidate layouts
```
Results: `eval/sensitivity/results.json`, `eval/sensitivity/combos.json`.

## 6. Evaluation

```bash
S=venv-serve/bin/python
# Cosine items (2,000, seed 42) and embeddings
$S scripts/cosine_eval.py build
CUDA_VISIBLE_DEVICES=0 $S scripts/cosine_eval.py embed --engine vllm --model /data/models/harrier-oss-v1-27b --label bf16-vllm
CUDA_VISIBLE_DEVICES=1 $S scripts/cosine_eval.py embed --engine st   --model /data/models/harrier-oss-v1-27b --label bf16-st
CUDA_VISIBLE_DEVICES=0 $S scripts/cosine_eval.py embed --engine vllm --model <variant> --label <variant>
$S scripts/cosine_eval.py compare --label <variant> --ref bf16-vllm          # gate: p5 >= 0.98 in every bucket
# Retrieval (mteb); ~1.5-2 h per model on one H100
CUDA_VISIBLE_DEVICES=1 $S scripts/run_mteb.py --engine st   --model /data/models/harrier-oss-v1-27b --label bf16-st
CUDA_VISIBLE_DEVICES=2 $S scripts/run_mteb.py --engine vllm --model /data/models/harrier-oss-v1-27b --label vllm-bf16
CUDA_VISIBLE_DEVICES=3 $S scripts/run_mteb.py --engine vllm --model <variant> --label vllm-<variant>
# 24 GB fit (on the RTX PRO 6000; budget emulates a 24 GB card)
CUDA_VISIBLE_DEVICES=0 $S scripts/context_probe.py --model <variant> --max-len 8192 --budget-gib 22.7 --chunked 2048 --enforce-eager
# End-to-end server check
CUDA_VISIBLE_DEVICES=0 scripts/serve_test.sh <variant> 8192 <utilization>
# Tables
$S scripts/make_results.py
```

Orchestration wrappers actually used (GPU assignment, timeouts): `scripts/orchestrate_*.sh`, `scripts/round2.sh`,
`scripts/round3.sh`, `scripts/chain_gpu0_*.sh`; Blackwell host: `scripts/bw_*.sh` (copies under `eval/blackwell/`).

## Workarounds (and why)

| Issue | Workaround | Where |
|---|---|---|
| transformers 5.17 maps `gemma3_text` to the multimodal `Gemma3Processor` (needs an image-processor config this repo doesn't have) | pass the tokenizer as `processor=` to `oneshot`; for sentence-transformers fall back to `AutoTokenizer` on that exact error | `quantize.py`, `st_compat.py` |
| llm-compressor warns it cannot find `lm_head` / check embeddings (no LM head by design with `AutoModel`) | expected; nothing to do | – |
| vLLM's `Gemma3TextModel` (-> `Gemma3Model`) has no `packed_modules_mapping`, so fused `qkv_proj` / `gate_up_proj` get no scheme and loading crashes | targets also name the fused modules; layouts never split fused parts | `quantize.py` |
| vLLM 0.31 stalls when a pooling request's length equals `max_model_len` with chunked prefill on | eval: chunked prefill off (identical inputs to the sentence-transformers reference). Serving: chunked prefill on with `--max-model-len` = client truncation length + 1 | `run_mteb.py`, `cosine_eval.py`, serve commands |
| vLLM 0.31 engine teardown can hang after work completes | write results, SIGKILL the EngineCore child, `os._exit` | `hard_exit.py` |
| vLLM 0.31 renamed the pooler's `normalize` to `use_activation` | `{"seq_pooling_type": "LAST", "use_activation": true}` | everywhere |
