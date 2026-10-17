#!/usr/bin/env bash
# GPU 0 chain: cosine embeddings (BF16 vLLM ref, W4A16), compare, then NVFP4A16 + NVFP4 builds.
set -u
cd /data/models/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
S=venv-serve/bin/python; Q=venv-quant/bin/python; M=/data/models/harrier-oss-v1-27b
run() { echo "[$(date -u +%FT%TZ)] START $1"; shift; "$@"; echo "[$(date -u +%FT%TZ)] exit=$?"; }
run cos-bf16-vllm $S scripts/cosine_eval.py embed --engine vllm --model $M --label bf16-vllm > logs/cos_embed_bf16-vllm.log 2>&1
run cos-w4a16    $S scripts/cosine_eval.py embed --engine vllm --model w4a16-gptq --label w4a16-gptq > logs/cos_embed_w4a16-gptq.log 2>&1
run cmp-w4a16    $S scripts/cosine_eval.py compare --label w4a16-gptq --ref bf16-vllm
run q-nvfp4a16   $Q -u scripts/quantize.py --variant nvfp4a16 --model $M --calib calib/calib.jsonl --out nvfp4a16 > logs/quant_nvfp4a16.log 2>&1
run q-nvfp4      $Q -u scripts/quantize.py --variant nvfp4 --model $M --calib calib/calib.jsonl --out nvfp4 > logs/quant_nvfp4.log 2>&1
