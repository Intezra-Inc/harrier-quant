#!/usr/bin/env bash
# GPU 0: NVFP4A16 and NVFP4 builds (target regexes include vLLM fused names, see quantize.py).
set -u
cd /data/models/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
Q=venv-quant/bin/python; M=/data/models/harrier-oss-v1-27b
run() { echo "[$(date -u +%FT%TZ)] START $1"; shift; "$@"; echo "[$(date -u +%FT%TZ)] exit=$?"; }
run q-nvfp4a16 $Q -u scripts/quantize.py --variant nvfp4a16 --model $M --calib calib/calib.jsonl --out nvfp4a16 > logs/quant_nvfp4a16.log 2>&1
run q-nvfp4    $Q -u scripts/quantize.py --variant nvfp4    --model $M --calib calib/calib.jsonl --out nvfp4    > logs/quant_nvfp4.log 2>&1
