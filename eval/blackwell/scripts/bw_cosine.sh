#!/usr/bin/env bash
# Blackwell (RTX PRO 6000, SM120): cosine embeddings for BF16 + all variants, one GPU each.
set -u
cd ~/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID HQ_ROOT=$HOME/harrier-quant
S=venv-serve/bin/python
job() { local g=$1 label=$2 model=$3; echo "[$(date -u +%FT%TZ)] START bw-$label gpu$g"; CUDA_VISIBLE_DEVICES=$g timeout 45m $S scripts/cosine_eval.py embed --engine vllm --model $model --label bw-$label; echo "[$(date -u +%FT%TZ)] END bw-$label exit=$?"; }
job 0 bf16       $HOME/harrier-oss-v1-27b > logs/cos_embed_bw-bf16.log 2>&1 &
job 1 nvfp4      nvfp4      > logs/cos_embed_bw-nvfp4.log 2>&1 &
job 2 nvfp4a16   nvfp4a16   > logs/cos_embed_bw-nvfp4a16.log 2>&1 &
job 3 w4a16-gptq w4a16-gptq > logs/cos_embed_bw-w4a16-gptq.log 2>&1 &
wait
echo "[$(date -u +%FT%TZ)] BW COSINE DONE"
