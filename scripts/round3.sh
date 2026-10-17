#!/usr/bin/env bash
# Round 3 (user-approved size increase after late12-attn missed only on multilingual p5):
#   GPU 0: late12-attn-down (INT4 g128)     GPU 2: late12-attn with INT4 g64 (after BF16 mteb finishes)
set -u
cd /data/models/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID
Q=venv-quant/bin/python; S=venv-serve/bin/python; M=/data/models/harrier-oss-v1-27b
log() { echo "[$(date -u +%FT%TZ)] $*"; }
build_eval() { local g=$1 layout=$2 v=$3; shift 3
  export CUDA_VISIBLE_DEVICES=$g
  log "START build $v gpu$g"
  timeout 2h $Q -u scripts/quantize.py --variant w4a16-gptq --layout $layout "$@" --model $M --calib calib/calib.jsonl --out $v > logs/quant_$v.log 2>&1
  log "END build $v exit=$?"
  [ -f $v/model.safetensors ] || [ -f $v/model.safetensors.index.json ] || return 1
  log "START cos $v"
  timeout 45m $S scripts/cosine_eval.py embed --engine vllm --model $v --label $v > logs/cos_embed_$v.log 2>&1
  log "END cos $v exit=$?"
  $S scripts/cosine_eval.py compare --label $v --ref bf16-vllm 2>&1 | grep -v Warn
}
gpu2() {
  while pgrep -f -- '[-]-label vllm-bf16$' > /dev/null; do sleep 30; done
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', ' '$2 ~ /CA:00/ {print $1}'); do
    log "killing orphaned process $p on GPU 2"; kill $p; done
  sleep 10
  build_eval 2 late12-attn w4a16-gptq-late12-attn-g64 --group-size 64
}
build_eval 0 late12-attn-down w4a16-gptq-late12-attn-down > logs/round3_gpu0.log 2>&1 &
gpu2 > logs/round3_gpu2.log 2>&1 &
wait
log "ROUND3 DONE"
