#!/usr/bin/env bash
# Evaluation orchestration (all jobs under timeouts; CUDA_DEVICE_ORDER=PCI_BUS_ID so indices match nvidia-smi).
#   GPU 0 (H100 NVL)  : cosine embeddings via vLLM for BF16 + all variants, then compare vs BF16
#   GPU 1 (H100 NVL)  : cosine embeddings via sentence-transformers BF16 (engine-drift reference)
#   GPU 2 (H100 PCIe) : mteb, BF16 via vLLM
#   GPU 3 (H100 PCIe) : mteb, each variant via vLLM, only after it passes the cosine gate (p5 >= 0.98)
# NVFP4 (W4A4) is embedded for cosine on Hopper for information only: vLLM runs it weight-only
# (Marlin) on SM90, so activations are not FP4-quantized. It gets no mteb run here.
set -u
cd /data/models/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID
S=venv-serve/bin/python; M=/data/models/harrier-oss-v1-27b
log() { echo "[$(date -u +%FT%TZ)] $*"; }
job() { local name=$1 to=$2; shift 2; log "START $name"; timeout "$to" "$@"; local rc=$?; log "END $name exit=$rc"; return $rc; }

gpu0() {
  export CUDA_VISIBLE_DEVICES=0
  job cos-bf16-vllm 45m $S scripts/cosine_eval.py embed --engine vllm --model $M --label bf16-vllm > logs/cos_embed_bf16-vllm.log 2>&1
  for v in w4a16-gptq nvfp4a16 nvfp4; do
    job cos-$v 45m $S scripts/cosine_eval.py embed --engine vllm --model $v --label $v > logs/cos_embed_$v.log 2>&1 &&
    job cmp-$v 5m $S scripts/cosine_eval.py compare --label $v --ref bf16-vllm > logs/cos_cmp_$v.log 2>&1
  done
}
gpu1() {
  export CUDA_VISIBLE_DEVICES=1
  job cos-bf16-st 60m $S scripts/cosine_eval.py embed --engine st --model $M --label bf16-st > logs/cos_embed_bf16-st.log 2>&1
}
gpu2() {
  export CUDA_VISIBLE_DEVICES=2
  job mteb-vllm-bf16 5h $S scripts/run_mteb.py --engine vllm --model $M --label vllm-bf16 > logs/mteb_vllm-bf16.log 2>&1
}
gpu3() {
  export CUDA_VISIBLE_DEVICES=3
  for v in w4a16-gptq nvfp4a16; do
    local f=eval/cosine/${v}_vs_bf16-vllm.json
    # wait for the cosine gate (or for gpu0 to give up on this variant)
    until [ -f "$f" ] || grep -q "END cos-$v exit=[1-9]" logs/orchestrate_gpu0.log 2>/dev/null; do sleep 20; done
    if [ -f "$f" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$f'))['pass'] else 1)"; then
      job mteb-vllm-$v 5h $S scripts/run_mteb.py --engine vllm --model $v --label vllm-$v > logs/mteb_vllm-$v.log 2>&1
    else
      log "SKIP mteb-vllm-$v (cosine gate failed or embedding failed)"
    fi
  done
}

gpu0 > logs/orchestrate_gpu0.log 2>&1 &
gpu1 > logs/orchestrate_gpu1.log 2>&1 &
gpu2 > logs/orchestrate_gpu2.log 2>&1 &
gpu3 > logs/orchestrate_gpu3.log 2>&1 &
wait
log "ALL DONE"
