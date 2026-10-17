#!/usr/bin/env bash
# Variant evaluation (relaunch after the EngineCore-orphan fix; BF16 jobs continue from orchestrate_eval.sh).
#   GPU 0 (NVL)  : cosine embeddings for each variant via vLLM, compare vs BF16-vLLM
#   GPU 3 (PCIe) : mteb via vLLM for each variant that passes the cosine gate
# START/END lines are written into each job's own log (the redirect covers the job wrapper).
set -u
cd /data/models/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID
S=venv-serve/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
job() { local name=$1 to=$2; shift 2; log "START $name"; timeout "$to" "$@"; local rc=$?; log "END $name exit=$rc"; return $rc; }

gpu0() {
  export CUDA_VISIBLE_DEVICES=0
  for v in w4a16-gptq nvfp4a16 nvfp4; do
    job cos-$v 45m $S scripts/cosine_eval.py embed --engine vllm --model $v --label $v > logs/cos_embed_$v.log 2>&1 &&
    job cmp-$v 5m $S scripts/cosine_eval.py compare --label $v --ref bf16-vllm > logs/cos_cmp_$v.log 2>&1
  done
}
gpu3() {
  export CUDA_VISIBLE_DEVICES=3
  for v in w4a16-gptq nvfp4a16; do
    local f=eval/cosine/${v}_vs_bf16-vllm.json
    until [ -f "$f" ] || grep -qE "END (cos|cmp)-$v exit=[1-9]" logs/cos_embed_$v.log logs/cos_cmp_$v.log 2>/dev/null; do sleep 20; done
    if [ -f "$f" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$f'))['pass'] else 1)"; then
      job mteb-vllm-$v 5h $S scripts/run_mteb.py --engine vllm --model $v --label vllm-$v > logs/mteb_vllm-$v.log 2>&1
    else
      log "SKIP mteb-vllm-$v (cosine gate failed or embedding failed)" | tee -a logs/mteb_vllm-$v.log
    fi
  done
}
gpu0 > logs/orchestrate_variants_gpu0.log 2>&1 &
gpu3 > logs/orchestrate_variants_gpu3.log 2>&1 &
wait
log "VARIANTS DONE"
