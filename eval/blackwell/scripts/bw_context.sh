#!/usr/bin/env bash
# 24 GB-card emulation on RTX PRO 6000 (SM120): grid over memory budget, CUDA graphs vs eager, context length.
# Each series stops at the first failing length.
set -u
cd ~/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID HQ_ROOT=$HOME/harrier-quant
S=venv-serve/bin/python
series() { local g=$1 v=$2 b=$3 e=$4; for L in 2048 4096 8192 16384; do
  local tag=${v}_b${b}_${e}_$L
  CUDA_VISIBLE_DEVICES=$g timeout 15m $S scripts/context_probe.py --model $v --max-len $L --budget-gib $b $([ $e = eager ] && echo --enforce-eager) > logs/ctx_$tag.log 2>&1
  local r=$(grep -ah "^RESULT" logs/ctx_$tag.log | tail -1)
  echo "CTX $tag $(grep -aoE "Available KV cache memory: -?[0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" logs/ctx_$tag.log | tr "\n" " ")| ${r:0:220}"
  echo "$r" | grep -q "\"ok\": true" || break
done; }
run_gpu() { local g=$1 v=$2; for b in 21.5 22.7; do for e in graphs eager; do series $g $v $b $e; done; done; }
run_gpu 1 w4a16-gptq > logs/bw_context_gpu1.log 2>&1 &
run_gpu 3 nvfp4 > logs/bw_context_gpu3.log 2>&1 &
wait; echo "CONTEXT DONE" >> logs/bw_context_gpu1.log
