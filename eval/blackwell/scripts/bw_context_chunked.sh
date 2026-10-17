#!/usr/bin/env bash
# 24 GB-card emulation, chunked prefill (2048-token chunks; max_model_len = L+1 to avoid the
# vLLM 0.31 stall when input length == max_model_len). Graph-mode configs run twice so the
# torch.compile cache is warm; only the second run counts.
set -u
cd ~/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID HQ_ROOT=$HOME/harrier-quant
S=venv-serve/bin/python
one() { local g=$1 v=$2 b=$3 e=$4 L=$5 tag=$6
  CUDA_VISIBLE_DEVICES=$g timeout 20m $S scripts/context_probe.py --model $v --max-len $L --budget-gib $b --chunked 2048 $([ $e = eager ] && echo --enforce-eager) > logs/ctxc_$tag.log 2>&1; }
series() { local g=$1 v=$2 b=$3 e=$4; for L in 8192 16384 32768; do
  local tag=${v}_b${b}_${e}_$L
  [ $e = graphs ] && one $g $v $b $e $L ${tag}_warmup
  one $g $v $b $e $L $tag
  local r=$(grep -ah "^RESULT" logs/ctxc_$tag.log | tail -1)
  echo "CTXC $tag $(grep -aoE "Available KV cache memory: -?[0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" logs/ctxc_$tag.log | tr "\n" " ")| ${r:0:200}"
  echo "$r" | grep -q "\"ok\": true" || break
done; }
run() { local g=$1 v=$2 b=$3; series $g $v $b eager; series $g $v $b graphs; }
run 0 w4a16-gptq 21.5 > logs/bwc_gpu0.log 2>&1 &
run 1 w4a16-gptq 22.7 > logs/bwc_gpu1.log 2>&1 &
run 2 nvfp4a16 21.5 > logs/bwc_gpu2.log 2>&1 &
run 3 nvfp4a16 22.7 > logs/bwc_gpu3.log 2>&1 &
wait; echo "CHUNKED CONTEXT DONE" >> logs/bwc_gpu0.log
