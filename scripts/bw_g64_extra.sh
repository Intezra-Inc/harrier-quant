#!/usr/bin/env bash
set -u
cd ~/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID HQ_ROOT=$HOME/harrier-quant
S=venv-serve/bin/python; V=w4a16-gptq-late12-attn-g64
one() { local g=$1 b=$2 L=$3; local tag=${V}_b${b}_eager_$L
  CUDA_VISIBLE_DEVICES=$g timeout 15m $S scripts/context_probe.py --model $V --max-len $L --budget-gib $b --chunked 2048 --enforce-eager > logs/ctxg64_$tag.log 2>&1
  echo "CTXG64 $tag $(grep -aoE "Available KV cache memory: -?[0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" logs/ctxg64_$tag.log | tr "\n" " ")| $(grep -ah "^RESULT" logs/ctxg64_$tag.log | tail -1 | cut -c1-170)"; }
one 0 21.5 2048 > logs/bwg64x_gpu0.log 2>&1 &
one 1 22.7 12288 > logs/bwg64x_gpu1.log 2>&1 &
wait
