#!/usr/bin/env bash
# Blackwell: does late12-attn-down (21.8 GiB weights) / late12-attn (~18.9 GiB) fit a 24 GB card?
# Eager + chunked prefill (best memory config found earlier); budgets 22.7 GiB (0.95) and 23.2 GiB (~0.97).
set -u
cd ~/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID HQ_ROOT=$HOME/harrier-quant
S=venv-serve/bin/python
probe() { local g=$1 v=$2 b=$3; shift 3; for L in "$@"; do
  local tag=${v}_b${b}_eager_$L
  CUDA_VISIBLE_DEVICES=$g timeout 15m $S scripts/context_probe.py --model $v --max-len $L --budget-gib $b --chunked 2048 --enforce-eager > logs/ctxr3_$tag.log 2>&1
  local r=$(grep -ah "^RESULT" logs/ctxr3_$tag.log | tail -1)
  echo "CTXR3 $tag $(grep -aoE "Available KV cache memory: -?[0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" logs/ctxr3_$tag.log | tr "\n" " ")| ${r:0:180}"
  echo "$r" | grep -q "\"ok\": true" || break
done; }
probe 0 w4a16-gptq-late12-attn-down 22.7 2048 4096 8192 > logs/bwr3_gpu0.log 2>&1 &
probe 1 w4a16-gptq-late12-attn-down 23.2 2048 4096 8192 16384 > logs/bwr3_gpu1.log 2>&1 &
( probe 2 w4a16-gptq-late12-attn 21.5 4096 8192 16384; probe 2 w4a16-gptq-late12-attn 22.7 8192 16384 32768 ) > logs/bwr3_gpu2.log 2>&1 &
( echo "[$(date -u +%FT%TZ)] START bw cos late12-attn-down"; CUDA_VISIBLE_DEVICES=3 timeout 90m $S scripts/cosine_eval.py embed --engine vllm --model w4a16-gptq-late12-attn-down --label bw-w4a16-gptq-late12-attn-down; echo "[$(date -u +%FT%TZ)] END bw cos exit=$?" ) > logs/bwr3_gpu3.log 2>&1 &
wait; echo "BWR3 DONE" >> logs/bwr3_gpu0.log
