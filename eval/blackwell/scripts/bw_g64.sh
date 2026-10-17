#!/usr/bin/env bash
# Blackwell: late12-attn-g64 cosine + 24 GB context probe (chunked prefill; eager and CUDA graphs, warm cache).
set -u
cd ~/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID HQ_ROOT=$HOME/harrier-quant
S=venv-serve/bin/python; V=w4a16-gptq-late12-attn-g64
probe() { local g=$1 b=$2 e=$3; shift 3; for L in "$@"; do
  local tag=${V}_b${b}_${e}_$L
  [ $e = graphs ] && CUDA_VISIBLE_DEVICES=$g timeout 15m $S scripts/context_probe.py --model $V --max-len $L --budget-gib $b --chunked 2048 > logs/ctxg64_${tag}_warmup.log 2>&1
  CUDA_VISIBLE_DEVICES=$g timeout 15m $S scripts/context_probe.py --model $V --max-len $L --budget-gib $b --chunked 2048 $([ $e = eager ] && echo --enforce-eager) > logs/ctxg64_$tag.log 2>&1
  local r=$(grep -ah "^RESULT" logs/ctxg64_$tag.log | tail -1)
  echo "CTXG64 $tag $(grep -aoE "Available KV cache memory: -?[0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" logs/ctxg64_$tag.log | tr "\n" " ")| ${r:0:170}"
  echo "$r" | grep -q "\"ok\": true" || break
done; }
probe 0 21.5 eager 4096 8192 16384 > logs/bwg64_gpu0.log 2>&1 &
probe 1 22.7 eager 8192 16384 32768 > logs/bwg64_gpu1.log 2>&1 &
probe 2 22.7 graphs 8192 16384 > logs/bwg64_gpu2.log 2>&1 &
( echo "[$(date -u +%FT%TZ)] START bw cos $V"; CUDA_VISIBLE_DEVICES=3 timeout 90m $S scripts/cosine_eval.py embed --engine vllm --model $V --label bw-$V > logs/cos_embed_bw-$V.log 2>&1; echo "[$(date -u +%FT%TZ)] END bw cos exit=$?"; $S scripts/cosine_eval.py compare --label bw-$V --ref bw-bf16 2>&1 | grep -v Warn ) > logs/bwg64_gpu3.log 2>&1 &
wait; echo "BWG64 DONE" >> logs/bwg64_gpu0.log
