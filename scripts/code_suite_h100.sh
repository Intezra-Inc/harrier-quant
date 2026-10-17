#!/usr/bin/env bash
# MTEB(Code, v1) + RTEB(Code, beta) public tasks via vLLM on H100: BF16 reference + two quantized variants.
# Same harness/prompts/truncation (8192) as the quantization eval; separate result folders per label.
set -u
cd /data/models/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID
S=venv-serve/bin/python
run() { local g=$1 model=$2 label=$3
  echo "[$(date -u +%FT%TZ)] START $label gpu$g"
  CUDA_VISIBLE_DEVICES=$g timeout 20h $S scripts/run_mteb.py --suite code --engine vllm --model $model --label $label > logs/mteb_$label.log 2>&1
  echo "[$(date -u +%FT%TZ)] END $label exit=$?"; }
run 0 /data/models/harrier-oss-v1-27b code-vllm-bf16 > logs/code_gpu0.log 2>&1 &
run 1 w4a16-gptq-late12-attn-g64 code-vllm-w4a16-gptq-late12-attn-g64 > logs/code_gpu1.log 2>&1 &
run 2 w4a16-gptq code-vllm-w4a16-gptq > logs/code_gpu2.log 2>&1 &
wait
echo "[$(date -u +%FT%TZ)] CODE SUITE DONE" >> logs/code_gpu0.log
