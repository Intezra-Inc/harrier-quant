#!/usr/bin/env bash
set -u
cd ~/harrier-quant
export CUDA_DEVICE_ORDER=PCI_BUS_ID
T=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits)
u() { python3 -c "print(round($1*1024/$T, 4))"; }   # GiB budget -> utilization on this 96 GB card
U24=$(u 22.7); U32=$(u 30.4)   # 0.95 x 24 GB card, 0.95 x 32 GB card
( CUDA_VISIBLE_DEVICES=0 timeout 20m scripts/serve_test.sh w4a16-gptq-late12-attn-g64 8192 $U24 8011; echo "rc=$?" ) > logs/serve_g64_8k.out 2>&1 &
( CUDA_VISIBLE_DEVICES=1 timeout 20m scripts/serve_test.sh w4a16-gptq-late12-attn-g64 12288 $U24 8012; echo "rc=$?" ) > logs/serve_g64_12k.out 2>&1 &
( CUDA_VISIBLE_DEVICES=2 timeout 20m scripts/serve_test.sh w4a16-gptq-late12-attn-down 16384 $U32 8013; echo "rc=$?" ) > logs/serve_down_16k_32gb.out 2>&1 &
wait; echo "SERVE TESTS DONE U24=$U24 U32=$U32" >> logs/serve_g64_8k.out
