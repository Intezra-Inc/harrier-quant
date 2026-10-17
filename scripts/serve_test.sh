#!/usr/bin/env bash
# End-to-end check of the recommended `vllm serve` command for a variant (OpenAI-compatible /v1/embeddings).
# Usage: CUDA_VISIBLE_DEVICES=0 scripts/serve_test.sh <model_dir> <max_len> <gpu_mem_util> [port]
#   gpu_mem_util is scaled on test hosts to emulate a 24 GB card (0.95 x 24 GB on the real card).
set -u
MODEL=$1; L=$2; UTIL=$3; PORT=${4:-8011}
cd "$(dirname "$0")/.."
S=venv-serve/bin
LOG=logs/serve_$(basename $MODEL)_$L.log
# The serve command recommended in the model card (only --gpu-memory-utilization differs on test hosts):
$S/vllm serve "$MODEL" --served-model-name harrier-test --port $PORT \
  --runner pooling --convert embed \
  --pooler-config '{"seq_pooling_type": "LAST", "use_activation": true}' \
  --max-model-len $((L + 1)) --enable-chunked-prefill --max-num-batched-tokens 2048 \
  --enforce-eager --gpu-memory-utilization $UTIL > $LOG 2>&1 &
PID=$!
for i in $(seq 1 120); do curl -sf localhost:$PORT/health > /dev/null && break; kill -0 $PID 2>/dev/null || { echo "SERVE FAILED to start"; tail -5 $LOG; exit 1; }; sleep 5; done
$S/python - "$MODEL" "$L" "$PORT" <<'EOF'
import json, sys, urllib.request
import numpy as np
from transformers import AutoTokenizer
model, L, port = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tok = AutoTokenizer.from_pretrained(model)
q = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat"
docs = ["As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "Definition of summit for English Language Learners: the highest point of a mountain."]
# a document of exactly L tokens (client-side truncation, as recommended)
long_ids = tok(" ".join(["The quick brown fox jumps over the lazy dog."] * (L // 8)), truncation=True, max_length=L)["input_ids"]
def post(inp):
    req = urllib.request.Request(f"http://localhost:{port}/v1/embeddings", method="POST",
                                 data=json.dumps({"model": "harrier-test", "input": inp}).encode(),
                                 headers={"Content-Type": "application/json"})
    return [d["embedding"] for d in json.load(urllib.request.urlopen(req, timeout=600))["data"]]
e = np.array(post([q] + docs))
s = e[0] @ e[1:].T
lng = post([long_ids])  # token-id input (exactly L tokens incl. <bos>/<eos>)
print("SERVE OK", json.dumps({"model": model, "max_len": L, "dim": e.shape[1], "norms": np.linalg.norm(e, axis=1).round(4).tolist(),
                               "q_doc_scores": (s * 100).round(2).tolist(), "long_doc_tokens": len(long_ids), "long_ok": len(lng[0]) == e.shape[1]}))
EOF
RC=$?
kill $PID; sleep 5; pkill -P $PID 2>/dev/null; kill -9 $PID 2>/dev/null
exit $RC
