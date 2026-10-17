#!/usr/bin/env python
"""Apply quantize.py's current target regexes to an existing checkpoint's config.json.

Checkpoints built before the qkv_proj/gate_up_proj alternatives were added to quantize.py
differ only in these regex strings (the alternatives match no HF module, so the weights are
identical). This rewrites the targets so vLLM can resolve its fused Gemma3 layers.
Usage: python patch_vllm_targets.py <checkpoint_dir>
"""
import json, os, re, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
# Round-1 target strings exactly as applied to w4a16-gptq/config.json (quantize.py has since moved to
# generated per-layout regexes; these literal strings are kept so this patch stays reproducible).
LINEARS = r"(self_attn\.(q|k|v|o|qkv)_proj|mlp\.(gate|up|down|gate_up)_proj)$"
EDGE = rf"re:.*layers\.([0-5]|5[6-9]|6[01])\.{LINEARS}"
MIDDLE = rf"re:.*layers\.([6-9]|[1-4][0-9]|5[0-5])\.{LINEARS}"

path = os.path.join(sys.argv[1], "config.json")
cfg = json.load(open(path))
groups = cfg["quantization_config"]["config_groups"]
for g in groups.values():
    (t,) = g["targets"]
    if re.search(r"\(\[0-5\]\|5\[6-9\]\|6\[01\]\)", t):
        g["targets"] = [EDGE]
    elif re.search(r"\(\[6-9\]\|\[1-4\]\[0-9\]\|5\[0-5\]\)", t):
        g["targets"] = [MIDDLE]
    else:
        raise SystemExit(f"unrecognised target {t}")
shutil.copy2(path, path + ".orig")
json.dump(cfg, open(path, "w"), indent=2)
print({k: g["targets"] for k, g in groups.items()})
