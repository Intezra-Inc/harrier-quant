#!/usr/bin/env python
"""Quantize microsoft/harrier-oss-v1-27b (Gemma3TextModel, no LM head) with llm-compressor.

Layer layout (--layout), shared by all schemes:
  edges (round 1)  : decoder layers 0-5 and 56-61 FP8_DYNAMIC, layers 6-55 4-bit
  late12-qkv       : layers 50-61 FP8_DYNAMIC + q/k/v_proj FP8_DYNAMIC in every layer, rest 4-bit
  late12-attn      : layers 50-61 FP8_DYNAMIC + q/k/v/o_proj FP8_DYNAMIC in every layer, rest 4-bit
  late12-attn-down : late12-attn + down_proj FP8_DYNAMIC in every layer
  (layer/module choices from scripts/sensitivity.py on calibration data, see eval/sensitivity/)
  FP8_DYNAMIC = per-channel FP8 weights, per-token dynamic FP8 activations.
  embed_tokens and all norms stay BF16 (never targeted).
  vLLM fuses q/k/v (qkv_proj) and gate/up (gate_up_proj); fused parts must share a scheme, so
  layouts only ever split them as a whole.

Variants:
  w4a16-gptq : GPTQ INT4 weights, group 128, symmetric, static act-order; calibrated
  nvfp4a16   : NVFP4 weights only (group 16, FP8 scales + global scale); data-free
  nvfp4      : NVFP4 weights + NVFP4 activations (W4A4); calibrated (activation global scales)

Usage:
  CUDA_VISIBLE_DEVICES=0 python quantize.py --variant w4a16-gptq \
      --model /data/models/harrier-oss-v1-27b --calib /data/models/harrier-quant/calib/calib.jsonl \
      --out /data/models/harrier-quant/harrier-oss-v1-27b-W4A16
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from compressed_tensors.quantization import preset_name_to_scheme
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from transformers import AutoModel, AutoTokenizer

SEED = 42
MAX_SEQ_LEN = 2048
# qkv_proj / gate_up_proj are vLLM's fused module names. They match nothing in the HF model, but
# vLLM's Gemma3TextModel (-> Gemma3Model) has no packed_modules_mapping, so without them vLLM
# matches its fused layers against the targets directly, finds no scheme, and leaves them unquantized
# (then crashes loading weight_scale / weight_packed into an unquantized layer).
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]
FUSED = [{"q", "k", "v"}, {"gate", "up"}]
LAYOUTS = {
    "edges": dict(fp8_layers=set(range(6)) | set(range(56, 62)), fp8_types=set()),
    "late12-qkv": dict(fp8_layers=set(range(50, 62)), fp8_types={"q", "k", "v"}),
    "late12-attn": dict(fp8_layers=set(range(50, 62)), fp8_types={"q", "k", "v", "o"}),
    "late12-attn-down": dict(fp8_layers=set(range(50, 62)), fp8_types={"q", "k", "v", "o", "down"}),
}


def _alt_types(types):
    """Regex alternation for module types, adding vLLM's fused names when both halves are present.

    qkv_proj / gate_up_proj are vLLM's fused module names. They match nothing in the HF model, but
    vLLM's Gemma3TextModel (-> Gemma3Model) has no packed_modules_mapping, so without them vLLM
    matches its fused layers against the targets directly, finds no scheme, and leaves them
    unquantized (then crashes loading weight_scale / weight_packed into an unquantized layer).
    """
    attn = [t for t in ["q", "k", "v", "o"] if t in types] + (["qkv"] if {"q", "k", "v"} <= types else [])
    mlp = [t for t in ["gate", "up", "down"] if t in types] + (["gate_up"] if {"gate", "up"} <= types else [])
    parts = []
    if attn:
        parts.append(rf"self_attn\.({'|'.join(attn)})_proj")
    if mlp:
        parts.append(rf"mlp\.({'|'.join(mlp)})_proj")
    return "(" + "|".join(parts) + ")$"


def _alt_layers(layers):
    return "(" + "|".join(str(i) for i in sorted(layers)) + ")"


def layout_targets(name, n_layers=62):
    lay = LAYOUTS[name]
    L, T = lay["fp8_layers"], lay["fp8_types"]
    for f in FUSED:
        assert f <= T or not (f & T), f"layout {name} splits fused modules {f}"
    fp8 = [rf"re:.*layers\.{_alt_layers(L)}\.{_alt_types(set(TYPES))}"]
    if T:
        fp8.append(rf"re:.*layers\.\d+\.{_alt_types(T)}")
    four = [rf"re:.*layers\.{_alt_layers(set(range(n_layers)) - L)}\.{_alt_types(set(TYPES) - T)}"]
    return fp8, four


# Files the sentence-transformers / mteb tooling needs that save_pretrained doesn't write.
PASSTHROUGH = ["modules.json", "config_sentence_transformers.json", "1_Pooling",
               "mteb_v2_eval_prompts.json"]

VARIANTS = {
    "w4a16-gptq": dict(scheme="W4A16", modifier="gptq", calibrated=True),
    "nvfp4a16": dict(scheme="NVFP4A16", modifier="rtn", calibrated=False),
    "nvfp4": dict(scheme="NVFP4", modifier="rtn", calibrated=True),
}


def build_recipe(variant, layout, group_size=None):
    v = VARIANTS[variant]
    fp8, four = layout_targets(layout)
    groups = {
        "fp8": preset_name_to_scheme("FP8_DYNAMIC", targets=fp8),
        "4bit": preset_name_to_scheme(v["scheme"], targets=four),
    }
    if group_size:  # INT4 only (NVFP4 group size is fixed at 16 by the format)
        assert v["scheme"] == "W4A16", "--group-size applies to the INT4 scheme only"
        groups["4bit"].weights.group_size = group_size
    if v["modifier"] == "gptq":
        return GPTQModifier(config_groups=groups, ignore=[], block_size=128,
                            dampening_frac=0.01, actorder="static")
    return QuantizationModifier(config_groups=groups, ignore=[])


def load_calib(path, tok, n):
    rows = [json.loads(l) for l in open(path)][:n]
    enc = [tok(r["text"], truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True) for r in rows]
    return Dataset.from_dict({"input_ids": [e["input_ids"] for e in enc],
                              "attention_mask": [e["attention_mask"] for e in enc]})


def check_targets(model, layout):
    """Every decoder Linear must be matched by exactly one group; also checks vLLM fused names."""
    import re
    fp8, four = layout_targets(layout, model.config.num_hidden_layers)
    rx = {"fp8": [re.compile(t[3:]) for t in fp8], "4bit": [re.compile(t[3:]) for t in four]}
    count = {"fp8": 0, "4bit": 0}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hit = [g for g, rs in rx.items() if any(r.match(name) for r in rs)]
            assert len(hit) == 1, f"{name} matched {hit}"
            count[hit[0]] += 1
    lay = LAYOUTS[layout]
    for li in range(model.config.num_hidden_layers):  # vLLM's fused names must land in the same group
        for fused, parts in [("qkv", {"q", "k", "v"}), ("gate_up", {"gate", "up"})]:
            g = [g for g, rs in rx.items() if any(r.match(f"layers.{li}.{'self_attn' if fused == 'qkv' else 'mlp'}.{fused}_proj") for r in rs)]
            exp = "fp8" if (li in lay["fp8_layers"] or parts <= lay["fp8_types"]) else "4bit"
            assert g == [exp], (li, fused, g, exp)
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--layout", default="edges", choices=LAYOUTS)
    ap.add_argument("--group-size", type=int, default=None, help="INT4 group size (default: preset, 128)")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    t0 = time.time()
    out = Path(args.out)

    tok = AutoTokenizer.from_pretrained(args.model)
    # AutoModel -> Gemma3TextModel (no LM head). Load on CPU; llm-compressor onloads layer by layer.
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    print(type(model).__name__, "layers:", model.config.num_hidden_layers)
    print("layout:", args.layout, "linears per group:", check_targets(model, args.layout))

    recipe = build_recipe(args.variant, args.layout, args.group_size)
    kw = {}
    if VARIANTS[args.variant]["calibrated"]:
        ds = load_calib(args.calib, tok, args.num_samples)
        kw = dict(dataset=ds, num_calibration_samples=len(ds), max_seq_length=MAX_SEQ_LEN,
                  pipeline="sequential", sequential_targets=["Gemma3DecoderLayer"])
        print(f"calibration: {len(ds)} samples, {sum(map(len, ds['input_ids']))} tokens")
    else:
        kw = dict(pipeline="datafree")

    # Pass the tokenizer explicitly: AutoProcessor maps gemma3_text to the multimodal
    # Gemma3Processor and fails (no preprocessor_config.json in this text-only repo).
    oneshot(model=model, recipe=recipe, processor=tok, **kw)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)
    for name in PASSTHROUGH:
        src = Path(args.model) / name
        if src.is_dir():
            shutil.copytree(src, out / name, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy2(src, out / name)

    meta = dict(variant=args.variant, seed=SEED, max_seq_len=MAX_SEQ_LEN,
                num_samples=args.num_samples if VARIANTS[args.variant]["calibrated"] else 0,
                layout=args.layout, group_size=args.group_size or "preset", targets=dict(zip(["fp8", "4bit"], layout_targets(args.layout))), recipe=str(recipe),
                minutes=round((time.time() - t0) / 60, 1))
    (out / "quantization_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
