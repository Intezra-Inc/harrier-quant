#!/usr/bin/env python
"""Per-layer / per-module-type quantization sensitivity of harrier-oss-v1-27b embeddings.

Simulated (fake) quantization in BF16 transformers, applied in place to the decoder Linear weights:
  int4 : symmetric INT4, group 128, round-to-nearest (proxy for the 4-bit middle layers)
  fp8  : FP8 E4M3, per-output-channel scale (weights of FP8_DYNAMIC layers; dynamic per-token
         activation quantization is not simulated)
Probe set: calibration samples only (calib/calib.jsonl), never eval data, so layer selection is
not tuned on the evaluation tasks. Metric: cosine of the L2-normalized last-token embedding vs BF16.

Experiments:
  bf16, all_int4, current_recipe (FP8 edges 0-5/56-61, INT4 middle),
  restore_layer_<i>: all INT4 except layer i in FP8 (all 62 layers),
  restore_type_<t>:  all INT4 except module type t in FP8 in every layer.
Output: eval/sensitivity/results.json

Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python sensitivity.py
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path("/data/models/harrier-quant")
MODEL = "/data/models/harrier-oss-v1-27b"
SEED = 42
N_PER_BUCKET = 16
MAX_LEN = 4096
TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def fq_int4(w, group=128):
    o, i = w.shape
    g = w.float().reshape(o, i // group, group)
    scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
    q = (g / scale).round().clamp(-8, 7)
    return (q * scale).reshape(o, i).to(w.dtype)


def fq_fp8(w):
    wf = w.float()
    scale = wf.abs().amax(1, keepdim=True).clamp(min=1e-8) / FP8_MAX
    return ((wf / scale).to(torch.float8_e4m3fn).float() * scale).to(w.dtype)


def main(combos=False):
    random.seed(SEED); torch.manual_seed(SEED)
    out_dir = ROOT / "eval" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(ROOT / "calib" / "calib.jsonl")]
    by = defaultdict(list)
    for r in rows:
        by[r["bucket"]].append(r)
    rng = random.Random(SEED)
    probe = []
    for b in sorted(by):
        probe += rng.sample(by[b], min(N_PER_BUCKET, len(by[b])))
    buckets = np.array([r["bucket"] for r in probe])

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = [tok(r["text"], truncation=True, max_length=MAX_LEN, return_tensors="pt")["input_ids"] for r in probe]
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    linears = {}  # (layer, type) -> module
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "layers." in name:
            li = int(name.split("layers.")[1].split(".")[0])
            linears[(li, name.rsplit(".", 1)[1])] = m
    n_layers = model.config.num_hidden_layers
    orig = {k: m.weight.data.to("cpu", copy=True) for k, m in linears.items()}

    def qweight(key, fmt):
        # recomputed on GPU each time: only the modules that change between experiments are touched
        w = orig[key].to("cuda", non_blocking=True)
        return w if fmt == "bf16" else (fq_int4(w) if fmt == "int4" else fq_fp8(w))

    state = {k: None for k in linears}

    def apply(assign):
        for k, fmt in assign.items():
            if state[k] != fmt:
                linears[k].weight.data.copy_(qweight(k, fmt), non_blocking=True)
                state[k] = fmt
        torch.cuda.synchronize()

    @torch.no_grad()
    def embed():
        embs = []
        for ids in enc:
            h = model(input_ids=ids.cuda()).last_hidden_state[0, -1].float()
            embs.append(F.normalize(h, dim=0).cpu())
        return torch.stack(embs)

    def assign_for(fp8_layers=(), fp8_types=(), default="int4"):
        return {k: ("fp8" if (k[0] in fp8_layers or k[1] in fp8_types) else default) for k in linears}

    def stats(e, ref):
        cos = (e * ref).sum(1).numpy()
        s = {"mean_1mcos": float((1 - cos).mean()), "median": float(np.median(cos)), "p5": float(np.percentile(cos, 5)),
             "min": float(cos.min())}
        for b in sorted(set(buckets)):
            c = cos[buckets == b]
            s[f"{b}_median"], s[f"{b}_min"] = float(np.median(c)), float(c.min())
        return s

    results = {"probe": {"n": len(probe), "buckets": {b: int((buckets == b).sum()) for b in set(buckets)},
                         "max_len": MAX_LEN, "tokens": int(sum(e.shape[1] for e in enc))}, "experiments": {}}
    t0 = time.time()
    apply({k: "bf16" for k in linears})
    ref = embed()
    print(f"bf16 reference: {time.time() - t0:.0f}s for {len(probe)} samples", flush=True)

    edges = set(range(6)) | set(range(n_layers - 6, n_layers))
    if combos:
        # Candidate recipes: current edge layout plus module types in FP8 in every layer.
        cands = {"cur": (), "cur+v": ("v_proj",), "cur+v+k": ("v_proj", "k_proj"), "cur+v+o": ("v_proj", "o_proj"),
                 "cur+attn": ("q_proj", "k_proj", "v_proj", "o_proj"), "cur+down": ("down_proj",),
                 "cur+v+down": ("v_proj", "down_proj"), "cur+v+o+down": ("v_proj", "o_proj", "down_proj"),
                 "cur+attn+down": ("q_proj", "k_proj", "v_proj", "o_proj", "down_proj")}
        late = set(range(n_layers - 12, n_layers))  # 50-61: the most sensitive block in the per-layer sweep
        experiments = [(f"combo_{n}", assign_for(fp8_layers=edges, fp8_types=set(t))) for n, t in cands.items()]
        for n, t in {"late12": (), "late12+v": ("v_proj",), "late12+v+o": ("v_proj", "o_proj"),
                     "late12+attn": ("q_proj", "k_proj", "v_proj", "o_proj"), "late12+v+down": ("v_proj", "down_proj")}.items():
            experiments.append((f"combo_{n}", assign_for(fp8_layers=late, fp8_types=set(t))))
    else:
        experiments = [("all_int4", assign_for()), ("current_recipe", assign_for(fp8_layers=edges))]
        experiments += [(f"restore_layer_{i}", assign_for(fp8_layers={i})) for i in range(n_layers)]
        experiments += [(f"restore_type_{t}", assign_for(fp8_types={t})) for t in TYPES]
        experiments += [("all_fp8", {k: "fp8" for k in linears})]

    for name, assign in experiments:
        t = time.time()
        apply(assign)
        results["experiments"][name] = s = stats(embed(), ref)
        print(f"{name:22s} mean(1-cos)={s['mean_1mcos']:.5f} median={s['median']:.5f} p5={s['p5']:.5f} "
              f"min={s['min']:.5f}  ({time.time() - t:.0f}s)", flush=True)
        (out_dir / ("combos.json" if combos else "results.json")).write_text(json.dumps(results, indent=2))

    if combos:
        return
    base = results["experiments"]["all_int4"]["mean_1mcos"]
    gain = {i: base - results["experiments"][f"restore_layer_{i}"]["mean_1mcos"] for i in range(n_layers)}
    results["layer_ranking"] = sorted(gain, key=gain.get, reverse=True)
    results["layer_gain"] = gain
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("top layers by FP8 gain:", results["layer_ranking"][:24], flush=True)


if __name__ == "__main__":
    import sys
    main(combos="--combos" in sys.argv)
