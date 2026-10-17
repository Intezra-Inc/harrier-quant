#!/usr/bin/env python
"""Build the Harrier quantization calibration set from permissively licensed public data.

Every source is pinned to an exact dataset revision, and sampling uses a single fixed
seed, so re-running this script reproduces calib.jsonl byte-for-byte (given the same
pyarrow/parquet row-group layout, which is a property of the pinned revision).

Rows are read one parquet row group at a time over HTTP (HfFileSystem), so the
multi-GB FineWeb / GitHub shards are never downloaded in full.

Usage: python build_calib.py --model /data/models/harrier-oss-v1-27b --out /data/models/harrier-quant/calib
"""
import argparse
import ast
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

SEED = 42
MAX_TOKENS = 2048

WEB_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
QA_QUERY = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
CODE_QUERY = "Instruct: Given a natural language description of a function, retrieve the code that implements it\nQuery: "

SOURCES = {
    "gooaq": dict(
        repo="sentence-transformers/gooaq", rev="b089f728748a068b7bc5234e5bcf5b25e3c8279c",
        license="Apache-2.0",
        note="Parquet mirror of allenai/gooaq (Apache-2.0, https://github.com/allenai/gooaq).",
    ),
    "aya": dict(
        repo="CohereLabs/aya_dataset", rev="f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
        license="Apache-2.0", note="Human-written multilingual prompts (inputs field only).",
    ),
    "fineweb2": dict(
        repo="HuggingFaceFW/fineweb-2", rev="af9c13333eb981300149d5ca60a8e9d659b276b9",
        license="ODC-By-1.0",
        note="Attribution-only licence; also subject to Common Crawl terms of use.",
    ),
    "fineweb_edu": dict(
        repo="HuggingFaceFW/fineweb-edu", rev="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        license="ODC-By-1.0",
        note="sample/10BT subset. Attribution-only licence; also subject to Common Crawl terms of use.",
    ),
    "code": dict(
        repo="codeparrot/github-code-clean", rev="c48d40f9e70f0196f8236901ee35807f7d6c44c0",
        license="Per-file: MIT / Apache-2.0 / BSD-2-Clause / BSD-3-Clause / ISC only",
        note="Collection is Apache-2.0; each sampled file was filtered on its own `license` field.",
    ),
}

PERMISSIVE_CODE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"}
# (aya language_code, fineweb-2 config, iso code)
LANGS = [("arb", "arb_Arab", "ar"), ("deu", "deu_Latn", "de"), ("spa", "spa_Latn", "es"),
         ("jpn", "jpn_Jpan", "ja"), ("hin", "hin_Deva", "hi"), ("zho", "cmn_Hani", "zh")]

fs = HfFileSystem()


def ds_path(src, path):
    s = SOURCES[src]
    return f"datasets/{s['repo']}@{s['rev']}/{path}"


def list_parquet(src, prefix):
    return sorted(p for p in fs.glob(ds_path(src, f"{prefix}/*.parquet")))


def read_row_group(path, rng, columns):
    """Read one rng-chosen row group from a remote parquet file."""
    with fs.open(path, "rb", block_size=8 * 2**20) as f:
        pf = pq.ParquetFile(f)
        rg = rng.randrange(pf.num_row_groups)
        return pf.read_row_group(rg, columns=columns).to_pylist(), rg


def rec(src, bucket, role, lang, text, **extra):
    s = SOURCES[src]
    return dict(source=src, dataset=s["repo"], revision=s["rev"], license=s["license"],
                bucket=bucket, role=role, lang=lang, text=text, **extra)


def build_gooaq(rng, n_pairs=100):
    files = list_parquet("gooaq", "pair")
    out, seen = [], set()
    while len(out) < 2 * n_pairs:
        path = files[rng.randrange(len(files))]
        rows, rg = read_row_group(path, rng, ["question", "answer"])
        rng.shuffle(rows)
        for r in rows:
            q, a = (r["question"] or "").strip(), (r["answer"] or "").strip()
            if len(q) < 10 or len(a) < 80 or q in seen:
                continue
            seen.add(q)
            pid = f"gooaq-{len(out) // 2}"
            out.append(rec("gooaq", "short_en", "query", "en", WEB_QUERY + q, pair_id=pid, origin=f"{path.split('/')[-1]}#rg{rg}"))
            out.append(rec("gooaq", "short_en", "passage", "en", a, pair_id=pid, origin=f"{path.split('/')[-1]}#rg{rg}"))
            if len(out) >= 2 * n_pairs:
                break
    return out


def build_multilingual(rng, n_query=10, n_passage=10):
    aya_files = list_parquet("aya", "data")
    aya_rows = []
    for p in aya_files:
        if "/train-" in p:
            with fs.open(p, "rb") as f:
                aya_rows += pq.read_table(f, columns=["inputs", "language_code", "annotation_type"]).to_pylist()
    out = []
    for aya_code, fw_cfg, iso in LANGS:
        cands = [r["inputs"].strip() for r in aya_rows
                 if r["language_code"] == aya_code and r["annotation_type"] == "original-annotations"
                 and 15 <= len(r["inputs"].strip()) <= 400]
        cands = sorted(set(cands))
        if len(cands) < n_query:
            raise SystemExit(f"Aya: only {len(cands)} candidates for {aya_code}")
        for t in rng.sample(cands, n_query):
            out.append(rec("aya", "multilingual", "query", iso, QA_QUERY + t))

        files = list_parquet("fineweb2", f"data/{fw_cfg}/train")
        path = files[rng.randrange(len(files))]
        rows, rg = read_row_group(path, rng, ["text", "url"])
        rows = [r for r in rows if 400 <= len(r["text"]) <= 6000]
        for r in rng.sample(rows, n_passage):
            out.append(rec("fineweb2", "multilingual", "passage", iso, r["text"].strip(),
                           url=r["url"], origin=f"{fw_cfg}/{path.split('/')[-1]}#rg{rg}"))
    return out


def build_fineweb_edu(rng, n_medium=66, n_long=46):
    files = list_parquet("fineweb_edu", "sample/10BT")
    out = []
    want = {"medium": (1500, 5000, n_medium), "long": (6000, 20000, n_long)}
    got = Counter()
    while any(got[k] < v[2] for k, v in want.items()):
        path = files[rng.randrange(len(files))]
        rows, rg = read_row_group(path, rng, ["text", "url"])
        rng.shuffle(rows)
        for r in rows:
            n = len(r["text"])
            for k, (lo, hi, cap) in want.items():
                if lo <= n <= hi and got[k] < cap:
                    got[k] += 1
                    out.append(rec("fineweb_edu", "long_en" if k == "long" else "short_en", "passage", "en",
                                   r["text"].strip(), url=r["url"], origin=f"{path.split('/')[-1]}#rg{rg}"))
                    break
    return out


def python_doc_pairs(code):
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            src = ast.get_source_segment(code, node)
            if not doc or not src:
                continue
            summary = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
            if 30 <= len(summary) <= 300 and 200 <= len(src) <= 6000:
                pairs.append((summary, src))
    return pairs


def build_code(rng, n_py_pairs=20, n_other_each=10):
    files = list_parquet("code", "data")
    other_langs = ["Java", "GO", "JavaScript", "TypeScript"]  # dataset spells Go as "GO"
    py, other = [], {l: [] for l in other_langs}
    seen_repos = set()
    for _ in range(200):  # cap: fail loudly instead of looping forever if a language is missing
        if len(py) >= n_py_pairs and all(len(v) >= n_other_each for v in other.values()):
            break
        path = files[rng.randrange(len(files))]
        rows, rg = read_row_group(path, rng, ["code", "repo_name", "path", "language", "license"])
        rng.shuffle(rows)
        for r in rows:
            if r["license"] not in PERMISSIVE_CODE or r["repo_name"] in seen_repos:
                continue
            meta = dict(repo=r["repo_name"], file=r["path"], file_license=r["license"],
                        origin=f"{path.split('/')[-1]}#rg{rg}")
            if r["language"] == "Python" and len(py) < n_py_pairs:
                pairs = python_doc_pairs(r["code"])
                if pairs:
                    seen_repos.add(r["repo_name"])
                    py.append((rng.choice(pairs), meta))
            elif r["language"] in other and len(other[r["language"]]) < n_other_each and 500 <= len(r["code"]) <= 20000:
                seen_repos.add(r["repo_name"])
                other[r["language"]].append((r["code"][:4000], meta))
    else:
        raise SystemExit(f"code: quotas not met: python={len(py)} " + str({k: len(v) for k, v in other.items()}))
    out = []
    for i, ((summary, src), meta) in enumerate(py):
        out.append(rec("code", "code", "query", "python", CODE_QUERY + summary, pair_id=f"code-{i}", **meta))
        out.append(rec("code", "code", "passage", "python", src, pair_id=f"code-{i}", **meta))
    for lang, items in other.items():
        for code, meta in items:
            out.append(rec("code", "code", "passage", lang.lower(), code, **meta))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Separate rng per source so changing one source's count doesn't reshuffle the others.
    records = []
    for name, fn in [("gooaq", build_gooaq), ("multilingual", build_multilingual),
                     ("code", build_code), ("fineweb_edu", build_fineweb_edu)]:
        part = fn(random.Random(f"{SEED}-{name}"))
        print(f"{name}: {len(part)} samples", flush=True)
        records += part

    # Exact-duplicate guard
    uniq, seen = [], set()
    for r in records:
        h = hashlib.sha256(r["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            r["sha256"] = h
            uniq.append(r)
    records = uniq
    random.Random(SEED).shuffle(records)

    tok = AutoTokenizer.from_pretrained(args.model)
    for i, r in enumerate(records):
        r["id"] = i
        n = len(tok(r["text"], add_special_tokens=True)["input_ids"])
        r["n_tokens_full"] = n
        r["n_tokens"] = min(n, MAX_TOKENS)

    with open(out / "calib.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by = lambda k: dict(sorted(Counter(r[k] for r in records).items()))
    stats = dict(seed=SEED, max_tokens=MAX_TOKENS, n=len(records),
                 total_tokens=sum(r["n_tokens"] for r in records),
                 truncated=sum(r["n_tokens_full"] > MAX_TOKENS for r in records),
                 by_source=by("source"), by_bucket=by("bucket"), by_role=by("role"), by_lang=by("lang"),
                 code_file_licenses=dict(Counter(r.get("file_license") for r in records if r["source"] == "code")),
                 prompts=dict(web=WEB_QUERY, qa=QA_QUERY, code=CODE_QUERY),
                 sha256_calib_jsonl=hashlib.sha256((out / "calib.jsonl").read_bytes()).hexdigest())
    (out / "calib_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    lines = ["# Calibration data sources and licences", "",
             f"{stats['n']} samples, seed {SEED}, truncated to {MAX_TOKENS} tokens at calibration time "
             f"({stats['total_tokens']} tokens total). Built by `build_calib.py`.", "",
             "Only permissively licensed sources are used (Apache-2.0, ODC-By-1.0, and per-file MIT/Apache/BSD/ISC code). "
             "No MS MARCO, Natural Questions, Wikipedia, MIRACL, CodeSearchNet, internal/private data, "
             "or data from the evaluation tasks.", "",
             "| Source | Dataset | Revision | Licence | Samples | Notes |", "|---|---|---|---|---|---|"]
    counts = Counter(r["source"] for r in records)
    for k, s in SOURCES.items():
        lines.append(f"| {k} | [{s['repo']}](https://huggingface.co/datasets/{s['repo']}) | `{s['rev']}` | {s['license']} | {counts[k]} | {s['note']} |")
    lines += ["", "Breakdown by bucket: " + ", ".join(f"{k}={v}" for k, v in stats["by_bucket"].items()),
              "Breakdown by language: " + ", ".join(f"{k}={v}" for k, v in stats["by_lang"].items()),
              "Code file licences: " + ", ".join(f"{k}={v}" for k, v in stats["code_file_licenses"].items()), "",
              "Query instruction prefixes (Harrier format `Instruct: <task>\\nQuery: <text>`; passages have no prefix):", ""]
    lines += [f"- `{p!r}`" for p in stats["prompts"].values()]
    lines += ["", "Every row of `calib.jsonl` records its source dataset, revision, licence, and origin "
              "(shard + row group; URL for web text; repo, path and licence for code)."]
    (out / "SOURCES.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
