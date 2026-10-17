# Calibration data sources and licences

512 samples, seed 42, truncated to 2048 tokens at calibration time (206837 tokens total). Built by `build_calib.py`.

Only permissively licensed sources are used (Apache-2.0, ODC-By-1.0, and per-file MIT/Apache/BSD/ISC code). No MS MARCO, Natural Questions, Wikipedia, MIRACL, CodeSearchNet, internal/private data, or data from the evaluation tasks.

| Source | Dataset | Revision | Licence | Samples | Notes |
|---|---|---|---|---|---|
| gooaq | [sentence-transformers/gooaq](https://huggingface.co/datasets/sentence-transformers/gooaq) | `b089f728748a068b7bc5234e5bcf5b25e3c8279c` | Apache-2.0 | 200 | Parquet mirror of allenai/gooaq (Apache-2.0, https://github.com/allenai/gooaq). |
| aya | [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` | Apache-2.0 | 60 | Human-written multilingual prompts (inputs field only). |
| fineweb2 | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | `af9c13333eb981300149d5ca60a8e9d659b276b9` | ODC-By-1.0 | 60 | Attribution-only licence; also subject to Common Crawl terms of use. |
| fineweb_edu | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By-1.0 | 112 | sample/10BT subset. Attribution-only licence; also subject to Common Crawl terms of use. |
| code | [codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) | `c48d40f9e70f0196f8236901ee35807f7d6c44c0` | Per-file: MIT / Apache-2.0 / BSD-2-Clause / BSD-3-Clause / ISC only | 80 | Collection is Apache-2.0; each sampled file was filtered on its own `license` field. |

Breakdown by bucket: code=80, long_en=46, multilingual=120, short_en=266
Breakdown by language: ar=20, de=20, en=312, es=20, go=10, hi=20, ja=20, java=10, javascript=10, python=40, typescript=10, zh=20
Code file licences: bsd-2-clause=7, bsd-3-clause=13, apache-2.0=47, mit=13

Query instruction prefixes (Harrier format `Instruct: <task>\nQuery: <text>`; passages have no prefix):

- `'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '`
- `'Instruct: Given a question, retrieve passages that answer the question\nQuery: '`
- `'Instruct: Given a natural language description of a function, retrieve the code that implements it\nQuery: '`

Every row of `calib.jsonl` records its source dataset, revision, licence, and origin (shard + row group; URL for web text; repo, path and licence for code).
