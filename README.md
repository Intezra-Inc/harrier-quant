# harrier-quant

4-bit quantization of microsoft/harrier-oss-v1-27b for 24 GB cards.

- [results.md](results.md): fidelity gate, retrieval and code-retrieval results, context limits; recommended variant `w4a16-gptq-late12-attn-g64`
- [COMMANDS.md](COMMANDS.md): how to reproduce
- `recipes/`, `scripts/`, `calib/` (calibration set), `eval/` (scores; embedding .npy files omitted), `logs/`

Quantized weights are not in this repo.

## Licence

The code and results in this repository are under the Apache License 2.0 (`LICENSE`). The base
model, [microsoft/harrier-oss-v1-27b](https://huggingface.co/microsoft/harrier-oss-v1-27b), is under
MIT.

Two folders contain text from public datasets, which stays under each dataset's own licence:

- `calib/calib.jsonl`: the calibration set. Every row records its source, revision and licence, and
  [`calib/SOURCES.md`](calib/SOURCES.md) lists them (Apache-2.0, ODC-By-1.0, and per-file
  MIT/Apache/BSD/ISC code).
- `eval/cosine/items.jsonl`: excerpts from nine MTEB evaluation tasks (ArguAna, BelebeleRetrieval,
  CodeSearchNetRetrieval, FiQA2018, LEMBSummScreenFDRetrieval, MLQARetrieval, NFCorpus, SCIDOCS,
  SciFact), used to compare embeddings across variants.
