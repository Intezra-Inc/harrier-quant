# harrier-quant

4-bit quantization of microsoft/harrier-oss-v1-27b for 24 GB cards.

- [results.md](results.md): fidelity gate, retrieval and code-retrieval results, context limits; recommended variant `w4a16-gptq-late12-attn-g64`
- [COMMANDS.md](COMMANDS.md): how to reproduce
- `recipes/`, `scripts/`, `calib/` (calibration set), `eval/` (scores; embedding .npy files omitted), `logs/`

Quantized weights are not in this repo.
