# static_embed

**static_embed** is an educational (but fully operatioal) library and and repository that shows how to use [Static Embedding](https://huggingface.co/blog/static-embeddings) with Rust. Its a library written in Rust (Candle + tokenizers) and exported to Python via [PyO3](https://pyo3.rs/) and [maturin](https://github.com/PyO3/maturin).

## Features

* Pure-Rust embedding implementation (no Python runtime dependencies at inference time)
* High performance embeddings
* CPU-only, self-contained model weights (downloaded on first use)
* Python bindings that expose an easy-to-use `Embedder` class

---

## Installation 

```bash
pip install static_embed
```
---

## Usage
```python
from static_embed import Embedder

# 1. Use the default public model (no args)
embedder = Embedder()

# 2. OR specify your own base-URL that hosts the weights/tokeniser
#    (must contain the same two files: ``model.safetensors`` & ``tokenizer.json``)
# custom_url = "https://my-cdn.example.com/static-retrieval-mrl-en-v1"
# embedder = Embedder(custom_url)

texts = ["Hello world!", "Rust + Python via PyO3"]
embeddings = embedder.embed(texts)

print(len(embeddings), "embeddings", "dimension", len(embeddings[0]))
```

---

## Benchmarking

To benchmark the embedding speed and resource usage, you can run the provided benchmark script. This script downloads a dataset, splits it into chunks, and measures embedding throughput and memory usage.

**Dependencies:**
- `datasets`
- `tqdm`
- `psutil`

Install them with:
```bash
pip install datasets tqdm psutil
```

**Run the benchmark:**
```bash
python python_files/benchmark.py
```

Example output:
```
Embedded 25,000 chunks in 1.54s → 16,205 chunks/sec
CPU util.         : 24.1 %
RAM used          : 16509 MB of 38655 MB
Process RSS       : 3102 MB
```

You can adjust the number of text chunks and batch size by editing the `TARGET_CHUNKS` and `BATCH_SIZE` variables at the top of the script.

---

## Project layout

```
.
├── Cargo.toml         # Rust crate manifest
├── src/               # Rust source (embedder logic + PyO3 bindings)
├── models/            # (Auto-downloaded) model weights live here
├── python_example.py  # Minimal demo script
├── test_embed.py      # Pytest verifying the binding
└── pyproject.toml     # Build configuration for maturin
```

---

## License
MIT © Alon Agmon 