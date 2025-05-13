# static_embed

**static_embed** is an educational (but fully operatioal) library and and repository that shows how to use Static Embedding with Rust.
Static sentence-embedding library written in Rust (Candle + tokenizers) and exported to Python via [PyO3](https://pyo3.rs/) and [maturin](https://github.com/PyO3/maturin).

## Features

* Pure-Rust embedding implementation (no Python runtime dependencies at inference time)
* CPU-only, self-contained model weights (downloaded on first use)
* Python bindings that expose an easy-to-use `Embedder` class

---

## Installation (for users)

### Prerequisites
* Python ≥ 3.8  (CPython)
* A Rust toolchain (stable) – install with `rustup`

### Quick install into the current **virtual env**
```bash
pip install maturin  # once
# From the repository root
maturin develop --release
```
This builds the Rust crate as a Python extension and installs it into the environment.

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

## Development workflow

1. **Set up a virtual environment**
```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
```
2. **Build the extension in editable mode**
```bash
maturin develop                      # or `--release` for optimised builds
```
3. **Run the Python example**
```bash
python python_example.py
```
4. **Run the test-suite**
```bash
pytest -q
```

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

## Publishing to PyPI
(Requires an API token in `$POETRY_PYPI_TOKEN_PYPI` or `~/.pypirc`)
```bash
maturin build --release --skip-auditwheel  # wheels in target/wheels/
maturin publish --skip-existing           # upload
```

---

## License
MIT © Your Name 