#![allow(unsafe_op_in_unsafe_fn, non_local_definitions)]
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

mod embedder;
pub use embedder::Embedder;
use embedder::Embedder as RustEmbedder;

#[pyclass(name = "Embedder")]
struct PyEmbedder {
    inner: RustEmbedder,
}

#[pymethods]
impl PyEmbedder {
    /// Construct a new `Embedder`.
    ///
    /// Parameters
    /// ----------
    /// model_url : str | None, optional
    ///     Base URL at which the model files (``model.safetensors`` and ``tokenizer.json``)
    ///     are hosted.  If omitted, the built-in default URL pointing at the public
    ///     Sentence-Transformers repository is used.
    #[new]
    #[pyo3(signature = (model_url = None))]
    fn new(model_url: Option<String>) -> PyResult<Self> {
        let builder_result = if let Some(url) = model_url.as_ref() {
            RustEmbedder::new_with_url(Some(url.as_str()))
        } else {
            RustEmbedder::new()
        };

        Ok(PyEmbedder {
            inner: builder_result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    /// Embed a list of strings and return a list of float vectors.
    fn embed(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner
            .embed_batch_vec(&refs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Python module definition
#[pymodule]
fn static_embed(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEmbedder>()?;
    Ok(())
}

// Unit tests remain usable from Rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedder_roundtrip() {
        let emb = PyEmbedder::new(None).unwrap();
        let result = emb.embed(vec!["Hello".to_string()]).unwrap();
        assert_eq!(result.len(), 1);
    }
}
