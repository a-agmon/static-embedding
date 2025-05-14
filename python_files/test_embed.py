def test_embed_basic():
    from static_embed import Embedder

    embedder = Embedder()
    texts = ["foo", "bar"]
    embeddings = embedder.embed(texts)

    assert len(embeddings) == len(texts)
    dim = len(embeddings[0])
    assert all(len(e) == dim for e in embeddings)
    # Check values are finite
    assert all(all(isinstance(x, float) and abs(x) < 10 for x in emb) for emb in embeddings)

# New test: constructing with an explicit URL (should behave identically)

def test_embed_with_custom_url():
    from static_embed import Embedder

    custom_url = "https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/main/0_StaticEmbedding"
    embedder = Embedder(custom_url)
    texts = ["foo", "bar"]
    embeddings = embedder.embed(texts)

    assert len(embeddings) == len(texts)
    dim = len(embeddings[0])
    assert all(len(e) == dim for e in embeddings) 