import sys
sys.stderr = sys.__stderr__ 

from static_embed import Embedder

if __name__ == "__main__":
    # Default model
    embedder = Embedder()

    # Uncomment to load from a custom location.
    # custom_url = "https://my-cdn.example.com/static-retrieval-mrl-en-v1"
    # embedder = Embedder(custom_url)

    texts = ["Hello world!", "Rust and Python interop via PyO3."]
    embeddings = embedder.embed(texts)

    print("Generated", len(embeddings), "embeddings with dimension", len(embeddings[0]))
    for text, emb in zip(texts, embeddings):
        print(f"{text} -> first 5 dims: {emb[:5]}") 