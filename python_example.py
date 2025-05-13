import sys
sys.stderr = sys.__stderr__ 

from static_embed import Embedder
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
    
    # Demonstrate cosine similarity
    if len(embeddings) >= 2:
        sim = cosine_similarity(embeddings[0], embeddings[1])
        print(f"\nCosine similarity between '{texts[0]}' and '{texts[1]}': {sim:.4f}")
        
    # Add a third text to show more comparisons
    new_text = "Hello everyone in the world!"
    new_embedding = embedder.embed([new_text])[0]
    
    # Compare with first text (should be highly similar)
    sim1 = cosine_similarity(embeddings[0], new_embedding)
    print(f"Cosine similarity between '{texts[0]}' and '{new_text}': {sim1:.4f}")
    
    # Compare with second text (should be less similar)
    sim2 = cosine_similarity(embeddings[1], new_embedding)
    print(f"Cosine similarity between '{texts[1]}' and '{new_text}': {sim2:.4f}") 