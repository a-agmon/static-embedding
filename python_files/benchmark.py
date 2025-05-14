import sys
sys.stderr = sys.__stderr__ 

from datasets import load_dataset
from static_embed import Embedder
from tqdm import tqdm
import time, psutil, os
from math import ceil

def chunk_text(text, max_chars=512):
    """Yield ~max_chars chunks split on word boundaries."""
    text = text.strip()
    while text:
        chunk = text[:max_chars]
        end   = chunk.rfind(' ')
        if end == -1:
            end = max_chars
        yield chunk[:end]
        text = text[end:].lstrip()

TARGET_CHUNKS = 25_000      # change to 1_000, 25_000, … as needed
chunks = []

# streaming=True ⇒ an IterableDataset → no Arrow cache, no FS error
ds_stream = load_dataset("tweet_eval", "sentiment",
                         split="train", streaming=True)

for row in tqdm(ds_stream, desc="Collecting chunks"):
    for c in chunk_text(row["text"]):
        chunks.append(c)
        if len(chunks) >= TARGET_CHUNKS:
            break
    if len(chunks) >= TARGET_CHUNKS:
        break

print(f"Collected {len(chunks):,} chunks "
      f"(avg length ≈{sum(map(len, chunks))//len(chunks)} chars)")

embedder = Embedder() # or Embedder(custom_url)

start = time.perf_counter()
embs  = embedder.embed(chunks)
elapsed = time.perf_counter() - start

print(f"\nEmbedded {len(embs):,} chunks in {elapsed:.2f}s "
      f"→ {len(embs)/elapsed:,.0f} chunks/sec")

# Quick resource snapshot
print(f"CPU util.         : {psutil.cpu_percent()} %")
print(f"RAM used          : {psutil.virtual_memory().used/1e6:.0f} MB "
      f"of {psutil.virtual_memory().total/1e6:.0f} MB")
print(f"Process RSS       : {psutil.Process(os.getpid()).memory_info().rss/1e6:.0f} MB")

BATCH_SIZE = 5_000          # 🔧 tweak for your GPU/CPU RAM
embedder   = Embedder()     # or Embedder(custom_url)

num_batches = ceil(len(chunks) / BATCH_SIZE)
all_embs    = []            # will hold the final embeddings

start = time.perf_counter()

for i in tqdm(range(num_batches), desc="Embedding"):
    batch = chunks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
    all_embs.extend(embedder.embed(batch))

elapsed = time.perf_counter() - start
assert len(all_embs) == len(chunks)

print(f"\nEmbedded {len(all_embs):,} chunks in {elapsed:.2f}s "
      f"→ {len(all_embs)/elapsed:,.0f} chunks/sec")

# Optional resource snapshot
print(f"CPU util.         : {psutil.cpu_percent()} %")
print(f"RAM used          : {psutil.virtual_memory().used/1e6:.0f} MB "
      f"of {psutil.virtual_memory().total/1e6:.0f} MB")
print(f"Process RSS       : {psutil.Process(os.getpid()).memory_info().rss/1e6:.0f} MB") 