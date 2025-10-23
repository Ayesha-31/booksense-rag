# make_embeddings.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 1️Load chunked data
INPUT = "data/booksense_chunks.csv"
df = pd.read_csv(INPUT)
print(f"Loaded {len(df):,} chunks")

# 2 Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3 Generate embeddings
embeddings = model.encode(df["chunk_text"].tolist(), show_progress_bar=True)

# 4Save to file
#We told NumPy to save your embeddings in a binary format, not as human-readable text.
#That’s why when we open the .npy file in a text editor, we see weird symbols like, To see actual vectors un check_embeddings.py file.
np.save("data/booksense_embeddings.npy", embeddings)
print(" Embeddings saved to data/booksense_embeddings.npy")
print("Shape:", embeddings.shape)
