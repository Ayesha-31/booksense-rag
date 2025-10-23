# insert_to_milvus_batched.py
import os, math
import pandas as pd, numpy as np
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from dotenv import load_dotenv

load_dotenv()

# ---- read secrets from env (recommended) ----
ZILLIZ_URI   = os.environ.get("ZILLIZ_URI")    # e.g., https://in03-....zilliz.com
ZILLIZ_USER  = os.environ.get("ZILLIZ_USER")   # e.g., db_xxx
ZILLIZ_TOKEN = os.environ.get("ZILLIZ_TOKEN")  # password/token

assert ZILLIZ_URI and ZILLIZ_USER and ZILLIZ_TOKEN, "Set ZILLIZ_URI, ZILLIZ_USER, ZILLIZ_TOKEN env vars."

connections.connect(
    alias="default",
    uri=ZILLIZ_URI,   # DO NOT append :19530 for Zilliz Cloud
    user=ZILLIZ_USER,
    password=ZILLIZ_TOKEN,
    secure=True,
)

COLLECTION = "booksense_chunks"
CSV = "data/booksense_chunks.csv"
NPY = "data/booksense_embeddings.npy"
BATCH = 1000

df = pd.read_csv(CSV)
emb = np.load(NPY)
assert len(df) == emb.shape[0], "CSV rows and embeddings must match"

# sanitize/nulls/lengths
df = df.fillna({"work_id": "", "book_title": "", "authors": "", "chunk_text": ""})
df["avg_rating"] = df["avg_rating"].fillna(0.0)

# enforce max lengths to avoid insert errors
df["work_id"]    = df["work_id"].astype(str).str.slice(0, 32)
df["book_title"] = df["book_title"].astype(str).str.slice(0, 256)
df["authors"]    = df["authors"].astype(str).str.slice(0, 256)
df["chunk_text"] = df["chunk_text"].astype(str).str.slice(0, 2000)

# (Re)create collection if missing
if not utility.has_collection(COLLECTION):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="work_id", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="book_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="avg_rating", dtype=DataType.FLOAT),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=int(emb.shape[1])),
    ]
    schema = CollectionSchema(fields, description="BookSense review chunks with metadata")
    col = Collection(COLLECTION, schema)
else:
    col = Collection(COLLECTION)

# Insert in batches
n = len(df)
steps = math.ceil(n / BATCH)
print(f"Inserting {n:,} rows in {steps} batches of {BATCH}…")

for i in tqdm(range(0, n, BATCH)):
    j = min(i + BATCH, n)
    rows = []
    for k in range(i, j):
        rows.append({
            "work_id":    str(df.at[k, "work_id"])[:32],
            "book_title": str(df.at[k, "book_title"])[:256],
            "authors":    str(df.at[k, "authors"])[:256],
            "avg_rating": float(df.at[k, "avg_rating"]) if pd.notna(df.at[k, "avg_rating"]) else 0.0,
            "chunk_text": str(df.at[k, "chunk_text"])[:2000],
            "embedding":  emb[k].tolist(),  # list[float] of length dim
        })
    col.insert(rows)

col.flush()  # ensure data persisted

# Build index (recommended for Serverless: AUTOINDEX; or skip to let Serverless handle)
try:
    col.create_index(
        field_name="embedding",
        index_params={"metric_type": "COSINE", "index_type": "AUTOINDEX"},
    )
except Exception as e:
    print(f"(index note) {e}")

col.load()
print("Batched insert complete, index built/verified, collection loaded.")
print("Entities:", col.num_entities)
