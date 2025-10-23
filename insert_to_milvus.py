# This wont work. our script tried to insert everything at once (~264 MB). 
# So if you run this script, it will fail with an error about the data size.
# So we need to batch the inserts and only build the index after all rows are in.
# So run insert_to_milvus_batched.py instead.
# So this script is just for reference.
#------------------------------------------------------------------------------------------------
# insert_to_milvus
import pandas as pd, numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

connections.connect("default", host="127.0.0.1", port="19530")

df = pd.read_csv("data/booksense_chunks.csv")
emb = np.load("data/booksense_embeddings.npy")
assert len(df) == emb.shape[0], "CSV rows and embeddings must match"

#Defining the collection (table)
#FieldSchema defines columns (like SQL schema).
#id → Milvus auto-generates it.
#work_id → identifies which book/review this chunk belongs to.
#chunk_text → the actual text snippet.
#embedding → your vector (dim = 384 for MiniLM).

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="work_id", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=emb.shape[1]),
]
schema = CollectionSchema(fields, description="BookSense review chunks")

col = Collection("booksense_chunks", schema)

#convert each column into plain Python lists because Milvus expects array-like input (not pandas DataFrames).

data = [
    df["work_id"].astype(str).tolist(),
    df["chunk_text"].tolist(),
    emb.tolist(),
]

#Inserting into Milvus
col.insert(data)

#Building the index
#This helps Milvus find similar chunks faster.
#This tells Milvus how to search efficiently:
#metric_type = COSINE → use cosine similarity to compare vectors.
#IVF_FLAT → an indexing algorithm for fast nearest-neighbor search.
#nlist=1024 → controls index granularity (you can tune this).
#Finally, .load() brings the collection into memory, ready for queries.

col.create_index(
    field_name="embedding",
    index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}},
)
col.load()
print(f"Inserted {len(df):,} chunks and built index in Milvus.")
