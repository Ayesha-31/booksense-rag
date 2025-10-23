# search_in_milvus.py
# Usage: python search_in_milvus.py "What do readers love about The Night Circus?"

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import argparse
import textwrap
import os, math
import pandas as pd, numpy as np

COLLECTION = "booksense_chunks"
# HOST = "127.0.0.1"
# PORT = "19530"
from dotenv import load_dotenv

load_dotenv()

# ---- read secrets from env (recommended) ----
ZILLIZ_URI   = os.environ.get("ZILLIZ_URI")    # e.g., https://in03-....zilliz.com
ZILLIZ_USER  = os.environ.get("ZILLIZ_USER")   # e.g., db_xxx
ZILLIZ_TOKEN = os.environ.get("ZILLIZ_TOKEN")  # password/token

assert ZILLIZ_URI and ZILLIZ_USER and ZILLIZ_TOKEN, "Set ZILLIZ_URI, ZILLIZ_USER, ZILLIZ_TOKEN env vars."

def connect():
    # connections.connect("default", host=HOST, port=PORT)
    connections.connect(
    alias="default",
    uri=ZILLIZ_URI,   # DO NOT append :19530 for Zilliz Cloud
    user=ZILLIZ_USER,
    password=ZILLIZ_TOKEN,
    secure=True,
)
    col = Collection(COLLECTION)   # must already exist and be indexed
    col.load()
    return col

#Converting the search query into a vector

def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Optional sanity check (uncomment if you want a hard fail on mismatch)
    # assert model.get_sentence_embedding_dimension() == 384
    return model


def pretty_snippet(s, n=300):
    s = " ".join(str(s).split())     # collapse whitespace
    return (s[:n] + "…") if len(s) > n else s

def search(col, model, query, k=5, min_score=None, output_fields=None):
    allowed_fields = {f.name for f in col.schema.fields}
    # ask for everything you plan to display
    wanted = ["work_id", "book_title", "authors", "avg_rating", "chunk_text"] if output_fields is None else output_fields
    output_fields = [f for f in wanted if f in allowed_fields]

    q_emb = model.encode(query, normalize_embeddings=True)

    res = col.search(
        data=[q_emb],
        anns_field="embedding",
        # For AUTOINDEX/HNSW on Serverless, you can omit params entirely
        param={"metric_type": "COSINE"},  # remove nprobe; it's IVF-specific
        limit=k,
        output_fields=output_fields,
    )

    hits = []
    for h in res[0]:
        score = float(h.distance)  # COSINE similarity in [0,1], higher is better
        if (min_score is not None) and (score < min_score):
            continue
        row = {f: h.entity.get(f) for f in output_fields if h.entity.get(f) is not None}
        row["score"] = score
        hits.append(row)
    return hits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="User question or search text")
    parser.add_argument("--k", type=int, default=5, help="top-K results")
    parser.add_argument("--min_score", type=float, default=None, help="optional cosine threshold")
    args = parser.parse_args()

    print("Connecting to Milvus…")
    col = connect()
    model = load_model()

    print(f"Searching: {args.query!r}")
    hits = search(col, model, args.query, k=args.k, min_score=args.min_score)

    if not hits:
        print("No results.")
        return

    print("\nTop results:\n" + "-"*80)
    for i, h in enumerate(hits, 1):
        title = h.get("book_title", "(unknown title)")
        authors = h.get("authors", "(unknown author)")
        work_id = h.get("work_id", "")
        avg = h.get("avg_rating", "")
        snippet = pretty_snippet(h.get("chunk_text", ""))
        print(f"{i}. score={h['score']:.3f} | work_id={work_id} | {title} — {authors} (avg {avg})")
        print(textwrap.fill(snippet, width=100))
        print("-"*80)

if __name__ == "__main__":
    main()
