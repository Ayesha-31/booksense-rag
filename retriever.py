# retriever.py
import os
import math
import json

# streamlit is optional for local; import defensively
try:
    import streamlit as st
except Exception:
    class _Stub:
        def __getattr__(self, _): return {}
        def cache_resource(self, *a, **k):
            def deco(f): return f
            return deco
        secrets = {}
    st = _Stub()

from dotenv import load_dotenv
from pymilvus import connections, Collection

load_dotenv()  # loads .env locally; no effect on Streamlit Cloud

# -----------------------------
# Config helpers (work local + cloud)
# -----------------------------
def _get(key, default=None):
    # 1) Streamlit Secrets (cloud) -> 2) env var (local) -> 3) default
    val = None
    try:
        val = st.secrets.get(key) if hasattr(st, "secrets") else None
    except Exception:
        val = None
    return val if val is not None else os.getenv(key, default)

ZILLIZ_URI   = _get("ZILLIZ_URI")
ZILLIZ_USER  = _get("ZILLIZ_USER")
ZILLIZ_TOKEN = _get("ZILLIZ_TOKEN")

# Choose embedder via env: EMBEDDER=sentence or EMBEDDER=fastembed
EMBEDDER_BACKEND = (_get("EMBEDDER", "sentence")).lower().strip()

# -----------------------------
# Embedding backends
# -----------------------------
@st.cache_resource
def _load_sentence_transformer(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # lazy import so local runs don’t fail if not installed
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

@st.cache_resource
def _load_fastembed(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # fastembed is CPU-only and lightweight; great fallback
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model_name)

def _embed(text: str):
    """
    Returns a Python list[float] matching the model used at indexing time.
    IMPORTANT: model_name must match what you used to build your Milvus vectors.
    """
    model_name = _get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    backend = EMBEDDER_BACKEND
    if backend == "sentence":
        try:
            m = _load_sentence_transformer(model_name)
            v = m.encode(text, normalize_embeddings=True)
            return v.tolist() if hasattr(v, "tolist") else list(v)
        except Exception:
            # auto-fallback to fastembed if sentence-transformers isn’t installed
            backend = "fastembed"

    if backend == "fastembed":
        m = _load_fastembed(model_name)
        # returns generator of np arrays
        return list(m.embed([text]))[0].tolist()

    # final fallback to sentence-transformers
    m = _load_sentence_transformer(model_name)
    v = m.encode(text, normalize_embeddings=True)
    return v.tolist() if hasattr(v, "tolist") else list(v)

# -----------------------------
# Milvus / Zilliz connection
# -----------------------------
@st.cache_resource
def connect():
    if not (ZILLIZ_URI and ZILLIZ_USER and ZILLIZ_TOKEN):
        raise RuntimeError(
            "Missing ZILLIZ_URI / ZILLIZ_USER / ZILLIZ_TOKEN.\n"
            "• Locally: put them in a .env file\n"
            "• Streamlit Cloud: set them in Settings → Secrets"
        )
    connections.connect(
        alias="default",
        uri=ZILLIZ_URI,      # DO NOT append :19530 for Zilliz Cloud
        user=ZILLIZ_USER,
        password=ZILLIZ_TOKEN,
        secure=True,
    )
    return True

# -----------------------------
# Public API
# -----------------------------
COLLECTION = _get("MILVUS_COLLECTION", "booksense_chunks")
VECTOR_FIELD = _get("VECTOR_FIELD", "embedding")  # change if your field name differs
METRIC_TYPE = _get("METRIC_TYPE", "COSINE")       # must match index metric

def get_top_chunks(query: str, k: int = 5, ef: int = 64):
    connect()
    col = Collection(COLLECTION)
    # If your index is IVF, use {"nprobe": 10}. For HNSW, "ef" is correct.
    params = {"metric_type": METRIC_TYPE, "params": {"ef": ef}}
    q_vec = _embed(query)

    res = col.search(
        data=[q_vec],
        anns_field=VECTOR_FIELD,
        param=params,
        limit=k,
        output_fields=["chunk_text", "book_title", "authors", "avg_rating", "work_id"],
    )[0]

    out = []
    for hit in res:
        get = hit.entity.get
        # Milvus returns a distance; for cosine, similarity ≈ (1 - distance)
        sim = 1.0 - float(hit.distance) if METRIC_TYPE.upper() == "COSINE" else -float(hit.distance)
        out.append({
            "chunk_text": get("chunk_text"),
            "book_title": get("book_title"),
            "authors": get("authors"),
            "avg_rating": get("avg_rating"),
            "work_id": get("work_id"),
            "score": sim,
        })
    return out
