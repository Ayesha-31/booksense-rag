# retriever.py
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import os,math

COLLECTION = "booksense_chunks"
from dotenv import load_dotenv

load_dotenv()

# ---- read secrets from env (recommended) ----
ZILLIZ_URI   = os.environ.get("ZILLIZ_URI")    # e.g., https://in03-....zilliz.com
ZILLIZ_USER  = os.environ.get("ZILLIZ_USER")   # e.g., db_xxx
ZILLIZ_TOKEN = os.environ.get("ZILLIZ_TOKEN")  # password/token

assert ZILLIZ_URI and ZILLIZ_USER and ZILLIZ_TOKEN, "Set ZILLIZ_URI, ZILLIZ_USER, ZILLIZ_TOKEN env vars."




_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_top_chunks(query, k=5, nprobe=10):
    connections.connect(
    alias="default",
    uri=ZILLIZ_URI,   # DO NOT append :19530 for Zilliz Cloud
    user=ZILLIZ_USER,
    password=ZILLIZ_TOKEN,
    secure=True,
)
    col = Collection(COLLECTION); col.load()
    q_emb = _model.encode(query, normalize_embeddings=True)  #Converts your input question (like “What do readers love about The Night Circus?”) into a 384-dimensional vector. Normalized for cosine similarity.
    # Performs the actual vector similarity search:
    #Compares your question’s embedding to all stored embeddings. Finds the closest ones (most semantically similar). Returns their metadata (text, author, title, etc.).
    res = col.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
        limit=k,
        output_fields=["chunk_text","book_title","authors","avg_rating","work_id"],
    )

    #Extracts results into a clean Python list of dictionaries — each with the review snippet and similarity score.
    hits = []
    for h in res[0]:
        item = {f: h.entity.get(f) for f in ["chunk_text","book_title","authors","avg_rating","work_id"]}
        item["score"] = float(h.distance)
        hits.append(item)
    return hits

# #That’s what will be passed to the LLM next.
#   {
#     "book_title": "The Night Circus",
#     "authors": "Erin Morgenstern",
#     "avg_rating": 4.0,
#     "chunk_text": "The Night Circus is a beautifully written book ...",
#     "score": 0.82
#   },
#   ...

