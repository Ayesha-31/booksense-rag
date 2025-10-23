# Book Reviews RAG Project

This project builds a Retrieval-Augmented Generation (RAG) system on top of **Goodreads book reviews**, allowing users to ask natural-language questions like:

> “What do readers think about The Alchemist?”  
> “Common complaints about Twilight?”  
> “Why do people love Harry Potter so much?”

---

Why Use RAG Instead of Search or Sentiment Analysis

Traditional methods like keyword search or sentiment analysis can show general patterns in text — for example, that most Night Circus reviews are positive and frequently mention words like magical or beautiful. However, these approaches only analyze surface-level information; they do not truly understand or explain it.

Retrieval-Augmented Generation (RAG) goes a step further by combining two processes:

Retrieval: Finds the most relevant text chunks from the dataset using a vector database like Milvus.

Generation: Uses a language model to read those chunks and generate a coherent, context-aware answer.

This allows RAG to provide meaningful, evidence-based explanations rather than raw statistics or keywords.
In short, search tells you what people say, while RAG explains why they say it—bridging the gap between retrieval and reasoning.


---
Installation & Environment Setup

1. Create a virtual environment
python -m venv venv
source venv/bin/activate      # on macOS / Linux
venv\Scripts\activate         # on Windows

2. Install required libraries
Run this inside your virtual environment:
>> pip install langchain-community langchain-chroma chromadb sentence-transformers tqdm

3. pip install sentence-transformers pymilvus tqdm


---

##  Project Workflow

1. **Data Ingestion** – Loaded `reviews.csv` and `works.csv` from the Goodreads dataset.  
2. **Cleaning & Filtering** – Removed duplicates, trimmed text, and kept reviews ≥ 80 characters.  
3. **Sampling** – Randomly selected 50,000 reviews for fast prototyping.  
4. **Merging** – Joined with `works.csv` to attach book titles and authors.  
5. **Embeddings + Vector DB** – Will use OpenAI or Sentence-Transformer embeddings and ChromaDB for semantic retrieval.  
6. **RAG Pipeline** – Uses embeddings to retrieve relevant reviews, then passes them to an LLM for context-aware answers.

---

## 🧹 Data Cleaning Notes

- Removed duplicate reviews  
- Dropped missing text entries  
- Kept ratings between **1–5**  
- **`rating = 0` → review with no user rating**  Keep them as the text rating are important for retrieval more than just rating scores.
- Minimum review length: 80 characters  

---

## 📁 Directory Structure
RAG/
├── data/
│   ├── booksense_sample.csv
│   ├── booksense_chunks.csv
├── venv/
├── chunk_reviews.py
├── index_booksense.py
├── query_booksense.py
├── cleaning_data.py
├── merge_sample_with_works.py
└── README.md


---

**Vector Model**

| Property                | Value                                             |
| ----------------------- | ------------------------------------------------- |
| **Model name**          | `all-MiniLM-L6-v2`                                |
| **Library**             | Sentence Transformers (Hugging Face)              |
| **Embedding dimension** | 384                                               |
| **Architecture**        | MiniLM (6-layer transformer, distilled from BERT) |
| **Max token length**    | **512 tokens**                                    |

Token ≠ character

1 token ≈ 0.75 words in English (depends on text)

512 tokens ≈ 350–400 words, or ~2000–2500 characters

---
## Chunking

chunk_size = how many characters (or tokens) each text segment contains before we cut it into a new chunk.
chunk_overlap = how many characters are repeated between chunks to preserve context flow.

Let’s say chunk_size=700 and chunk_overlap=120.
Chunk 1 → 0–700 chars  
Chunk 2 → 580–1280 chars  
Chunk 3 → 1160–1860 chars

700 characters ≈ 120–140 words ≈ ~150 tokens
That’s well below the 512-token limit

1.Each long book review is split into smaller, meaningful text segments (chunks) that can be efficiently processed by an embedding model later.

2.We use LangChain’s **RecursiveCharacterTextSplitter** to automatically break reviews into chunks of around 700 characters, with a 120-character overlap to preserve context across boundaries.

3.Each chunk inherits metadata such as work_id, book_title, authors, and rating from the original review, and is assigned a unique chunk_id for traceability.


| Step                         | Description                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------- |
| 1️⃣ Load & clean data        | Remove empty reviews and strip whitespace.                                      |
| 2️⃣ Split text               | Use `RecursiveCharacterTextSplitter` (`chunk_size=700`, `chunk_overlap=120`).   |
| 3️⃣ Preserve metadata        | Each chunk keeps original book and review info.                                 |
| 4️⃣ Create structured output | Store all chunks in a new file `booksense_chunks.csv` for downstream embedding. |

*Chunking Results*

Splitting reviews into chunks...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 50000/50000 [00:04<00:00, 10366.09it/s]
✅ Saved 132,033 chunks to data/booksense_chunks.csv

After applying the LangChain RecursiveCharacterTextSplitter (chunk_size=700, chunk_overlap=120) to 50,000 cleaned book reviews, the process produced:
Total reviews processed: 50,000
Total chunks created: 132,033
Average chunks per review: ~2.6
Chunk storage file: data/booksense_chunks.csv

Each chunk contains:
A segment of the original review_text (~700 characters),
Metadata fields like work_id, book_title, authors, rating, and date_added,
A new chunk_id and chunk_text column for reference and traceability.
These smaller, context-preserving text chunks will be used in the next stage, embedding generation and vector storage (Milvus), to enable high-accuracy semantic search and retrieval.

Average chunks per review: 4.58
Max chunks in a single review: 33

---

## Embedding

Each row of embeddings is a 384-dimensional vector representing one chunk’s meaning.

---

## Inserting into Milvus

We have Generated embeddings in the previous step 
Now we’re telling Milvus:

“Here’s a table (collection). Store each review’s text, its embedding vector, and an ID so I can search later.”

Let’s set up Docker + Milvus together

1. Verify Docker is Running 
>> docker ps

2.Run Milvus in Docker - start Milvus as a container:


docker run -d \
  --name milvus \
  --platform linux/amd64 \
  -p 19530:19530 \
  -p 9091:9091 \
  -e ETCD_USE_EMBED=true \
  -e ROCKSMQ_USE_EMBED=true \
  -e MINIO_USE_EMBED=false \
  -e MINIO_ADDRESS=milvus-minio:9000 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin \
  -e MINIO_BUCKET_NAME=milvus-bucket \
  --link milvus-minio \
  milvusdb/milvus:v2.4.4 \
  milvus run standalone


This will:
Download the Milvus image (~2 GB the first time).
Run it in the background.
Expose port 19530 (for Python client) and 9091 (for metrics).

3. 
>>from pymilvus import connections
>>connections.connect("default", host="127.0.0.1", port="19530")
This opens a live connection to the Milvus service that’s running inside Docker (on port 19530).

4. Reading our data
>>df = pd.read_csv("data/booksense_chunks.csv")
>>emb = np.load("data/booksense_embeddings.npy")
>>booksense_chunks.csv → contains metadata (work_id, chunk_text).

booksense_embeddings.npy → contains each chunk’s 384-dimensional embedding vector.
The assert line just checks they have the same number of rows.

5.Defining the collection (table)
>> fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="work_id", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=emb.shape[1]),
]
schema = CollectionSchema(fields, description="BookSense review chunks")
col = Collection("booksense_chunks", schema)

--
FieldSchema defines columns (like SQL schema).

id → Milvus auto-generates it.

work_id → identifies which book/review this chunk belongs to.

chunk_text → the actual text snippet.

embedding → your vector (dim = 384 for MiniLM).

You then create the collection booksense_chunks with this schema.

6. convert each column into plain Python lists because Milvus expects array-like input (not pandas DataFrames).

7. #Inserting into Milvus
>> col.insert(data)

8. Building the index

This helps Milvus find similar chunks faster.
This tells Milvus how to search efficiently:
metric_type = COSINE → use cosine similarity to compare vectors.
IVF_FLAT → an indexing algorithm for fast nearest-neighbor search.
nlist=1024 → controls index granularity (you can tune this).
Finally, .load() brings the collection into memory, ready for queries.

col.create_index(
    field_name="embedding",
    index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}},
)
col.load()
---
**Integrating Milvus into our pipeline**

Here’s a sketch of how your RAG pipeline changes when you switch to Milvus:

Install pymilvus (Python SDK for Milvus).

Start Milvus server (Standalone or Lite for now).

Define a collection schema (vector field + metadata fields).

Insert embeddings + metadata into Milvus.

Search by providing query embedding vector + optional filters.

Retrieve metadata (title, authors, image_url) from Milvus results.

Data Flow: Storing Embeddings into Milvus
flowchart TD
    A[Review Dataset (CSV)] --> B[Chunking: Split long reviews into smaller text pieces]
    B --> C[Embedding Generation<br/>SentenceTransformer (MiniLM-L6-v2)]
    C --> D[( Embeddings .npy File)]
    B --> E[(Chunk Metadata CSV)]

    D & E --> F[ insert_to_milvus.py Script]
    F --> G[ Milvus Collection<br/>(booksense_chunks)]
    G --> H[Index Creation<br/>IVF_FLAT, COSINE]
    H --> I[ Ready for Semantic Search!]

    style A fill:#d6eaff,stroke:#1e90ff,stroke-width:1px
    style C fill:#fef3c7,stroke:#fbbf24,stroke-width:1px
    style G fill:#e5e7eb,stroke:#4b5563,stroke-width:1px
    style I fill:#bbf7d0,stroke:#22c55e,stroke-width:1px

** Explanation Summary **

After chunking the reviews and generating embeddings using SentenceTransformer, we store each chunk (text + vector) in Milvus, a high-performance vector database.
The insert_to_milvus.py script:

Connects to Milvus running locally (via Docker).

Creates a collection schema (booksense_chunks) with fields for IDs, book IDs, text, and embeddings.

Inserts all data and builds a COSINE similarity index using IVF_FLAT.

Loads the collection into memory, enabling real-time semantic search.

Done Inserting itno Milvus --

(venv) (base) ayeshatabassum@Ayeshas-MacBook RAG % python insert_to_milvus_batched.py
Inserting 132,033 rows in 67 batches of 2000…
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 67/67 [00:10<00:00,  6.11it/s]
Batched insert complete, index built, collection loaded.
(venv) (base) ayeshatabassum@Ayeshas-MacBook RAG % 

------

## Next Steps

- [ ] Merge cleaned reviews with works metadata  
- [ ] Generate embeddings  
- [ ] Build retrieval + generation pipeline  
- [ ] Deploy on Streamlit or Gradio  

---

###  Author
**Ayesha Tabassum Shaik**  
*Data Analyst → Data Scientist / ML Engineer Track*

