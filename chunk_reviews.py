import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Input / Output paths
INPUT = "data/booksense_sample.csv"
OUTPUT = "data/booksense_chunks.csv"

#Converts review_text to a string (to avoid NaN issues)
#Removes empty reviews

print("Loading data...")
df = pd.read_csv(INPUT)
df["review_text"] = df["review_text"].astype(str).str.strip()
df = df[df["review_text"].str.len() > 0]

# LangChain splitter — good defaults
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,     # target characters per chunk
    chunk_overlap=120,  # overlap to preserve context continuity
)

print("Splitting reviews into chunks...")
records = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    text = row["review_text"]
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        rec = row.to_dict()
        rec["chunk_id"] = i + 1
        rec["chunk_text"] = chunk
        records.append(rec)

#df.iterrows() → goes row by row through your DataFrame df.
#Each row contains all the columns of that review (like work_id, rating, review_text, etc.).
#_ → means we’re ignoring the index (since we don’t need it).
#tqdm(df.iterrows(), total=len(df)) → tqdem(...)shows a progress bar while processing each row.
#If your DataFrame has 50,000 reviews, this loop runs 50,000 times, once per review.
#tqdm is a library that shows a progress bar in the console.
#total=len(df) → tells tqdm how many iterations to expect, so it knows when to stop.

#For each review:
#text = row["review_text"] → gets the review text from that row.
#chunks = splitter.split_text(text) → splits the review text into chunks.
#splitter is a LangChain object that splits the text into chunks.
#i, chunk → loops through each chunk.
#rec = row.to_dict() → converts the row to a dictionary so we can store it.
#rec["chunk_id"] = i + 1 → adds a unique ID to each chunk.


chunks_df = pd.DataFrame(records)
chunks_df.to_csv(OUTPUT, index=False)

print(f"Saved {len(chunks_df):,} chunks to {OUTPUT}")
