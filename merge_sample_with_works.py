# merge_sample_with_works.py
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
SAMPLE_IN = BASE / "data" / "reviews_sample.csv"
WORKS_IN  = BASE / "data" / "works.csv"
OUT       = BASE / "data" / "booksense_sample.csv"

print("Loading works metadata...")
use_cols = [
    "work_id", "original_title", "author",
    "image_url", "avg_rating", "ratings_count", "similar_books"
]
works = pd.read_csv(WORKS_IN, usecols=use_cols)

# Rename columns for consistency
works = works.rename(columns={
    "original_title": "book_title",
    "author": "authors"
})

print("Loading 50k sample...")
sample = pd.read_csv(SAMPLE_IN)

print("Merging on work_id...")
merged = sample.merge(works, on="work_id", how="left")

# Reorder columns neatly
keep = [
    "work_id", "book_title", "authors", "image_url",
    "avg_rating", "ratings_count", "similar_books",
    "rating", "date_added", "review_text"
]
merged = merged[[c for c in keep if c in merged.columns]]

# Drop rows without a valid title
merged = merged[merged["book_title"].astype(str).str.strip().str.len() > 0]

merged.to_csv(OUT, index=False)
print(f"Saved {len(merged):,} rows -> {OUT}")
