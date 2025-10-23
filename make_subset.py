# make_subset.py
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
IN = BASE / "data" / "reviews.csv"
OUT = BASE / "data" / "reviews_long.csv"

MIN_LEN = 80                # keep reviews with at least 80 characters
CHUNKSIZE = 200_000         # process 200k rows at a time
USE_COLS = ["work_id", "rating", "date_added", "review_text"]

print(f"Reading from: {IN}")
if not IN.exists():
    raise FileNotFoundError(f"Input not found: {IN}")

first_chunk = True
written = 0

for chunk in pd.read_csv(IN, usecols=USE_COLS, chunksize=CHUNKSIZE):
    if "review_text" not in chunk.columns:
        raise ValueError("Column 'review_text' not found. Check USE_COLS list.")
    chunk["review_text"] = chunk["review_text"].astype(str).str.strip()
    part = chunk[chunk["review_text"].str.len() >= MIN_LEN]
    if len(part) == 0:
        continue
    part.to_csv(OUT, index=False, mode="w" if first_chunk else "a", header=first_chunk)
    written += len(part)
    first_chunk = False

print(f"✅ Done. Wrote {written} filtered rows to {OUT}")
