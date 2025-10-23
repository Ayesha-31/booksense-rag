import pandas as pd

# -------------------- STEP 1: Load & Inspect --------------------
print("Loading dataset...")
df = pd.read_csv("data/reviews_long.csv")

print("\nShape:", df.shape)
print("\nColumns:\n", df.columns.tolist())

print("\nMissing values:\n", df.isna().sum())

print("\nRating stats:")
print(df['rating'].describe())

# Calculate review length
df["review_len"] = df["review_text"].astype(str).str.len()
print("\nReview length stats:")
print(df["review_len"].describe())

# Show a few sample rows
print("\nSample reviews:")
print(df.sample(5, random_state=42)[["work_id", "rating", "review_len", "review_text"]])

# -------------------- STEP 2: Cleaning --------------------
print("\n🧹 Cleaning data...")

# Drop missing reviews (should be none)
df = df.dropna(subset=["review_text"])

# Fill missing ratings with 0 (or could drop them)
df["rating"] = df["rating"].fillna(0)

# Keep only valid ratings (1–5 or 0 if unknown)
df = df[df["rating"].between(1, 5) | (df["rating"] == 0)] #mmissing rating -27,648 reviews without a rating were replaced with 0.

# Drop duplicate review texts
df = df.drop_duplicates(subset=["review_text"])

# Recalculate review length after cleaning
df["review_len"] = df["review_text"].astype(str).str.len()

# Optional: filter again by min length
df = df[df["review_len"] >= 80]

print("\nAfter cleaning:")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# -------------------- STEP 3: Save Cleaned File --------------------
out_path = "data/reviews_long_clean.csv"
df.to_csv(out_path, index=False)
print(f"\nCleaned data saved to {out_path}")
