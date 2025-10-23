import pandas as pd

# Read the cleaned 1M review dataset
df = pd.read_csv("data/reviews_long_clean.csv")

# Take a reproducible random sample of 50k rows
sample = df.sample(n=50000, random_state=42)

# Save it as a smaller dataset
sample.to_csv("data/reviews_sample.csv", index=False)

print(f"Created sample with {len(sample)} rows → data/reviews_sample.csv")
