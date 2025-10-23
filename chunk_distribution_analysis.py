import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/booksense_chunks.csv")

# Compute chunk counts per review
chunk_counts = df.groupby("work_id")["chunk_id"].nunique()

# Plot style
sns.set(style="whitegrid", palette="deep", font_scale=1.1)
plt.figure(figsize=(10, 5))

# Plot histogram with KDE overlay
sns.histplot(chunk_counts, bins=range(1, chunk_counts.max() + 1), kde=True, color="#4C72B0")

# Labels and title
plt.title("📊 Distribution of Chunks per Review", fontsize=15, weight="bold", pad=15)
plt.xlabel("Number of chunks per review", fontsize=12)
plt.ylabel("Number of reviews", fontsize=12)

# Annotate key statistics
plt.axvline(chunk_counts.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {chunk_counts.mean():.2f}")
plt.legend()
plt.tight_layout()
plt.show()
