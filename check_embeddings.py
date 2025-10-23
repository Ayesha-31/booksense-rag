import numpy as np
embeddings = np.load("data/booksense_embeddings.npy")
print(embeddings.shape)
print(embeddings[0][:10])  # first 10 values of the first embedding
