# encoding=utf-8

import sys
import numpy as np

filename = sys.argv[1]
with open(filename, 'r') as file:
    file.readline()
    embeddings = []
    for line in file.readlines():
        str = line.split()
        features = [float(dim) for dim in str[1:]]
        embeddings.append(features)

print(embeddings)
embeddings = np.array(embeddings, dtype=np.float32)


def plot_with_labels(low_dim_embs, filename='tsne.png'):
    plt.figure(figsize=(18, 18))  # in inches
    for i in range(len(low_dim_embs)):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)

    plt.savefig(filename)


try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    print(embeddings.shape)
    low_dim_embs = tsne.fit_transform(embeddings)
    plot_with_labels(low_dim_embs)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
    pass
