
# Agglomerative Clustering
# We will start by clustering the random data points we just created.

#
# The Agglomerative Clustering class will require two inputs:
#
# n_clusters: The number of clusters to form as well as the number of centroids to generate.
# Value will be: 4
# linkage: Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation.
# The algorithm will merge the pairs of cluster that minimize this criterion.
# Value will be: 'complete'
# Note: It is recommended you try everything with 'average' as well
#
# Save the result to a variable called agglom .
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

# Create random data (replace this with your actual data)
X1, y1 = make_blobs(n_samples=50, centers=4, random_state=42)

# Create an AgglomerativeClustering instance
agglom = AgglomerativeClustering(n_clusters=4, linkage='average')
agglom.fit(X1)

plt.figure(figsize=(6, 4))

# Scale the data points to a range of [0, 1]
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)
X1 = (X1 - x_min) / (x_max - x_min)

# Display data points with cluster labels using colormap
for i in range(X1.shape[0]):
    # Scale cluster labels to the [0, 1] range
    normalized_label = agglom.labels_[i] / (np.max(agglom.labels_) + 1)

    plt.text(X1[i, 0], X1[i, 1], str(agglom.labels_[i]),
             color=plt.cm.nipy_spectral(normalized_label),
             fontdict={'weight': 'bold', 'size': 9})

# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
plt.show()
