

# Setting up K-Means
# Now that we have our random data, let's set up our K-Means Clustering.

#
# The KMeans class has many parameters that can be used, but we will be using these three:
#
# init: Initialization method of the centroids.
# Value will be: "k-means++"
# k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
# n_clusters: The number of clusters to form as well as the number of centroids to generate.
# Value will be: 4 (since we have 4 centers)
# n_init: Number of time the k-means algorithm will be run with different centroid seeds.
# The final results will be the best output of n_init consecutive runs in terms of inertia.
# Value will be: 12
# Initialize KMeans with these parameters, where the output parameter is called k_means.

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



np.random.seed(0)


X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.title("the scatter of the Cluster")
# plt.show()

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# Now let's fit the KMeans model with the feature matrix we created above, X .
print(k_means.fit(X))


# Now let's grab the labels for each point in the model using KMeans' .labels_ attribute and save it as k_means_labels .
k_means_labels = k_means.labels_
print(k_means_labels)


# We will also get the coordinates of the cluster centers using KMeans' .cluster_centers_ and save it as k_means_cluster_centers .

k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers)