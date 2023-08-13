
# Introduction
# There are many models for clustering out there. In this notebook,
# we will be presenting the model that is considered one of the simplest models amongst them.
# Despite its simplicity, the K-means is vastly used for clustering in many data science applications,
# it is especially useful if you need to quickly discover insights from unlabeled data.
# In this notebook, you will learn how to use k-Means for customer segmentation.
#
# Some real-world applications of k-means:
#
# Customer segmentation
# Understand what the visitors of a website are trying to accomplish
# Pattern recognition
# Machine learning
# Data compression
# In this notebook we practice k-means clustering with 2 examples:
#
# k-means on a random generated dataset
# Using k-means for customer segmentation


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# #
# First we need to set a random seed. Use numpy's random.seed() function, where the seed will be set to 0.

np.random.seed(0)
# k-Means on a randomly generated dataset¶
# Let's create our own dataset for this lab!

# Next we will be making random clusters of points by using the make_blobs class.
# The make_blobs class can take in many inputs, but we will be using these specific ones.


# #
#
#
# Input
#
# n_samples: The total number of points equally divided among clusters.
# Value will be: 5000
# centers: The number of centers to generate, or the fixed center locations.
# Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
# cluster_std: The standard deviation of the clusters.
# Value will be: 0.9
#
# Output
# X: Array of shape [n_samples, n_features]. (Feature Matrix)
# The generated samples.
# y: Array of shape [n_samples]. (Response Vector)
# The integer labels for cluster membership of each sample.


X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.title("the scatter of the Cluster")
plt.show()

