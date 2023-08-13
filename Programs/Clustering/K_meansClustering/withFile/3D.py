import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/Cust_Segmentation.csv")
print(df.head())


X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)


# In our example (if we didn't have access to the k-means algorithm),
# it would be the same as guessing that each customer group would have certain age,
# income, education, etc, with multiple tests and experiments. However, using the
# K-means clustering we can do all this process much easier.
#
# Let's apply k-means on our dataset, and take a look at cluster labels.
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)
print("\n to the other line")

# Insights
# We assign the labels to each row in the dataframe.

df["Clus_km"] = labels
print(df.head(5))

#
print("We can easily check the centroid values by averaging the features in each cluster.\n ",df.groupby('Clus_km').mean())


fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:, 2], X[:, 0], X[:, 3], c=labels.astype(np.float64))

plt.show()