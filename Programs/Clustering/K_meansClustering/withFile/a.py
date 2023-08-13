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

df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/Cust_Segmentation.csv")
print(df.head())


# Pre-processing
# As you can see, Address in this dataset is a categorical variable. The k-means algorithm isn't
# directly applicable to categorical variables because
# the Euclidean distance function isn't really meaningful for discrete variables. So, let's drop this feature and run clustering.


# df = cust_df.drop('Address', axis=1)
# print(df.head())

#

# Normalizing over the standard deviation
# Now let's normalize the dataset. But why do we need normalization in the first place? Normalization is a
# statistical method that helps mathematical-based algorithms to
# interpret features with different magnitudes and distributions equally. We use StandardScaler() to normalize our dataset.



X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)