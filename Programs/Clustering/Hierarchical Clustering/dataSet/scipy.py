import numpy as np
import pandas as pd
import scipy
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import pylab
import scipy.cluster.hierarchy


filename= "/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/cars_clus.csv"
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape, "\n")

#
# print(pdf.head())

# Data Cleaning
# Let's clean the dataset by dropping the rows that have null value:'

print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
# print ("Shape of dataset after cleaning: ", pdf.size)
# pdf.head(5)

# feature selection

featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# Normalization
# Now we can normalize the feature set. MinMaxScaler transforms features by scaling each feature to a given range.
# It is by default (0, 1).
# That is, this estimator scales and translates each feature individually such that it is between zero and one.


x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
# print(feature_mtx [0:5])

leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
print(D)