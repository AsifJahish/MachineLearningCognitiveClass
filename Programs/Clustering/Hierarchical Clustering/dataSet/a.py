import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

filename= "/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/cars_clus.csv"
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape, "\n")


print(pdf.head())