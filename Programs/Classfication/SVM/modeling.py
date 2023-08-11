

#  The SVM algorithm offers a choice of kernel functions for performing its processing.
#  Basically, mapping data into a higher dimensional space is called kernelling.
#  The mathematical function used for the transformation is known as the kernel function, and can be of different types, such as:
#
# 1.Linear
# 2.Polynomial
# 3.Radial basis function (RBF)
# 4.Sigmoid
# Each of these functions has its characteristics, its pros and cons, and its equation,
# but as there's no easy way of knowing which function performs best with any given dataset.
# We usually choose different functions in turn and compare the results. Let's just use the default,
# RBF (Radial Basis Function) for this lab.


from sklearn import svm
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

cell_df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/cell_samples.csv")
# print(cell_df.head())

# print(cell_df.shape)
#
print(cell_df.dtypes)


#
# It looks like the BareNuc column includes some values that are not numerical. We can drop those rows:

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print("after change the BareNuc data Type",cell_df.dtypes)


feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
print(X[0:5])

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
print(y [0:5])


#


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# After being fitted, the model can then be used to predict new values:

yhat = clf.predict(X_test)
print(yhat [0:5])

