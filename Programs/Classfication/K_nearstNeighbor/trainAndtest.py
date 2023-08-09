import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

df= pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/teleCust1000t.csv")

# print(df.head())
# # Let’s see how many of each class is in our data set
#
# print(df.columns)

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print(X[0:5])
# is based on the distance of data points:
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])



# Train Test Split
# Out of Sample Accuracy is the percentage of correct predictions that the model makes on data that the model has NOT
# trained on.
# Doing a train and test on the same dataset will most likely have low out-of-sample accuracy,
# due to the likelihood of our model overfitting.
#
# It is important that our models have a high, out-of-sample accuracy,
# because the purpose of any model, of course, is to make correct predictions on unknown data.
# how can we improve out-of-sample accuracy? One way is to use an evaluation approach called Train/Test Split.
# Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive.
# After which, you train with the training set and test with the testing set.
#
# This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not
# part of the dataset that has been used to train the model. It is more realistic for the real world problems.

y = df['custcat'].values


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

