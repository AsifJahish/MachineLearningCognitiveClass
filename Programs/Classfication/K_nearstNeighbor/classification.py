from sklearn.neighbors import KNeighborsClassifier
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




y = df['custcat'].values


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



k = 4
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)


k = 4
# Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

# Print some information about the trained model
print("Trained KNeighborsClassifier model:")
print("Number of neighbors (k):", neigh.n_neighbors)
print("Effective metric:", neigh.effective_metric_)
print("Classes:", neigh.classes_)
