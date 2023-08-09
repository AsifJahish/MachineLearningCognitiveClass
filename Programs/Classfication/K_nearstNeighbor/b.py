import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

df= pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/teleCust1000t.csv")

print(df.head())
# Let’s see how many of each class is in our data set

print(df.columns)



X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print(X[0:5])

# What are our labels?

y = df['custcat'].values
print(y[0:5])
