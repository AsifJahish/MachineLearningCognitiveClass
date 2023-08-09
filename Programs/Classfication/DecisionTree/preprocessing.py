
# Feature Matrix (X):
# A feature matrix, often denoted as "X," is a structured representation of your data's features or attributes.
# Each row in this matrix corresponds to an individual sample or observation, while each column represents
# a specific feature or attribute of that sample. In simpler terms, X contains the input data that you will use
# to make predictions or perform analysis.
#
# For instance, if you have a dataset with information about various drugs including their
# chemical properties (such as molecular weight, solubility, etc.), each row could represent a different drug,
# and each column could represent a specific chemical property. The feature matrix X would hold this data.
#
# Response Vector (y):
# The response vector, often denoted as "y," contains the target variable or the values you want
# to predict or analyze based on the features in the feature matrix. In supervised learning scenarios, y
# typically represents the dependent variable, the outcome, or the label you're trying to predict.
# Each element in the response vector corresponds to the target value for the corresponding row in the feature matrix.
#
# Continuing with the drug example, if you're trying to predict the effectiveness of a drug based on
# its chemical properties, the response vector y might contain the effectiveness scores or labels for each drug.



# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

my_data= pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/drug200.csv")
# print(my_data.head())


# what is the size of the Data and is (200 ,6) which means that 6 column and 200 rows
print(my_data.shape)


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

