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
print(my_data.head())


# what is the size of the Data and is (200 ,6) which means that 6 column and 200 rows
print(my_data.shape)