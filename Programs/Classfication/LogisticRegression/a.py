import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt


churn_df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/ChurnData.csv")
print(churn_df.head())

print(churn_df.shape)