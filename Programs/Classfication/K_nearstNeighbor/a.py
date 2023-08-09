

# K-Nearest Neighbors is a supervised learning algorithm. Where the data is 'trained' with data points corresponding
# to their classification. To predict the class of a given data point, it takes into account the classes of the 'K'
# nearest data points
# and chooses the class in which the majority of the 'K' nearest data points belong to as the predicted class.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

df= pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/teleCust1000t.csv")

print(df.head())
# Let’s see how many of each class is in our data set

print(df['custcat'].value_counts())

df.hist(column='income', bins=50)
plt.title("see how is our data")
plt.show()