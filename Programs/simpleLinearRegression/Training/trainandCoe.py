from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/FuelConsumption.csv")


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
print(regr.fit(train_x, train_y))


# Y= a0+a1.X1
#
# a0= Intercept:  [125.00741465]
# a1= Coefficients:  [[39.14788573]]


print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
