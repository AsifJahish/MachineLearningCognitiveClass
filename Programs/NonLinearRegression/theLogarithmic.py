import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/FuelConsumption.csv")



cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

X = np.arange(-5.0, 5.0, 0.1)

Y = np.log(X)

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.title("the Logarithmic Regression")
plt.show()


# Logarithmic
# The response  𝑦
#   is a results of applying the logarithmic map from the input  𝑥
#   to the output  𝑦
#  . It is one of the simplest form of log(): i.e.
# 𝑦=log(𝑥)
#
# Please consider that instead of  𝑥
#  , we can use  𝑋
#  , which can be a polynomial representation of the  𝑥
#   values. In general form it would be written as
# 𝑦=log(𝑋)