# Non-linear regression is a method to model the non-linear relationship between the independent variables  𝑥
#   and the dependent variable  𝑦
#  . Essentially any relationship that is not linear can be termed as non-linear,
#  and is usually represented by the polynomial of  𝑘
#   degrees (maximum power of  𝑥
#  ). For example:
#
#  𝑦=𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑
#
# Non-linear functions can have elements like exponentials, logarithms, fractions, and so on. For example:
# 𝑦=log(𝑥)
#
# We can have a function that's even more complicated such as :
# 𝑦=log(𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑)


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/FuelConsumption.csv")



cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.title("the nonLinear Regresssion betwween the data ")
plt.show()