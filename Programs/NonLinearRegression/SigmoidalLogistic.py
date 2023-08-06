import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/FuelConsumption.csv")



cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


X = np.arange(-5.0, 5.0, 0.1)


Y = 1-4/(1+np.power(3, X-2))

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')

plt.title("th SigmoidalLogistic")
plt.show()


# Sigmoidal/Logistic¶
#
#
#
#
#
#
# 𝑌=𝑎+𝑏1+𝑐(𝑋−𝑑)