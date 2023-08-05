import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/FuelConsumption.csv")


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color= 'blue')
plt.xlabel("engine Size")
plt.ylabel("Co2 Emission")
plt.title("Engine Size as CorBon Emission")

plt.show()