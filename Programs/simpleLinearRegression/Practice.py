import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/FuelConsumption.csv")

# Practice
# Plot CYLINDER vs the Emission, to see how linear is their relationship is:
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")

plt.title("Practice Scatter Plot: CYLINDERS vs Emission")
plt.show()
