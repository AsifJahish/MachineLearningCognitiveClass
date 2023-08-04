import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/FuelConsumption.csv")

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Create a figure and axes for the subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot the first scatter plot in the first subplot
axes[0].scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
axes[0].set_xlabel("FUELCONSUMPTION_COMB")
axes[0].set_ylabel("Emission")

# Plot the second scatter plot in the second subplot
axes[1].scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
axes[1].set_xlabel("Engine size")
axes[1].set_ylabel("Emission")

# Adjust the layout to prevent overlapping of subplots
plt.tight_layout()

# Show the plot with both scatter plots
plt.show()
