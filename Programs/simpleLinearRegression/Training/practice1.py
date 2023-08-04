from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Load the data
df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/FuelConsumption.csv")

# Select the relevant features and target variable
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Split the data into training and testing sets
train, test = train_test_split(cdf, test_size=0.2, random_state=42)
train_x = train[["FUELCONSUMPTION_COMB"]]
train_y = train["CO2EMISSIONS"]
test_x = test[["FUELCONSUMPTION_COMB"]]
test_y = test["CO2EMISSIONS"]

# Create a linear regression model
regr = linear_model.LinearRegression()

# Fit the model to the training data
regr.fit(train_x, train_y)

# Make predictions on the test data
predictions = regr.predict(test_x)

# Evaluate the model using R-squared (coefficient of determination)
r2 = r2_score(test_y, predictions)
print("R-squared score:", r2)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(test_y, predictions)
print("Mean Absolute Error: %.2f" % mae)

# R-squared score: 0.8071474868274242
# Mean Absolute Error: 20.44
