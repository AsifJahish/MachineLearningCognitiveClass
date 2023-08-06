import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/china_gdp.csv")
print(df.head(10))

X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0

# Assuming the column names are "Year" and "GDP" in the DataFrame df
x_data = df['Year'].values
y_data = df['GDP'].values
# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)
# Logistic function
Y_pred = sigmoid(x_data, beta_1, beta_2)

# Plot initial prediction against datapoints
plt.plot(x_data, Y_pred * 15000000000000.)
plt.plot(x_data, y_data, 'ro')
plt.xlabel("Year")
plt.ylabel("GDP")
plt.show()
