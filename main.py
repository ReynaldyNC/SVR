import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR

# Read dataset
data = pd.read_csv('data/Salary_Data.csv')

# Separate attribute and label
x = data['YearsExperience']
y = data['Salary']

# Change attribute shape
x = x[:, np.newaxis]

# Build model with C, gamma, and kernel params
model = SVR(C=1000, gamma=0.05, kernel='rbf')

# Train model
model.fit(x, y)

# Visualize model
plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.show()
