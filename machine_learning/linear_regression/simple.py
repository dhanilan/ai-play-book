# Import the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("doing something")
# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Plot the data and the regression line
plt.scatter(X, y)
plt.plot(X_new, y_pred, "r-")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Linear Regression")
plt.show()