# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature (in this case, the 3rd feature, BMI)
X = diabetes.data[:,[2]]
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
linear_model = LinearRegression()

# Fit the model to the training data
linear_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = linear_model.predict(X_test)

# Calculate and print model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Linear Regression:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Hyperparameter tuning for Lasso regularization
lasso_model = Lasso()

# Define a parameter grid for alpha (regularization strength)
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}

# Perform grid search with cross-validation
lasso_grid_search = GridSearchCV(lasso_model, param_grid, cv=5)
lasso_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_alpha_lasso = lasso_grid_search.best_params_['alpha']

# Train the Lasso model with the best alpha
lasso_model = Lasso(alpha=best_alpha_lasso)
lasso_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_lasso = lasso_model.predict(X_test)

# Calculate and print model performance metrics for Lasso
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print("\nLasso Regression:")
print(f"Best Alpha: {best_alpha_lasso}")
print(f"Mean Squared Error (MSE): {mse_lasso}")
print(f"R-squared (R2): {r2_lasso}")

# Hyperparameter tuning for Ridge regularization
ridge_model = Ridge()

# Define a parameter grid for alpha (regularization strength)
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}

# Perform grid search with cross-validation
ridge_grid_search = GridSearchCV(ridge_model, param_grid, cv=5)
ridge_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_alpha_ridge = ridge_grid_search.best_params_['alpha']

# Train the Ridge model with the best alpha
ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_ridge = ridge_model.predict(X_test)

# Calculate and print model performance metrics for Ridge
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print("\nRidge Regression:")
print(f"Best Alpha: {best_alpha_ridge}")
print(f"Mean Squared Error (MSE): {mse_ridge}")
print(f"R-squared (R2): {r2_ridge}")

# # Plot the data and the regression lines
# plt.scatter(X_test, y_test, color='black', label='Test Data')
# plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Linear Regression')
# plt.plot(X_test, y_pred_lasso, color='red', linewidth=3, label=f'Lasso (alpha={best_alpha_lasso})')
# plt.plot(X_test, y_pred_ridge, color='green', linewidth=3, label=f'Ridge (alpha={best_alpha_ridge})')
# plt.xlabel("BMI")
# plt.ylabel("Disease Progression")
# plt.title("Linear Regression and Regularization on Diabetes Dataset")
# plt.legend()
# plt.show()
