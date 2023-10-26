from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import numpy as np


X = np.array([[10],[20],[30],[40],[50]])

# Create a MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the feature data
X_scaled = scaler.fit_transform(X)


print(X_scaled)

x_predict = [[25]]

# use the same method to scale the new values
x_scaled_predict = scaler.transform(x_predict)

print(x_scaled_predict)