# Import necessary libraries
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

# Sample movie dataset
X = np.array([[100, 3000],
              [150, 3500],
              [80, 2500],
              [200, 4000],
              [50, 1500],
              [120, 3200],
              [180, 3800],
              [90, 2800],
              [130, 3300],
              [160, 3600]])
y = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])

# Define the number of folds for K-fold cross-validation
num_folds = 5

# Initialize K-fold cross-validation
kf = KFold(n_splits=num_folds,shuffle=True)

# Initialize a logistic regression model
model = LogisticRegression()

# Perform K-fold cross-validation
scores = cross_val_score(model, X, y, cv=kf)

# Print the cross-validation scores
for fold, score in enumerate(scores):
    print(f"Fold {fold + 1}: Accuracy = {score:.2f}")

# Calculate the average accuracy across all folds
average_accuracy = np.mean(scores)
print(f"Average Accuracy: {average_accuracy:.2f}")
