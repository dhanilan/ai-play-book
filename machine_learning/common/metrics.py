# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample movie dataset
data = {
    'Budget': [100, 150, 80, 200, 50, 120, 180, 90, 130, 160],
    'Theaters': [3000, 3500, 2500, 4000, 1500, 3200, 3800, 2800, 3300, 3600],
    'Blockbuster': [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Define the features and target variable
X = df[['Budget', 'Theaters']]
y = df['Blockbuster']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
