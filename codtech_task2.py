# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Adult Income dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Define column names based on the dataset documentation
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]

# Load the dataset into a DataFrame
data = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Drop unnecessary columns
data.drop(['fnlwgt', 'education', 'native_country'], axis=1, inplace=True)

# Encode categorical columns
data['sex'] = data['sex'].map({'Male': 0, 'Female': 1})
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})

# One-hot encode other categorical columns
data = pd.get_dummies(data, columns=['workclass', 'marital_status', 'occupation', 'relationship', 'race'], drop_first=True)

# Define features and target
X = data.drop('income', axis=1)
y = data['income']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
