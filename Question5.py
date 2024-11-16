import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load your dataset
# df = pd.read_csv('patient_data.csv')

# For demonstration, let's create a sample dataset
np.random.seed(42)
df = pd.DataFrame({
    'Symptom1': np.random.rand(100),
    'Symptom2': np.random.rand(100),
    'Age': np.random.randint(20, 70, 100),
    'MedicalHistory': np.random.randint(0, 2, 100),
    'Disease': np.random.randint(0, 2, 100)
})

# Split the dataset into features and target variable
X = df[['Symptom1', 'Symptom2', 'Age', 'MedicalHistory']]
y = df['Disease']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Find the optimal K value using cross-validation
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot the cross-validation scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Optimal K Value Selection')
plt.show()

# Choose the optimal K value
optimal_k = k_values[np.argmax(cv_scores)]
print(f'The optimal K value is {optimal_k}')

# Train the K-NN model with the optimal K value
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
