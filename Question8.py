import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your dataset
# df = pd.read_csv('emails.csv')

# For demonstration, let's create a sample dataset
np.random.seed(42)
emails = ['Free money now', 'Win a free iPhone', 'Meeting at 10am', 'Your invoice is attached', 'Click here to claim your prize']
labels = [1, 1, 0, 0, 1]  # 1 for spam, 0 for not spam
df = pd.DataFrame({'Email': emails, 'Label': labels})

# Split the dataset into features and target variable
X = df['Email']
y = df['Label']

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Initialize and train the linear SVM model
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

# Make predictions with the linear SVM model
y_pred_linear = linear_svm.predict(X_test)

# Evaluate the linear SVM model
accuracy_linear = accuracy_score(y_test, y_pred_linear)
precision_linear = precision_score(y_test, y_pred_linear)
recall_linear = recall_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear)

# Print evaluation metrics for the linear SVM model
print('Linear SVM Model:')
print(f'Accuracy: {accuracy_linear:.2f}')
print(f'Precision: {precision_linear:.2f}')
print(f'Recall: {recall_linear:.2f}')
print(f'F1 Score: {f1_linear:.2f}')

# Initialize and train the non-linear SVM model with RBF kernel
non_linear_svm = SVC(kernel='rbf')
non_linear_svm.fit(X_train, y_train)

# Make predictions with the non-linear SVM model
y_pred_non_linear = non_linear_svm.predict(X_test)

# Evaluate the non-linear SVM model
accuracy_non_linear = accuracy_score(y_test, y_pred_non_linear)
precision_non_linear = precision_score(y_test, y_pred_non_linear)
recall_non_linear = recall_score(y_test, y_pred_non_linear)
f1_non_linear = f1_score(y_test, y_pred_non_linear)

# Print evaluation metrics for the non-linear SVM model
print('\nNon-Linear SVM Model:')
print(f'Accuracy: {accuracy_non_linear:.2f}')
print(f'Precision: {precision_non_linear:.2f}')
print(f'Recall: {recall_non_linear:.2f}')
print(f'F1 Score: {f1_non_linear:.2f}')
