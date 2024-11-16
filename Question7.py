import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your dataset
# df = pd.read_csv('loan_data.csv')

# For demonstration, let's create a sample dataset
np.random.seed(42)
df = pd.DataFrame({
    'CreditScore': np.random.randint(300, 850, 100),
    'IncomeLevel': np.random.randint(20000, 100000, 100),
    'EmploymentStatus': np.random.randint(0, 2, 100),
    'Default': np.random.randint(0, 2, 100)
})

# Split the dataset into features and target variable
X = df[['CreditScore', 'IncomeLevel', 'EmploymentStatus']]
y = df['Default']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

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

# Print the coefficients
coefficients = log_reg.coef_[0]
feature_names = ['CreditScore', 'IncomeLevel', 'EmploymentStatus']
for feature, coef in zip(feature_names, coefficients):
    print(f'{feature}: {coef:.2f}')

# Interpret the coefficients
print("\nInterpretation of Coefficients:")
for feature, coef in zip(feature_names, coefficients):
    if coef > 0:
        print(f'An increase in {feature} is associated with an increase in the probability of defaulting.')
    else:
        print(f'An increase in {feature} is associated with a decrease in the probability of defaulting.')
