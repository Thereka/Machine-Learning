# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Prepare the dataset
# For demonstration, we'll create a sample DataFrame
data = {
    'study_hours': [5, 8, 12, 6, 10, 7, 9, 13, 6, 5],
    'attendance_rate': [85, 90, 95, 80, 92, 88, 91, 97, 84, 82],
    'socioeconomic_background': ['low', 'medium', 'high', 'low', 'medium', 'medium', 'high', 'high', 'low', 'medium'],
    'test_score': [60, 75, 85, 65, 80, 72, 90, 88, 63, 67]
}

df = pd.DataFrame(data)

# Step 2: Encode categorical variable 'socioeconomic_background'
df_encoded = pd.get_dummies(df, columns=['socioeconomic_background'], drop_first=True)

# Step 3: Define the independent variables (X) and dependent variable (y)
X = df_encoded.drop('test_score', axis=1)
y = df_encoded['test_score']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Fit a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 8: Interpret the coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)

# Intercept of the model (bias term)
print("\nIntercept:", model.intercept_)
