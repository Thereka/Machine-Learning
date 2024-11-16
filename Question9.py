import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load your dataset
# df = pd.read_csv('real_estate_data.csv')

# For demonstration, let's create a sample dataset
np.random.seed(42)
df = pd.DataFrame({
    'Location': np.random.randint(1, 5, 100),
    'Size': np.random.randint(500, 3500, 100),
    'Bedrooms': np.random.randint(1, 5, 100),
    'Price': np.random.randint(100000, 500000, 100)
})

# Convert categorical variable 'Location' to dummy variables
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Split the dataset into features and target variable
X = df.drop('Price', axis=1)
y = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print evaluation metrics
print(f'R-squared: {r2:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f'Cross-Validation R-squared Scores: {cv_scores}')
print(f'Average Cross-Validation R-squared Score: {cv_scores.mean():.2f}')
