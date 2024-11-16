import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your historical weather dataset
# df = pd.read_csv('weather_data.csv')

# For demonstration, let's create a sample dataset
np.random.seed(42)
dates = pd.date_range(start='1/1/2020', periods=365, freq='D')
temperatures = 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 2, 365)
df = pd.DataFrame({'Date': dates, 'Temperature': temperatures})

# Feature engineering to capture seasonal variations
df['DayOfYear'] = df['Date'].dt.dayofyear
df['Sin_DayOfYear'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['Cos_DayOfYear'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

# Split the dataset into features and target variable
X = df[['DayOfYear', 'Sin_DayOfYear', 'Cos_DayOfYear']]
y = df['Temperature']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared: {r2:.2f}')

# Plot the actual vs predicted temperatures
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Temperature'], label='Actual Temperature')
plt.plot(df['Date'][X_test.index], y_pred, label='Predicted Temperature', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Actual vs Predicted Temperatures')
plt.legend()
plt.show()
