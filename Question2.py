import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Step 1: Generate a sample time series dataset (for demonstration purposes)
# Assume we have monthly electricity consumption data over 5 years (60 months)
np.random.seed(42)

# Generate a time series: monthly data for 5 years
dates = pd.date_range(start='2018-01-01', periods=60, freq='M')
electricity_consumption = 100 + 0.5 * np.arange(60) + 10 * np.sin(2 * np.pi * dates.month / 12) + np.random.normal(scale=5, size=60)

# Create a DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Electricity_Consumption': electricity_consumption
})

# Step 2: Feature Engineering for Time Series Model
df['Month'] = df['Date'].dt.month  # Extract the month to capture seasonality
df['Year'] = df['Date'].dt.year    # Extract the year to capture trend
df['Time'] = np.arange(len(df))    # Time variable: simply an index for linear trend

# Step 3: Set up X (features) and y (target)
X = df[['Time', 'Month']]  # Time and month as features
y = df['Electricity_Consumption']  # Target variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Fit a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plotting actual vs predicted values
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Electricity_Consumption'], label='Actual Consumption', color='blue')
plt.plot(df['Date'][len(X_train):], y_pred, label='Predicted Consumption', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.title('Electricity Consumption Prediction')
plt.legend()
plt.show()

# Step 8: Coefficients of the model
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)

# Print the intercept
print(f"\nIntercept: {model.intercept_}")
