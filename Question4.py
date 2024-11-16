import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load your updated dataset
# df = pd.read_csv('updated_market_data.csv')

# For demonstration, let's create a sample dataset
np.random.seed(42)
df = pd.DataFrame({
    'MarketTrend': np.random.rand(100),
    'EconomicIndicator': np.random.rand(100),
    'InvestmentOpportunity': np.random.randint(0, 2, 100)
})

# Split the dataset into training and testing sets
X = df[['MarketTrend', 'EconomicIndicator']]
y = df['InvestmentOpportunity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

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

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=['MarketTrend', 'EconomicIndicator'], class_names=['No', 'Yes'], filled=True)
plt.title('Decision Tree for Investment Opportunities')
plt.show()

# Print the decision tree rules
tree_rules = export_text(clf, feature_names=['MarketTrend', 'EconomicIndicator'])
print(tree_rules)
