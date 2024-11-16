import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your gene expression dataset
# df = pd.read_csv('gene_expression_data.csv')

# For demonstration, let's create a sample dataset
np.random.seed(42)
df = pd.DataFrame(np.random.rand(100, 1000), columns=[f'Gene_{i}' for i in range(1000)])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Perform PCA
pca = PCA(n_components=10)  # Adjust the number of components as needed
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=principal_components, columns=[f'PC_{i+1}' for i in range(10)])

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), pca.explained_variance_ratio_, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, 11), np.cumsum(pca.explained_variance_ratio_), where='mid', label='Cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Components')
plt.legend(loc='best')
plt.show()

# Plot the first two principal components
plt.figure(figsize=(10, 6))
plt.scatter(pc_df['PC_1'], pc_df['PC_2'], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - First Two Principal Components')
plt.show()
