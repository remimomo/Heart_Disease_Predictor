################## Exploratory Data analyses ####################
#### Step 1: Load the Dataset and View Basic Information
# First, we'll load the dataset and take a look at its basic structure.
import pandas as pd

# Load the dataset
file_path = '/Users/remimomo/Documents/heart_disease_project/data/heart-disease.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Display basic information about the dataset
print(data.info())

# Summary statistics
print(data.describe())

#### Step 2: Check for Missing Values
# We will check if there are any missing values in the dataset.
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

#### Step 3: Distribution of the Target Variable
# Let's visualise the distribution of the target variable to understand the balance of classes.
import seaborn as sns
import matplotlib.pyplot as plt

# Visualise the distribution of the target variable
sns.countplot(x='target', data=data)
plt.title('Distribution of Heart Disease')
plt.xlabel('Heart Disease (1: Yes, 0: No)')
plt.ylabel('Count')
plt.show()

#### Step 4: Statistical Analysis of Features
# We'll look at the distribution of some key features to understand their spread and central tendencies.
# Histograms for continuous variables
data.hist(figsize=(12, 12))
plt.show()

# Box plots for continuous variables
plt.figure(figsize=(12, 8))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.title('Boxplot of Features')
plt.show()

#### Step 5: Correlation Matrix
# A correlation matrix helps identify relationships between features and the target variable.
# Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

#### Step 6: Pairwise Relationships
# Visualizing pairwise relationships between features can reveal more about the data #structure.
# Pairplot for selected features
selected_features = ['age', 'trestbps', 'chol', 'thalach', 'target']
sns.pairplot(data[selected_features], hue='target')
plt.show()

#### Step 7: Outlier Detection
# Identifying outliers can be important, especially in features related to medical attributes.
# Box plots to detect outliers for a few important features
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x='target', y='age', data=data)
plt.title('Age vs. Target')

plt.subplot(1, 3, 2)
sns.boxplot(x='target', y='chol', data=data)
plt.title('Cholesterol vs. Target')

plt.subplot(1, 3, 3)
sns.boxplot(x='target', y='thalach', data=data)
plt.title('Maximum Heart Rate vs. Target')

plt.show()
