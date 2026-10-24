import pandas as pd

# Load the dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

# Remove duplicate records
data = data.drop_duplicates()

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Separate input features and target
X = data.drop(columns=["G3"])
y = data["G3"]

print("\nInput features shape:", X.shape)
print("Target shape:", y.shape)

print("\nData preparation completed successfully!")