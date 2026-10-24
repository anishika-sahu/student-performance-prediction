import pandas as pd

# Load the student dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

# Display the first 5 rows
print(data.head())

# Display dataset information
print("\nDataset shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())
print("\nFinal grade (G3) statistics:")
print(data["G3"].describe())

print("\nFinal grade examples:")
print(data["G3"].head(10).tolist())