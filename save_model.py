import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

# Separate features and target
X = data.drop(columns=["G3"])
y = data["G3"]

# Identify column types
categorical_columns = X.select_dtypes(include=["object"]).columns
numeric_columns = X.select_dtypes(exclude=["object"]).columns

# Preprocess the data
preprocessor = ColumnTransformer(transformers=[
    ("numeric", SimpleImputer(strategy="median"), numeric_columns),
    ("categorical", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_columns)
])

# Create the model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train the model on the full dataset
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_model.pkl")

print("Model trained and saved successfully!")
print("Saved file: student_model.pkl")