import pandas as pd
import joblib

# Load the saved model
model = joblib.load("student_model.pkl")

# Load the dataset to get one example student
data = pd.read_csv("data/student-mat.csv", sep=";")

# Select one student's details, excluding the final grade
student = data.drop(columns=["G3"]).iloc[[0]]

# Predict the final grade
prediction = model.predict(student)

print("Student details:")
print(student)

print("\nPredicted final grade:", round(prediction[0], 2))
print("Actual final grade:", data.loc[0, "G3"])