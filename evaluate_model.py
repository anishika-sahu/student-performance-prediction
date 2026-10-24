import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

# Separate input features and target
X = data.drop("G3", axis=1)
y = data["G3"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the saved model
model = joblib.load("student_model.pkl")

# Predict grades
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Evaluation")
print("----------------")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Compare actual and predicted grades
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Student Grades")
plt.plot([0, 20], [0, 20], "r--")
plt.tight_layout()

# Save the graph
plt.savefig("grade_comparison.png")
plt.show()

print("Graph saved as grade_comparison.png")