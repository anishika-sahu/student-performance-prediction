import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

model = load_model()

# Page title
st.title("🎓 Student Performance Prediction")
st.write("Predict a student's final grade using machine learning.")

st.info("Enter the student's details below to predict their final grade.")

# Student details
st.header("Student Details")

school = st.selectbox("School", ["GP", "MS"])
sex = st.selectbox("Sex", ["F", "M"])
age = st.slider("Age", 15, 22, 18)
address = st.selectbox("Address", ["U", "R"])
famsize = st.selectbox("Family Size", ["GT3", "LE3"])
Pstatus = st.selectbox("Parents' Cohabitation Status", ["T", "A"])

Medu = st.selectbox("Mother's Education (0-4)", [0, 1, 2, 3, 4])
Fedu = st.selectbox("Father's Education (0-4)", [0, 1, 2, 3, 4])

Mjob = st.selectbox(
    "Mother's Job",
    ["teacher", "health", "services", "at_home", "other"]
)
Fjob = st.selectbox(
    "Father's Job",
    ["teacher", "health", "services", "at_home", "other"]
)

reason = st.selectbox(
    "Reason for Choosing School",
    ["home", "reputation", "course", "other"]
)
guardian = st.selectbox(
    "Guardian",
    ["mother", "father", "other"]
)

traveltime = st.slider("Travel Time (1-4)", 1, 4, 1)
studytime = st.slider("Weekly Study Time (1-4)", 1, 4, 2)
failures = st.slider("Past Class Failures (0-4)", 0, 4, 0)

schoolsup = st.selectbox("Extra School Support", ["no", "yes"])
famsup = st.selectbox("Family Educational Support", ["no", "yes"])
paid = st.selectbox("Extra Paid Classes", ["no", "yes"])
activities = st.selectbox("Extracurricular Activities", ["no", "yes"])
nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
higher = st.selectbox("Wants Higher Education", ["yes", "no"])
internet = st.selectbox("Internet Access at Home", ["yes", "no"])
romantic = st.selectbox("In a Romantic Relationship", ["no", "yes"])

famrel = st.slider("Family Relationship Quality (1-5)", 1, 5, 4)
freetime = st.slider("Free Time After School (1-5)", 1, 5, 3)
goout = st.slider("Going Out with Friends (1-5)", 1, 5, 3)
Dalc = st.slider("Workday Alcohol Consumption (1-5)", 1, 5, 1)
Walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 1)
health = st.slider("Health Status (1-5)", 1, 5, 3)
absences = st.number_input(
    "School Absences",
    min_value=0,
    max_value=100,
    value=0
)

G1 = st.slider("First Period Grade (0-20)", 0, 20, 10)
G2 = st.slider("Second Period Grade (0-20)", 0, 20, 10)

# Prepare student data in the same column order as training
student = pd.DataFrame([{
    "school": school,
    "sex": sex,
    "age": age,
    "address": address,
    "famsize": famsize,
    "Pstatus": Pstatus,
    "Medu": Medu,
    "Fedu": Fedu,
    "Mjob": Mjob,
    "Fjob": Fjob,
    "reason": reason,
    "guardian": guardian,
    "traveltime": traveltime,
    "studytime": studytime,
    "failures": failures,
    "schoolsup": schoolsup,
    "famsup": famsup,
    "paid": paid,
    "activities": activities,
    "nursery": nursery,
    "higher": higher,
    "internet": internet,
    "romantic": romantic,
    "famrel": famrel,
    "freetime": freetime,
    "goout": goout,
    "Dalc": Dalc,
    "Walc": Walc,
    "health": health,
    "absences": absences,
    "G1": G1,
    "G2": G2
}])

# Predict final grade
if st.button("Predict Final Grade"):
    try:
        prediction = model.predict(student)[0]
        prediction = max(0, min(20, prediction))

        st.success(f"Predicted Final Grade: {prediction:.1f} / 20")
        st.caption(
            "This is an estimate from the trained model, "
            "not a guaranteed result."
        )
    except Exception as error:
        st.error(f"Prediction error: {error}")