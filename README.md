# 🎓 Student Performance Prediction

A Machine Learning project that predicts a student's final academic grade using Python and Streamlit.

## 📌 About the Project

This project uses student academic and personal information to estimate their final grade (G3) out of 20.

A Random Forest Regression model is trained using the Student Performance dataset.

The project provides an interactive web application where users can enter student details and receive a predicted final grade.

## 🚀 Features

- Predict student final grades using Machine Learning
- Interactive Streamlit web application
- Data preprocessing and model training
- Model evaluation using regression metrics
- User-friendly interface

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## 📂 Project Structure

```text
Student-Performance-ML/
│
├── Data/
│   └── student-mat.csv
│
├── src/
│   ├── app.py
│   ├── check_data.py
│   ├── prepare_data.py
│   ├── train_modal.py
│   ├── save_model.py
│   ├── predict_student.py
│   └── evaluate_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/anishika-sahu/student-performance-prediction.git
```

### 2. Open the project folder

```bash
cd student-performance-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run src/app.py
```

The application will open in your web browser.

## 📊 Dataset

The project uses the Student Performance dataset.

The target variable is `G3`, which represents the student's final grade out of 20.

## 🎯 Project Objective

The objective of this project is to understand how Machine Learning can be used to analyze student performance and estimate academic outcomes.

## 👩‍💻 Author

Anshika Sahu
