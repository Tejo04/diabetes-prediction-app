# app.py

import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load models and scaler
# -----------------------------
models = {
    "Random Forest": joblib.load("rf_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl"),
    "XGBoost": joblib.load("xgb_model.pkl"),
    "Logistic Regression": joblib.load("logreg_model.pkl")
}

scaler = joblib.load("scaler.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter your health data below to predict diabetes risk.")

# -----------------------------
# Input fields (User-friendly)
# -----------------------------
age = st.number_input("Age", 1, 120, value=30)
glucose = st.number_input("Fasting Glucose (mg/dL)", 0, 300, value=120)
blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 200, value=70)
bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, value=25.0)

# Hidden complex inputs – using smart defaults
pregnancies = 1
skin_thickness = 20
insulin = 85
dpf = 0.5

# Optional: Advanced mode toggle
if st.checkbox("Show advanced medical inputs"):
    pregnancies = st.number_input("Pregnancies", 0, 20, value=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, value=20)
    insulin = st.number_input("Insulin Level (mu U/ml)", 0, 1000, value=85)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, value=0.5)

# -----------------------------
# Model selection
# -----------------------------
selected_model_name = st.selectbox("Choose a model to predict with:", list(models.keys()))
selected_model = models[selected_model_name]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = selected_model.predict(input_scaled)[0]

    st.markdown(f"### 🔍 Prediction using **{selected_model_name}**")
    st.markdown("---")
    if prediction == 1:
        st.error("🚨 The model predicts: Diabetic")
    else:
        st.success("✅ The model predicts: Not Diabetic")
