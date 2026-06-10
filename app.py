import streamlit as st
import pickle

st.title("❤️ Heart Disease Predictor")

model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

male = st.selectbox("Gender", [0, 1])
age = st.number_input("Age")
education = st.number_input("Education")
currentSmoker = st.selectbox("Current Smoker", [0, 1])
cigsPerDay = st.number_input("Cigarettes Per Day")
BPMeds = st.selectbox("BP Medication", [0, 1])
prevalentStroke = st.selectbox("Previous Stroke", [0, 1])
prevalentHyp = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])

totChol = st.number_input("Total Cholesterol")
sysBP = st.number_input("Systolic BP")
diaBP = st.number_input("Diastolic BP")
BMI = st.number_input("BMI")
heartRate = st.number_input("Heart Rate")
glucose = st.number_input("Glucose")

if st.button("Predict"):

    data = [[
        male,
        age,
        education,
        currentSmoker,
        cigsPerDay,
        BPMeds,
        prevalentStroke,
        prevalentHyp,
        diabetes,
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose
    ]]

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")