import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open("cardio_risk_model.pkl","rb"))
scaler = pickle.load(open("cardio_scaler.pkl","rb"))

st.set_page_config(page_title="CardioRisk AI",page_icon="❤️",layout="centered")

st.title("❤️ CardioRisk AI")
st.subheader("Clinical Cardiovascular Risk Stratification Platform")
st.write("AI-assisted cardiovascular risk assessment using calibrated Random Forest model with probability estimation.")


male = st.selectbox("Gender",[0,1])
age = st.number_input("Age",min_value=1,max_value=120,value=50)
education = st.number_input("Education Level",value=1)
currentSmoker = st.selectbox("Current Smoker",[0,1])
cigsPerDay = st.number_input("Cigarettes Per Day",value=0)
BPMeds = st.selectbox("Blood Pressure Medication",[0,1])
prevalentStroke = st.selectbox("Previous Stroke",[0,1])
prevalentHyp = st.selectbox("Hypertension",[0,1])
diabetes = st.selectbox("Diabetes",[0,1])
totChol = st.number_input("Total Cholesterol",value=200)
sysBP = st.number_input("Systolic Blood Pressure",value=120)
diaBP = st.number_input("Diastolic Blood Pressure",value=80)
BMI = st.number_input("BMI",value=25.0)
heartRate = st.number_input("Heart Rate",value=70)
glucose = st.number_input("Glucose",value=90)


if st.button("Analyze Cardiovascular Risk"):

    patient = pd.DataFrame([[
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

    ]],

    columns=[
        "male",
        "age",
        "education",
        "currentSmoker",
        "cigsPerDay",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose"
    ])
    
    scaled = scaler.transform(patient)

    probability = model.predict_proba(scaled)[0][1]

    error = 1.96*np.sqrt(probability*(1-probability)/100)

    lower = max(0,probability-error)

    upper = min(1,probability+error)


    if probability < 0.10:
        risk = "LOW"
        st.success("🟢 LOW CARDIOVASCULAR RISK")

    elif probability < 0.20:
        risk = "MEDIUM"
        st.warning("🟡 MEDIUM CARDIOVASCULAR RISK")

    else:
        risk = "HIGH"
        st.error("🔴 HIGH CARDIOVASCULAR RISK")


    st.metric("10-Year Cardiovascular Risk",f"{probability*100:.2f}%")

    st.write(f"### Risk Category: {risk}")

    st.write(f"**95% Confidence Interval:** {lower*100:.2f}% - {upper*100:.2f}%")

    st.progress(float(probability))

    st.info("This system is an AI-based decision support prototype and not a medical diagnosis tool.")