import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# load .env to env vars
API_URL = os.getenv("API_URL")

st.set_page_config(
    page_title="MedBuddy.ML",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1000px;
}
h1 {
    color: #16A085;
    font-weight: 700;
}
div[data-testid="stButton"] > button {
    background-color: #16A085;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background-color: #117A65;
    color: white;
    border: none;
}
div[data-testid="stMetric"] {
    background-color: rgba(22, 160, 133, 0.08);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(22, 160, 133, 0.2);
}
div[data-testid="stSelectbox"],
div[data-testid="stNumberInput"] {
    margin-bottom: 6px;
}
hr {
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🧬 MedBuddy.ML")
st.write("Heart Disease Risk Predictor 🫀")

st.subheader("Enter patient details and click **Predict**")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age",1,120,52)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)",[0,1])
    cp = st.number_input("Chest pain Type (cp)",0,3,0)
    trestbps = st.number_input("Resting Blood Pressure",0,250,125)
    chol = st.number_input("Cholestrol", 0, 600, 212)


with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", [0, 1])
    restecg = st.number_input("Resting ECG (restecg)",0,2,1)
    thalach = st.number_input("Max Heart Rate (thalach)",0,250,168)
    exang = st.selectbox("Exercide Induced Angina", [0,1])

with col3:
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0) 
    slope = st.number_input("Slope", 0, 2, 2)
    ca = st.number_input("Number of Major Vessels (ca)", 0, 4, 0)
    thal = st.number_input("Thal", 0, 3, 2) 

if st.button("Predict🔎"):
    input_data = {
         "age": age,
         "sex": sex,
         "cp": cp,
         "trestbps": trestbps,
         "chol": chol,
         "fbs": fbs,
         "restecg": restecg,
         "thalach": thalach,
         "exang": exang,
         "oldpeak": oldpeak,
         "slope": slope,
         "ca": ca,
         "thal": thal
    }


    response = requests.post(API_URL, json=input_data)

    if response.status_code !=200:
        st.error("Something went wrong! Try again later...")
    else:
        result = response.json()
        prediction = result["prediction"]    
        probability = result["probability"]
        diagnosis = result["diagnosis"]

        st.divider()

        st.metric(
        label="Heart Disease Probability",
        value=f"{probability:.2f}"
        )

        if prediction == 1:
            st.error(f"⚠️ Model Prediction {diagnosis}")
        else:
            st.success(f" ✅ Model Prediction {diagnosis}")    
