import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model/insurance_cost_model.pkl")

model = load_model()

st.title("🩺 Health Insurance Cost Predictor")
st.write("Enter details below to predict the insurance claim cost.")

# ------------------ Inputs ------------------ #
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

with col2:
    bloodpressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=90)
    diabetic = st.selectbox("Diabetic", ["Yes", "No"])
    smoker = st.selectbox("Smoker", ["Yes", "No"])

children = st.number_input("Children", min_value=0, max_value=10, value=0)
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Match your training column names exactly
input_data = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "bloodpressure": bloodpressure,
    "diabetic": diabetic,
    "children": children,
    "smoker": smoker,
    "region": region
}])

# ------------------ Prediction ------------------ #
if st.button("Predict Insurance Cost"):
    try:
        pred = model.predict(input_data)[0]
        st.success(f"✅ Predicted Insurance Claim Cost: {pred:.2f}")
    except Exception as e:
        st.error("Something went wrong while predicting.")
        st.exception(e)

st.markdown("---")
st.caption("Model: Tuned RandomForestRegressor (GridSearchCV)")
