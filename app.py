import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('loan_model.pkl')

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.write("Fill the following details to check if your loan is likely to be approved.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
credit_history = st.selectbox("Credit History", ["1", "0"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in 1000s)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)

# Transform inputs
total_income = applicant_income + coapplicant_income

applicant_income_log = np.log(applicant_income + 1)
loan_amount_log = np.log(loan_amount + 1)
loan_term_log = np.log(loan_amount_term + 1)
total_income_log = np.log(total_income + 1)

# Encoding inputs
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = int(credit_history)

dependents = 3 if dependents == "3+" else int(dependents)

property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
property_area = property_map[property_area]

# Create input DataFrame
input_data = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'Credit_History': credit_history,
    'Property_Area': property_area,
    'ApplicantIncomelog': applicant_income_log,
    'LoanAmountlog': loan_amount_log,
    'Loan_Amount_Term_log': loan_term_log,
    'Total_Income_log': total_income_log
}

input_df = pd.DataFrame([input_data])  # ✅ keeps column names

# Predict on click
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ Loan is likely to be Approved.")
    else:
        st.error("❌ Loan is likely to be Rejected.")
