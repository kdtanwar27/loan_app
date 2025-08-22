import streamlit as st
import numpy as np
import tensorflow as tf


# Load the Keras model
#model = tf.keras.models.load_model("loan_model.h5")
model = tf.keras.models.load_model("loan_model_tf")

st.title("💸 Loan Default Risk Prediction")
st.markdown("Enter borrower details to assess the risk of default.")

# --- Input Fields ---
int_rate = st.number_input("Interest Rate (e.g., 0.11 for 11%)", min_value=0.0, max_value=1.0, value=0.122640)
installment = st.number_input("Monthly Installment", min_value=0.0, value=319.089431)
log_annual_inc = st.number_input("Log of Annual Income", min_value=0.0, value=10.932117)
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=12.606679)
fico = st.number_input("FICO Credit Score", min_value=300, max_value=850, step=1, value=710)
days_with_cr_line = st.number_input("Days with Credit Line", min_value=0.0, value=4566.767197)
revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=16919.6)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=46.799236)
inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0, step=1, value=1)
delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0, step=1, value=0)
pub_rec = st.number_input("Public Records (e.g., bankruptcies)", min_value=0, step=1, value=0)
not_fully_paid = st.radio("Loan Fully Paid?", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 0 else "No")

# Purpose selection (encoded)
purpose_map = {
    "Credit Card": 0,
    "Debt Consolidation": 1,
    "Educational": 2,
    "Major Purchase": 3,
    "Small Business": 4,
    "All Other": 5
}
purpose = st.selectbox("Purpose of Loan", options=list(purpose_map.keys()))
purpose_encoded = purpose_map[purpose]

# --- Predict Button ---
if st.button("Evaluate Loan Risk"):
    input_data = np.array([[
        int_rate, installment, log_annual_inc, dti, fico,
        days_with_cr_line, revol_bal, revol_util,
        inq_last_6mths, delinq_2yrs, pub_rec,
        not_fully_paid, purpose_encoded
    ]])

    prediction = model.predict(input_data)[0][0]  # sigmoid output
    risk = "High" if prediction > 0.5 else "Low"

    st.subheader("🔍 Prediction Result:")
    st.success(f"**Loan Risk: {risk}** (Score: {prediction:.2f})")
