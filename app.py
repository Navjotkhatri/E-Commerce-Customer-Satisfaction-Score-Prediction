import streamlit as st
import pandas as pd

from predict import predict_csat
import joblib

dropdowns = joblib.load("models/dropdown_values.pkl")
st.set_page_config(
    page_title="Customer Satisfaction Prediction",
    page_icon="😊",
    layout="wide"
)

st.title("🛍 Customer Satisfaction (CSAT) Prediction")

st.write("Enter customer interaction details below.")

channel = st.selectbox(
    "Channel",
    dropdowns["channel_name"]
)

category = st.selectbox(
    "Category",
    dropdowns["category"]
)

subcategory = st.selectbox(
    "Sub-category",
    dropdowns["Sub-category"]
)

remarks = st.text_area("Customer Remarks")

issue_reported = st.text_input(
    "Issue Reported Time",
    "31/08/2023 10:00"
)

issue_responded = st.text_input(
    "Issue Responded Time",
    "31/08/2023 10:30"
)

survey_date = st.text_input(
    "Survey Response Date",
    "31-Aug-23"
)

tenure = st.selectbox(
    "Tenure Bucket",
    dropdowns["Tenure Bucket"]
)

shift = st.selectbox(
    "Agent Shift",
    [
        "Morning",
        "Evening",
        "Night",
        "Split"
    ]
)

if st.button("Predict CSAT"):

    sample = pd.DataFrame({

        "Unique id":["Demo"],

        "channel_name":[channel],

        "category":[category],

        "Sub-category":[subcategory],

        "Customer Remarks":[remarks],

        "Order_id":["Demo"],

        "order_date_time":[None],

        "Issue_reported at":[issue_reported],

        "issue_responded":[issue_responded],

        "Survey_response_Date":[survey_date],

        "Customer_City":[None],

        "Product_category":[None],

        "Item_price":[None],

        "connected_handling_time":[None],

        "Tenure Bucket":[tenure],

        "Agent Shift":[shift]

    })

    prediction = predict_csat(sample)

    st.success(f"🎯 Predicted CSAT Score : {prediction}")