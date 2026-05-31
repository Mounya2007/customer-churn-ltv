# Customer Churn Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("../data/customers_featured.csv")
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    return df

# Load models
@st.cache_resource
def load_models():
    with open("../models/best_model.pkl", "rb") as f:
        churn_model = pickle.load(f)
    with open("../models/ltv_model.pkl", "rb") as f:
        ltv_model = pickle.load(f)
    with open("../models/feature_names_v2.pkl", "rb") as f:
        churn_features = pickle.load(f)
    return churn_model, ltv_model, churn_features

df = load_data()
df["LTV"] = df["tenure"] * df["MonthlyCharges"]
churn_model, ltv_model, churn_features = load_models()

# Title
st.title("Customer Churn Prediction & LTV Dashboard")
st.markdown("---")

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", "7,043")
with col2:
    st.metric("Churn Rate", "26.54%")
with col3:
    st.metric("Avg Monthly Charges", "$64.76")
with col4:
    st.metric("Avg Tenure", "32.37 months")

st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", 
    ["Overview", "Churn Analysis", "LTV Analysis", "Predict Customer"])

if page == "Overview":
    st.header("Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn distribution
        churn_counts = df["Churn"].value_counts()
        fig = px.pie(values=churn_counts.values, 
                    names=["Not Churned", "Churned"],
                    title="Churn Distribution",
                    color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig)
    
    with col2:
        # LTV distribution
        fig = px.histogram(df, x="LTV", 
                          title="LTV Distribution",
                          color_discrete_sequence=["#3498db"])
        st.plotly_chart(fig)

elif page == "Churn Analysis":
    st.header("Churn Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by contract
        fig = px.bar(df.groupby("Contract_Two year")["Churn"].mean().reset_index(),
                    x="Contract_Two year", y="Churn",
                    title="Churn Rate by Contract Type",
                    color_discrete_sequence=["#e74c3c"])
        st.plotly_chart(fig)
    
    with col2:
        # Churn by tenure
        fig = px.box(df, x="Churn", y="tenure",
                    title="Tenure vs Churn",
                    color_discrete_sequence=["#3498db"])
        st.plotly_chart(fig)
    
    # Monthly charges vs churn
    fig = px.box(df, x="Churn", y="MonthlyCharges",
                title="Monthly Charges vs Churn",
                color_discrete_sequence=["#9b59b6"])
    st.plotly_chart(fig)

elif page == "LTV Analysis":
    st.header("LTV Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # LTV segments
        df["LTV_Segment"] = pd.cut(df["LTV"],
                                   bins=[0, 500, 2000, 5000, 9000],
                                   labels=["Low", "Medium", "High", "Premium"])
        segment_counts = df["LTV_Segment"].value_counts()
        fig = px.pie(values=segment_counts.values,
                    names=segment_counts.index,
                    title="LTV Segments",
                    color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig)
    
    with col2:
        # LTV vs tenure
        fig = px.scatter(df, x="tenure", y="LTV",
                        color="Churn",
                        title="LTV vs Tenure",
                        color_continuous_scale="RdYlGn")
        st.plotly_chart(fig)

elif page == "Predict Customer":
    st.header("Predict Customer Churn & LTV")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 119, 65)
        total_charges = st.number_input("Total Charges ($)", value=780.0)
        total_services = st.slider("Total Services", 0, 7, 3)
    
    with col2:
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        is_new_customer = st.selectbox("New Customer", [0, 1])
        is_long_term = st.selectbox("Long Term Customer", [0, 1])
        is_high_spender = st.selectbox("High Spender", [0, 1])
    
    charges_per_month = monthly_charges / (tenure + 1)
    
    if st.button("Predict"):
        # Prepare input
        input_dict = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "TotalServices": total_services,
            "ChargesPerMonth": charges_per_month,
            "IsNewCustomer": is_new_customer,
            "IsLongTermCustomer": is_long_term,
            "IsHighSpender": is_high_spender,
            "SeniorCitizen": senior_citizen
        }
        
        input_df = pd.DataFrame([input_dict])
        
        for col in churn_features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[churn_features]
        
        # Churn prediction
        churn_pred = churn_model.predict(input_df)[0]
        churn_prob = churn_model.predict_proba(input_df)[0]
        
        # LTV prediction
        ltv_input = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "TotalServices": total_services,
            "ChargesPerMonth": charges_per_month,
            "IsNewCustomer": is_new_customer,
            "IsLongTermCustomer": is_long_term,
            "IsHighSpender": is_high_spender
        }])
        ltv_pred = float(ltv_model.predict(ltv_input)[0])
        
        # Display results
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if churn_pred == 1:
                st.error("Will Churn!")
            else:
                st.success("Will Not Churn!")
        
        with col2:
            st.metric("Churn Probability", f"{round(churn_prob[1]*100, 2)}%")
        
        with col3:
            st.metric("Predicted LTV", f"${round(ltv_pred, 2)}")