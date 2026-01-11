import sys
import os

# --- FIX 1: Add 'src' to path so Python can find 'preprocessing.py' ---
sys.path.append(os.path.abspath("src"))
# -----------------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go  # <--- Added for Gauge Chart
import numpy as np

# Page Config
st.set_page_config(
    page_title="Financial Risk Platform",
    page_icon="🏦",
    layout="wide"
)

# Load Models
@st.cache_resource
def load_models():
    try:
        clf = joblib.load('models/best_classifier.pkl')
        reg = joblib.load('models/best_regressor.pkl')
        le = joblib.load('models/label_encoder.pkl')
        return clf, reg, le
    except FileNotFoundError:
        return None, None, None

clf_model, reg_model, label_encoder = load_models()

# Sidebar Navigation
st.sidebar.title("FinRisk AI Platform")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=80)
nav = st.sidebar.radio("Navigate", ["Project Overview", "Data Analytics", "Live Prediction"])

# --- PAGE 1: OVERVIEW ---
if nav == "Project Overview":
    st.title("🏦 Financial Risk Assessment Platform")
    st.markdown("""
    ### Problem Statement
    Financial institutions struggle with manual underwriting. This platform uses AI to:
    1. **Classify Eligibility:** Determine if a user can afford a loan (Eligible / High Risk / Not Eligible).
    2. **Predict Limit:** Calculate the maximum safe monthly EMI amount.
    
    ### Dataset
    * **400,000 Records**
    * **22 Features** (Demographics, Financials, Employment)
    * **5 Scenarios:** E-commerce, Home Appliances, Vehicle, Personal, Education.
    """)

# --- PAGE 2: ANALYTICS ---
elif nav == "Data Analytics":
    st.title("📊 Data Insights")
    
    try:
        # Load Data (Low memory fix)
        df = pd.read_csv('data/emi_dataset.csv', low_memory=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loan Scenario Distribution")
            scenario_counts = df['emi_scenario'].value_counts().reset_index()
            scenario_counts.columns = ['Scenario', 'Count']
            fig1 = px.pie(scenario_counts, names='Scenario', values='Count', title='Applications by Loan Type')
            st.plotly_chart(fig1, width="stretch")
            
        with col2:
            st.subheader("Risk Classification")
            risk_counts = df['emi_eligibility'].value_counts().reset_index()
            risk_counts.columns = ['Status', 'Count']
            fig2 = px.bar(risk_counts, x='Status', y='Count', color='Status', title='Eligibility Counts')
            st.plotly_chart(fig2, width="stretch")
            
        st.subheader("Income vs Max Safe EMI")
        sample_df = df.sample(min(1000, len(df)))
        fig3 = px.scatter(sample_df, x='monthly_salary', y='max_monthly_emi', 
                          color='emi_eligibility', title='Salary vs EMI Capacity (Sample)')
        st.plotly_chart(fig3, width="stretch")
        
    except FileNotFoundError:
        st.error("Dataset not found in `data/emi_dataset.csv`. Please upload it to view analytics.")

# --- PAGE 3: PREDICTION ---
elif nav == "Live Prediction":
    st.title("⚡ Real-Time Risk Assessment")
    
    if clf_model is None:
        st.error("⚠️ Models not loaded! Please run `python src/train.py` first.")
    else:
        with st.form("prediction_form"):
            st.subheader("1. Applicant Details")
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Age", 20, 70, 30)
            gender = c2.selectbox("Gender", ["Male", "Female"])
            marital = c3.selectbox("Marital Status", ["Single", "Married"])
            edu = c1.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            emp_type = c2.selectbox("Employment", ["Private", "Government", "Self-employed"])
            comp_type = c3.selectbox("Company Type", ["MNC", "Startup", "SME", "Public Sector"])
            exp = c1.slider("Years of Experience", 0, 40, 5)
            family = c2.slider("Family Size", 1, 10, 3)
            dependents = c3.slider("Dependents", 0, 5, 1)

            st.subheader("2. Financial Profile")
            c4, c5 = st.columns(2)
            salary = c4.number_input("Monthly Salary (INR)", 10000, 500000, 50000)
            credit_score = c5.number_input("Credit Score (300-900)", 300, 900, 750)
            bank_bal = c4.number_input("Bank Balance", 0, 1000000, 50000)
            emergency = c5.number_input("Emergency Fund", 0, 500000, 20000)

            st.subheader("3. Monthly Expenses")
            c6, c7, c8 = st.columns(3)
            rent = c6.number_input("Monthly Rent", 0, 100000, 5000)
            school = c7.number_input("School Fees", 0, 50000, 2000)
            college = c8.number_input("College Fees", 0, 50000, 0)
            travel = c6.number_input("Travel Expenses", 0, 20000, 3000)
            groceries = c7.number_input("Groceries/Utilities", 0, 30000, 5000)
            other_exp = c8.number_input("Other Expenses", 0, 20000, 2000)
            current_emi = st.number_input("Existing EMI Obligations", 0, 100000, 0)

            st.subheader("4. Loan Request")
            c9, c10, c11 = st.columns(3)
            scenario = c9.selectbox("Loan Scenario", [
                'E-commerce Shopping EMI', 'Home Appliances EMI', 
                'Vehicle EMI', 'Personal Loan EMI', 'Education EMI'
            ])
            req_amt = c10.number_input("Requested Amount", 5000, 2000000, 50000)
            req_tenure = c11.number_input("Tenure (Months)", 3, 84, 12)

            submit_btn = st.form_submit_button("Assess Risk")

        if submit_btn:
            # Build Input
            input_data = {
                'age': age, 'gender': gender, 'marital_status': marital, 'education': edu,
                'monthly_salary': salary, 'employment_type': emp_type, 'years_of_employment': exp,
                'company_type': comp_type, 'house_type': 'Rented' if rent > 0 else 'Own',
                'family_size': family, 'dependents': dependents,
                'monthly_rent': rent, 'school_fees': school, 'college_fees': college,
                'travel_expenses': travel, 'groceries_utilities': groceries,
                'other_monthly_expenses': other_exp, 'current_emi_amount': current_emi,
                'existing_loans': 'Yes' if current_emi > 0 else 'No',
                'credit_score': credit_score, 'bank_balance': bank_bal, 'emergency_fund': emergency,
                'emi_scenario': scenario, 'requested_amount': req_amt, 'requested_tenure': req_tenure
            }

            input_df = pd.DataFrame([input_data])

            # Predict
            pred_idx = clf_model.predict(input_df)[0]
            eligibility = label_encoder.inverse_transform([pred_idx])[0]
            max_emi = reg_model.predict(input_df)[0]
            
            # Requested EMI Calculation
            req_emi_calc = req_amt / req_tenure

            # --- Results Layout ---
            st.divider()
            
            # 1. Top Cards
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown("### 📋 Eligibility Status")
                if eligibility == "Eligible":
                    st.success(f"✅ **{eligibility}**")
                elif eligibility == "High_Risk":
                    st.warning(f"⚠️ **{eligibility}**")
                else:
                    st.error(f"❌ **{eligibility}**")
            
            with col_res2:
                st.markdown("### 💰 Max Safe EMI Limit")
                st.info(f"**₹ {max_emi:,.2f}**")

            # 2. Speedometer / Gauge Chart
            st.subheader("EMI Safety Meter")
            
            # Define Range max (Dynamic scaling)
            max_range = max(max_emi * 1.5, req_emi_calc * 1.2)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = req_emi_calc,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Requested vs Safe Limit", 'font': {'size': 20}},
                delta = {'reference': max_emi, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [0, max_range], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        # Green Zone (0 to Max Safe EMI)
                        {'range': [0, max_emi], 'color': "rgba(0, 200, 0, 0.4)"},  
                        # Red Zone (Max Safe EMI to Limit)
                        {'range': [max_emi, max_range], 'color': "rgba(200, 0, 0, 0.4)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_emi
                    }
                }
            ))
            st.plotly_chart(fig_gauge, width="stretch")

            # 3. Text Comparison
            if req_emi_calc > max_emi:
                st.error(f"🚨 **Alert:** Your requested EMI (₹{req_emi_calc:,.0f}) exceeds the safe limit by ₹{req_emi_calc - max_emi:,.0f}.")
            else:
                st.success(f"✅ **Safe:** Your requested EMI (₹{req_emi_calc:,.0f}) is comfortably within the safe limit.")
