import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the trained model
xgb_model = joblib.load("xgboost_best_model.pkl")

# Load the columns used in training (assumed to be stored)
columns = joblib.load("xgb_model_columns.pkl")  # You must save this during model training

# Categorical options (based on your original dataset)
job_list = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
            'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
marital_list = ['divorced', 'married', 'single']
edu_list = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
            'professional.course', 'university.degree']
default_list = ['no', 'yes']
housing_list = ['no', 'yes']
loan_list = ['no', 'yes']
contact_list = ['cellular', 'telephone']
month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_list = ['mon', 'tue', 'wed', 'thu', 'fri']
poutcome_list = ['failure', 'nonexistent', 'success']

st.set_page_config(page_title="CampaignSense Pro", layout="wide")
st.title("🚀 CampaignSense Pro: Full-Feature Marketing Optimizer")

st.markdown("Fill in the campaign details below. The model will predict whether your campaign is likely to succeed and offer insights.")

with st.form("campaign_form"):
    st.subheader("📋 Campaign Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Client Age", 18, 95, 35)
        job = st.selectbox("Job", job_list)
        marital = st.selectbox("Marital Status", marital_list)
        education = st.selectbox("Education Level", edu_list)
        default = st.selectbox("Credit Default?", default_list)
        housing = st.selectbox("Housing Loan?", housing_list)

    with col2:
        loan = st.selectbox("Personal Loan?", loan_list)
        contact = st.selectbox("Contact Type", contact_list)
        month = st.selectbox("Last Contact Month", month_list)
        day_of_week = st.selectbox("Last Contact Day", day_list)
        poutcome = st.selectbox("Previous Outcome", poutcome_list)
        campaign = st.number_input("# Contacts This Campaign", 1, 50, 1)

    with col3:
        pdays = st.number_input("Days Since Last Contact", 0, 999, 999)
        previous = st.number_input("# Previous Contacts", 0, 10, 0)
        emp_var_rate = st.slider("Employment Variation Rate", -3.0, 2.0, 1.1)
        cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.994)
        cons_conf_idx = st.slider("Consumer Confidence Index", -50.0, -20.0, -36.4)
        euribor3m = st.slider("3-Month Euribor Rate", 0.5, 5.0, 4.8)
        nr_employed = st.slider("# Employed in Economy", 4000, 5500, 5191)

    submitted = st.form_submit_button("🔮 Predict Campaign Success")

if submitted:
    # Create raw input dataframe
    raw_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome],
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed]
    })

    # One-hot encode user input
    user_df_encoded = pd.get_dummies(raw_data)

    # Align with model training columns
    user_df_encoded = user_df_encoded.reindex(columns=columns, fill_value=0)

    # Predict
    prediction = xgb_model.predict(user_df_encoded)[0]
    probability = xgb_model.predict_proba(user_df_encoded)[0][1] * 100

    # Display result
    if prediction == 1:
        st.success(f"✅ This campaign is likely to SUCCEED!\nSuccess Probability: {probability:.2f}%")
    else:
        st.error(f"❌ This campaign might not perform well.\nSuccess Probability: {probability:.2f}%")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Success Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if prediction == 1 else "red"},
            'steps': [
                {'range': [0, 50], 'color': "#FFCCCC"},
                {'range': [50, 80], 'color': "#FFE066"},
                {'range': [80, 100], 'color': "#B6F2A6"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.info("Model used: Full-feature XGBoost with one-hot encoded campaign metadata")
