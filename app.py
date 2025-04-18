import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load models and columns
xgb_model = joblib.load("xgboost_best_model.pkl")
sentiment_model = joblib.load("sentiment_logistic_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
columns = joblib.load("xgb_model_columns.pkl")

# Categorical options
job_list = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
marital_list = ['divorced', 'married', 'single']
edu_list = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree']
default_list = ['no', 'yes']
housing_list = ['no', 'yes']
loan_list = ['no', 'yes']
contact_list = ['cellular', 'telephone']
month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_list = ['mon', 'tue', 'wed', 'thu', 'fri']
poutcome_list = ['failure', 'nonexistent', 'success']

st.set_page_config(page_title="CampaignSense Pro", layout="wide")
st.title("🚀 CampaignSense Pro: AI Marketing Suite")

st.markdown("A complete AI-powered tool to predict marketing campaign success and analyze user sentiment. ✨")

tab1, tab2 = st.tabs(["📈 Campaign Predictor", "💬 Sentiment Analyzer"])

# -------------------------------
# 📈 Campaign Predictor Tab
# -------------------------------
with tab1:
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
        raw_data = pd.DataFrame({
            'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
            'default': [default], 'housing': [housing], 'loan': [loan], 'contact': [contact],
            'month': [month], 'day_of_week': [day_of_week], 'campaign': [campaign],
            'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome],
            'emp.var.rate': [emp_var_rate], 'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx], 'euribor3m': [euribor3m], 'nr.employed': [nr_employed]
        })

        encoded = pd.get_dummies(raw_data)
        encoded = encoded.reindex(columns=columns, fill_value=0)

        prediction = xgb_model.predict(encoded)[0]
        probability = xgb_model.predict_proba(encoded)[0][1] * 100

        st.markdown("---")
        st.subheader("📊 Campaign Report")

        if prediction == 1:
            st.success(f"✅ This campaign is likely to SUCCEED!\nSuccess Probability: {probability:.2f}%")
            st.markdown(f"""
            - Job segment like **{job}** responds well.
            - Timing in **{month.upper()}** shows strong engagement trends.
            - Confidence boosted by **positive economic indicators**.
            """)
        else:
            st.error(f"❌ This campaign may UNDERPERFORM.\nSuccess Probability: {probability:.2f}%")
            st.markdown(f"""
            - High last contact delay (pdays = {pdays}) or low previous engagement.
            - Consider adjusting your outreach channel or timing (**{contact}** in **{month.upper()}**).
            - Try sentiment-optimized messaging → Check Tab 2 to craft better content.
            """)

        gauge = go.Figure(go.Indicator(
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
        st.plotly_chart(gauge, use_container_width=True)

        benchmark = pd.DataFrame({
            'Campaign': ['Your Campaign', 'Benchmark Success'],
            'Score': [probability, 73.5]
        })
        bar = px.bar(benchmark, x='Campaign', y='Score', color='Campaign',
                     color_discrete_sequence=['#1dd1a1', '#8395a7'],
                     title="📉 Success Comparison")
        st.plotly_chart(bar, use_container_width=True)

# -------------------------------
# 💬 Sentiment Analyzer Tab
# -------------------------------
with tab2:
    st.subheader("🧠 Sentiment Analyzer for Campaign Messaging")
    st.markdown("Paste a message, tweet, or email copy to analyze tone before sending.")

    user_text = st.text_area("✍️ Enter message or tweet text here:", height=120)
    analyze = st.button("📥 Analyze Sentiment")

    if analyze:
        if user_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            vector = tfidf_vectorizer.transform([user_text])
            sentiment = sentiment_model.predict(vector)[0]
            confidence = sentiment_model.predict_proba(vector)[0][sentiment] * 100

            if sentiment == 1:
                st.success(f"🌟 Positive Sentiment Detected ({confidence:.2f}%)")
                st.markdown("Great tone for engagement! This message may increase conversion chances.")
            else:
                st.error(f"⚠️ Negative Sentiment Detected ({confidence:.2f}%)")
                st.markdown("Consider softening the message or focusing more on benefits.")

            pie = go.Figure(data=[
                go.Pie(labels=['Negative', 'Positive'], values=[100 - confidence, confidence], hole=0.4)
            ])
            pie.update_layout(title="Sentiment Distribution", showlegend=True)
            st.plotly_chart(pie, use_container_width=True)

            feedback = """
            ### 💡 Message Optimization Tips
            - Use action words ("Get started", "Explore", "Unlock")
            - Be benefit-driven (highlight value, not just features)
            - Keep tone friendly and enthusiastic
            """
            st.markdown(feedback)
