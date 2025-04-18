import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud

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

# Profanity list
profanity_words = ["bitch", "asshole", "stupid", "idiot", "dumb", "hate", "ugly", "fool"]

def contains_profanity(text):
    return any(re.search(rf"\\b{word}\\b", text.lower()) for word in profanity_words)

st.set_page_config(page_title="CampaignSense Ultra", layout="wide")
st.title("🚀 CampaignSense Ultra: Complete AI Marketing Suite")

st.markdown("Analyze campaign success, customer sentiment, and visualize everything beautifully with data-backed insights.")

tab1, tab2 = st.tabs(["📈 Campaign Predictor", "💬 Sentiment Analyzer"])

# Campaign Predictor Tab
with tab1:
    with st.form("campaign_form"):
        st.header("🎯 Marketing Campaign Input Form")
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

        submitted = st.form_submit_button("🔍 Predict Campaign Success")

    if submitted:
        st.markdown("---")
        st.subheader("📊 Campaign Report & Visuals")

        input_df = pd.DataFrame({
            'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
            'default': [default], 'housing': [housing], 'loan': [loan], 'contact': [contact],
            'month': [month], 'day_of_week': [day_of_week], 'campaign': [campaign],
            'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome],
            'emp.var.rate': [emp_var_rate], 'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx], 'euribor3m': [euribor3m], 'nr.employed': [nr_employed]
        })

        encoded = pd.get_dummies(input_df)
        encoded = encoded.reindex(columns=columns, fill_value=0)
        prediction = xgb_model.predict(encoded)[0]
        prob = xgb_model.predict_proba(encoded)[0][1] * 100

        colA, colB = st.columns(2)

        with colA:
            if prediction == 1:
                st.success(f"✅ Success Likely! Confidence: {prob:.2f}%")
            else:
                st.error(f"❌ Campaign Risk Alert! Success Chance: {prob:.2f}%")

        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                title={'text': "Success Probability (%)"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1DD1A1"}}
            ))
            st.plotly_chart(fig, use_container_width=True)

        bar_data = pd.DataFrame({
            'Category': ['Your Campaign', 'Benchmark Avg'],
            'Probability': [prob, 74.2]
        })
        bar_fig = px.bar(bar_data, x='Category', y='Probability', color='Category',
                         title='Your Score vs Industry Benchmark', text_auto=True)
        st.plotly_chart(bar_fig, use_container_width=True)

        radar_data = pd.DataFrame({
            'Metrics': ['emp.var.rate', 'euribor3m', 'campaign', 'pdays', 'previous'],
            'Value': [emp_var_rate, euribor3m, campaign, pdays, previous]
        })
        radar_fig = px.line_polar(radar_data, r='Value', theta='Metrics', line_close=True,
                                   title="📌 Key Metric Spread", color_discrete_sequence=['#00cec9'])
        st.plotly_chart(radar_fig, use_container_width=True)

# Sentiment Analyzer Tab
with tab2:
    st.header("💬 Campaign Sentiment Analyzer")
    st.markdown("Test your campaign message for emotional tone and effectiveness before launching.")

    text = st.text_area("Paste campaign message or tweet here:", height=150)
    run_sentiment = st.button("🔎 Analyze Sentiment")

    if run_sentiment:
        if not text.strip():
            st.warning("Please enter a valid message.")
        else:
            flag_profanity = contains_profanity(text)
            vec = tfidf_vectorizer.transform([text])
            sent = sentiment_model.predict(vec)[0]
            conf = sentiment_model.predict_proba(vec)[0][sent] * 100

            pie = go.Figure(data=[go.Pie(labels=['Negative', 'Positive'], values=[100-conf, conf], hole=0.4)])
            pie.update_layout(title="Sentiment Composition")
            st.plotly_chart(pie, use_container_width=True)

            wordcloud = WordCloud(width=600, height=300, background_color='white').generate(text)
            st.subheader("🌥 Word Cloud Representation")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            if sent == 1:
                st.success(f"🌟 Positive Tone Detected ({conf:.2f}%)")
            else:
                st.error(f"⚠️ Caution: Negative Sentiment ({conf:.2f}%)")

            if flag_profanity:
                st.warning("🚫 Profanity detected. Rephrase for a professional tone.")

            st.markdown("""
            ### 🔧 Optimization Suggestions
            - Add benefits and call-to-action words
            - Remove aggressive/negative language
            - Make it more personal and goal-driven
            """)
