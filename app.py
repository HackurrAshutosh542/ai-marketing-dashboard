import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load models and vectorizer
xgb_model = joblib.load("xgboost_best_model.pkl")
sentiment_model = joblib.load("sentiment_logistic_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI setup
st.set_page_config(page_title="CampaignSense | AI Marketing Optimizer", layout="wide")

st.title("📊 CampaignSense: AI-Powered Marketing Optimizer")

st.markdown("""
Welcome to your personalized AI-powered marketing cockpit! 🚀
Choose a tab to either predict your campaign success or analyze the sentiment of any text, tweet, or customer review.
""")

# Tabs
tab1, tab2 = st.tabs(["📈 Campaign Predictor", "💬 Sentiment Analyzer"])

# -------------------------------
# 📈 TAB 1: Campaign Success Predictor
# -------------------------------
with tab1:
    st.header("🎯 Predict Campaign Success with AI")
    st.markdown("Enter the following details about your campaign:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Client Age", 18, 95, 35)
        campaign = st.number_input("# Contacts During Campaign", 1, 50, 1)
    
    with col2:
        pdays = st.number_input("Days Since Last Contact", 0, 999, 999)
        previous = st.number_input("# Previous Contacts", 0, 10, 0)

    with col3:
        emp_var_rate = st.slider("Employment Variation Rate", -3.0, 2.0, 1.1)
        euribor3m = st.slider("3-Month Euribor Rate", 0.5, 5.0, 4.8)

    input_data = np.array([[age, campaign, pdays, previous, emp_var_rate, euribor3m]])

    if st.button("🔮 Predict Campaign Success"):
        prediction = xgb_model.predict(input_data)[0]
        probability = xgb_model.predict_proba(input_data)[0][1] * 100

        # Output summary card
        if prediction == 1:
            st.success("✅ Your campaign is likely to be **successful**!")
            st.markdown("""
            - Great choice of targeting or timing.
            - You can consider increasing email frequency slightly for better reach.
            - 📢 Tip: Success is often higher in **May–July**.
            """)
        else:
            st.error("❌ Your campaign might **not perform well**.")
            st.markdown("""
            - Consider adjusting targeting age group or contact frequency.
            - Review past contact performance (`pdays`, `previous`).
            - 📢 Tip: Test using **multichannel outreach**.
            """)

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Success Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "green" if prediction == 1 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "#FFCCCC"},
                    {'range': [50, 80], 'color': "#FFE066"},
                    {'range': [80, 100], 'color': "#B6F2A6"},
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 💬 TAB 2: Sentiment Analyzer
# -------------------------------
with tab2:
    st.header("💬 Analyze Tweet, Review or Message Sentiment")
    user_text = st.text_area("Paste the text you want to analyze:")

    if st.button("🧠 Analyze Sentiment"):
        if user_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            processed = tfidf_vectorizer.transform([user_text])
            sentiment_pred = sentiment_model.predict(processed)[0]
            confidence = sentiment_model.predict_proba(processed)[0][sentiment_pred] * 100

            # Show result
            if sentiment_pred == 1:
                st.success(f"🌟 Positive Sentiment ({confidence:.2f}% confidence)")
                st.markdown("People are likely responding well to this! 🎉")
            else:
                st.error(f"⚠️ Negative Sentiment ({confidence:.2f}% confidence)")
                st.markdown("Consider improving tone, clarity or value offer.")

            # Pie chart
            fig2 = go.Figure(data=[
                go.Pie(labels=['Negative', 'Positive'], values=[100-confidence, confidence],
                       hole=0.4, marker_colors=['#FF6B6B', '#1DD1A1'])
            ])
            fig2.update_layout(title_text="Sentiment Breakdown")
            st.plotly_chart(fig2, use_container_width=True)
