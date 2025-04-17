import streamlit as st
import joblib
import numpy as np

# Load models
xgb_model = joblib.load("xgboost_best_model.pkl")
sentiment_model = joblib.load("sentiment_logistic_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit config
st.set_page_config(page_title="AI Marketing Optimizer", layout="wide")

st.title("🚀 AI-Powered Marketing Campaign Optimizer")

# 👋 Welcome Message
st.markdown("""
Welcome to the **AI-Powered Marketing Dashboard**!  
Use the tabs above to:
- 📈 Predict marketing campaign success
- 💬 Analyze sentiment of tweets or reviews

⚡ Powered by XGBoost & Logistic Regression models
""")

# Tabs
tab1, tab2 = st.tabs(["📈 Campaign Success Predictor", "💬 Sentiment Analyzer"])

# -------------------------------
# 📈 Tab 1: Campaign Prediction
# -------------------------------
with tab1:
    st.header("📊 Predict Campaign Success")

    age = st.slider("Client Age", 18, 95, 35)
    campaign = st.number_input("Number of Contacts", 1, 50, 1)
    pdays = st.number_input("Days Since Last Contact", 0, 999, 999)
    previous = st.number_input("Number of Previous Contacts", 0, 10, 0)
    emp_var_rate = st.slider("Employment Variation Rate", -3.0, 2.0, 1.1)
    euribor3m = st.slider("3-Month Euribor Rate", 0.5, 5.0, 4.8)

    input_data = np.array([[age, campaign, pdays, previous, emp_var_rate, euribor3m]])

    if st.button("Predict Campaign Success"):
        prediction = xgb_model.predict(input_data)[0]
        st.success(f"📢 Prediction: {'Success ✅' if prediction == 1 else 'Not Successful ❌'}")

# -------------------------------
# 💬 Tab 2: Sentiment Analyzer
# -------------------------------
with tab2:
    st.header("🧠 Analyze Tweet or Review Sentiment")
    user_text = st.text_area("Enter tweet, comment or review below:")

    if st.button("Analyze Sentiment"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            processed = tfidf_vectorizer.transform([user_text])
            sentiment_pred = sentiment_model.predict(processed)[0]
            if sentiment_pred == 1:
                st.success("🌟 Positive Sentiment Detected!")
            else:
                st.error("⚠️ Negative Sentiment Detected.")

