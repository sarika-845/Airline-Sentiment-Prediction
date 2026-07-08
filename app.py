import streamlit as st
import joblib

# Page setup
st.set_page_config(page_title="Airline Sentiment Analysis", page_icon="✈️")

# Title
st.title("✈️ Airline Sentiment Analysis")
st.write("Enter a tweet or text to predict sentiment (Negative, Neutral, Positive).")

# Load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# User input
user_input = st.text_area("📝 Enter text here:")

# Prediction button with emojis
if st.button("🔍 🚀 Predict Sentiment 🎯"):
    if user_input.strip() != "":
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        if prediction.lower() == "negative":
            st.error(f"❌ Predicted Sentiment: {prediction}")
        elif prediction.lower() == "positive":
            st.success(f"✅ Predicted Sentiment: {prediction}")
        else:
            st.warning(f"⚠️ Predicted Sentiment: {prediction}")
    else:
        st.warning("⚠️ Please enter some text.")
