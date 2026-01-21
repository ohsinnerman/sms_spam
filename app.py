import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Page Config ---
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered"
)

# --- Load Models ---
@st.cache_resource
def load_models():
    """Load model and vectorizer with caching."""
    if not os.path.exists('spam_model.pkl') or not os.path.exists('vectorizer.pkl'):
        st.error("Model artifacts not found. Please run `generate_artifacts.py` first.")
        return None, None
    
    try:
        model = joblib.load('spam_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, vectorizer = load_models()

# --- UI Layout ---
st.title("📩 SMS Spam Detection System")
st.markdown("""
<style>
.stTextArea textarea {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.write("Enter an SMS message below to check if it's **Spam** or **Ham** (Legitimate).")

# Input
user_input = st.text_area("Message Content", height=150, placeholder="Type your message here...")

# Sidebar for controls and info
with st.sidebar:
    st.header("Settings & Info")
    threshold = st.slider("Spam Threshold", 0.0, 1.0, 0.5, 0.05, 
                        help="Probability threshold to classify as Spam. Higher means stricter.")
    
    st.info(
        "**Note on Bias**: This model is trained on the SMS Spam Collection dataset. "
        "It may be biased towards lengthier spam messages typical of that dataset "
        "and might not catch all modern, short spam messages."
    )
    st.caption("Model: Multinomial Naive Bayes | Vectorizer: CountVectorizer")

# Prediction Logic
if st.button("Check Spam", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message to check.")
    elif model and vectorizer:
        try:
            # Transform and Predict
            X_input = vectorizer.transform([user_input])
            # get probability of class 1 (spam)
            spam_prob = model.predict_proba(X_input)[0][1]
            
            # Determine label based on threshold
            is_spam = spam_prob >= threshold
            
            # Display Result
            st.markdown("---")
            if is_spam:
                st.error(f"🚨 **SPAM DETECTED**")
                st.metric("Confidence (Spam Probability)", f"{spam_prob:.2%}")
                st.write("This message has been flagged as potential spam.")
            else:
                st.success(f"✅ **HAM (Legitimate)**")
                st.metric("Confidence (Ham Probability)", f"{1 - spam_prob:.2%}")
                st.write("This message appears to be safe.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit & Scikit-learn*")
