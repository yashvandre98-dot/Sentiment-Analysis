import streamlit as st
from transformers import pipeline

# Set up the page title
st.title("Sentiment Analysis Tool")
st.write("Enter text below to see if it's Positive or Negative.")

# Load the model (cached so it doesn't reload every time you click a button)
@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis")

obj = load_pipeline()

# User Input
user_text = st.text_input("Input Text:", "i don't like this project")

if st.button("Analyze"):
    result = obj(user_text)[0]
    label = result['label']
    score = result['score']

    if label == "POSITIVE":
        st.success(f"Result: {label} (Confidence: {score:.2f})")
    else:
        st.error(f"Result: {label} (Confidence: {score:.2f})")
