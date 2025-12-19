import streamlit as st
from transformers import pipeline

# 1. Page Configuration (Sets the browser tab title and icon)
st.set_page_config(
    page_title="Sentify AI",
    page_icon="🎭",
    layout="centered"
)

# 2. Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar / Logo Area
with st.sidebar:
    st.title("Settings ⚙️")
    st.info("This app uses a DistilBERT model to analyze sentiment in real-time.")
    st.markdown("---")
    st.write("Created by: [Your Name]")

# 4. Header Section
st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=1000", use_container_width=True)
st.title("🎭 Sentify AI")
st.subheader("Understand the emotion behind your text instantly.")

# 5. Model Loading (Cached)
@st.cache_resource
def load_model():
    # Adding a spinner so users know it's loading the heavy model
    return pipeline("sentiment-analysis")

with st.spinner("Initializing AI Model..."):
    analyzer = load_model()

# 6. Main Interface
st.markdown("---")
user_input = st.text_area("✍️ Enter your text here:", placeholder="e.g., I'm really enjoying this new project!")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze_btn = st.button("Analyze Sentiment ✨")

if analyze_btn:
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Thinking..."):
            result = analyzer(user_input)[0]
            label = result['label']
            score = result['score']

        # 7. Display Results with visual flair
        st.markdown("### Analysis Result:")
        
        if label == "POSITIVE":
            st.balloons()
            st.success(f"### Positive 😄")
            st.progress(score)
            st.write(f"Confidence Score: **{score:.2%}**")
        else:
            st.error(f"### Negative 😟")
            st.progress(score)
            st.write(f"Confidence Score: **{score:.2%}**")

st.markdown("---")
st.caption("Sentify AI v1.0 | Powered by Hugging Face Transformers")
