
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬"
)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

st.title("💬 Sentiment Analyzer")
st.write("Enter a sentence to find out whether its sentiment is positive or negative.")

text = st.text_area(
    "Enter your sentence:",
    placeholder="Example: I really love this product!"
)

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter a sentence first.")
    else:
        try:
            with st.spinner("Analyzing sentiment..."):
                classifier = load_model()
                result = classifier(text.strip())[0]

            label = result["label"].upper()
            score = result["score"]

            st.subheader("Result")
            st.write(f"**Sentiment: {label}**")
            st.write(f"Confidence: {score:.1%}")

        except Exception as e:
            st.error("The analysis failed. Please try again.")
            st.exception(e)
