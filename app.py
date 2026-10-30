
# ==========================================================
# CS4106 - Foundations of Generative AI
# Student Name: Saadullah
# Student ID: b020039su
# Project: Sentiment Analyser
# ==========================================================

import streamlit as st
from transformers import pipeline


# ----------------------------------------------------------
# 1. Streamlit Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analyser - CS4106",
    page_icon="😊",
    layout="wide"
)

# Required student identification
st.title("Sentiment Analyser")
st.subheader("Student: Saadullah | ID: b020039su")

st.markdown(
    "Analyze product reviews, identify possible sarcasm, "
    "and compare two reviews side by side."
)

st.divider()


# ----------------------------------------------------------
# 2. Hugging Face Model Configuration
# ----------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


@st.cache_resource
def load_sentiment_model():
    """
    Load and cache the Hugging Face sentiment pipeline.
    Streamlit reuses the loaded model across reruns.
    """
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME
    )
    return classifier


# ----------------------------------------------------------
# 3. Rule-Based Sarcasm Detection
# ----------------------------------------------------------

def check_sarcasm(text):
    """
    A simple heuristic sarcasm checker.

    This is NOT a trained sarcasm detection model.
    It looks for common sarcastic expressions.
    """

    sarcasm_phrases = [
        "oh great",
        "what a masterpiece",
        "absolutely wonderful",
        "just fantastic",
        "love how",
        "pure unadulterated joy",
        "what a brilliant",
        "amazing job",
        "thanks a lot",
        "best product ever",
        "works perfectly",
        "endless hours of joy"
    ]

    lower_text = text.lower()

    matched_phrases = [
        phrase for phrase in sarcasm_phrases
        if phrase in lower_text
    ]

    if matched_phrases:
        return True, ", ".join(matched_phrases)

    return False, "No common sarcasm indicators detected"


# ----------------------------------------------------------
# 4. Sentiment Analysis Function
# ----------------------------------------------------------

def analyze_review(text, classifier):

    if not text or not text.strip():
        return None, None

    # Run Hugging Face sentiment inference
    result = classifier(text[:512])[0]

    sentiment = result["label"]
    confidence = result["score"]

    # Run independent sarcasm heuristic
    sarcasm_detected, sarcasm_reason = check_sarcasm(text)

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "sarcasm": sarcasm_detected,
        "sarcasm_reason": sarcasm_reason
    }, None


# ----------------------------------------------------------
# 5. Load the Model
# ----------------------------------------------------------

try:
    with st.spinner("Loading Hugging Face sentiment model..."):
        classifier = load_sentiment_model()

    st.success("Sentiment analysis model loaded successfully!")

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.info("Check your internet connection and installed libraries.")
    st.stop()


# ----------------------------------------------------------
# 6. Sidebar - Select Application Mode
# ----------------------------------------------------------

st.sidebar.title("Application Settings")

mode = st.sidebar.selectbox(
    "Choose Analysis Mode",
    [
        "Single Text Analysis",
        "Dual Input Comparison"
    ]
)

st.sidebar.divider()

st.sidebar.write("Model Information")
st.sidebar.caption(MODEL_NAME)

st.sidebar.info(
    "Sarcasm detection uses simple linguistic "
    "rules, while sentiment classification "
    "uses the Hugging Face model."
)


# ----------------------------------------------------------
# 7. Reusable Result Display
# ----------------------------------------------------------

def display_result(review, result, heading):

    st.subheader(heading)

    st.write("**Review:**")
    st.write(review)

    # Sentiment result
    if result["sentiment"] == "POSITIVE":
        st.success("Sentiment: POSITIVE")
    else:
        st.error("Sentiment: NEGATIVE")

    st.metric(
        "Confidence Score",
        f"{result['confidence'] * 100:.2f}%"
    )

    # Sarcasm result
    if result["sarcasm"]:
        st.warning(
            "Possible sarcasm detected!"
        )
        st.write(
            "**Matching expressions:**",
            result["sarcasm_reason"]
        )
    else:
        st.info(
            "No common sarcasm indicators detected."
        )


# ----------------------------------------------------------
# 8. Single Text Analysis Mode
# ----------------------------------------------------------

if mode == "Single Text Analysis":

    st.header("Single Review Analysis")

    review = st.text_area(
        "Enter your product review:",
        placeholder="Example: Oh great, another phone that stops working!",
        height=130
    )

    if st.button("Analyze Review", use_container_width=True):

        if review.strip():

            with st.spinner("Analyzing your review..."):

                result, error = analyze_review(
                    review,
                    classifier
                )

            display_result(
                review,
                result,
                "Analysis Result"
            )

        else:
            st.warning("Please enter a review first.")


# ----------------------------------------------------------
# 9. Dual Input Comparison Mode
# ----------------------------------------------------------

elif mode == "Dual Input Comparison":

    st.header("Compare Two Product Reviews")

    col1, col2 = st.columns(2)

    with col1:
        review1 = st.text_area(
            "Enter First Review:",
            placeholder="Example: This phone has excellent battery life!",
            height=150
        )

    with col2:
        review2 = st.text_area(
            "Enter Second Review:",
            placeholder="Example: Oh wonderful, my phone dies every hour!",
            height=150
        )

    if st.button(
        "Compare Reviews",
        use_container_width=True
    ):

        if review1.strip() and review2.strip():

            with st.spinner("Comparing both reviews..."):

                result1, _ = analyze_review(
                    review1,
                    classifier
                )

                result2, _ = analyze_review(
                    review2,
                    classifier
                )

            # Display results side by side
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                display_result(
                    review1,
                    result1,
                    "Review 1 Results"
                )

            with result_col2:
                display_result(
                    review2,
                    result2,
                    "Review 2 Results"
                )

            st.divider()
            st.subheader("Comparison Summary")

            if result1["sentiment"] == result2["sentiment"]:
                st.info(
                    "Both reviews have the same predicted sentiment."
                )
            else:
                st.info(
                    "The two reviews have different predicted sentiments."
                )

            if result1["sarcasm"] != result2["sarcasm"]:
                st.write(
                    "The sarcasm heuristic produced different "
                    "results for the two reviews."
                )

        else:
            st.warning(
                "Please enter both reviews before comparing."
            )


# ----------------------------------------------------------
# 10. Footer
# ----------------------------------------------------------

st.divider()

st.caption(
    "CS4106 Foundations of Generative AI | "
    "Saadullah | b020039su"
)
