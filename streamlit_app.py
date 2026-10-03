import streamlit as st
import joblib

# Load trained model
model = joblib.load("spam_detector_model.pkl")

# Page configuration
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# Title
st.title("🛡️ AI Based Spam Email/SMS Detector")
st.write("Enter an email or SMS message below and let the AI classify it.")

st.divider()

# Message input
message = st.text_area(
    "📩 Enter your message",
    placeholder="Example: Congratulations! You have won a cash prize. Claim now.",
    height=180
)

# Analyze button
if st.button("🔍 Analyze Message", use_container_width=True):

    if not message.strip():
        st.warning("⚠️ Please enter a message first.")

    else:
        prediction = model.predict([message])[0]

        probabilities = model.predict_proba([message])[0]
        confidence = max(probabilities) * 100

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE DETECTED")
        else:
            st.success("✅ SAFE / NOT SPAM")

        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )

        st.info(
            f"Prediction: **{prediction.upper()}**"
        )

st.divider()

st.caption("AI Based Spam Email/SMS Detector | TF-IDF + Multinomial Naive Bayes")