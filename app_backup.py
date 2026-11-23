import streamlit as st
import pickle

# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Custom design
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    max-width: 850px;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #14213d;
}

.subtitle {
    color: #64748b;
    font-size: 17px;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# History
if "history" not in st.session_state:
    st.session_state.history = []

# Header
st.markdown(
    '<div class="main-title">📧 Email Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered SMS message classification</div>',
    unsafe_allow_html=True
)

st.divider()

st.write(
    "Enter a message below to check whether "
    "the machine learning model classifies it as Spam or Ham."
)

# Input form
with st.form("spam_form"):

    message = st.text_area(
        "Enter your message",
        placeholder="Type or paste your message here...",
        height=150
    )

    submitted = st.form_submit_button("🔍 Analyze Message")

# Prediction
if submitted:

    if not message.strip():

        st.warning("Please enter a message.")

    else:

        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)[0]

        probabilities = model.predict_proba(message_vector)[0]

        classes = model.classes_

        confidence = max(probabilities) * 100

        if prediction == "spam":

            st.error("🚨 SPAM MESSAGE")

            st.write(
                "The model classified this message as Spam."
            )

        else:

            st.success("✅ NOT SPAM (HAM)")

            st.write(
                "The model classified this message as Ham."
            )

        st.metric(
            "Prediction confidence",
            f"{confidence:.2f}%"
        )

        st.caption(
            "Confidence is the model's estimated probability "
            "for its predicted class, not a guarantee of correctness."
        )

        st.progress(float(max(probabilities)))

        # Save history
        st.session_state.history.insert(
            0,
            {
                "message": message,
                "prediction": prediction,
                "confidence": confidence
            }
        )

# History section
st.divider()

st.subheader("📜 Prediction History")

if st.session_state.history:

    if st.button("🗑️ Clear History"):

        st.session_state.history = []

        st.rerun()

    for item in st.session_state.history:

        with st.expander(item["prediction"].upper()):

            st.write("Message:")
            st.write(item["message"])

            st.write(
                "Prediction:",
                item["prediction"].upper()
            )

            st.write(
                f"Confidence: {item['confidence']:.2f}%"
            )

else:

    st.info("No predictions yet.")

# About section
st.divider()

with st.expander("ℹ️ About this project"):

    st.write("""
    This project uses Machine Learning to classify
    SMS messages into Spam and Ham categories.

    Model: Multinomial Naive Bayes

    Text processing: TF-IDF Vectorization

    Dataset: SMS Spam Collection

    This is an educational classifier and should not
    be used as the sole basis for security decisions.
    """)