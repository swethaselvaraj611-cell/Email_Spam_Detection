import streamlit as st
import pickle
import pandas as pd
from datetime import datetime

# ---------------- LOAD MODEL ----------------

with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MailGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "input_message" not in st.session_state:
    st.session_state.input_message = ""

# ---------------- CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
    padding: 20px;
}

.stButton button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}

div[data-testid="stSidebar"] {
    background-color: #eef2ff;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🛡️ MailGuard AI")
st.sidebar.caption("Intelligent Message Security")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📜 History", "ℹ️ About"]
)

st.sidebar.divider()

st.sidebar.caption("AI | Machine Learning | NLP")

# ---------------- HOME ----------------

if page == "🏠 Home":

    st.title("🛡️ MailGuard AI")

    st.markdown(
        "### Intelligent Spam Detection System"
    )

    st.write(
        "Analyze messages using Machine Learning "
        "and identify potentially unwanted messages."
    )

    st.divider()

    # Statistics

    total = len(st.session_state.history)

    spam_count = sum(
        item["Prediction"] == "SPAM"
        for item in st.session_state.history
    )

    ham_count = sum(
        item["Prediction"] == "NOT SPAM"
        for item in st.session_state.history
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Analyzed", total)
    col2.metric("Spam Detected", spam_count)
    col3.metric("Not Spam", ham_count)

    st.divider()

    # Quick examples

    st.markdown("### ⚡ Quick Test Messages")

    st.write("Click a sample to load it into the message box.")

    sample1, sample2 = st.columns(2)

    with sample1:

        if st.button(
            "🚨 Load Spam Example",
            use_container_width=True
        ):

            st.session_state.input_message = (
                "Congratulations! You have won a free prize. "
                "Claim your reward now!"
            )

    with sample2:

        if st.button(
            "✅ Load Normal Example",
            use_container_width=True
        ):

            st.session_state.input_message = (
                "Hi, are you coming to college tomorrow? "
                "Please bring your notebook."
            )

    st.divider()

    # Input form

    st.markdown("### 🔍 Message Analyzer")

    with st.form("message_form"):

        message = st.text_area(
            "Enter your message",
            key="input_message",
            height=180,
            placeholder="Paste or type your message here..."
        )

        analyze = st.form_submit_button(
            "Analyze Message",
            use_container_width=True
        )

    # Analyze

    if analyze:

        if not message.strip():

            st.warning("Please enter a message first.")

        else:

            message_vector = vectorizer.transform([message])

            prediction = model.predict(message_vector)[0]

            probabilities = model.predict_proba(message_vector)[0]

            index = list(model.classes_).index(prediction)

            confidence = probabilities[index] * 100

            label = (
                "SPAM"
                if prediction.lower() == "spam"
                else "NOT SPAM"
            )

            record = {
                "Date": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),
                "Message": message,
                "Prediction": label,
                "Confidence": f"{confidence:.2f}%"
            }

            st.session_state.history.append(record)

            st.session_state.latest_result = {
                "Prediction": label,
                "Confidence": confidence,
                "Message": message
            }

    # Display result

    if st.session_state.latest_result:

        result = st.session_state.latest_result

        st.divider()

        st.markdown("### 📌 Latest Analysis")

        if result["Prediction"] == "SPAM":

            st.error("🚨 SPAM MESSAGE DETECTED")

            st.write(
                "The model classified this message as Spam."
            )

        else:

            st.success("✅ NOT SPAM")

            st.write(
                "The model classified this message as Not Spam."
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Model Confidence",
                f"{result['Confidence']:.2f}%"
            )

        with col2:

            st.metric(
                "Prediction",
                result["Prediction"]
            )

        st.progress(
            float(result["Confidence"] / 100)
        )

        st.caption(
            "Confidence is an estimated model probability, "
            "not a guarantee of message safety."
        )

    st.divider()

    st.markdown("### ⚙️ Model Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Algorithm", "Naive Bayes")
    col2.metric("Feature Extraction", "TF-IDF")
    col3.metric("Dataset", "SMS Spam Collection")

# ---------------- HISTORY ----------------

elif page == "📜 History":

    st.title("📜 Analysis History")

    st.write(
        "View and download messages analyzed during this session."
    )

    st.divider()

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download CSV Report",
            data=csv_data,
            file_name="MailGuard_Analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        if st.button(
            "🗑️ Clear History",
            use_container_width=True
        ):

            st.session_state.history = []
            st.session_state.latest_result = None

            st.rerun()

    else:

        st.info(
            "No analysis records found. "
            "Analyze messages from the Home page."
        )

# ---------------- ABOUT ----------------

elif page == "ℹ️ About":

    st.title("ℹ️ About MailGuard AI")

    st.markdown(
        "### Machine Learning Based Spam Detection"
    )

    st.write(
        "MailGuard AI is a text classification application "
        "that uses Machine Learning to classify messages "
        "into Spam and Not Spam categories."
    )

    st.divider()

    st.markdown("### 🧰 Technologies")

    st.markdown("""
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - TF-IDF Vectorization
    - Multinomial Naive Bayes
    """)

    st.divider()

    st.markdown("### 🔄 Workflow")

    st.markdown("""
    1. User enters a message.
    2. Text is converted into numerical features using TF-IDF.
    3. The trained ML model processes the features.
    4. The model predicts Spam or Not Spam.
    5. Prediction and confidence are displayed.
    6. The result can be stored in session history.
    """)

    st.divider()

    st.markdown("### 🎯 Objective")

    st.write(
        "To develop a Machine Learning application that "
        "can automatically classify messages and assist "
        "users in identifying potentially unwanted messages."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "MailGuard AI | Machine Learning Project"
)