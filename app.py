import numpy as np
import pickle
import streamlit as st

# Load trained model
with open("classifier.pkl", "rb") as file:
    classifier = pickle.load(file)

st.set_page_config(page_title="Banknote Authenticator", layout="centered")

st.title("💵 Banknote Authentication App")
st.write("Predict whether a banknote is **Authentic** or **Fake** using ML.")

st.markdown("---")

# Input fields (numeric only)
variance = st.number_input("Variance", step=0.01)
skewness = st.number_input("Skewness", step=0.01)
curtosis = st.number_input("Curtosis", step=0.01)
entropy = st.number_input("Entropy", step=0.01)

if st.button("Predict"):
    features = np.array([[variance, skewness, curtosis, entropy]])
    prediction = classifier.predict(features)[0]

    if prediction == 0:
        st.success("✅ The banknote is **Authentic**")
    else:
        st.error("❌ The banknote is **Fake**")

st.markdown("---")
st.caption("Built using Machine Learning and Streamlit")
