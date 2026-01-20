import streamlit as st
import requests

# Backend URL
URL = "http://127.0.0.1:8000"

# Page config
st.set_page_config(page_title="Neurodiversity Classifier", layout="centered")

st.title("🧠 Neurodiversity Classifier App")
st.markdown("Enter the user information and choose the number of clusters to generate an interpretable prediction.")

# -------------------------
# Cluster selection
# -------------------------
st.subheader("Clustering Configuration")
n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=4, step=1)

# -------------------------
# Input Features
# -------------------------
st.subheader("Input Features")

age = st.number_input("Age", min_value=5, max_value=80, value=12)

gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=0)

handedness = st.selectbox("Handedness", options=[0, 1], format_func=lambda x: "Left" if x == 0 else "Right", index=1)

verbal_iq = st.number_input("Verbal IQ", min_value=50, max_value=150, value=100)
performance_iq = st.number_input("Performance IQ", min_value=50, max_value=150, value=100)
full4_iq = st.number_input("Full IQ", min_value=50, max_value=150, value=100)

adhd_index = st.number_input("ADHD Index", min_value=0, max_value=100, value=50)
inattentive = st.number_input("Inattentive Score", min_value=0, max_value=100, value=50)
hyper_impulsive = st.number_input("Hyper-Impulsive Score", min_value=0, max_value=100, value=50)

# -------------------------
# Payload
# -------------------------
payload = {
    "age": age,
    "gender": gender,
    "handedness": handedness,
    "verbal_iq": verbal_iq,
    "performance_iq": performance_iq,
    "full4_iq": full4_iq,
    "adhd_index": adhd_index,
    "inattentive": inattentive,
    "hyper_impulsive": hyper_impulsive,
}

# -------------------------
# Prediction Button
# -------------------------
if st.button("🔍 Classify", use_container_width=True):
    try:
        with st.spinner("Running clustering model..."):
            r = requests.post(f"{URL}/predict", params={"n_clusters": n_clusters}, json=payload, timeout=60)

        r.raise_for_status()
        result = r.json()

        st.success("Prediction successful!")

        # -------------------------
        # Results Display
        # -------------------------
        if "group" in result:
            st.metric("Predicted Cluster Group", result["group"])

            st.subheader("📊 Cluster Profile (Average)")
            st.json(result["cluster_profile"])

            st.subheader("👤 Your Values")
            st.json(result["user_values"])

            st.subheader("📈 How You Compare to the Cluster")
            st.json(result["differences"])

            st.subheader("📝 Interpretation")
            for line in result["interpretation"]:
                st.write("•", line)

        else:
            st.warning("Unexpected response format:")
            st.json(result)

    except requests.exceptions.RequestException as e:
        st.error("Failed to connect to the backend API.")
        st.exception(e)
