from tkinter import Scale

import streamlit as st
import numpy as np
import joblib


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("knn_model.pkl")
scaler= joblib.load("scaler.pkl")
# ---------------- STYLE ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#0f172a,#111827);
    color:white;
}

.title{
    font-size:60px;
    font-weight:700;
    text-align:center;
    color:#60a5fa;
    margin-top:20px;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#cbd5e1;
    margin-bottom:40px;
}

.result-good{
    background-color:#052e16;
    padding:20px;
    border-radius:15px;
    color:#86efac;
    font-size:25px;
    text-align:center;
    font-weight:bold;
}

.result-bad{
    background-color:#450a0a;
    padding:20px;
    border-radius:15px;
    color:#fca5a5;
    font-size:25px;
    text-align:center;
    font-weight:bold;
}

.stButton>button{
    width:100%;
    background-color:#2563eb;
    color:white;
    border-radius:12px;
    height:50px;
    font-size:20px;
    border:none;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    '<p class="title">🩺 Breast Cancer Detection</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-powered tumor prediction using KNN Machine Learning</p>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Model Info")

st.sidebar.success("Model Accuracy: 97%")

st.sidebar.write("""
Features Used:
- radius_mean
- texture_mean
- perimeter_worst
- concave_points_worst
- area_worst
""")

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("radius_mean")
    texture_mean = st.number_input("texture_mean")
    perimeter_worst = st.number_input("perimeter_worst")

with col2:
    concave_points_worst = st.number_input("concave_points_worst")
    area_worst = st.number_input("area_worst")

# ---------------- PREDICT ----------------
if st.button("Predict Tumor"):

    features = np.array([[
        radius_mean,
        texture_mean,
        perimeter_worst,
        concave_points_worst,
        area_worst
    ]])

    prediction = model.predict(scaler.transform(features))

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.markdown(
            '<div class="result-bad">⚠️ Malignant Tumor Detected</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="result-good">✅ Benign Tumor</div>',
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <center>
        Built with ❤️ using Streamlit & Scikit-Learn
    </center>
    """,
    unsafe_allow_html=True
)