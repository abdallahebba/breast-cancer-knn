import streamlit as st


import numpy as np
import pandas as pd
import joblib

# load model
model = joblib.load("knn_model.pkl")

# title
st.title("Breast Cancer Detection App")

st.write("Enter tumor features")

# inputs
radius_mean = st.number_input("radius_mean")
texture_mean = st.number_input("texture_mean")
perimeter_worst = st.number_input("perimeter_worst")
concave_points_worst = st.number_input("concave_points_worst")
area_worst = st.number_input("area_worst")

# predict
if st.button("Predict"):

    features = np.array([[
        radius_mean,
        texture_mean,
        perimeter_worst,
        concave_points_worst,
        area_worst
    ]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Malignant Tumor")
    else:
        st.success("Benign Tumor")