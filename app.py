import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model and scaler (Keras 3 compatible)
model = tf.keras.models.load_model("neural_network_model.keras")
scaler = joblib.load("scaler.pkl")

st.title("Energy Efficiency Prediction App")
st.write("Predict Heating Load using building parameters")

relative_compactness = st.number_input("Relative Compactness", 0.6, 1.0, 0.8)
surface_area = st.number_input("Surface Area", 400.0, 900.0, 500.0)
wall_area = st.number_input("Wall Area", 200.0, 400.0, 300.0)
roof_area = st.number_input("Roof Area", 100.0, 200.0, 110.0)
overall_height = st.number_input("Overall Height", 3.5, 7.0, 7.0)
orientation = st.selectbox("Orientation", [2, 3, 4, 5])
glazing_area = st.slider("Glazing Area", 0.0, 0.4, 0.2)
glazing_dist = st.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5])

input_data = np.array([[relative_compactness, surface_area, wall_area,
                         roof_area, overall_height, orientation,
                         glazing_area, glazing_dist]])

input_scaled = scaler.transform(input_data)

if st.button("Predict Heating Load"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Heating Load: {prediction[0][0]:.2f}")
