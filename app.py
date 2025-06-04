import streamlit as st
import joblib
import numpy as np

# Load the saved regression model
model = joblib.load('linear_regression_model.pkl')

# If you used a scaler, load it too (optional)
# scaler = joblib.load('scaler.pkl')

st.title("Linear Regression Model Prediction")

# Define your input features here as Streamlit inputs
# Example with 3 features: span, Runs_per_Year, NO_Ratio
# Change these according to your actual feature names and count

span = st.number_input("Enter Span (years):", min_value=0.0, step=0.1)
runs_per_year = st.number_input("Enter Runs per Year:", min_value=0.0, step=0.1)
no_ratio = st.number_input("Enter NO Ratio:", min_value=0.0, step=0.01)

# Create feature array for prediction
features = np.array([[span, runs_per_year, no_ratio]])

# If you scaled features during training, scale input here as well
# features_scaled = scaler.transform(features)

if st.button("Predict"):
    # prediction = model.predict(features_scaled)  # If scaled
    prediction = model.predict(features)  # If not scaled
    st.write(f"Predicted value: {prediction[0]:.2f}")
