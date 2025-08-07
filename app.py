import streamlit as st
import pandas as pd
import joblib

# Load the model and features
model = joblib.load("best_model(1).pkl")
features = joblib.load("feature_columns(1).pkl")

# Streamlit UI
st.set_page_config(page_title="AQI Predictor", layout="centered")
st.title("🌫️ Air Quality Index (AQI) Predictor")
st.write("Upload pollutant concentration data to get AQI prediction.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:", input_df.head())

        # Ensure correct columns
        if set(features).issubset(input_df.columns):
            input_data = input_df[features]
            prediction = model.predict(input_data)

            st.subheader("📈 Predicted AQI Values:")
            input_df["Predicted_AQI"] = prediction
            st.dataframe(input_df)

        else:
            st.error(f"CSV must contain the following columns: {features}")

    except Exception as e:
        st.error(f"Error: {e}")
