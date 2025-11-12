# app.py
"""
🌾 LSTM Crop Forecasting App
Uses pre-trained LSTM models (per District–Crop–Metric) for forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(page_title="🌾 LSTM Crop Forecast", layout="wide")

# ---------------------------
# CONFIGURATION
# ---------------------------

DATA_PATH = os.path.join("data", "District wise Crop Statistics from 1952-53 to 2023-24.csv")
MODELS_DIR = "models"
N_STEPS_DEFAULT = 3  # Lookback window used in training

# ---------------------------
# LOAD DATA
# ---------------------------

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ---------------------------
# DATA CLEANING
# ---------------------------

def preprocess(df):
    df.columns = [c.strip() for c in df.columns]
    if "Agriculture Year" in df.columns:
        df["Year"] = df["Agriculture Year"].astype(str).str[:4].astype(int) + 1
    for col in ["Area", "Production", "Productivity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["District", "Crop", "Year"])
    df["District"] = df["District"].str.strip().str.lower()
    df["Crop"] = df["Crop"].str.strip().str.lower()
    return df

df = preprocess(df)

# ---------------------------
# SIDEBAR UI
# ---------------------------

st.sidebar.header("⚙️ Settings")

district = st.sidebar.selectbox("🏙️ Select District", ["Alappuzha", "Kannur", "Thiruvanathapuram"])
crop = st.sidebar.selectbox("🌾 Select Crop", ["Paddy", "Banana", "Rubber"])
metric = st.sidebar.radio("📈 Select Metric", ["Area", "Production", "Productivity"])
forecast_year = st.sidebar.slider("🔮 Forecast up to Year", 2025, 2035, 2028)
n_steps = st.sidebar.number_input("LSTM Lookback (training window)", 1, 10, N_STEPS_DEFAULT)

# ---------------------------
# LOAD MODEL + SCALER
# ---------------------------

def load_lstm_model_and_scaler(district, crop, metric):
    # Map display names to file-safe keys
    district_aliases = {"Thiruvananthapuram": "tvm", "Alappuzha": "alp", "Kannur": "kannur"}
    crop_aliases = {"Paddy": "paddy", "Banana": "banana", "Rubber": "rubber"}

    d_key = district_aliases.get(district, district.lower())
    c_key = crop_aliases.get(crop, crop.lower())
    m_key = metric.lower()

    model_path = os.path.join("models", f"{d_key}_{c_key}_{m_key}_lstm.h5")
    scaler_path = os.path.join("models", f"{d_key}_{c_key}_{m_key}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(f"❌ Model or scaler not found for {district}-{crop}-{metric}")
        st.stop()

    # ✅ FIX: Load model without compiling
    model = load_model(model_path, compile=False)

    # Load corresponding scaler
    scaler = joblib.load(scaler_path)

    return model, scaler
# ---------------------------
# FORECAST FUNCTION
# ---------------------------

def forecast_future(model, scaler, values, n_steps, horizon):
    """Forecast future values using pre-trained LSTM model."""
    log_values = np.log1p(values).reshape(-1, 1)
    scaled = scaler.transform(log_values)
    last_seq = scaled[-n_steps:].reshape(1, n_steps, 1)

    preds_scaled = []
    for _ in range(horizon):
        yhat = model.predict(last_seq, verbose=0)
        preds_scaled.append(yhat[0, 0])
        last_seq = np.append(last_seq[:, 1:, :], [[yhat]], axis=1)

    preds = np.expm1(scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))).ravel()
    return preds

# ---------------------------
# FILTER DATA FOR SELECTION
# ---------------------------

subset = df[(df["District"] == district) & (df["Crop"] == crop)].copy()
subset = subset.dropna(subset=[metric])
subset = subset.sort_values("Year")

if subset.empty:
    st.error(f"No historical data for {district}-{crop}.")
    st.stop()

years = subset["Year"].values
values = subset[metric].astype(float).values

if len(values) < n_steps:
    st.error("Not enough data points for forecasting.")
    st.stop()

# ---------------------------
# PLOT HISTORICAL
# ---------------------------

st.subheader(f"📊 Historical {metric.capitalize()} for {crop.capitalize()} in {district.upper()}")
st.line_chart(pd.DataFrame({metric.capitalize(): values}, index=years))

# ---------------------------
# RUN FORECAST
# ---------------------------

if st.button("🔮 Run Forecast"):
    last_year = int(years[-1])
    horizon = int(forecast_year - last_year)
    if horizon <= 0:
        st.warning("Forecast year must be greater than the last data year.")
        st.stop()

    preds = forecast_future(model, scaler, values, n_steps, horizon)
    future_years = np.arange(last_year + 1, forecast_year + 1)
    forecast_df = pd.DataFrame({"Year": future_years, f"Forecast_{metric.capitalize()}": preds})

    # Plot combined chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, values, marker="o", label="Historical")
    ax.plot(future_years, preds, marker="x", linestyle="--", label="Forecast")
    ax.set_title(f"{metric.capitalize()} Forecast for {crop.capitalize()} ({district.upper()})")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    st.pyplot(fig)

    # Show forecast table
    st.subheader("📈 Forecast Table")
    st.dataframe(forecast_df.style.format({f"Forecast_{metric.capitalize()}": "{:,.2f}"}))

    # Download as CSV
    csv_data = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv_data,
        file_name=f"{district}_{crop}_{metric}_forecast.csv",
        mime="text/csv",
    )
