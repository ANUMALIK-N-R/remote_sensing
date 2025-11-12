# app.py
"""
Streamlit app to forecast crop time-series using pre-trained LSTM models.
Compatible with model names like:
  Alappuzha_Banana_Area_lstm.h5
  Alappuzha_Banana_Area_scaler.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

st.set_page_config(page_title="Crop Forecast (LSTM)", layout="wide")

# ---------------- CONFIG ----------------
DATA_FILENAME = os.path.join("data", "District wise Crop Statistics from 1952-53 to 2023-24.csv")
MODELS_DIR = os.path.join("models")
N_STEPS_DEFAULT = 3

ALLOWED_DISTRICTS = ["Thiruvananthapuram", "Alappuzha", "Kannur"]
ALLOWED_CROPS = ["Paddy", "Banana", "Rubber"]
ALLOWED_METRICS = ["Area", "Production", "Productivity"]

# ---------------- HELPERS ----------------
def load_csv(path):
    return pd.read_csv(path)

def preprocess_df(df):
    """Simplify cleaning and derive Year."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in ["District", "Crop"]:
        df[c] = df[c].astype(str).str.strip()

    for col in ["Area", "Production", "Productivity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    if "Agriculture Year" in df.columns:
        df["Year"] = df["Agriculture Year"].astype(str).str[:4].astype(int) + 1

    return df

def safe_load_model_and_scaler(district, crop, metric):
    """Load model & scaler for given combination using ProperCase filenames."""
    model_fname = f"{district}_{crop}_{metric}_lstm.h5"
    scaler_fname = f"{district}_{crop}_{metric}_scaler.pkl"
    model_path = os.path.join(MODELS_DIR, model_fname)
    scaler_path = os.path.join(MODELS_DIR, scaler_fname)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, model_path, scaler_path

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler, model_path, scaler_path

def forecast_with_model(model, scaler, series_values, n_steps, horizon):
    """Forecast future values using trained LSTM."""
    series_log = np.log1p(series_values.reshape(-1, 1))
    scaled_all = scaler.transform(series_log)
    last_seq = scaled_all[-n_steps:].reshape(1, n_steps, 1)

    preds_scaled = []
    for _ in range(horizon):
        yhat = model.predict(last_seq, verbose=0)
        preds_scaled.append(yhat[0, 0])
        # correct dimension handling
        last_seq = np.append(last_seq[:, 1:, :], yhat.reshape(1, 1, 1), axis=1)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_log = scaler.inverse_transform(preds_scaled)
    preds = np.expm1(preds_log).ravel()
    return preds

# ---------------- UI ----------------
st.title("🌾 Crop Forecast (LSTM)")
st.markdown("Forecast Area, Production, or Productivity using pretrained LSTM models.")

# Load data
if not os.path.exists(DATA_FILENAME):
    st.error(f"CSV not found at {DATA_FILENAME}. Place your file in ./data/")
    st.stop()

df = preprocess_df(load_csv(DATA_FILENAME))
df = df[df["District"].isin(ALLOWED_DISTRICTS) & df["Crop"].isin(ALLOWED_CROPS)]

# Sidebar
district_choice = st.sidebar.selectbox("Select District", ALLOWED_DISTRICTS)
crop_choice = st.sidebar.selectbox("Select Crop", ALLOWED_CROPS)
metric_choice = st.sidebar.radio("Metric", ALLOWED_METRICS, index=1)
n_steps=3

last_year = int(df["Year"].max())
forecast_year = st.sidebar.slider("Forecast up to year", last_year + 1, last_year + 20, last_year + 3)

# Filter historical data
subset = df[(df["District"] == district_choice) & (df["Crop"] == crop_choice)].dropna(subset=[metric_choice])
subset = subset.sort_values("Year")

if subset.empty:
    st.error(f"No historical data for {district_choice} - {crop_choice}.")
    st.stop()

years = subset["Year"].values
values = subset[metric_choice].values

st.subheader(f"📈 Historical {metric_choice}: {crop_choice} ({district_choice})")
st.line_chart(pd.DataFrame({metric_choice: values}, index=years))

# Load model
model, scaler, model_path, scaler_path = safe_load_model_and_scaler(district_choice, crop_choice, metric_choice)
if model is None or scaler is None:
    st.error(f"❌ Model or scaler not found for {district_choice}-{crop_choice}-{metric_choice}")
    st.write("Expected model:", model_path)
    st.write("Expected scaler:", scaler_path)
    st.stop()

st.success(f"✅ Loaded model: {os.path.basename(model_path)}")

# Run forecast
if st.button("Run Forecast"):
    horizon = forecast_year - years[-1]
    if horizon <= 0:
        st.warning("Forecast year must be greater than last year.")
        st.stop()

    preds = forecast_with_model(model, scaler, values, n_steps, horizon)
    future_years = np.arange(years[-1] + 1, forecast_year + 1)
    forecast_df = pd.DataFrame({"Year": future_years, f"Forecast_{metric_choice}": preds})

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, values, label="Historical", marker="o")
    ax.plot(future_years, preds, label="Forecast", linestyle="--", marker="x")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric_choice)
    ax.set_title(f"{metric_choice} Forecast: {crop_choice} ({district_choice})")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("📊 Forecast Table")
    st.dataframe(forecast_df.style.format({f"Forecast_{metric_choice}": "{:,.2f}"}))

    st.download_button(
        "Download Forecast CSV",
        forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{district_choice}_{crop_choice}_{metric_choice}.csv",
        mime="text/csv"
    )
