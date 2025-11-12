# app.py
"""
Streamlit app to forecast crop time-series using pre-trained LSTM models.
Assumes:
 - CSV placed at ./data/<your csv>.csv (see DATA_FILENAME)
 - Pretrained models + scalers in ./models/ named:
     {district_key}_{crop_key}_{metric_key}_lstm.h5
     {district_key}_{crop_key}_{metric_key}_scaler.pkl
   where key = name.strip().lower().replace(' ', '_')
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Crop Forecast (LSTM)", layout="wide")

# ---------------- CONFIG ----------------
DATA_FILENAME = os.path.join("data", "District wise Crop Statistics from 1952-53 to 2023-24.csv")
# if your file has a different name, change DATA_FILENAME accordingly

MODELS_DIR = os.path.join("models")
N_STEPS_DEFAULT = 3

# If you used abbreviations in filenames (e.g. 'tvm' / 'ekm') map full display names to file keys here:
# Example: "Thiruvananthapuram" displayed in dropdown will map to "tvm" for filenames.
ALIASES = {
    # "Thiruvananthapuram": "tvm",
    # "Ernakulam": "ekm",
    # "Kannur": "kannur",
    # Add/edit as needed. If left empty the app uses normalized full names as keys.
}

# ---------------- HELPERS ----------------
def key_for_name(s: str) -> str:
    """Normalized filename key for a name."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    # if exact display name mapped to alias, use that
    if s in ALIASES:
        return ALIASES[s]
    return s.lower().replace(" ", "_")

def load_csv(path):
    if path.startswith("http://") or path.startswith("https://"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path)
    return df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare dataframe similar to your notebook."""
    df = df.copy()
    # strip header names
    df.columns = [c.strip() for c in df.columns]

    # normalize text columns
    if "District" in df.columns:
        df["District"] = df["District"].astype(str).str.strip()

    if "Crop" in df.columns:
        df["Crop"] = df["Crop"].astype(str).str.strip()

    # Numeric cleanup
    for col in ["Area", "Production", "Productivity"]:
        if col in df.columns:
            # remove commas and coerce
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").replace("", np.nan), errors="coerce")

    # Fill derived metrics where two present (same formulas you used)
    if set(["Area", "Production", "Productivity"]).issubset(df.columns):
        mask_prodmiss = df["Productivity"].isnull() & df["Area"].notnull() & df["Production"].notnull()
        df.loc[mask_prodmiss, "Productivity"] = (df.loc[mask_prodmiss, "Production"] * 1000) / df.loc[mask_prodmiss, "Area"]

        mask_prodnmiss = df["Production"].isnull() & df["Area"].notnull() & df["Productivity"].notnull()
        df.loc[mask_prodnmiss, "Production"] = (df.loc[mask_prodnmiss, "Area"] * df.loc[mask_prodnmiss, "Productivity"]) / 1000

        mask_areamiss = df["Area"].isnull() & df["Production"].notnull() & df["Productivity"].notnull()
        df.loc[mask_areamiss, "Area"] = (df.loc[mask_areamiss, "Production"] * 1000) / df.loc[mask_areamiss, "Productivity"]

    # Derive Year from "Agriculture Year" like '1952-53' -> 1953 as in your notebook
    if "Agriculture Year" in df.columns:
        # extract first 4 digits then +1
        try:
            df["Year"] = df["Agriculture Year"].astype(str).str[:4].astype(int) + 1
        except Exception:
            # fallback: split by '-' and take first number
            df["Year"] = df["Agriculture Year"].astype(str).str.extract(r"(\d{4})")[0].astype(float).fillna(method="ffill").astype(int) + 1

    # standardize display names (Title case) for dropdowns while preserving original values for mapping
    df["District_display"] = df["District"].astype(str).str.strip()
    df["Crop_display"] = df["Crop"].astype(str).str.strip()

    # Optionally title-case them for nicer UI (but keep original for key mapping)
    df["District_display"] = df["District_display"].str.replace(r"\s+", " ", regex=True)
    df["Crop_display"] = df["Crop_display"].str.replace(r"\s+", " ", regex=True)

    return df

def safe_load_model_and_scaler(district_key, crop_key, metric_key):
    model_fname = f"{district_key}_{crop_key}_{metric_key}_lstm.h5"
    scaler_fname = f"{district_key}_{crop_key}_{metric_key}_scaler.pkl"
    model_path = os.path.join(MODELS_DIR, model_fname)
    scaler_path = os.path.join(MODELS_DIR, scaler_fname)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, model_path, scaler_path

    # load model without compilation to avoid Keras legacy metric deserialization issues
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler, model_path, scaler_path

def forecast_with_model(model, scaler, series_values: np.ndarray, n_steps: int, horizon: int):
    """Expect series_values as 1D numpy array of raw numbers (no log or scaling)."""
    # Apply the same preprocessing used during training: log1p then scaler.transform
    series_log = np.log1p(series_values.reshape(-1, 1))
    scaled_all = scaler.transform(series_log)  # scaler must have been fit on log1p during training

    # prepare the rolling forecast
    if len(scaled_all) < n_steps:
        raise ValueError("Not enough points for the given n_steps")

    last_seq = scaled_all[-n_steps:].reshape(1, n_steps, 1)
    preds_scaled = []
    for _ in range(horizon):
        yhat = model.predict(last_seq, verbose=0)
        preds_scaled.append(yhat[0, 0])
        # append and roll
        last_seq = np.append(last_seq[:, 1:, :], [[yhat]], axis=1)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_log = scaler.inverse_transform(preds_scaled)
    preds = np.expm1(preds_log).ravel()
    return preds

def evaluate_preds(actual, predicted):
    if len(actual) == 0:
        return np.nan, np.nan, np.nan
    mae = mean_absolute_error(actual, predicted)
    rmse = sqrt(mean_squared_error(actual, predicted))
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.array(actual) != 0
        if mask.sum() == 0:
            mape = np.nan
        else:
            mape = np.mean(np.abs((np.array(actual)[mask] - np.array(predicted)[mask]) / np.array(actual)[mask])) * 100
    return mae, rmse, mape

# ---------------- UI ----------------
st.title("🌾 Crop Time-Series Forecast (pretrained LSTM)")
st.markdown("Select District, Crop and Metric. Forecast uses your pre-trained models in ./models/.")

# Load and preprocess CSV
if not os.path.exists(DATA_FILENAME):
    st.error(f"CSV not found at {DATA_FILENAME}. Put your CSV into ./data/ and update DATA_FILENAME if needed.")
    st.stop()

raw_df = load_csv(DATA_FILENAME)
df = preprocess_df(raw_df)

# build district list (display)
districts = sorted(df["District_display"].dropna().unique())
district_choice = st.sidebar.selectbox("Select District", districts)

# build crop list for selected district
crops_available = sorted(df[df["District_display"] == district_choice]["Crop_display"].dropna().unique())
crop_choice = st.sidebar.selectbox("Select Crop", crops_available)

# metric choice
metric_choice = st.sidebar.radio("Metric", options=["Area", "Production", "Productivity"], index=1)

# n_steps and forecast year
n_steps = st.sidebar.number_input("LSTM lookback (n_steps used during training)", min_value=1, max_value=10, value=N_STEPS_DEFAULT)
last_data_year = int(df["Year"].max()) if "Year" in df.columns else None
min_year = last_data_year if last_data_year is not None else 2023
forecast_year = st.sidebar.slider("Forecast up to year (inclusive)", min_value=min_year, max_value=min_year + 20, value=min_year + 3)

st.sidebar.markdown("---")
st.sidebar.write("Models directory:", MODELS_DIR)
st.sidebar.write("Filename key rule: `name.strip().lower().replace(' ', '_')`")

# show available crops/districts debug if requested
if st.sidebar.checkbox("Show available crops per district (debug)", False):
    avail = df.groupby("District_display")["Crop_display"].unique().reset_index()
    st.dataframe(avail)

# Filter historical subset
subset = df[(df["District_display"] == district_choice) & (df["Crop_display"] == crop_choice)].copy()
subset = subset.dropna(subset=[metric_choice])
subset = subset.sort_values("Year")

if subset.empty:
    st.error(f"No historical data for {district_choice} - {crop_choice} for metric {metric_choice}.")
    # show quick diagnostics
    st.write("Available crops for this district:", df[df["District_display"] == district_choice]["Crop_display"].unique())
    st.stop()

years = subset["Year"].values
values = subset[metric_choice].astype(float).values

st.subheader(f"Historical {metric_choice} — {crop_choice} ({district_choice})")
st.line_chart(pd.DataFrame({metric_choice: values}, index=years))

# Build filename keys
district_key = key_for_name(subset["District"].iloc[0])
crop_key = key_for_name(subset["Crop"].iloc[0])
metric_key = key_for_name(metric_choice)

# load model+scaler (safe)
model, scaler, expected_model_path, expected_scaler_path = safe_load_model_and_scaler(district_key, crop_key, metric_key)
if model is None or scaler is None:
    st.error("Model or scaler not found for selection.")
    st.write("Expected model path:", expected_model_path)
    st.write("Expected scaler path:", expected_scaler_path)
    st.stop()

st.success(f"Loaded model: {os.path.basename(expected_model_path)}")

# compute forecast horizon
last_year = int(years[-1])
horizon = int(forecast_year - last_year)
if horizon <= 0:
    st.warning("Forecast year must be greater than last historical year.")
    st.stop()

# show evaluation on last portion (optional): do an in-sample rolling forecast to evaluate if you have holdout
if st.checkbox("Show model test evaluation (rolling forecast on last 20%)", False):
    # build scaled series, split last 20% as test if possible
    series = values
    if len(series) < (n_steps + 2):
        st.info("Not enough points to evaluate rolling forecast.")
    else:
        split_idx = max(int(len(series) * 0.8), n_steps + 1)
        train_series = series[:split_idx]
        test_series = series[split_idx:]
        # We will simulate the training approach: fit scaler on full train_series log and then rolling predict test
        series_log = np.log1p(series.reshape(-1, 1))
        # scaler provided was fit on full training historically; for evaluation we re-use provided scaler
        scaled = scaler.transform(series_log)
        preds = []
        for i in range(len(test_series)):
            train_end = split_idx + i
            train_data_scaled = scaled[:train_end]
            # prepare sequences
            X = []
            for j in range(n_steps, len(train_data_scaled)):
                X.append(train_data_scaled[j-n_steps:j, 0])
            X = np.array(X).reshape(-1, n_steps, 1)
            if X.size == 0:
                break
            # train? we don't retrain model; we just predict next using pretrained model with last n_steps
            last_seq = train_data_scaled[-n_steps:].reshape(1, n_steps, 1)
            yhat = model.predict(last_seq, verbose=0)
            preds.append(yhat[0, 0])
            # append yhat to scaled as if it were observed
            scaled = np.vstack([scaled, yhat.reshape(1,1)])

        if len(preds) > 0:
            preds = np.array(preds).reshape(-1, 1)
            preds_inv = np.expm1(scaler.inverse_transform(preds)).ravel()
            actuals = test_series[: len(preds)]
            mae, rmse, mape = evaluate_preds(actuals, preds_inv)
            st.write("Evaluation on last portion -> MAE: {:.3f}, RMSE: {:.3f}, MAPE%: {:.2f}".format(mae, rmse, mape))
            eval_df = pd.DataFrame({"Year": years[split_idx: split_idx + len(preds)], "Actual": actuals, "Predicted": preds_inv})
            st.dataframe(eval_df)

# Run forecast
if st.button("Run Forecast"):
    preds = forecast_with_model(model, scaler, values, n_steps, horizon)
    future_years = np.arange(last_year + 1, forecast_year + 1)
    forecast_df = pd.DataFrame({"Year": future_years, f"Forecast_{metric_choice}": preds})

    # Plot historical + forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, values, marker="o", label="Historical")
    ax.plot(future_years, preds, marker="x", linestyle="--", label="Forecast")
    ax.set_title(f"{metric_choice} Forecast for {crop_choice} ({district_choice})")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric_choice)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Forecast table")
    st.dataframe(forecast_df.style.format({f"Forecast_{metric_choice}": "{:,.2f}"}))

    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast CSV", data=csv_bytes, file_name=f"forecast_{district_key}_{crop_key}_{metric_key}.csv", mime="text/csv")

    st.success("Forecast completed.")
