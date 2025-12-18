
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import load_model # For LSTM
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Stock Price Forecasting Dashboard")
st.title("Apple Stock Price Forecasting Dashboard")

# --- 2. Load Data and Models ---
# Use st.cache_resource for models and large objects to avoid reloading on each rerun
@st.cache_resource
def load_data_and_models():
    try:
        df = joblib.load('df.joblib')
        arima_model = joblib.load('arima_model.joblib')
        sarima_model = joblib.load('sarima_model.joblib')
        random_forest_model = joblib.load('random_forest_model.joblib')
        xgboost_model = joblib.load('xgboost_model.joblib')
        min_max_scaler = joblib.load('min_max_scaler.joblib')
        lstm_model = load_model('lstm_model_saved.keras')

        # Feature lists for ML models (consistent with original notebook)
        rf_xgb_features = [
            "Open", "High", "Low", "Volume",
            "MA07", "MA30", "Volatility", "Daily_Returns"
        ]

        return df, arima_model, sarima_model, random_forest_model, xgboost_model, min_max_scaler, lstm_model, rf_xgb_features
    except Exception as e:
        st.error(f"Error loading models or data: {e}. Please ensure all .joblib and .keras files are in the same directory.")
        return None, None, None, None, None, None, None, None

df, arima_model, sarima_model, random_forest_model, xgboost_model, min_max_scaler, lstm_model, rf_xgb_features = load_data_and_models()

# Stop the app if models/data failed to load
if df is None:
    st.stop()

# --- 3. Forecasting Logic Functions ---

def generate_future_dates(last_date, periods):
    """Generates future business day dates starting from the day after last_date."""
    return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="B")

def get_arima_sarima_forecast(model, forecast_steps=30):
    """Generates forecasts and confidence intervals for ARIMA/SARIMA models."""
    forecast_res = model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    return forecast_mean, conf_int

def get_ml_forecast(model, historical_df_for_features, features, forecast_steps=30):
    """
    Generates future forecasts for ML models.
    NOTE: This is a simplification. Future features are approximated by repeating
    the last known feature set from historical_df_for_features.
    A more robust solution would involve predicting these features or sourcing external data.
    """
    last_known_features = historical_df_for_features[features].iloc[-1]
    
    # Create a DataFrame of future features by repeating the last known features
    future_features_df = pd.DataFrame([last_known_features] * forecast_steps, columns=features)
    
    future_predictions = model.predict(future_features_df)
    return future_predictions

def get_lstm_forecast(model, historical_scaled_data, scaler, lookback, forecast_steps=30):
    """Generates future forecasts for LSTM models using a rolling prediction window."""
    current_sequence = historical_scaled_data[-lookback:].reshape(1, lookback, 1)
    
    future_predictions_scaled = []
    
    for _ in range(forecast_steps):
        next_pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        future_predictions_scaled.append(next_pred_scaled)
        
        # Update the sequence: remove the first element, add the new prediction
        current_sequence = np.append(current_sequence[:, 1:, :], [[[next_pred_scaled]]], axis=1)
        
    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1)).flatten()
    return future_predictions

# --- 4. Sidebar for Navigation and Settings ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Historical Data Analysis", "Model Forecasting"])

# Define LSTM LOOKBACK constant
LSTM_LOOKBACK = 60 

# Slider for number of future days to forecast
FORECAST_STEPS = st.sidebar.slider("Number of Future Days to Forecast", 1, 90, 30)

# --- 5. Main Content Area ---

if page_selection == "Historical Data Analysis":
    st.header("Historical Data Analysis")

    st.subheader("Raw Data and Basic Statistics")
    st.write(df.tail())
    st.write(df.describe())

    st.subheader("Apple Stock Price with Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Close"], label="Close Price")
    ax.plot(df["MA07"], label="7-Day MA")
    ax.plot(df["MA30"], label="30-Day MA")
    ax.set_title("Apple Stock Price with Moving Averages")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.subheader("Volume and Volatility")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df.index, df["Volume"], alpha=0.3, label="Volume")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(df.index, df["Volatility"], color="red", label="Volatility")
    ax2.legend(loc="upper right")
    ax.set_title("Volume and Volatility")
    st.pyplot(fig)

    st.subheader("Distribution of Adjusted Close Price")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["Adj Close"], kde=True, ax=ax)
        ax.set_title("Distribution of Adjusted Close Price")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df["Adj Close"], ax=ax)
        ax.set_title("Boxplot of Adjusted Close Price")
        st.pyplot(fig)

    st.subheader("Year-wise Close Price Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    # Ensure 'Year' column exists for plotting
    if "Year" not in df.columns:
        df["Year"] = df.index.year
    sns.boxplot(x="Year", y="Close", data=df, ax=ax)
    ax.set_title("Year-wise Close Price Distribution")
    st.pyplot(fig)

    st.subheader("Seasonal Decomposition")
    # Period for seasonal decomposition (252 trading days in a year)
    try:
        # Check if df has enough data points for the period
        if len(df["Close"]) >= 2 * 252: # At least 2 full periods needed for meaningful decomposition
            decomposition = seasonal_decompose(df["Close"], model="multiplicative", period=252)
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            axes[0].plot(decomposition.observed); axes[0].set_title("Observed")
            axes[1].plot(decomposition.trend); axes[1].set_title("Trend")
            axes[2].plot(decomposition.seasonal); axes[2].set_title("Seasonality")
            axes[3].plot(decomposition.resid); axes[3].set_title("Residuals")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Not enough data points to perform seasonal decomposition with period 252. Need at least 504 data points.")
    except Exception as e:
        st.warning(f"Could not perform seasonal decomposition: {e}. (Ensure enough data points for the chosen period)")

elif page_selection == "Model Forecasting":
    st.header("Stock Price Forecasts")

    model_choice = st.selectbox(
        "Select Forecasting Model",
        ("ARIMA", "SARIMA", "Random Forest", "XGBoost", "LSTM")
    )

    last_historical_date = df.index[-1]
    future_dates = generate_future_dates(last_historical_date, FORECAST_STEPS)

    st.subheader(f"{model_choice} Forecast for the Next {FORECAST_STEPS} Trading Days")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Close"].tail(200), label="Historical Prices", color="black") # Show recent 200 historical data points

    if model_choice == "ARIMA":
        forecast_mean, conf_int = get_arima_sarima_forecast(arima_model, FORECAST_STEPS)
        ax.plot(future_dates, forecast_mean, label="ARIMA Forecast", linestyle="--", color="red")
        ax.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="pink", alpha=0.3, label="Confidence Interval")
    elif model_choice == "SARIMA":
        forecast_mean, conf_int = get_arima_sarima_forecast(sarima_model, FORECAST_STEPS)
        ax.plot(future_dates, forecast_mean, label="SARIMA Forecast", linestyle="--", color="green")
        ax.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="lightgreen", alpha=0.3, label="Confidence Interval")
    elif model_choice == "Random Forest":
        # Prepare the df for ML models by dropping NaNs created by rolling features
        rf_df_for_forecast = df[rf_xgb_features + ["Close"]].dropna()
        forecast_predictions = get_ml_forecast(random_forest_model, rf_df_for_forecast, rf_xgb_features, FORECAST_STEPS)
        ax.plot(future_dates, forecast_predictions, label="Random Forest Forecast", linestyle="--", color="purple")
    elif model_choice == "XGBoost":
        # Prepare the df for ML models by dropping NaNs created by rolling features
        xgb_df_for_forecast = df[rf_xgb_features + ["Close"]].dropna()
        forecast_predictions = get_ml_forecast(xgboost_model, xgb_df_for_forecast, rf_xgb_features, FORECAST_STEPS)
        ax.plot(future_dates, forecast_predictions, label="XGBoost Forecast", linestyle="--", color="orange")
    elif model_choice == "LSTM":
        # LSTM needs scaled data and a lookback window for sequence generation
        close_data = df[['Close']].values
        scaled_close_for_lstm = min_max_scaler.transform(close_data)
        
        forecast_predictions = get_lstm_forecast(lstm_model, scaled_close_for_lstm, min_max_scaler, LSTM_LOOKBACK, FORECAST_STEPS)
        ax.plot(future_dates, forecast_predictions, label="LSTM Forecast", linestyle="--", color="blue")

    ax.set_title(f"Apple Stock Price - {FORECAST_STEPS} Day Forecast ({model_choice})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Model Comparison (RMSE & MAPE - on Test Data)")
    # Hardcoding evaluation metrics from the notebook for quick display
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("ARIMA", value=f"RMSE: {33.39:.2f}", delta=f"MAPE: {0.1154:.4f}")
    with col2:
        st.metric("SARIMA", value=f"RMSE: {33.48:.2f}", delta=f"MAPE: {0.1157:.4f}")
    with col3:
        st.metric("Random Forest", value=f"RMSE: {32.31:.2f}", delta=f"MAPE: {0.0905:.4f}")
    with col4:
        st.metric("XGBoost", value=f"RMSE: {32.54:.2f}", delta=f"MAPE: {0.0916:.4f}")
    with col5:
        st.metric("LSTM", value=f"RMSE: {7.64:.2f}", delta=f"MAPE: {0.0293:.4f}")

    st.markdown("""
        **Note on ML Model Future Features**:
        For Random Forest and XGBoost, future features (Open, High, Low, Volume, MAs, Volatility, Daily_Returns)
        are currently approximated by repeating the last observed values for the forecast horizon.
        A more accurate forecasting system would involve predicting these features or sourcing external economic indicators.
        
        **Note on ARIMA/SARIMA confidence intervals**:
        The confidence intervals are generated directly by the `get_forecast` method of the `statsmodels` models.
    """)
