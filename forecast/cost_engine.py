# Cost-forecasting model for DC Energy Chat Assistant
# Implements LightGBM-based load forecaster with tariff integration
# Target: MAPE ≤ 5% accuracy for cost forecasts

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "forecast/models/"
LIGHTGBM_MODEL_FILE = "lightgbm_load_forecaster.txt"
FEATURE_SCALER_FILE = "feature_scaler.joblib"

def create_lag_features(df: pd.DataFrame, power_col: str = 'power_kw') -> pd.DataFrame:
    """
    Create lag features for time series forecasting.
    
    Args:
        df: DataFrame with hourly power data and datetime index
        power_col: Column name for power values
    
    Returns:
        DataFrame with lag features: lag_1h, lag_24h, lag_168h
    """
    df = df.copy()
    
    # Create lag features (1h, 24h, 168h = 1 week)
    df['lag_1h'] = df[power_col].shift(1)
    df['lag_24h'] = df[power_col].shift(24)
    df['lag_168h'] = df[power_col].shift(168)
    
    # Time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Rolling statistics (moving averages)
    df['rolling_mean_24h'] = df[power_col].rolling(window=24, min_periods=1).mean()
    df['rolling_std_24h'] = df[power_col].rolling(window=24, min_periods=1).std()
    
    return df

def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 5-minute power data to hourly averages.
    
    Args:
        df: DataFrame with 5-minute telemetry data
    
    Returns:
        DataFrame with hourly resampled data
    """
    try:
        # Make a copy to avoid modifying original
        df_copy = df.copy()
        
        # Convert timestamp to datetime
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        df_copy.set_index('timestamp', inplace=True)
        
        # Group by data center and device, then resample
        hourly_data = []
        for (data_center, device_id), group in df_copy.groupby(['data_center', 'device_id']):
            hourly_group = group.resample('1h').agg({
                'power_kw': 'mean',
                'voltage_v': 'mean',
                'current_a': 'mean',
                'location': 'first'
            })
            # Only keep rows with data
            hourly_group = hourly_group.dropna(subset=['power_kw'])
            
            if not hourly_group.empty:
                hourly_group['data_center'] = data_center
                hourly_group['device_id'] = device_id
                hourly_data.append(hourly_group)
        
        # Combine all devices
        if hourly_data:
            result = pd.concat(hourly_data, ignore_index=False)
            # Aggregate by data center and hour for total facility load
            facility_hourly = result.groupby(result.index).agg({
                'power_kw': 'sum',
                'voltage_v': 'mean',
                'current_a': 'sum'
            })
            
            logger.info(f"Resampled to {len(facility_hourly)} hourly records")
            return facility_hourly
        
        logger.warning("No hourly data generated from resampling")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error in resample_to_hourly: {str(e)}")
        return pd.DataFrame()

def prepare_training_data(df_telemetry: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Prepare training data with features for LightGBM.
    
    Args:
        df_telemetry: Raw telemetry DataFrame
    
    Returns:
        Tuple of (prepared_df, feature_columns)
    """
    # Resample to hourly
    hourly_df = resample_to_hourly(df_telemetry)
    
    if hourly_df.empty:
        return pd.DataFrame(), []
    
    # Create lag features
    featured_df = create_lag_features(hourly_df, 'power_kw')
    
    # Define feature columns for model
    feature_columns = [
        'lag_1h', 'lag_24h', 'lag_168h',
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'rolling_mean_24h', 'rolling_std_24h'
    ]
    
    # Drop rows with NaN values (due to lag features)
    featured_df.dropna(subset=feature_columns, inplace=True)
    
    return featured_df, feature_columns

def train_lightgbm_model(df_prepared: pd.DataFrame, feature_columns: list) -> lgb.Booster:
    """
    Train LightGBM model for power load forecasting.
    
    Args:
        df_prepared: DataFrame with features and target
        feature_columns: List of feature column names
    
    Returns:
        Trained LightGBM model
    """
    if df_prepared.empty:
        raise ValueError("No data available for training")
    
    # Prepare features and target
    X = df_prepared[feature_columns]
    y = df_prepared['power_kw']
    
    # Chronological split for validation (last 20% for validation)
    split_idx = int(len(df_prepared) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # LightGBM parameters optimized for time series
    params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    # Log validation performance
    val_pred = model.predict(X_val)
    mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
    logger.info(f"Validation MAPE: {mape:.2f}%")
    
    return model

def load_trained_model() -> Optional[lgb.Booster]:
    """Load trained LightGBM model from disk."""
    model_path = os.path.join(MODEL_PATH, LIGHTGBM_MODEL_FILE)
    if os.path.exists(model_path):
        return lgb.Booster(model_file=model_path)
    return None

def save_trained_model(model: lgb.Booster):
    """Save trained LightGBM model to disk."""
    os.makedirs(MODEL_PATH, exist_ok=True)
    model_path = os.path.join(MODEL_PATH, LIGHTGBM_MODEL_FILE)
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

def load_tariff_data(tariff_path: str = "data/sample_tariff.csv") -> pd.DataFrame:
    """
    Load utility tariff data.
    
    Args:
        tariff_path: Path to tariff CSV file
    
    Returns:
        DataFrame with tariff rates by hour
    """
    if not os.path.exists(tariff_path):
        logger.warning(f"Tariff file not found: {tariff_path}")
        return pd.DataFrame()
    
    df_tariff = pd.read_csv(tariff_path)
    return df_tariff

def generate_power_forecast(
    model: lgb.Booster,
    last_data: pd.DataFrame,
    feature_columns: list,
    horizon_h: int = 24
) -> pd.DataFrame:
    """
    Generate power load forecast using trained model.
    
    Args:
        model: Trained LightGBM model
        last_data: Recent historical data for creating features
        feature_columns: List of feature column names
        horizon_h: Forecast horizon in hours
    
    Returns:
        DataFrame with hourly power forecasts
    """
    # Start from the last timestamp
    last_timestamp = last_data.index[-1]
    forecast_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=horizon_h,
        freq='1H'
    )
    
    # Initialize forecast results
    forecasts = []
    working_data = last_data.copy()
    
    # Generate iterative forecasts
    for i, forecast_time in enumerate(forecast_timestamps):
        # Create features for this timestamp
        current_hour = forecast_time.hour
        current_dow = forecast_time.dayofweek
        current_month = forecast_time.month
        is_weekend = 1 if current_dow >= 5 else 0
        
        # Get lag values
        lag_1h = working_data['power_kw'].iloc[-1] if len(working_data) > 0 else 0
        lag_24h = working_data['power_kw'].iloc[-24] if len(working_data) >= 24 else lag_1h
        lag_168h = working_data['power_kw'].iloc[-168] if len(working_data) >= 168 else lag_1h
        
        # Rolling statistics
        recent_power = working_data['power_kw'].tail(24)
        rolling_mean_24h = recent_power.mean()
        rolling_std_24h = recent_power.std()
        
        # Create feature vector
        features = {
            'lag_1h': lag_1h,
            'lag_24h': lag_24h,
            'lag_168h': lag_168h,
            'hour_of_day': current_hour,
            'day_of_week': current_dow,
            'month': current_month,
            'is_weekend': is_weekend,
            'rolling_mean_24h': rolling_mean_24h,
            'rolling_std_24h': rolling_std_24h
        }
        
        # Predict
        feature_vector = [features[col] for col in feature_columns]
        prediction = model.predict([feature_vector])[0]
        
        forecasts.append({
            'timestamp': forecast_time,
            'predicted_power_kw': max(0, prediction)  # Ensure non-negative
        })
        
        # Update working data with prediction for next iteration
        new_row = pd.DataFrame({
            'power_kw': [prediction],
            'voltage_v': [working_data['voltage_v'].iloc[-1]],
            'current_a': [prediction * working_data['voltage_v'].iloc[-1] / 1000]
        }, index=[forecast_time])
        
        working_data = pd.concat([working_data, new_row])
    
    return pd.DataFrame(forecasts)

def calculate_cost_forecast(
    power_forecast: pd.DataFrame,
    df_tariff: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate cost forecast by merging power forecast with tariff rates.
    
    Args:
        power_forecast: DataFrame with power predictions
        df_tariff: Tariff rates DataFrame
    
    Returns:
        DataFrame with cost forecasts
    """
    if power_forecast.empty or df_tariff.empty:
        return pd.DataFrame()
    
    # Add hour column to power forecast
    power_forecast['hour'] = power_forecast['timestamp'].dt.hour
    
    # Merge with tariff rates
    forecast_with_rates = power_forecast.merge(
        df_tariff[['hour', 'rate_usd_per_kwh', 'demand_charge_usd_per_kw']],
        on='hour',
        how='left'
    )
    
    # Calculate energy cost (kWh × rate)
    forecast_with_rates['energy_cost_usd'] = (
        forecast_with_rates['predicted_power_kw'] * 
        forecast_with_rates['rate_usd_per_kwh']
    )
    
    # Calculate demand cost (peak kW × demand charge spread over forecast period)
    peak_demand_kw = forecast_with_rates['predicted_power_kw'].max()
    demand_charge_per_hour = (
        peak_demand_kw * 
        forecast_with_rates['demand_charge_usd_per_kw'].iloc[0] / 
        len(forecast_with_rates)
    )
    
    forecast_with_rates['demand_cost_usd'] = demand_charge_per_hour
    forecast_with_rates['total_cost_usd'] = (
        forecast_with_rates['energy_cost_usd'] + 
        forecast_with_rates['demand_cost_usd']
    )
    
    return forecast_with_rates[
        ['timestamp', 'predicted_power_kw', 'energy_cost_usd', 
         'demand_cost_usd', 'total_cost_usd']
    ]

def cost_forecast(
    df_usage: pd.DataFrame, 
    df_tariff: pd.DataFrame, 
    horizon_h: int = 24
) -> pd.DataFrame:
    """
    Generate cost forecast for DC energy usage using LightGBM.
    
    Args:
        df_usage: Historical power telemetry DataFrame
        df_tariff: Tariff structure DataFrame
        horizon_h: Forecast horizon in hours (default 24h)
    
    Returns:
        DataFrame with forecast: ['timestamp', 'total_cost_usd', 'predicted_power_kw']
    """
    try:
        # Load or train model
        model = load_trained_model()
        
        if model is None:
            logger.info("No trained model found. Training new model...")
            prepared_data, feature_columns = prepare_training_data(df_usage)
            
            if prepared_data.empty:
                logger.error("No data available for training")
                return pd.DataFrame()
            
            model = train_lightgbm_model(prepared_data, feature_columns)
            save_trained_model(model)
        else:
            # Prepare data to get feature columns
            prepared_data, feature_columns = prepare_training_data(df_usage)
            if prepared_data.empty:
                logger.error("No data available for forecasting")
                return pd.DataFrame()
        
        # Generate power forecast
        power_forecast = generate_power_forecast(
            model, prepared_data.tail(200), feature_columns, horizon_h
        )
        
        # Calculate cost forecast
        cost_forecast_df = calculate_cost_forecast(power_forecast, df_tariff)
        
        logger.info(f"Generated {len(cost_forecast_df)} hour cost forecast")
        return cost_forecast_df
        
    except Exception as e:
        logger.error(f"Error in cost_forecast: {str(e)}")
        return pd.DataFrame()

def load_openinfra_telemetry(csv_path: str) -> pd.DataFrame:
    """
    Load and clean OpenInfra power telemetry CSV data.
    
    Args:
        csv_path: Path to telemetry CSV file
    
    Returns:
        DataFrame with cleaned telemetry data
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['timestamp', 'power_kw', 'device_id', 'data_center']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Data cleaning
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['power_kw'] = pd.to_numeric(df['power_kw'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=['power_kw'])
        df = df[df['power_kw'] >= 0]  # Remove negative power values
        
        logger.info(f"Loaded {len(df)} telemetry records from {csv_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading telemetry data: {str(e)}")
        return pd.DataFrame()

def validate_forecast_accuracy(
    actual_costs: pd.DataFrame, 
    predicted_costs: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics.
    
    Args:
        actual_costs: DataFrame with actual cost data
        predicted_costs: DataFrame with predicted cost data
    
    Returns:
        Dict with MAPE, RMSE, and other accuracy metrics
    """
    try:
        # Align dataframes on timestamp
        merged = pd.merge(
            actual_costs[['timestamp', 'actual_cost']],
            predicted_costs[['timestamp', 'total_cost_usd']],
            on='timestamp',
            how='inner'
        )
        
        if len(merged) == 0:
            return {'error': 'No overlapping data for validation'}
        
        actual = merged['actual_cost']
        predicted = merged['total_cost_usd']
        
        # Calculate metrics
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))
        
        return {
            'mape_percent': float(mape),
            'rmse_usd': float(rmse),
            'mae_usd': float(mae),
            'samples': len(merged)
        }
        
    except Exception as e:
        logger.error(f"Error calculating accuracy metrics: {str(e)}")
        return {'error': str(e)} 