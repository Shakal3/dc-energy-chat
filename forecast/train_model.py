#!/usr/bin/env python3
"""
Standalone training script for DC Energy Cost Forecasting Model
Usage: python forecast/train_model.py
Implements weekly retraining automation with validation
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from forecast.cost_engine import (
    load_openinfra_telemetry, 
    prepare_training_data,
    train_lightgbm_model,
    save_trained_model,
    load_trained_model,
    generate_power_forecast,
    validate_forecast_accuracy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_model_performance(model, prepared_data, feature_columns, threshold_mape=5.0):
    """
    Validate model performance against MAPE threshold.
    
    Args:
        model: Trained LightGBM model
        prepared_data: Prepared dataset with features
        feature_columns: List of feature column names
        threshold_mape: Maximum acceptable MAPE percentage
    
    Returns:
        Dict with validation results
    """
    try:
        # Use last 30% for validation
        val_start_idx = int(len(prepared_data) * 0.7)
        val_data = prepared_data.iloc[val_start_idx:]
        
        if len(val_data) < 24:  # Need at least 24 hours for meaningful validation
            logger.warning("Insufficient validation data")
            return {'valid': False, 'reason': 'insufficient_data'}
        
        # Prepare validation features and targets
        X_val = val_data[feature_columns]
        y_val = val_data['power_kw']
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        mae = np.mean(np.abs(y_val - y_pred))
        
        # Check if model meets performance threshold
        meets_threshold = mape <= threshold_mape
        
        logger.info(f"Validation Results - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        return {
            'valid': meets_threshold,
            'mape': float(mape),
            'rmse': float(rmse),
            'mae': float(mae),
            'threshold_met': meets_threshold,
            'samples': len(val_data)
        }
        
    except Exception as e:
        logger.error(f"Error in model validation: {str(e)}")
        return {'valid': False, 'reason': str(e)}

def retrain_model(telemetry_path='data/openinfra/demo_telemetry.csv'):
    """
    Complete model retraining pipeline.
    
    Args:
        telemetry_path: Path to telemetry data CSV
    
    Returns:
        Dict with training results
    """
    logger.info("Starting model retraining pipeline...")
    
    try:
        # Load and prepare data
        logger.info(f"Loading telemetry data from {telemetry_path}")
        df_telemetry = load_openinfra_telemetry(telemetry_path)
        
        if df_telemetry.empty:
            logger.error("No telemetry data loaded")
            return {'success': False, 'error': 'no_data'}
        
        logger.info(f"Loaded {len(df_telemetry)} telemetry records")
        
        # Prepare training data
        logger.info("Preparing training data with feature engineering...")
        prepared_data, feature_columns = prepare_training_data(df_telemetry)
        
        if prepared_data.empty:
            logger.error("No prepared training data available")
            return {'success': False, 'error': 'data_preparation_failed'}
        
        logger.info(f"Prepared {len(prepared_data)} training samples with {len(feature_columns)} features")
        
        # Train model
        logger.info("Training LightGBM model...")
        model = train_lightgbm_model(prepared_data, feature_columns)
        
        # Validate model performance
        logger.info("Validating model performance...")
        validation_results = validate_model_performance(model, prepared_data, feature_columns)
        
        if not validation_results['valid']:
            logger.warning(f"Model validation failed: {validation_results.get('reason', 'performance_threshold')}")
            if validation_results.get('mape', float('inf')) > 10.0:
                logger.error("Model performance is too poor to deploy")
                return {'success': False, 'error': 'poor_performance', 'validation': validation_results}
        
        # Save model
        logger.info("Saving trained model...")
        save_trained_model(model)
        
        # Create training summary
        training_summary = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data_samples': len(prepared_data),
            'features': feature_columns,
            'validation': validation_results,
            'model_path': 'forecast/models/lightgbm_load_forecaster.txt'
        }
        
        logger.info("Model retraining completed successfully!")
        logger.info(f"Validation MAPE: {validation_results.get('mape', 'N/A'):.2f}%")
        
        return training_summary
        
    except Exception as e:
        logger.error(f"Error in model retraining: {str(e)}")
        return {'success': False, 'error': str(e)}

def test_model_inference():
    """
    Test the trained model with sample inference.
    """
    logger.info("Testing model inference...")
    
    try:
        # Load trained model
        model = load_trained_model()
        if model is None:
            logger.error("No trained model found")
            return False
        
        # Load sample data for testing
        df_telemetry = load_openinfra_telemetry('data/openinfra/demo_telemetry.csv')
        if df_telemetry.empty:
            logger.error("No test data available")
            return False
        
        # Prepare data
        prepared_data, feature_columns = prepare_training_data(df_telemetry)
        if prepared_data.empty:
            logger.error("No prepared test data")
            return False
        
        # Generate test forecast
        test_forecast = generate_power_forecast(
            model, 
            prepared_data.tail(50), 
            feature_columns, 
            horizon_h=6
        )
        
        if not test_forecast.empty:
            logger.info(f"Test inference successful - generated {len(test_forecast)} predictions")
            logger.info(f"Sample prediction: {test_forecast.iloc[0]['predicted_power_kw']:.2f} kW")
            return True
        else:
            logger.error("Test inference failed - no predictions generated")
            return False
            
    except Exception as e:
        logger.error(f"Error in test inference: {str(e)}")
        return False

def main():
    """
    Main training script entry point.
    """
    logger.info("DC Energy Cost Forecasting - Model Training")
    logger.info("=" * 50)
    
    # Check if data exists
    telemetry_path = 'data/openinfra/demo_telemetry.csv'
    if not os.path.exists(telemetry_path):
        logger.error(f"Telemetry data not found: {telemetry_path}")
        sys.exit(1)
    
    # Run training pipeline
    results = retrain_model(telemetry_path)
    
    if results['success']:
        logger.info("✅ Training completed successfully")
        
        # Test inference
        if test_model_inference():
            logger.info("✅ Model inference test passed")
        else:
            logger.warning("⚠️  Model inference test failed")
        
        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Data samples: {results['data_samples']}")
        print(f"Features: {len(results['features'])}")
        print(f"Validation MAPE: {results['validation'].get('mape', 'N/A'):.2f}%")
        print(f"Validation RMSE: {results['validation'].get('rmse', 'N/A'):.2f}")
        print(f"Model saved: {results['model_path']}")
        print("=" * 50)
        
    else:
        logger.error(f"❌ Training failed: {results.get('error', 'unknown')}")
        sys.exit(1)

if __name__ == "__main__":
    main() 