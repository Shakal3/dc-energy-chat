#!/usr/bin/env python3
"""
Demo script for DC Energy Cost Forecasting System
Demonstrates LightGBM-based forecasting with sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from forecast.cost_engine import (
    cost_forecast,
    load_openinfra_telemetry,
    load_tariff_data,
    prepare_training_data,
    train_lightgbm_model,
    save_trained_model,
    load_trained_model
)

def main():
    """Demonstrate the complete forecasting system."""
    
    print("🔋 DC Energy Cost Forecasting System Demo")
    print("=" * 50)
    
    # 1. Load sample data
    print("\n📊 Loading Sample Data...")
    df_usage = load_openinfra_telemetry('data/openinfra/realistic_telemetry.csv')
    df_tariff = load_tariff_data('data/sample_tariff.csv')
    
    if df_usage.empty:
        print("❌ No telemetry data found. Please ensure data/openinfra/sample_telemetry.csv exists.")
        return
    
    if df_tariff.empty:
        print("❌ No tariff data found. Please ensure data/sample_tariff.csv exists.")
        return
    
    print(f"✅ Loaded {len(df_usage)} telemetry records")
    print(f"✅ Loaded {len(df_tariff)} tariff rates")
    
    # 2. Generate forecasts for different horizons
    print("\n🔮 Generating Cost Forecasts...")
    
    horizons = [6, 24, 168]  # 6 hours, 1 day, 1 week
    horizon_names = ["6 hours", "24 hours (1 day)", "168 hours (1 week)"]
    
    for horizon_h, horizon_name in zip(horizons, horizon_names):
        print(f"\n📈 Forecasting for {horizon_name}:")
        
        try:
            forecast_df = cost_forecast(df_usage, df_tariff, horizon_h=horizon_h)
            
            if not forecast_df.empty:
                # Calculate summary statistics
                total_cost = forecast_df['total_cost_usd'].sum()
                avg_hourly_cost = forecast_df['total_cost_usd'].mean()
                peak_power = forecast_df['predicted_power_kw'].max()
                avg_power = forecast_df['predicted_power_kw'].mean()
                
                print(f"  💰 Total Cost: ${total_cost:.2f}")
                print(f"  📊 Avg Hourly Cost: ${avg_hourly_cost:.2f}")
                print(f"  ⚡ Avg Power: {avg_power:.1f} kW")
                print(f"  🔋 Peak Power: {peak_power:.1f} kW")
                
                # Show first few predictions
                print(f"  📋 First 3 hours:")
                for i in range(min(3, len(forecast_df))):
                    row = forecast_df.iloc[i]
                    timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M')
                    power = row['predicted_power_kw']
                    cost = row['total_cost_usd']
                    print(f"    {timestamp}: {power:.1f} kW → ${cost:.2f}")
            else:
                print("  ❌ Forecast generation failed")
                
        except Exception as e:
            print(f"  ❌ Error generating forecast: {str(e)}")
    
    # 3. Test API-style interaction
    print("\n🤖 Testing Chat API Style Interaction...")
    
    sample_queries = [
        "What will my energy costs be tomorrow?",
        "Predict power usage for next week",
        "Show me cost forecast for next 6 hours"
    ]
    
    for query in sample_queries:
        print(f"\n❓ Query: '{query}'")
        
        # Simulate API intent detection
        from chat_api.routes import detect_forecast_intent, format_forecast_response
        
        intent = detect_forecast_intent(query)
        
        if intent['is_forecast']:
            print(f"✅ Detected forecast intent (horizon: {intent['horizon_h']}h)")
            
            try:
                forecast_df = cost_forecast(df_usage, df_tariff, intent['horizon_h'])
                response = format_forecast_response(forecast_df, query)
                
                if response['forecast_available']:
                    print("💬 Response Preview:")
                    # Show truncated response
                    response_lines = response['response'].split('\n')
                    for line in response_lines[:6]:  # First 6 lines
                        if line.strip():
                            print(f"  {line.strip()}")
                    
                    if len(response_lines) > 6:
                        print("  ...")
                else:
                    print("❌ Forecast not available")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        else:
            print("ℹ️  Not a forecast query")
    
    # 4. Show model information
    print("\n🔧 Model Information...")
    
    model = load_trained_model()
    if model is not None:
        print("✅ Trained LightGBM model available")
        print(f"  📁 Model path: forecast/models/lightgbm_load_forecaster.txt")
        
        # Try to show feature importance if available
        try:
            feature_importance = model.feature_importance(importance_type='gain')
            feature_names = [
                'lag_1h', 'lag_24h', 'lag_168h',
                'hour_of_day', 'day_of_week', 'month', 'is_weekend',
                'rolling_mean_24h', 'rolling_std_24h'
            ]
            
            print("  📊 Top 3 Most Important Features:")
            importance_pairs = list(zip(feature_names, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(importance_pairs[:3]):
                print(f"    {i+1}. {feature}: {importance:.0f}")
                
        except Exception as e:
            print(f"  ⚠️  Could not retrieve feature importance: {e}")
    else:
        print("⚠️  No trained model found - will train on first forecast request")
    
    # 5. Performance summary
    print("\n🎯 System Performance Summary:")
    print("  🏆 Target Accuracy: MAPE ≤ 5%")
    print("  ⚡ Algorithm: LightGBM Gradient Boosting")
    print("  🔄 Features: Lag values (1h, 24h, 168h) + Time features")
    print("  💾 Model Persistence: Automatic save/load")
    print("  🔗 Integration: Flask API with natural language responses")
    
    print("\n✨ Demo completed successfully!")
    print("💡 Try querying the API at: POST /chat with JSON: {'query': 'What will my costs be tomorrow?'}")

if __name__ == "__main__":
    main() 