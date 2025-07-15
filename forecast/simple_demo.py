#!/usr/bin/env python3
"""
Simple demo for DC Energy Cost Forecasting
Works with small datasets and shows basic functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simple_cost_calculation():
    """Demonstrate basic cost calculation without ML training."""
    
    print("🔋 Simple DC Energy Cost Calculation Demo")
    print("=" * 50)
    
    # Load sample data
    print("\n📊 Loading Sample Data...")
    try:
        df_usage = pd.read_csv('data/openinfra/sample_telemetry.csv')
        df_tariff = pd.read_csv('data/sample_tariff.csv')
        
        print(f"✅ Loaded {len(df_usage)} telemetry records")
        print(f"✅ Loaded {len(df_tariff)} tariff rates")
        
        # Convert timestamp
        df_usage['timestamp'] = pd.to_datetime(df_usage['timestamp'])
        
        # Calculate current power usage
        total_current_power = df_usage['power_kw'].sum()
        avg_device_power = df_usage['power_kw'].mean()
        
        print(f"\n⚡ Current Power Analysis:")
        print(f"  Total Facility Power: {total_current_power:.1f} kW")
        print(f"  Average Device Power: {avg_device_power:.1f} kW")
        print(f"  Number of Devices: {df_usage['device_id'].nunique()}")
        
        # Simple cost projection
        print(f"\n💰 Simple Cost Projection (next 24 hours):")
        
        # Use different hourly rates
        peak_rate = df_tariff['rate_usd_per_kwh'].max()
        off_peak_rate = df_tariff['rate_usd_per_kwh'].min()
        avg_rate = df_tariff['rate_usd_per_kwh'].mean()
        
        print(f"  Peak Rate: ${peak_rate:.2f}/kWh")
        print(f"  Off-Peak Rate: ${off_peak_rate:.2f}/kWh")
        print(f"  Average Rate: ${avg_rate:.2f}/kWh")
        
        # Calculate costs for different scenarios
        daily_energy_kwh = total_current_power * 24  # 24 hours
        
        peak_cost = daily_energy_kwh * peak_rate
        off_peak_cost = daily_energy_kwh * off_peak_rate
        avg_cost = daily_energy_kwh * avg_rate
        
        print(f"\n📈 24-Hour Cost Scenarios:")
        print(f"  If all peak hours: ${peak_cost:.2f}")
        print(f"  If all off-peak hours: ${off_peak_cost:.2f}")
        print(f"  Mixed rate average: ${avg_cost:.2f}")
        
        # Show hourly breakdown
        print(f"\n🕐 Sample Hourly Costs (first 6 hours):")
        for i in range(6):
            hour = i
            tariff_rate = df_tariff[df_tariff['hour'] == hour]['rate_usd_per_kwh'].iloc[0]
            hourly_cost = total_current_power * tariff_rate
            period = df_tariff[df_tariff['hour'] == hour]['period_type'].iloc[0]
            print(f"  Hour {hour:2d} ({period:8s}): {total_current_power:.1f} kW × ${tariff_rate:.2f} = ${hourly_cost:.2f}")
        
        # Intent detection demo
        print(f"\n🤖 Intent Detection Demo:")
        test_queries = [
            "What will my energy costs be tomorrow?",
            "How much power are we using?",
            "Predict usage for next week",
            "Hello, how are you?"
        ]
        
        for query in test_queries:
            intent = detect_simple_intent(query)
            print(f"  '{query}' → {intent}")
        
        print(f"\n✨ Simple demo completed!")
        print(f"💡 This shows basic cost calculation while the full ML system trains.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def detect_simple_intent(query):
    """Simple intent detection without imports."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['cost', 'price', 'bill', 'expensive']):
        if any(word in query_lower for word in ['tomorrow', 'next', 'future']):
            return "forecast_cost"
        else:
            return "current_cost"
    elif any(word in query_lower for word in ['power', 'usage', 'energy', 'kw']):
        return "power_usage"
    else:
        return "general_question"

if __name__ == "__main__":
    simple_cost_calculation() 