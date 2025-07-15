-- TimescaleDB initialization script for DC Energy Chat
-- TODO: Add proper indexes and retention policies for production
-- TODO: Create hypertables for efficient time-series queries

-- Create extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Power telemetry hypertable
CREATE TABLE IF NOT EXISTS power_telemetry (
    timestamp TIMESTAMPTZ NOT NULL,
    device_id VARCHAR(50) NOT NULL,
    location VARCHAR(100),
    data_center VARCHAR(50),
    power_kw NUMERIC(10,3),
    voltage_v NUMERIC(8,2),
    current_a NUMERIC(8,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable (time-series optimized)
SELECT create_hypertable(
    'power_telemetry', 
    'timestamp',
    if_not_exists => TRUE
);

-- Tariff rates table
CREATE TABLE IF NOT EXISTS tariff_rates (
    id SERIAL PRIMARY KEY,
    hour_start INTEGER CHECK (hour_start >= 0 AND hour_start <= 23),
    hour_end INTEGER CHECK (hour_end >= 0 AND hour_end <= 23),
    rate_per_kwh NUMERIC(10,6),
    season VARCHAR(20),
    effective_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cost forecasts table  
CREATE TABLE IF NOT EXISTS cost_forecasts (
    id SERIAL PRIMARY KEY,
    forecast_timestamp TIMESTAMPTZ NOT NULL,
    target_timestamp TIMESTAMPTZ NOT NULL,
    predicted_cost_usd NUMERIC(10,2),
    confidence_low NUMERIC(10,2),
    confidence_high NUMERIC(10,2),
    model_version VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add some sample tariff data
INSERT INTO tariff_rates (hour_start, hour_end, rate_per_kwh, season, effective_date) VALUES
(0, 6, 0.08, 'all', '2024-01-01'),     -- Off-peak
(6, 18, 0.15, 'all', '2024-01-01'),    -- Peak  
(18, 24, 0.12, 'all', '2024-01-01')    -- Semi-peak
ON CONFLICT DO NOTHING;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_power_telemetry_device_time 
    ON power_telemetry (device_id, timestamp DESC);
    
CREATE INDEX IF NOT EXISTS idx_power_telemetry_location_time
    ON power_telemetry (location, timestamp DESC);

-- TODO: Add retention policy for old data
-- SELECT add_retention_policy('power_telemetry', INTERVAL '1 year'); 