# utils/zone_features.py

import pandas as pd

import pandas as pd

def generate_zone_sensor_features(sensor_df):
    if 'timestamp' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])

    zone_stats = sensor_df.groupby("zone_id").agg(
        mean_value=("reading_value", "mean"),
        max_value=("reading_value", "max"),
        min_value=("reading_value", "min"),
        anomaly_count=("anomaly_flag", "sum"),
        sensor_count=("sensor_type", "count")
    ).reset_index()

    return zone_stats
