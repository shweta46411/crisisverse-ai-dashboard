# utils/anomaly_detector.py

import pandas as pd

def detect_zscore_anomalies(sensor_df, threshold=2):
    df = sensor_df.copy()
    
    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Rolling statistics on the actual reading_value
    df["rolling_mean"] = df["reading_value"].rolling(window=10, min_periods=1).mean()
    df["rolling_std"] = df["reading_value"].rolling(window=10, min_periods=1).std()

    # Z-score
    df["z_score"] = (df["reading_value"] - df["rolling_mean"]) / df["rolling_std"]
    df["anomaly_flag"] = df["z_score"].abs() > threshold

    return df
