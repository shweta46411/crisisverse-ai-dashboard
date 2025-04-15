import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Disaster type keyword matching
def get_disaster_type_from_text(text):
    text = text.lower()
    if "earthquake" in text:
        return "earthquake"
    elif "fire" in text:
        return "fire"
    elif "flood" in text:
        return "flood"
    elif "hurricane" in text:
        return "hurricane"
    elif "industrial" in text or "explosion" in text or "chemical" in text:
        return "industrial accident"
    else:
        return "unknown"

# Lat/lon to 3D cartesian conversion for KDTree
EARTH_RADIUS_KM = 6371.0
def latlon_to_cartesian(lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)
    x = EARTH_RADIUS_KM * np.cos(lat) * np.cos(lon)
    y = EARTH_RADIUS_KM * np.cos(lat) * np.sin(lon)
    z = EARTH_RADIUS_KM * np.sin(lat)
    return np.vstack((x, y, z)).T

# Main fake news detection function
def detect_fake_news(social_media, disaster_events, time_window_hours=3, distance_km=20, debug=False):
    social_media["detected_disaster_type"] = social_media["text"].apply(get_disaster_type_from_text)
    social_media["is_verified_event"] = False

    for dtype in disaster_events["disaster_type"].unique():
        sm_subset = social_media[social_media["detected_disaster_type"] == dtype].copy()
        ev_subset = disaster_events[disaster_events["disaster_type"] == dtype].copy()

        if sm_subset.empty or ev_subset.empty:
            continue

        sm_coords = latlon_to_cartesian(sm_subset["latitude"].values, sm_subset["longitude"].values)
        ev_coords = latlon_to_cartesian(ev_subset["latitude"].values, ev_subset["longitude"].values)

        tree = cKDTree(ev_coords)
        distances, indices = tree.query(sm_coords, distance_upper_bound=distance_km)

        matched = distances != np.inf
        matched_indices = np.where(matched)[0]

        for i in matched_indices:
            event_idx = indices[i]
            if event_idx < len(ev_subset):
                tweet_time = sm_subset.iloc[i]["timestamp"]
                event_time = ev_subset.iloc[event_idx]["date"]
                time_diff_hr = abs((tweet_time - event_time).total_seconds()) / 3600
                if time_diff_hr <= time_window_hours:
                    social_media.loc[sm_subset.index[i], "is_verified_event"] = True
                    if debug:
                        print(f"[✓] Matched: tweet @ {tweet_time} ↔ event @ {event_time} | dist = {distances[i]:.1f} km, time_diff = {time_diff_hr:.1f} hr")

    social_media["is_potential_fake"] = ~social_media["is_verified_event"]
    return social_media

# Convert sensor data to disaster_events-like format
def extract_sensor_disasters(sensor_df):
    active = sensor_df[sensor_df["status"] == "active"]
    mapping = {
        "seismic": "earthquake",
        "flood": "flood",
        "temp": "fire",
        "humidity": "fire"
    }
    active = active[active["sensor_type"].isin(mapping.keys())]
    active["disaster_type"] = active["sensor_type"].map(mapping)
    sensor_disasters = active.rename(columns={"timestamp": "date"})[["date", "latitude", "longitude", "disaster_type"]]
    sensor_disasters["event_id"] = range(1, len(sensor_disasters) + 1)
    sensor_disasters["location"] = "unknown"
    sensor_disasters["severity"] = None
    sensor_disasters["casualties"] = None
    sensor_disasters["economic_loss_million_usd"] = None
    sensor_disasters["duration_hours"] = None
    return sensor_disasters
