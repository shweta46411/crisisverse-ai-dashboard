import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.spatial import cKDTree

def latlon_to_cartesian(lat, lon):
    EARTH_RADIUS_KM = 6371.0
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = EARTH_RADIUS_KM * np.cos(lat_rad) * np.cos(lon_rad)
    y = EARTH_RADIUS_KM * np.cos(lat_rad) * np.sin(lon_rad)
    z = EARTH_RADIUS_KM * np.sin(lat_rad)
    return np.vstack((x, y, z)).T

def process_data(city_map, energy_consumption, disaster_events):
    # 1. Extract building_id, coordinates, and type from GeoJSON
    building_coords = []
    for feature in city_map["features"]:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        if props["name"].startswith("Location"):
            building_id = int(props["name"].replace("Location ", ""))
            building_coords.append({
                "building_id": building_id,
                "longitude": coords[0],
                "latitude": coords[1],
                "type": props.get("type", "unknown")
            })

    buildings_df = pd.DataFrame(building_coords)

    # 2. Merge energy data with building info (including type)
    energy = energy_consumption.merge(buildings_df, on="building_id", how="left")

    # 3. Aggregate energy consumption hourly
    energy["hour"] = energy["timestamp"].dt.floor("H")
    energy_hourly = energy.groupby(["building_id", "hour"]).agg(
        energy_kwh=("energy_kwh", "mean"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        type=("type", "first")
    ).reset_index()

    # 4. Coordinate conversion
    coords = latlon_to_cartesian(energy_hourly["latitude"], energy_hourly["longitude"])
    tree = cKDTree(coords)

    # 5. Match disasters with affected buildings
    affected_records = []
    for _, row in disaster_events.iterrows():
        event_time = row["date"]
        event_lat = row["latitude"]
        event_lon = row["longitude"]
        event_type = row["disaster_type"]
        event_id = row["event_id"]

        event_coord = latlon_to_cartesian([event_lat], [event_lon])[0]
        idxs = tree.query_ball_point(event_coord, r=5)  # 5km radius

        time_window_start = event_time - timedelta(hours=1)
        time_window_end = event_time + timedelta(hours=1)

        affected = energy_hourly.iloc[idxs]
        affected = affected[
            (affected["hour"] >= time_window_start) &
            (affected["hour"] <= time_window_end)
        ].copy()
        affected["event_id"] = event_id
        affected["event_type"] = event_type
        affected["event_time"] = event_time

        affected_records.append(affected)

    # 6. Combine affected records
    affected_df = pd.concat(affected_records, ignore_index=True)

    # 7. Compare with average energy per building
    building_avg = energy_hourly.groupby("building_id")["energy_kwh"].mean().reset_index(name="avg_energy")
    affected_df = affected_df.merge(building_avg, on="building_id", how="left")
    affected_df["energy_diff"] = affected_df["energy_kwh"] - affected_df["avg_energy"]
    affected_df["anomaly_type"] = np.where(
        affected_df["energy_diff"] < -50, "drop",
        np.where(affected_df["energy_diff"] > 50, "surge", "normal")
    )

    # 8. Summary by building type
    anomaly_summary = affected_df.groupby(["type", "anomaly_type"]).size().unstack(fill_value=0)
    anomaly_summary["total"] = anomaly_summary.sum(axis=1)
    anomaly_summary["drop_rate"] = anomaly_summary.get("drop", 0) / anomaly_summary["total"]
    anomaly_summary["surge_rate"] = anomaly_summary.get("surge", 0) / anomaly_summary["total"]
    anomaly_summary["normal_rate"] = anomaly_summary.get("normal", 0) / anomaly_summary["total"]

    return anomaly_summary, affected_df
