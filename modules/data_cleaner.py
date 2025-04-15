import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Step 1: Generate zone bounding boxes from disaster dataset
def generate_zone_bounding_boxes(disaster_df):
    disaster_df = disaster_df.dropna(subset=['latitude', 'longitude', 'location'])

    bbox_df = disaster_df.groupby("location").agg({
        "latitude": ["min", "max"],
        "longitude": ["min", "max"]
    }).reset_index()

    bbox_df.columns = ["Zone", "Lat_Min", "Lat_Max", "Lon_Min", "Lon_Max"]
    return bbox_df


# Step 2: Assign zone from bounding box
def assign_zone(lat, lon, bbox_df):
    for _, row in bbox_df.iterrows():
        if row['Lat_Min'] <= lat <= row['Lat_Max'] and row['Lon_Min'] <= lon <= row['Lon_Max']:
            return row['Zone']
    return "Unknown"


# Step 3: Apply to a dataframe (e.g. sensor or tweet)
def assign_zones_to_df(df, bbox_df):
    df = df.dropna(subset=["latitude", "longitude"])
    df["zone"] = df.apply(lambda row: assign_zone(row["latitude"], row["longitude"], bbox_df), axis=1)
    return df
import geopandas as gpd
import pandas as pd

# === Assign Zones Based on City Map (GeoSpatial Join) ===
def assign_zones_geospatial(df, zones_gdf):
    """
    Assigns each row (with lat/lon) to a zone from the GeoJSON map using spatial join.
    """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=zones_gdf.crs)
    joined = gpd.sjoin(gdf, zones_gdf, how="left", predicate="within")
    
    if 'Zone' in joined.columns:
        df["zone"] = joined["Zone"]
    elif 'name' in joined.columns:
        df["zone"] = joined["name"]
    elif 'id' in joined.columns:
        df["zone"] = joined["id"]
    else:
        df["zone"] = "Unknown"

    return df
import pandas as pd

# === Clean Sensor Data ===
def clean_sensor_data_inclusive(df):
    """
    Cleans sensor data by removing invalid timestamps, filtering by active status,
    and removing extreme outliers based on IQR method.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["status"] == "active"]

    Q1 = df["reading_value"].quantile(0.25)
    Q3 = df["reading_value"].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    df = df[(df["reading_value"] >= lower) & (df["reading_value"] <= upper)]
    return df
# === Clean Social Media Data ===
def clean_social_media_data(df):
    """
    Cleans social media tweets by parsing timestamps and removing duplicate texts.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.drop_duplicates(subset="text")
    df = df.dropna(subset=["latitude", "longitude"])
    return df


# === Geospatial Zone Assignment ===
def assign_zones_geospatial(df, zones_gdf):
    """Assigns city zones to each lat/lon point using a GeoDataFrame"""
    df = df.copy()
    df = df.dropna(subset=["latitude", "longitude"])

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    joined = gpd.sjoin(gdf, zones_gdf, how="left", predicate="within")
    df["zone"] = joined["name"]  # assumes 'name' column in geojson defines zone name like A/B/C/D
    return df

# === Sensor Data Cleaning ===
def clean_sensor_data_inclusive(df):
    df = df.copy()
    df = df[df["status"] == "active"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["reading_value", "latitude", "longitude"])
    
    Q1 = df["reading_value"].quantile(0.25)
    Q3 = df["reading_value"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df["reading_value"] >= lower) & (df["reading_value"] <= upper)]

    return df

# === Tweet Data Cleaning ===
def clean_social_media_data(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.drop_duplicates(subset="text")
    df = df.dropna(subset=["latitude", "longitude"])
    return df

# === Zone Summary ===
def generate_zone_summary(sensor_df, tweets_df, disaster_df=None):
    # Clean the inputs
    sensor_df = clean_sensor_data_inclusive(sensor_df)
    tweets_df = clean_social_media_data(tweets_df)

    # Sensor Summary
    sensor_summary = sensor_df.groupby("zone").agg({
        "reading_value": ["mean", "max", "count"],
        "sensor_type": lambda x: x.value_counts().index[0]
    }).reset_index()
    sensor_summary.columns = ["Zone", "Avg Reading", "Max Reading", "Sensor Count", "Top Sensor Type"]

    # Tweet Summary
    tweet_summary = tweets_df.groupby("zone").size().reset_index(name="Tweet Count")
    tweet_summary.rename(columns={"zone": "Zone"}, inplace=True)

    # Merge sensor + tweet summaries
    zone_data = pd.merge(sensor_summary, tweet_summary, on="Zone", how="outer")

    # Optional: Disaster Summary (if data available)
    if disaster_df is not None and "location" in disaster_df.columns:
        disaster_summary = disaster_df.groupby("location").size().reset_index(name="Disaster Events")
        disaster_summary.rename(columns={"location": "Zone"}, inplace=True)
        zone_data = pd.merge(zone_data, disaster_summary, on="Zone", how="left")

    # Fill missing columns with 0s where needed
    if "Tweet Count" not in zone_data.columns:
        zone_data["Tweet Count"] = 0
    else:
        zone_data["Tweet Count"] = zone_data["Tweet Count"].fillna(0)

    if "Disaster Events" in zone_data.columns:
        zone_data["Disaster Events"] = zone_data["Disaster Events"].fillna(0)
    else:
        zone_data["Disaster Events"] = 0

    return zone_data
