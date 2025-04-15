# data_loader.py

import pandas as pd
import json
import os

DATA_DIR = "data"

def load_disaster_events():
    return pd.read_csv(os.path.join(DATA_DIR, "disaster_events.csv"))

def load_sensor_readings():
    return pd.read_csv(os.path.join(DATA_DIR, "sensor_readings.csv"))

def load_social_media():
    return pd.read_csv(os.path.join(DATA_DIR, "social_media_stream.csv"))

def load_weather_data():
    return pd.read_csv(os.path.join(DATA_DIR, "weather_historical.csv"))

def load_city_map():
    with open(os.path.join(DATA_DIR, "city_map.geojson"), "r") as f:
        return json.load(f)

def load_energy_data():
    return pd.read_csv(os.path.join(DATA_DIR, "energy_consumption.csv"))

def load_transportation_data():
    return pd.read_csv(os.path.join(DATA_DIR, "transportation.csv"))

def load_events_calendar():
    return pd.read_csv(os.path.join(DATA_DIR, "events_calendar.csv"))

def load_economic_activity():
    return pd.read_csv(os.path.join(DATA_DIR, "economic_activity.csv"))

def load_business_reviews():
    return pd.read_csv(os.path.join(DATA_DIR, "local_business_reviews.csv"))
