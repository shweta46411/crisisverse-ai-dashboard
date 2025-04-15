# utils/zone_mapper.py

from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

# utils/zone_mapper.py

from sklearn.neighbors import KNeighborsClassifier

def assign_zones_to_sensors_knn(sensor_df, disaster_df):
    # Use labeled disaster zones for training
    zone_train = disaster_df[['latitude', 'longitude', 'location']].dropna()
    zone_train = zone_train[zone_train['location'].str.contains('Zone')]

    # Train KNN on known zone coordinates
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(zone_train[['latitude', 'longitude']], zone_train['location'])

    # Predict zone for each sensor
    sensor_df['zone_id'] = knn.predict(sensor_df[['latitude', 'longitude']])
    return sensor_df

