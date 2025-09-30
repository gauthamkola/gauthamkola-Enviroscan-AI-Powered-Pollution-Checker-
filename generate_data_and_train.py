import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')
import requests
from datetime import datetime
import osmnx as ox

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# API keys
OPENAQ_API_KEY = “use your own api key“
OPENWEATHER_API_KEY = “use your own api key“  # Not used for historical weather, but kept

headers = {"X-API-Key": OPENAQ_API_KEY}

# Cities with lat/lon
cities = {
    'Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    'Chennai': (13.0827, 80.2707),
    'Kolkata': (22.5726, 88.3639),
    'Bangalore': (12.9716, 77.5946),
    'Hyderabad': (17.3850, 78.4867),
    'Pune': (18.5204, 73.8567),
    'Ahmedabad': (23.0225, 72.5714),
    'Jaipur': (26.9124, 75.7873),
    'Lucknow': (26.8467, 80.9462)
}

# Monthly average weather data (temp in °C, humidity in %, wind_speed in m/s)
monthly_weather = {
    'Delhi': {
        1: {'temp': 14.4, 'humidity': 80, 'wind_speed': 2.7},
        2: {'temp': 18.3, 'humidity': 69, 'wind_speed': 3.6},
        3: {'temp': 23.3, 'humidity': 59, 'wind_speed': 4.0},
        4: {'temp': 29.4, 'humidity': 44, 'wind_speed': 4.0},
        5: {'temp': 32.8, 'humidity': 42, 'wind_speed': 4.5},
        6: {'temp': 33.9, 'humidity': 54, 'wind_speed': 4.9},
        7: {'temp': 31.7, 'humidity': 74, 'wind_speed': 3.6},
        8: {'temp': 31.1, 'humidity': 76, 'wind_speed': 3.6},
        9: {'temp': 30.0, 'humidity': 70, 'wind_speed': 3.1},
        10: {'temp': 26.7, 'humidity': 63, 'wind_speed': 1.8},
        11: {'temp': 20.6, 'humidity': 67, 'wind_speed': 1.8},
        12: {'temp': 15.6, 'humidity': 74, 'wind_speed': 2.2},
    },
    'Mumbai': {
        1: {'temp': 24.4, 'humidity': 57, 'wind_speed': 3.6},
        2: {'temp': 25.0, 'humidity': 56, 'wind_speed': 3.6},
        3: {'temp': 27.2, 'humidity': 60, 'wind_speed': 4.0},
        4: {'temp': 28.9, 'humidity': 67, 'wind_speed': 4.5},
        5: {'temp': 30.6, 'humidity': 70, 'wind_speed': 5.4},
        6: {'temp': 29.4, 'humidity': 77, 'wind_speed': 6.3},
        7: {'temp': 27.8, 'humidity': 83, 'wind_speed': 7.2},
        8: {'temp': 27.8, 'humidity': 82, 'wind_speed': 6.7},
        9: {'temp': 27.8, 'humidity': 81, 'wind_speed': 4.5},
        10: {'temp': 28.9, 'humidity': 69, 'wind_speed': 3.6},
        11: {'temp': 27.8, 'humidity': 58, 'wind_speed': 3.6},
        12: {'temp': 25.6, 'humidity': 57, 'wind_speed': 3.1},
    },
    'Chennai': {
        1: {'temp': 25.6, 'humidity': 74, 'wind_speed': 4.0},
        2: {'temp': 26.7, 'humidity': 70, 'wind_speed': 4.5},
        3: {'temp': 29.4, 'humidity': 70, 'wind_speed': 5.4},
        4: {'temp': 31.7, 'humidity': 69, 'wind_speed': 6.3},
        5: {'temp': 33.3, 'humidity': 64, 'wind_speed': 6.7},
        6: {'temp': 32.8, 'humidity': 60, 'wind_speed': 7.2},
        7: {'temp': 31.1, 'humidity': 65, 'wind_speed': 6.7},
        8: {'temp': 30.6, 'humidity': 69, 'wind_speed': 6.3},
        9: {'temp': 30.0, 'humidity': 73, 'wind_speed': 5.4},
        10: {'temp': 28.9, 'humidity': 77, 'wind_speed': 4.5},
        11: {'temp': 27.2, 'humidity': 81, 'wind_speed': 4.5},
        12: {'temp': 26.1, 'humidity': 78, 'wind_speed': 4.5},
    },
    'Kolkata': {
        1: {'temp': 18.9, 'humidity': 71, 'wind_speed': 1.8},
        2: {'temp': 22.8, 'humidity': 66, 'wind_speed': 1.8},
        3: {'temp': 27.8, 'humidity': 64, 'wind_speed': 2.7},
        4: {'temp': 30.6, 'humidity': 69, 'wind_speed': 4.9},
        5: {'temp': 31.1, 'humidity': 74, 'wind_speed': 4.9},
        6: {'temp': 30.6, 'humidity': 80, 'wind_speed': 4.0},
        7: {'temp': 30.0, 'humidity': 84, 'wind_speed': 3.6},
        8: {'temp': 30.0, 'humidity': 85, 'wind_speed': 3.1},
        9: {'temp': 29.4, 'humidity': 84, 'wind_speed': 2.7},
        10: {'temp': 28.3, 'humidity': 79, 'wind_speed': 1.8},
        11: {'temp': 24.4, 'humidity': 72, 'wind_speed': 1.3},
        12: {'temp': 20.6, 'humidity': 73, 'wind_speed': 1.8},
    },
    'Bangalore': {
        1: {'temp': 22.8, 'humidity': 58, 'wind_speed': 1.8},
        2: {'temp': 24.4, 'humidity': 48, 'wind_speed': 1.8},
        3: {'temp': 27.2, 'humidity': 44, 'wind_speed': 1.8},
        4: {'temp': 28.9, 'humidity': 52, 'wind_speed': 1.3},
        5: {'temp': 27.8, 'humidity': 65, 'wind_speed': 2.2},
        6: {'temp': 25.6, 'humidity': 74, 'wind_speed': 4.0},
        7: {'temp': 24.4, 'humidity': 78, 'wind_speed': 4.0},
        8: {'temp': 24.4, 'humidity': 79, 'wind_speed': 3.6},
        9: {'temp': 24.4, 'humidity': 76, 'wind_speed': 2.7},
        10: {'temp': 24.4, 'humidity': 73, 'wind_speed': 1.8},
        11: {'temp': 23.3, 'humidity': 72, 'wind_speed': 1.8},
        12: {'temp': 22.8, 'humidity': 68, 'wind_speed': 1.8},
    },
    'Hyderabad': {
        1: {'temp': 23.3, 'humidity': 53, 'wind_speed': 2.7},
        2: {'temp': 25.6, 'humidity': 45, 'wind_speed': 3.1},
        3: {'temp': 29.4, 'humidity': 39, 'wind_speed': 3.1},
        4: {'temp': 32.2, 'humidity': 39, 'wind_speed': 3.1},
        5: {'temp': 33.9, 'humidity': 39, 'wind_speed': 4.0},
        6: {'temp': 30.0, 'humidity': 61, 'wind_speed': 6.3},
        7: {'temp': 27.2, 'humidity': 73, 'wind_speed': 6.7},
        8: {'temp': 27.2, 'humidity': 75, 'wind_speed': 5.8},
        9: {'temp': 27.2, 'humidity': 76, 'wind_speed': 4.0},
        10: {'temp': 26.7, 'humidity': 67, 'wind_speed': 3.1},
        11: {'temp': 24.4, 'humidity': 60, 'wind_speed': 2.7},
        12: {'temp': 22.8, 'humidity': 58, 'wind_speed': 2.2},
    },
    'Pune': {
        1: {'temp': 20.6, 'humidity': 60, 'wind_speed': 0.4},
        2: {'temp': 22.8, 'humidity': 49, 'wind_speed': 0.9},
        3: {'temp': 25.6, 'humidity': 41, 'wind_speed': 1.3},
        4: {'temp': 28.9, 'humidity': 37, 'wind_speed': 1.8},
        5: {'temp': 30.0, 'humidity': 49, 'wind_speed': 2.7},
        6: {'temp': 27.2, 'humidity': 72, 'wind_speed': 3.1},
        7: {'temp': 25.0, 'humidity': 82, 'wind_speed': 2.7},
        8: {'temp': 24.4, 'humidity': 84, 'wind_speed': 2.7},
        9: {'temp': 25.0, 'humidity': 81, 'wind_speed': 1.8},
        10: {'temp': 25.6, 'humidity': 73, 'wind_speed': 0.9},
        11: {'temp': 22.8, 'humidity': 66, 'wind_speed': 0.4},
        12: {'temp': 21.1, 'humidity': 66, 'wind_speed': 0.4},
    },
    'Ahmedabad': {
        1: {'temp': 20.0, 'humidity': 50, 'wind_speed': 3.6},
        2: {'temp': 23.3, 'humidity': 41, 'wind_speed': 3.6},
        3: {'temp': 27.8, 'humidity': 33, 'wind_speed': 3.6},
        4: {'temp': 32.8, 'humidity': 34, 'wind_speed': 4.0},
        5: {'temp': 35.6, 'humidity': 39, 'wind_speed': 5.8},
        6: {'temp': 33.3, 'humidity': 59, 'wind_speed': 6.3},
        7: {'temp': 30.0, 'humidity': 75, 'wind_speed': 5.8},
        8: {'temp': 28.9, 'humidity': 80, 'wind_speed': 5.4},
        9: {'temp': 29.4, 'humidity': 77, 'wind_speed': 3.6},
        10: {'temp': 28.9, 'humidity': 58, 'wind_speed': 2.2},
        11: {'temp': 25.0, 'humidity': 51, 'wind_speed': 2.7},
        12: {'temp': 21.1, 'humidity': 51, 'wind_speed': 3.6},
    },
    'Jaipur': {
        1: {'temp': 15.6, 'humidity': 55, 'wind_speed': 3.1},
        2: {'temp': 20.0, 'humidity': 42, 'wind_speed': 3.6},
        3: {'temp': 25.0, 'humidity': 33, 'wind_speed': 4.5},
        4: {'temp': 31.1, 'humidity': 25, 'wind_speed': 4.5},
        5: {'temp': 34.4, 'humidity': 26, 'wind_speed': 5.8},
        6: {'temp': 34.4, 'humidity': 43, 'wind_speed': 6.3},
        7: {'temp': 31.1, 'humidity': 70, 'wind_speed': 5.4},
        8: {'temp': 29.4, 'humidity': 79, 'wind_speed': 4.9},
        9: {'temp': 29.4, 'humidity': 66, 'wind_speed': 4.0},
        10: {'temp': 27.8, 'humidity': 40, 'wind_speed': 2.7},
        11: {'temp': 22.2, 'humidity': 41, 'wind_speed': 2.2},
        12: {'temp': 17.8, 'humidity': 48, 'wind_speed': 2.2},
    },
    'Lucknow': {
        1: {'temp': 14.4, 'humidity': 78, 'wind_speed': 2.7},
        2: {'temp': 18.9, 'humidity': 67, 'wind_speed': 2.7},
        3: {'temp': 23.9, 'humidity': 54, 'wind_speed': 3.6},
        4: {'temp': 29.4, 'humidity': 40, 'wind_speed': 4.0},
        5: {'temp': 32.8, 'humidity': 46, 'wind_speed': 4.0},
        6: {'temp': 33.3, 'humidity': 60, 'wind_speed': 4.5},
        7: {'temp': 30.6, 'humidity': 83, 'wind_speed': 3.6},
        8: {'temp': 30.0, 'humidity': 85, 'wind_speed': 3.1},
        9: {'temp': 29.4, 'humidity': 80, 'wind_speed': 3.1},
        10: {'temp': 26.7, 'humidity': 70, 'wind_speed': 1.8},
        11: {'temp': 21.1, 'humidity': 66, 'wind_speed': 1.3},
        12: {'temp': 16.1, 'humidity': 75, 'wind_speed': 1.8},
    },
}

# Fetch locations with all parameters
print("Fetching locations from OpenAQ...")
params_list = "pm25,pm10,no2,so2,o3,co"
url = f"https://api.openaq.org/v3/locations?country=IN&parameters={params_list}&limit=20"
response = requests.get(url, headers=headers)
if response.status_code == 200:
    locations = response.json()['results']
    print(f"Fetched {len(locations)} locations")
else:
    print(f"Failed to fetch locations: {response.status_code}")
    locations = []

# Collect data
data = []
for i, loc in enumerate(locations):
    print(f"Processing location {i+1}/{len(locations)}: {loc['name']}")
    city = loc.get('city') or loc['name']
    lat = loc['coordinates']['latitude']
    lon = loc['coordinates']['longitude']
    closest_city = min(cities.keys(), key=lambda c: (cities[c][0] - lat)*2 + (cities[c][1] - lon)*2)

    # Get sensors dict
    sensors_dict = {s['parameter']['name']: s['id'] for s in loc['sensors'] if s['parameter']['name'] in params_list.split(',')}

    # Fetch measurements for each sensor
    meas_dfs = []
    for param, sensor_id in sensors_dict.items():
        meas_url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements/hourly?date_from=2024-01-01T00:00:00Z&date_to=2024-02-01T00:00:00Z&limit=100"
        meas_response = requests.get(meas_url, headers=headers)
        if meas_response.status_code == 200:
            meas = meas_response.json()['results']
            if meas:
                param_df = pd.DataFrame(meas)
                param_df['parameter'] = param
                param_df['value'] = param_df['value']
                param_df['datetime'] = pd.to_datetime(param_df['period'].apply(lambda x: x['datetimeFrom']['utc']))
                param_df = param_df[['datetime', 'parameter', 'value']]
                meas_dfs.append(param_df)

    if meas_dfs:
        temp_df = pd.concat(meas_dfs)
        temp_df = temp_df.pivot(index='datetime', columns='parameter', values='value').reset_index()
        temp_df['city'] = closest_city 
        temp_df['latitude'] = lat
        temp_df['longitude'] = lon
        temp_df['month'] = temp_df['datetime'].dt.month
        temp_df['temperature'] = temp_df['month'].apply(lambda m: monthly_weather[closest_city][m]['temp'])
        temp_df['humidity'] = temp_df['month'].apply(lambda m: monthly_weather[closest_city][m]['humidity'])
        temp_df['wind_speed'] = temp_df['month'].apply(lambda m: monthly_weather[closest_city][m]['wind_speed'])
        temp_df['season'] = temp_df['month'].apply(lambda m: "Dry" if m in [1,2,3,11,12] else "Wet")
        for p in ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']:
            if p in temp_df.columns:
                temp_df[p] = temp_df[p].fillna(0)
            else:
                temp_df[p] = 0
        temp_df = temp_df.rename(columns={'pm25': 'pm2_5'})
        temp_df['aqi'] = np.clip((temp_df['pm2_5'] / 50).round().astype(int), 1, 5)
        temp_df['pollution_source'] = temp_df.apply(lambda row: "Vehicular" if row['no2'] > 70 and row['wind_speed'] < 5 else "Industrial" if row['so2'] > 40 and row['pm10'] > 100 else "Agricultural" if row['pm10'] > 90 and row['season'] == "Dry" else "Burning" if row['co'] > 2 and row['aqi'] >= 4 else "Natural", axis=1)
        temp_df['temp_humidity_index'] = temp_df['temperature'] * temp_df['humidity'] / 100
        temp_df['pollution_wind_ratio'] = (temp_df['pm2_5'] + temp_df['pm10'] + temp_df['no2'] + temp_df['so2'] + temp_df['o3'] + temp_df['co']) / (temp_df['wind_speed'] + 0.1)
        data.append(temp_df)

if data:
    df = pd.concat(data, ignore_index=True)
else:
    df = pd.DataFrame()

# Sample to 500
if len(df) > 500:
    df = df.sample(500, random_state=42)
df['location_id'] = range(1, len(df) + 1)
df.to_csv("data/labeled_data.csv", index=False)
print(f"✅ Generated {len(df)} samples using real data from OpenAQ")

# Module 2: Data Cleaning & Feature Engineering
print("Cleaning and engineering features...")
df = df.fillna(0)  # Fill missing
# OSMnx features
unique_locs = df[['latitude', 'longitude']].drop_duplicates()

def get_osm_features(lat, lon):
    point = (lat, lon)
    try:
        tags = {'landuse': 'industrial'}
        gdf = ox.geometries_from_point(point, tags, dist=2000)
        industrial_count = len(gdf)
    except:
        industrial_count = 0
    try:
        G = ox.graph_from_point(point, dist=2000, network_type='drive')
        road_length = sum(data.get('length', 0) for u, v, data in G.edges(data=True))
    except:
        road_length = 0
    return industrial_count, road_length

features = unique_locs.apply(lambda row: get_osm_features(row['latitude'], row['longitude']), axis=1).tolist()
unique_locs['industrial_count'], unique_locs['road_length'] = zip(*features) if features else (0, 0)
df = df.merge(unique_locs, on=['latitude', 'longitude'])

# Module 3: Source Labeling (already done, but can refine with new features)
df['pollution_source'] = df.apply(lambda row: "Industrial" if row['industrial_count'] > 0 and row['so2'] > 30 else "Vehicular" if row['road_length'] > 5000 and row['no2'] > 50 else row['pollution_source'], axis=1)  # Example refine

# Module 4: Model Training
print("Training models...")
features = ["pm2_5", "pm10", "no2", "so2", "o3", "co", "temperature", "humidity", "wind_speed", "latitude", "longitude", "industrial_count", "road_length"]
X = df[features]
y = df["pollution_source"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "DecisionTree": DecisionTreeClassifier(max_depth=15, min_samples_split=5, random_state=42)
}

best_model = None
best_accuracy = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name} accuracy: {acc:.3f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save best model
joblib.dump(best_model, "models/model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(le, "models/label_encoder.joblib")

print(f"✅ Best model trained with accuracy: {best_accuracy:.3f}")
print("✅ All files prepared successfully!")
