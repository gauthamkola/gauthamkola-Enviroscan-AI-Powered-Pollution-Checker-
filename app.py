import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import joblib
from datetime import datetime

# Page config
st.set_page_config(layout="wide", page_title="🌍 Pollution Source Prediction")

# Title
st.title("🌍 Pollution Source Prediction System")
st.markdown("Real-time pollution monitoring and source identification using Machine Learning")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/labeled_data.csv")
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    return df
df = load_data()

# Sidebar
st.sidebar.header("🔧 Control Panel")

# Filters
sources = df["pollution_source"].unique().tolist()
selected_sources = st.sidebar.multiselect(
    "Pollution Sources",
    options=sources,
    default=sources
)

cities = df["city"].unique().tolist()
selected_cities = st.sidebar.multiselect(
    "Cities",
    options=cities,
    default=cities[:5]
)

pollutants = ["pm2_5", "pm10", "no2", "so2", "o3", "co", "aqi"]
selected_pollutant = st.sidebar.selectbox("Primary Pollutant", pollutants)

# Filter data
df_filtered = df[df["pollution_source"].isin(selected_sources) & df["city"].isin(selected_cities)]

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Locations", len(df_filtered))
with col2:
    st.metric("Avg AQI", f"{df_filtered['aqi'].mean():.1f}")
with col3:
    high_risk = len(df_filtered[df_filtered["aqi"] > 3])
    st.metric("High Risk Areas", high_risk)
with col4:
    st.metric("Dominant Source", df_filtered["pollution_source"].mode()[0] if len(df_filtered) > 0 else "N/A")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🗺 Map", "📊 Analytics", "⚠ Alerts", "🤖 Predict"])

with tab1:
    st.subheader("Pollution Sources Map")
    
    # Create map
    m = folium.Map(
        location=[df_filtered["latitude"].mean(), df_filtered["longitude"].mean()],
        zoom_start=5
    )
    
    # Colors
    colors = {
        "Vehicular": "red",
        "Industrial": "purple", 
        "Agricultural": "orange",
        "Burning": "darkred",
        "Natural": "green"
    }
    
    # Add markers
    for _, row in df_filtered.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            popup=f"{row['city']}<br>Source: {row['pollution_source']}<br>AQI: {row['aqi']}",
            color=colors.get(row["pollution_source"], "gray"),
            fill=True,
            fillOpacity=0.7
        ).add_to(m)
    
    st_folium(m, height=500)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Source distribution
        fig_pie = px.pie(
            values=df_filtered["pollution_source"].value_counts().values,
            names=df_filtered["pollution_source"].value_counts().index,
            title="Source Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Pollutant levels
        pollutant_cols = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]
        avg_by_source = df_filtered.groupby("pollution_source")[pollutant_cols].mean()
        
        fig_bar = px.bar(
            avg_by_source.T,
            title="Avg Pollutants by Source"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.subheader("⚠ Pollution Alerts")
    
    # High risk locations
    high_risk_df = df_filtered[df_filtered["aqi"] > 3]
    
    if len(high_risk_df) > 0:
        st.error(f"🚨 {len(high_risk_df)} locations exceed safe thresholds!")
        st.dataframe(high_risk_df[["city", "pollution_source", "aqi", "pm2_5", "pm10"]])
    else:
        st.success("✅ All locations within safe levels")

with tab4:
    st.subheader("🤖 Predict Pollution Source")
    
    # Load model
    model = joblib.load("models/model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    le = joblib.load("models/label_encoder.joblib")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pm25 = st.number_input("PM2.5", value=50.0)
        pm10 = st.number_input("PM10", value=80.0)
        no2 = st.number_input("NO2", value=40.0)
        so2 = st.number_input("SO2", value=20.0)
    
    with col2:
        o3 = st.number_input("O3", value=35.0)
        co = st.number_input("CO", value=1.5)
        temp = st.number_input("Temperature", value=28.0)
        humidity = st.number_input("Humidity", value=65.0)
    
    with col3:
        wind = st.number_input("Wind Speed", value=3.5)
        lat = st.number_input("Latitude", value=20.59)
        lon = st.number_input("Longitude", value=78.96)
    
    # For new features, set default 0
    industrial_count = 0
    road_length = 0
    
    if st.button("Predict Source", type="primary"):
        # Predict
        input_data = [[pm25, pm10, no2, so2, o3, co, temp, humidity, wind, lat, lon, industrial_count, road_length]]
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        source = le.inverse_transform([prediction])[0]
        
        st.success(f"### Predicted Source: *{source}*")
        
        # Get probabilities
        proba = model.predict_proba(input_scaled)[0]
        
        # Show confidence
        fig = px.bar(
            x=le.classes_,
            y=proba * 100,
            labels={'x': 'Source', 'y': 'Confidence (%)'},
            title="Prediction Confidence"
        )
        st.plotly_chart(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
