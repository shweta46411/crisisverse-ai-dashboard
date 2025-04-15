import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import st_folium
import folium
import json
import openai
import streamlit as st

from data_loader import (
    load_sensor_readings,
    load_disaster_events,
    load_city_map
)
from utils.fake_news_utils import detect_fake_news, extract_sensor_disasters

from utils.zone_mapper import assign_zones_to_sensors_knn
from utils.anomaly_detector import detect_zscore_anomalies
from utils.zone_features import generate_zone_sensor_features
import matplotlib as mpl


# Streamlit settings
st.set_page_config(page_title="Crisisverse AI", layout="wide")

mpl.rcParams.update({
    "axes.titlesize": 9,
    "axes.labelsize": 6,
    "xtick.labelsize": 4,
    "ytick.labelsize": 4,
    "legend.fontsize": 4
})
st.markdown("""
    <style>
    /* Fix header cutoff issue */
    .main > div:first-child {
        padding-top: 2rem;
    }
    /* Prevent emoji + text overlap */
    h1 {
        line-height: 1.5 !important;
        font-size: 2.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stPlotlyChart, .stImage, .stAltairChart, .stVegaLiteChart, .stPyplot {
            max-width: 80% !important;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)
st.title("🧠 CrisisMind AI Dashboard")
st.markdown("The brain of a smart city during a multi-disaster scenario.")
# Sidebar Navigation
selected_tab = st.sidebar.radio("📂 Choose a View", [
    "📍 Risk Zones",
    "📊 Zone Intelligence",
    # "📈 Crisis Timeline",
    "📌 Disaster Explorer",
    "🌍 Disaster Event Map",
    "📰 Fake News Detection",
    # "⚡ Energy Impact",
    # "🚦 Transport Delays",
    # "✅ Recommendations"

])

# --------------------------------
# 📍 RISK ZONES TAB
# --------------------------------
if selected_tab == "📍 Risk Zones":
    st.header("📍 Risk Zones – Sensor & Disaster Overview")

    sensor_df = load_sensor_readings()
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    disaster_df = load_disaster_events()
    disaster_df['date'] = pd.to_datetime(disaster_df['date'])

    st.markdown("### 🛰️ Sensor Network Overview")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=sensor_df, x='sensor_type', order=sensor_df['sensor_type'].value_counts().index, palette='viridis', ax=ax1)
        ax1.set_title("Sensor Count by Type")
        ax1.set_xlabel("Sensor Type")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

    with col2:
        status_counts = sensor_df['status'].value_counts()
        labels = [f"{label} ({val})" for label, val in zip(status_counts.index, status_counts.values)]
        sizes = status_counts.values
        colors = sns.color_palette('pastel')[:len(labels)]
        explode = [0.04] * len(labels)

        fig2, ax2 = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode,
            shadow=False, colors=colors, textprops={'fontsize': 9}
        )
        centre_circle = plt.Circle((0, 0), 0.65, fc='white')
        fig2.gca().add_artist(centre_circle)
        ax2.set_title("Sensor Status", fontsize=12, weight='bold')
        st.pyplot(fig2)

    st.markdown("### 🧮 Sensor Health Summary")
    with st.expander("See Stacked Bar Chart of Sensor Type vs Status"):
        pivot = pd.crosstab(sensor_df['sensor_type'], sensor_df['status'])
        fig3, ax3 = plt.subplots(figsize=(4, 2))
        pivot.plot(kind='bar', stacked=True, colormap='Set2', ax=ax3)
        ax3.set_title("Sensor Type vs Status Distribution")
        ax3.set_xlabel("Sensor Type")
        ax3.set_ylabel("Count")
        ax3.tick_params(axis='x', rotation=45)
        st.pyplot(fig3)

    st.markdown("---")
    st.markdown("### 🌊 Disaster Event Insights")

    col3, col4 = st.columns(2)

    with col3:
        fig4, ax4 = plt.subplots(figsize=(4, 2))
        sns.countplot(data=disaster_df, x='disaster_type', order=disaster_df['disaster_type'].value_counts().index, palette='flare', ax=ax4)
        ax4.set_title("Disaster Count by Type")
        ax4.set_xlabel("Disaster Type")
        ax4.set_ylabel("Count")
        ax4.tick_params(axis='x', rotation=45)
        st.pyplot(fig4)

    with col4:
        fig5, ax5 = plt.subplots(figsize=(4, 2))
        sns.countplot(data=disaster_df, x='location', order=disaster_df['location'].value_counts().index, palette='crest', ax=ax5)
        ax5.set_title("Disasters by Zone")
        ax5.set_xlabel("Zone")
        ax5.set_ylabel("Count")
        st.pyplot(fig5)

    st.markdown("### ⏳ Temporal & Heatmap Patterns")
    with st.expander("📅 Temporal Distribution & Heatmap"):
        fig6, ax6 = plt.subplots(figsize=(7, 1))
        sns.histplot(disaster_df['date'], bins=60, kde=True, color='darkred', ax=ax6)
        ax6.set_title("Disaster Events Over Time")
        ax6.set_xlabel("Date")
        ax6.set_ylabel("Count")
        st.pyplot(fig6)

        type_zone = pd.crosstab(disaster_df['disaster_type'], disaster_df['location'])
        fig7, ax7 = plt.subplots(figsize=(4, 2))
        sns.heatmap(type_zone, annot=True, cmap='YlOrBr', fmt='d', ax=ax7)
        ax7.set_title("Disaster Type vs Zone")
        st.pyplot(fig7)

    st.markdown("### 🧠 Key Insights")
    
elif selected_tab == "📊 Zone Intelligence":
    st.header("📊 Zone Intelligence")

    # Load and prepare data
    sensor_df = load_sensor_readings()
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    disaster_df = load_disaster_events()
    disaster_df['date'] = pd.to_datetime(disaster_df['date'])

    # Assign zones via KNN and detect anomalies
    sensor_df = assign_zones_to_sensors_knn(sensor_df, disaster_df)
    sensor_df = detect_zscore_anomalies(sensor_df)

    # Generate zone-level features
    zone_feature_df = generate_zone_sensor_features(sensor_df)
    zone_feature_df["name"] = zone_feature_df["zone_id"]

    # Classify zones based on anomaly count
    zone_feature_df["risk_level"] = pd.cut(
        zone_feature_df["anomaly_count"],
        bins=[-1, 50, 100, 150, float("inf")],
        labels=["Low", "Moderate", "High", "Critical"]
    )

    # Show top risky zones
    st.markdown("### 🚨 Top Risky Zones")
    st.dataframe(zone_feature_df.sort_values("anomaly_count", ascending=False).head(5))

    # Load GeoJSON
    with open("data/city_map.geojson", "r") as f:
        city_map = json.load(f)

    # st.markdown("### 🗺️ Zone Intelligence Map")

    # # Create map
    # m = folium.Map(location=[37.77, -122.42], zoom_start=12)

    # # Choropleth by anomaly count
    # folium.Choropleth(
    #     geo_data=city_map,
    #     name="Anomaly Map",
    #     data=zone_feature_df,
    #     columns=["name", "anomaly_count"],
    #     key_on="feature.properties.name",
    #     fill_color="YlOrRd",
    #     fill_opacity=0.7,
    #     line_opacity=0.2,
    #     legend_name="Anomaly Count by Zone",
    #     highlight=True,
    # ).add_to(m)

    # # Tooltip for each zone
    # folium.GeoJson(
    #     data=city_map,
    #     name="Interactive Zones",
    #     tooltip=folium.GeoJsonTooltip(
    #         fields=["name"],
    #         aliases=["Zone"],
    #         sticky=True,
    #         labels=True,
    #         style="background-color: white; font-size: 12px; padding: 5px;"
    #     ),
    # ).add_to(m)

    # st_folium(m, width=950, height=550)
elif selected_tab == "📈 Crisis Timeline":
    st.header("📈 Crisis Timeline")
    st.info("This section will show cascading disasters and sensor alerts over time.")
elif selected_tab == "📌 Disaster Explorer":
    st.header("📌 Disaster Explorer – Interactive EDA Dashboard")

    # Load and prepare data
    disaster_df = load_disaster_events()
    disaster_df['date'] = pd.to_datetime(disaster_df['date'])
    disaster_df['year'] = disaster_df['date'].dt.year
    disaster_df['month'] = disaster_df['date'].dt.month

    # 🎛️ Filters (inside main page)
    st.subheader("🎛️ Filter Disasters")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_year = st.multiselect("Select Year(s)", sorted(disaster_df['year'].unique()), default=sorted(disaster_df['year'].unique()))
    with col2:
        selected_type = st.multiselect("Select Disaster Type(s)", disaster_df['disaster_type'].unique(), default=disaster_df['disaster_type'].unique())
    with col3:
        selected_zone = st.multiselect("Select Zone(s)", disaster_df['location'].unique(), default=disaster_df['location'].unique())

    # Filter application
    filtered_df = disaster_df[
        (disaster_df['year'].isin(selected_year)) &
        (disaster_df['disaster_type'].isin(selected_type)) &
        (disaster_df['location'].isin(selected_zone))
    ]

    # Handle empty results
    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filters. Please update your selections to view the dashboard.")
    else:
        # 📊 Overview KPIs
        st.subheader("📊 Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", len(filtered_df))
        col2.metric("Total Casualties", int(filtered_df['casualties'].sum()))
        col3.metric("Economic Loss (M)", f"${int(filtered_df['economic_loss_million_usd'].sum()):,}")

        st.divider()

        # 📅 Events per Year
        st.subheader("📅 Yearly Event Count")
        fig1, ax1 = plt.subplots(figsize=(4, 2))
        filtered_df.groupby("year").size().plot(kind="bar", ax=ax1, color='skyblue')
        ax1.set_ylabel("Event Count", fontsize=9)
        ax1.set_title("Events per Year", fontsize=11)
        ax1.tick_params(axis='x', labelsize=8)
        st.pyplot(fig1)
        st.divider()

        # 📈 Monthly Trends
        st.subheader("📈 Monthly Trends by Disaster Type")
        monthly = filtered_df.groupby(['month', 'disaster_type']).size().unstack().fillna(0)
        fig2, ax2 = plt.subplots(figsize=(5, 1))
        monthly.plot(ax=ax2)
        ax2.set_title("Seasonal Trends", fontsize=11)
        ax2.set_xlabel("Month", fontsize=9)
        ax2.set_ylabel("Count", fontsize=9)
        ax2.legend(fontsize=7)
        st.pyplot(fig2)
        st.divider()

        # 🌪️ Disaster Type Pie
        st.subheader("🌪️ Disaster Type Distribution")
        fig3, ax3 = plt.subplots(figsize=(2, 2))
        filtered_df['disaster_type'].value_counts().plot.pie(
            autopct='%1.1f%%', startangle=90, shadow=False, ax=ax3
        )
        ax3.set_ylabel("")
        ax3.set_title("Share by Type", fontsize=11)
        st.pyplot(fig3)
        st.divider()

        # 📍 Events by Zone
        st.subheader("📍 Disaster Events by Zone")
        fig4, ax4 = plt.subplots(figsize=(4, 2))
        sns.countplot(data=filtered_df, x='location', palette='Set2', ax=ax4)
        ax4.set_title("Disasters by Zone", fontsize=11)
        ax4.tick_params(axis='x', labelsize=9)
        st.pyplot(fig4)
        st.divider()

        # 💥 Severity vs Casualties
        with st.expander("💥 Severity vs Casualties"):
            fig5, ax5 = plt.subplots(figsize=(7, 3))
            sns.boxplot(data=filtered_df, x="severity", y="casualties", ax=ax5)
            ax5.set_title("Casualties Across Severity Levels", fontsize=11)
            st.pyplot(fig5)
        st.divider()

        # 💸 Economic Loss by Type
        with st.expander("💸 Economic Loss by Disaster Type"):
            econ = filtered_df.groupby("disaster_type")["economic_loss_million_usd"].sum().sort_values(ascending=False)
            fig6, ax6 = plt.subplots(figsize=(4, 2))
            econ.plot(kind="bar", ax=ax6, color='orange')
            ax6.set_ylabel("Total Loss (M USD)", fontsize=9)
            ax6.set_title("Economic Loss by Disaster Type", fontsize=11)
            st.pyplot(fig6)
        st.divider()

        # 📊 Avg Loss Table
        st.subheader("📊 Avg Economic Loss per Event")
        avg_loss = filtered_df.groupby("disaster_type")["economic_loss_million_usd"].mean().round(2)
        st.dataframe(avg_loss.reset_index().rename(columns={"economic_loss_million_usd": "Avg Loss (M)"}))
elif selected_tab == "🌍 Disaster Event Map":
    st.header("🌍 Disaster Risk Map (Color-Coded by Severity)")

    # Load and validate disaster data
    disaster_df = load_disaster_events()
    disaster_df['date'] = pd.to_datetime(disaster_df['date'], errors='coerce')
    disaster_df = disaster_df.dropna(subset=['latitude', 'longitude'])

    if disaster_df.empty:
        st.warning("⚠️ Disaster dataset appears to be empty.")
    else:
        disaster_df['year'] = disaster_df['date'].dt.year
        disaster_df['severity'] = disaster_df['severity'].fillna(5).astype(int)

        # Sidebar Filters
        selected_year = st.selectbox("Select Year", sorted(disaster_df['year'].dropna().unique(), reverse=True))
        selected_type = st.selectbox("Select Disaster Type", sorted(disaster_df['disaster_type'].dropna().unique()))

        filtered = disaster_df[
            (disaster_df['year'] == selected_year) &
            (disaster_df['disaster_type'] == selected_type)
        ]

        if filtered.empty:
            st.warning("⚠️ No records found for selected year and disaster type.")
        else:
            # Severity color palette
            severity_color = {
                1: "green", 2: "green", 3: "green",
                4: "orange", 5: "orange", 6: "orange",
                7: "red", 8: "red", 9: "red"
            }

            m = folium.Map(
                location=[filtered['latitude'].mean(), filtered['longitude'].mean()],
                zoom_start=11
            )

            for _, row in filtered.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=6,
                    color=severity_color.get(int(row['severity']), "gray"),
                    fill=True,
                    fill_opacity=0.8,
                    popup=folium.Popup(
                        f"<b>{row['disaster_type']}</b><br>"
                        f"Zone: {row['location']}<br>"
                        f"Date: {row['date'].date()}<br>"
                        f"Severity: {row['severity']}",
                        max_width=250
                    )
                ).add_to(m)

            st.markdown("🟢 Green = Low | 🟠 Medium | 🔴 High Severity")
            st_folium(m, width=950, height=550)
elif selected_tab == "📰 Fake News Detection":
    st.header("📰 Early Warnings & Misinformation Detection")

    # Load data
    sensor_df = pd.read_csv("data/essential_data/sensor_readings.csv")
    disaster_df = pd.read_csv("data/disaster_events.csv")
    social_df = pd.read_csv("data/essential_data/social_media_stream.csv")

    # Ensure datetime parsing
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    disaster_df['date'] = pd.to_datetime(disaster_df['date'])
    social_df['timestamp'] = pd.to_datetime(social_df['timestamp'])

    # Extract disaster events from sensors
    from utils.fake_news_utils import detect_fake_news, extract_sensor_disasters
    sensor_disasters = extract_sensor_disasters(sensor_df)

    # Merge historical and sensor disasters
    combined_events = pd.concat([disaster_df, sensor_disasters], ignore_index=True)

    # Run fake news detection
    result_df = detect_fake_news(social_df, combined_events)

    # Toggle filters
    st.markdown("### 🕵️ Tweet Classification")
    filter_option = st.radio("Choose tweets to view:", ["All", "Verified", "Potential Fake"])

    if filter_option == "Verified":
        display_df = result_df[result_df["is_verified_event"]]
    elif filter_option == "Potential Fake":
        display_df = result_df[result_df["is_potential_fake"]]
    else:
        display_df = result_df

    st.markdown(f"### 📢 Showing {len(display_df)} tweets")
    st.dataframe(display_df[["timestamp", "text", "latitude", "longitude", "detected_disaster_type", "is_verified_event"]].sort_values("timestamp", ascending=False), use_container_width=True)

    st.info("""
    • ✅ Verified = Tweet matched to a real disaster (location & time window)
    • ⚠️ Potential Fake = No match to real disaster event
    • Uses both sensor and reported event datasets for validation
    """)
