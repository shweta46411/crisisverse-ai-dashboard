import openai
import streamlit as st  # <-- to access secrets

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def classify_tweet(tweet):
    prompt = f"Classify the following tweet as 'Real' or 'Fake' and explain in 1 line:\n\nTweet: {tweet}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a misinformation expert for city emergencies."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.3
    )
    return response.choices[0].message.content
from openai import OpenAI
import streamlit as st
import os

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def summarize_zone_stats(zone, hour, avg, max_val, count, sensor_type):
    prompt = f"""
    You are an AI risk analyst summarizing crisis activity in a smart city zone.

    Zone: {zone}
    Hour: {hour}:00
    Average Sensor Reading: {avg:.2f}
    Max Sensor Reading: {max_val:.2f}
    Sensor Count: {int(count)}
    Dominant Sensor Type: {sensor_type}

    Generate a brief, clear summary (2-3 lines) explaining the situation and potential risks in plain English.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an emergency response analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=120
    )

    return response.choices[0].message.content.strip()
