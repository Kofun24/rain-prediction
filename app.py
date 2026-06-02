import streamlit as st
import pandas as pd
import joblib
import os


MODEL_PATH = "models/rain_prediction_model.pkl"


st.set_page_config(
    page_title="Rain Prediction App",
    page_icon="🌧️",
    layout="centered"
)


st.title("🌧️ Rain Prediction App")
st.write("This app predicts whether it will rain tomorrow using weather details.")


if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run 03_model_building.ipynb first.")
    st.stop()


model = joblib.load(MODEL_PATH)


st.sidebar.header("Enter Weather Details")


location = st.sidebar.selectbox(
    "Location",
    [
        "Albury", "BadgerysCreek", "Cobar", "CoffsHarbour", "Moree",
        "Newcastle", "NorahHead", "NorfolkIsland", "Penrith", "Richmond",
        "Sydney", "SydneyAirport", "WaggaWagga", "Williamtown", "Wollongong",
        "Canberra", "Tuggeranong", "MountGinini", "Ballarat", "Bendigo",
        "Sale", "MelbourneAirport", "Melbourne", "Mildura", "Nhil",
        "Portland", "Watsonia", "Dartmoor", "Brisbane", "Cairns",
        "GoldCoast", "Townsville", "Adelaide", "MountGambier", "Nuriootpa",
        "Woomera", "Albany", "Witchcliffe", "PearceRAAF", "PerthAirport",
        "Perth", "SalmonGums", "Walpole", "Hobart", "Launceston",
        "AliceSprings", "Darwin", "Katherine", "Uluru"
    ]
)

min_temp = st.sidebar.number_input("Minimum Temperature", value=13.4)
max_temp = st.sidebar.number_input("Maximum Temperature", value=22.9)
rainfall = st.sidebar.number_input("Rainfall", value=0.6)

wind_gust_dir = st.sidebar.selectbox(
    "Wind Gust Direction",
    ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
)

wind_gust_speed = st.sidebar.number_input("Wind Gust Speed", value=44.0)

wind_dir_9am = st.sidebar.selectbox(
    "Wind Direction 9am",
    ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
)

wind_dir_3pm = st.sidebar.selectbox(
    "Wind Direction 3pm",
    ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
)

wind_speed_9am = st.sidebar.number_input("Wind Speed 9am", value=20.0)
wind_speed_3pm = st.sidebar.number_input("Wind Speed 3pm", value=24.0)

humidity_9am = st.sidebar.number_input("Humidity 9am", value=71.0)
humidity_3pm = st.sidebar.number_input("Humidity 3pm", value=22.0)

pressure_9am = st.sidebar.number_input("Pressure 9am", value=1007.7)
pressure_3pm = st.sidebar.number_input("Pressure 3pm", value=1007.1)

temp_9am = st.sidebar.number_input("Temperature 9am", value=16.9)
temp_3pm = st.sidebar.number_input("Temperature 3pm", value=21.8)

rain_today = st.sidebar.selectbox("Rain Today", ["No", "Yes"])

year = st.sidebar.number_input("Year", value=2008)
month = st.sidebar.slider("Month", 1, 12, 12)
day = st.sidebar.slider("Day", 1, 31, 1)


if st.button("Predict Rain Tomorrow"):
    input_data = pd.DataFrame([{
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today,
        "Year": year,
        "Month": month,
        "Day": day
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🌧️ It may rain tomorrow.")
    else:
        st.success("☀️ It may not rain tomorrow.")

    st.write("Rain Probability:", round(probability * 100, 2), "%")