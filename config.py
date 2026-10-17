import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, '.env'))

# Datasets
CSV_PATH = os.path.join(BASE_DIR, 'data', 'city_day.csv')
STATION_DAY_PATH = os.path.join(BASE_DIR, 'data', 'station_day.csv')
STATIONS_PATH = os.path.join(BASE_DIR, 'data', 'stations.csv')

# Model artifacts directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Static images directory
IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')

# The 12 pollutant features the model expects as input (in order)
FEATURE_COLUMNS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]

# --- OpenWeatherMap (live AQI feature) ---
# Add your key to a .env file in the project root:  OPENWEATHER_API_KEY=xxxxx
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')

GEOCODE_URL = 'https://api.openweathermap.org/geo/1.0/direct'
AIR_POLLUTION_URL = 'https://api.openweathermap.org/data/2.5/air_pollution'
AIR_POLLUTION_FORECAST_URL = 'https://api.openweathermap.org/data/2.5/air_pollution/forecast'
WEATHER_URL = 'https://api.openweathermap.org/data/2.5/weather'

# OpenWeather returns a 1-5 air quality index, which is its own scale
# (not the 0-500 Indian AQI the model predicts). Labels per their docs.
OWM_AQI_LEVELS = {
    1: ('Good',      '#00b050'),
    2: ('Fair',      '#92d050'),
    3: ('Moderate',  '#ffff00'),
    4: ('Poor',      '#ff9900'),
    5: ('Very Poor', '#ff0000'),
}

# AQI bucket thresholds (official Indian AQI standard)
AQI_BUCKETS = [
    (0,   50,  'Good',          '#00b050'),
    (51,  100, 'Satisfactory',  '#92d050'),
    (101, 200, 'Moderate',      '#ffff00'),
    (201, 300, 'Poor',          '#ff9900'),
    (301, 400, 'Very Poor',     '#ff0000'),
    (401, 500, 'Severe',        '#990000'),
]
