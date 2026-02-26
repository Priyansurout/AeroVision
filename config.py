import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# AQI bucket thresholds (official Indian AQI standard)
AQI_BUCKETS = [
    (0,   50,  'Good',          '#00b050'),
    (51,  100, 'Satisfactory',  '#92d050'),
    (101, 200, 'Moderate',      '#ffff00'),
    (201, 300, 'Poor',          '#ff9900'),
    (301, 400, 'Very Poor',     '#ff0000'),
    (401, 500, 'Severe',        '#990000'),
]
