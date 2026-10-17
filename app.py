"""
AeroVision - Flask Web Application
Run: python app.py
Make sure to run train_model.py first to generate model artifacts.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from config import MODELS_DIR, FEATURE_COLUMNS, AQI_BUCKETS, OPENWEATHER_API_KEY
import live_aqi
from live_aqi import LiveAQIError

app = Flask(__name__)

# --- Load all artifacts at startup ---
def load_artifact(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{filename}' not found in models/. Run 'python train_model.py' first."
        )
    with open(path, 'rb') as f:
        return pickle.load(f)

rf_model = load_artifact('rf_model.pkl')
xgb_model = load_artifact('xgb_model.pkl')
gb_model = load_artifact('gb_model.pkl')
et_model = load_artifact('et_model.pkl')
scaler = load_artifact('scaler.pkl')
metrics = load_artifact('model_metrics.pkl')
feat_imp = load_artifact('feature_importance.pkl')
city_summary = load_artifact('city_summary.pkl')
city_monthly = load_artifact('city_monthly.pkl')


def get_aqi_bucket(aqi_value):
    """Map numeric AQI to bucket label and color."""
    for low, high, label, color in AQI_BUCKETS:
        if low <= aqi_value <= high:
            return label, color
    return 'Severe', '#990000'


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        # Collect 12 pollutant values from form
        values = []
        for col in FEATURE_COLUMNS:
            val = float(request.form.get(col, 0))
            values.append(val)

        # Scale using the saved scaler
        input_array = np.array(values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Predict with all models
        xgb_aqi = round(float(xgb_model.predict(input_scaled)[0]), 1)
        rf_aqi = round(float(rf_model.predict(input_scaled)[0]), 1)
        gb_aqi = round(float(gb_model.predict(input_scaled)[0]), 1)
        et_aqi = round(float(et_model.predict(input_scaled)[0]), 1)

        bucket, color = get_aqi_bucket(xgb_aqi)

        result = {
            'aqi': xgb_aqi,
            'rf_aqi': rf_aqi,
            'gb_aqi': gb_aqi,
            'et_aqi': et_aqi,
            'bucket': bucket,
            'color': color,
            'inputs': dict(zip(FEATURE_COLUMNS, values)),
        }

    return render_template('predict.html', result=result, features=FEATURE_COLUMNS)


@app.route('/dashboard')
def dashboard():
    cities = city_summary['City'].tolist()
    summary_data = city_summary.to_dict('records')
    return render_template('dashboard.html', cities=cities, summary=summary_data)


@app.route('/api/city_trend/<city>')
def city_trend(city):
    """Return monthly AQI data for a specific city as JSON."""
    data = city_monthly[city_monthly['City'] == city]
    return jsonify({
        'labels': data['YearMonth'].tolist(),
        'values': data['AQI'].tolist(),
    })


@app.route('/live')
def live():
    """Live AQI page - data is loaded client-side from /api/live."""
    return render_template('live.html', api_key_configured=bool(OPENWEATHER_API_KEY))


@app.route('/api/city_search')
def api_city_search():
    """Autocomplete suggestions for the live search box (?q=del)."""
    query = (request.args.get('q') or '').strip()
    if len(query) < 2:
        return jsonify({'results': []})

    try:
        return jsonify({'results': live_aqi.search_cities(query)})
    except LiveAQIError as e:
        return jsonify({'error': str(e), 'results': []}), 502


@app.route('/api/live')
def api_live():
    """
    Live air quality for a city (?city=Delhi) or coordinates (?lat=&lon=).

    Returns OpenWeather's own 1-5 index plus this project's model prediction
    on the 0-500 scale, so the two can be shown side by side.
    """
    city = (request.args.get('city') or '').strip()
    lat, lon = request.args.get('lat'), request.args.get('lon')

    try:
        if lat and lon:
            location = {
                'name': request.args.get('label') or f'{lat}, {lon}',
                'country': '', 'state': '',
                'lat': float(lat), 'lon': float(lon),
            }
        elif city:
            location = live_aqi.geocode_city(city)
        else:
            return jsonify({'error': 'Provide a city name or lat/lon.'}), 400

        pollution = live_aqi.fetch_air_pollution(location['lat'], location['lon'])
        forecast = live_aqi.fetch_forecast(location['lat'], location['lon'])
        weather = live_aqi.fetch_weather(location['lat'], location['lon'])
    except LiveAQIError as e:
        return jsonify({'error': str(e)}), 502
    except ValueError:
        return jsonify({'error': 'Invalid coordinates.'}), 400

    # Score the live reading with the trained model
    features = live_aqi.build_feature_vector(pollution['components'])
    scaled = scaler.transform(np.array(features).reshape(1, -1))
    predicted_aqi = round(float(xgb_model.predict(scaled)[0]), 1)
    bucket, color = get_aqi_bucket(predicted_aqi)

    return jsonify({
        'location': location,
        'owm': {
            'index': pollution['owm_index'],
            'label': pollution['owm_label'],
            'color': pollution['owm_color'],
        },
        'components': {k: round(v, 2) for k, v in pollution['components'].items()},
        'model': {
            'aqi': predicted_aqi,
            'bucket': bucket,
            'color': color,
            'features': dict(zip(FEATURE_COLUMNS, features)),
        },
        'weather': weather,
        'forecast': forecast,
        'timestamp': pollution['timestamp'],
    })


@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html', metrics=metrics, feat_imp=feat_imp)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
