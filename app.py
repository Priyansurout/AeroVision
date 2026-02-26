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
from config import MODELS_DIR, FEATURE_COLUMNS, AQI_BUCKETS

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

        # Predict with both models
        xgb_aqi = round(float(xgb_model.predict(input_scaled)[0]), 1)
        rf_aqi = round(float(rf_model.predict(input_scaled)[0]), 1)

        bucket, color = get_aqi_bucket(xgb_aqi)

        result = {
            'aqi': xgb_aqi,
            'rf_aqi': rf_aqi,
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


@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html', metrics=metrics, feat_imp=feat_imp)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
