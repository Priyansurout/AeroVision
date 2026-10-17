"""
Live AQI data from OpenWeatherMap.

Uses two free endpoints on a standard API key:
  - Geocoding API    : city name -> lat/lon
  - Air Pollution API: lat/lon   -> AQI index + pollutant concentrations

The Air Pollution API gives 8 of the 12 pollutants the model expects, so
build_feature_vector() fills the gaps (see notes there) to let the trained
model score live readings on the 0-500 Indian AQI scale.
"""

import requests

from config import (
    AIR_POLLUTION_FORECAST_URL,
    AIR_POLLUTION_URL,
    FEATURE_COLUMNS,
    GEOCODE_URL,
    OPENWEATHER_API_KEY,
    OWM_AQI_LEVELS,
    WEATHER_URL,
)

TIMEOUT = 10


class LiveAQIError(Exception):
    """Raised when live data can't be fetched, with a user-safe message."""


def _get(url, params):
    if not OPENWEATHER_API_KEY:
        raise LiveAQIError(
            'No OpenWeatherMap API key configured. '
            'Add OPENWEATHER_API_KEY to your .env file and restart the app.'
        )

    try:
        resp = requests.get(
            url, params={**params, 'appid': OPENWEATHER_API_KEY}, timeout=TIMEOUT
        )
    except requests.exceptions.Timeout:
        raise LiveAQIError('OpenWeatherMap took too long to respond. Try again.')
    except requests.exceptions.RequestException:
        raise LiveAQIError('Could not reach OpenWeatherMap. Check your connection.')

    if resp.status_code == 401:
        raise LiveAQIError(
            'OpenWeatherMap rejected the API key. New keys can take a couple of '
            'hours to activate.'
        )
    if resp.status_code == 429:
        raise LiveAQIError('API rate limit reached. Wait a minute and try again.')
    if not resp.ok:
        raise LiveAQIError(f'OpenWeatherMap returned an error (HTTP {resp.status_code}).')

    return resp.json()


def _label(entry):
    """'City, State, Country' label, skipping the parts the API omits."""
    return ', '.join(
        part for part in (entry['name'], entry.get('state'), entry.get('country'))
        if part
    )


def search_cities(query, limit=6):
    """Autocomplete suggestions for a partial city name."""
    results = _get(GEOCODE_URL, {'q': query, 'limit': limit})

    suggestions, seen = [], set()
    for r in results:
        label = _label(r)
        if label in seen:
            continue
        seen.add(label)
        suggestions.append({
            'label': label,
            'name': r['name'],
            'state': r.get('state', ''),
            'country': r.get('country', ''),
            'lat': r['lat'],
            'lon': r['lon'],
        })
    return suggestions


def geocode_city(city):
    """Resolve a city name to the best matching location."""
    results = search_cities(city, limit=1)
    if not results:
        raise LiveAQIError(f"No city found matching '{city}'. Try 'City, CountryCode'.")
    return results[0]


def build_feature_vector(components):
    """
    Map OpenWeather pollutant components to the model's 12 features.

    Unit fixes and gap fills:
      - CO: OpenWeather reports ug/m3, the training data is in mg/m3.
      - NOx: not reported, approximated as NO + NO2.
      - Benzene/Toluene/Xylene: not reported, set to 0 (their dataset medians
        are ~1-3 and they are the least important features in the model).
    """
    values = {
        'PM2.5': components.get('pm2_5', 0.0),
        'PM10': components.get('pm10', 0.0),
        'NO': components.get('no', 0.0),
        'NO2': components.get('no2', 0.0),
        'NH3': components.get('nh3', 0.0),
        'CO': components.get('co', 0.0) / 1000.0,
        'SO2': components.get('so2', 0.0),
        'O3': components.get('o3', 0.0),
        'Benzene': 0.0,
        'Toluene': 0.0,
        'Xylene': 0.0,
    }
    values['NOx'] = values['NO'] + values['NO2']

    return [round(values[col], 2) for col in FEATURE_COLUMNS]


def fetch_air_pollution(lat, lon):
    """Current pollution reading for a coordinate."""
    data = _get(AIR_POLLUTION_URL, {'lat': lat, 'lon': lon})
    entries = data.get('list') or []
    if not entries:
        raise LiveAQIError('OpenWeatherMap has no air quality data for this location.')

    current = entries[0]
    index = current['main']['aqi']
    label, color = OWM_AQI_LEVELS.get(index, ('Unknown', '#6b7280'))

    return {
        'owm_index': index,
        'owm_label': label,
        'owm_color': color,
        'components': current['components'],
        'timestamp': current['dt'],
    }


def fetch_forecast(lat, lon, hours=24):
    """Hourly pollution forecast, trimmed to the next `hours` points."""
    data = _get(AIR_POLLUTION_FORECAST_URL, {'lat': lat, 'lon': lon})
    entries = (data.get('list') or [])[:hours]

    return [
        {
            'timestamp': e['dt'],
            'owm_index': e['main']['aqi'],
            'pm2_5': round(e['components'].get('pm2_5', 0.0), 1),
            'pm10': round(e['components'].get('pm10', 0.0), 1),
        }
        for e in entries
    ]


def fetch_weather(lat, lon):
    """Current weather, shown alongside the AQI for context. Never fatal."""
    try:
        data = _get(WEATHER_URL, {'lat': lat, 'lon': lon, 'units': 'metric'})
    except LiveAQIError:
        return None

    weather = (data.get('weather') or [{}])[0]
    return {
        'temp': round(data['main']['temp'], 1),
        'humidity': data['main']['humidity'],
        'wind_speed': data.get('wind', {}).get('speed'),
        'description': weather.get('description', '').title(),
        'icon': weather.get('icon'),
    }
