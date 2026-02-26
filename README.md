# AeroVision - AI-Powered Air Quality Prediction & Monitoring System

A machine learning web application that predicts **Air Quality Index (AQI)** from pollutant concentrations and provides interactive dashboards to monitor air pollution trends across **26 Indian cities**.

Built with **Flask**, **scikit-learn**, **XGBoost**, **Bootstrap 5**, and **Chart.js**.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [What is AQI?](#what-is-aqi)
- [Dataset](#dataset)
- [Pollutants Explained](#pollutants-explained)
- [ML Pipeline](#ml-pipeline)
  - [1. Data Loading & Cleaning](#1-data-loading--cleaning)
  - [2. Handling Missing Values](#2-handling-missing-values)
  - [3. Outlier Removal (IQR Method)](#3-outlier-removal-iqr-method)
  - [4. Feature Engineering & Scaling](#4-feature-engineering--scaling)
  - [5. Model Training](#5-model-training)
  - [6. Model Evaluation](#6-model-evaluation)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Features](#features)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)

---

## Problem Statement

Air pollution is one of the most critical environmental challenges in India. Cities like Delhi, Ahmedabad, and Patna frequently experience hazardous pollution levels, directly impacting public health.

**Goal:** Build a system that can:
1. **Predict** the AQI value given pollutant concentrations (PM2.5, NO2, CO, etc.)
2. **Classify** the predicted AQI into health categories (Good, Moderate, Poor, Severe, etc.)
3. **Visualize** pollution trends across cities to identify patterns and hotspots

This helps in understanding which pollutants are the biggest contributors to poor air quality, enabling better policy decisions and public awareness.

---

## What is AQI?

The **Air Quality Index (AQI)** is a standardized scale from **0 to 500** that communicates how polluted the air is and what health effects might be a concern.

| AQI Range | Category       | Health Impact                                      | Color  |
|-----------|----------------|-----------------------------------------------------|--------|
| 0 - 50    | Good           | Minimal impact                                      | Green  |
| 51 - 100  | Satisfactory   | Minor breathing discomfort to sensitive people       | Light Green |
| 101 - 200 | Moderate       | Breathing discomfort to people with lung/heart disease | Yellow |
| 201 - 300 | Poor           | Breathing discomfort to most people on prolonged exposure | Orange |
| 301 - 400 | Very Poor      | Respiratory illness on prolonged exposure             | Red    |
| 401 - 500 | Severe         | Affects healthy people, serious impact on those with existing diseases | Dark Red |

AQI is calculated based on the **worst sub-index** among individual pollutant concentrations (PM2.5, PM10, NO2, SO2, CO, O3, NH3). In this project, we use ML to predict the overall AQI from all 12 pollutant values simultaneously.

---

## Dataset

**Source:** [Kaggle - Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

| Property        | Value                          |
|-----------------|--------------------------------|
| File Used       | `city_day.csv`                 |
| Total Records   | 29,531 daily readings          |
| Cities Covered  | 26 Indian cities               |
| Time Period     | January 2015 - July 2020       |
| Features        | 12 pollutants + AQI + AQI_Bucket |

---

## Pollutants Explained

Understanding what each pollutant is and where it comes from:

| Pollutant | Full Name | Source | Health Effect |
|-----------|-----------|--------|---------------|
| **PM2.5** | Particulate Matter (< 2.5 microns) | Vehicle exhaust, construction, burning | Penetrates deep into lungs, causes respiratory and cardiovascular diseases |
| **PM10** | Particulate Matter (< 10 microns) | Dust, pollen, industrial emissions | Irritates airways, aggravates asthma |
| **NO** | Nitric Oxide | Vehicle engines, power plants | Converts to NO2, contributes to smog |
| **NO2** | Nitrogen Dioxide | Traffic, fossil fuel combustion | Inflames airways, reduces immunity to lung infections |
| **NOx** | Nitrogen Oxides (NO + NO2) | Combustion processes | Precursor to ground-level ozone and acid rain |
| **NH3** | Ammonia | Agriculture (fertilizers, livestock) | Forms fine particulate matter, irritates eyes and lungs |
| **CO** | Carbon Monoxide | Incomplete combustion of fuels | Reduces oxygen delivery in blood, toxic at high levels |
| **SO2** | Sulfur Dioxide | Coal/oil burning, industrial processes | Causes acid rain, triggers asthma attacks |
| **O3** | Ozone (ground-level) | Formed by NOx + VOCs in sunlight | Good in upper atmosphere, harmful at ground level (smog) |
| **Benzene** | Benzene | Vehicle exhaust, tobacco smoke, paints | Known carcinogen (causes cancer) |
| **Toluene** | Toluene | Paint thinners, adhesives, gasoline | Affects nervous system, causes dizziness |
| **Xylene** | Xylene | Paints, varnishes, printing inks | Irritates skin, eyes, and respiratory tract |

---

## ML Pipeline

The complete machine learning workflow follows the industry-standard approach:

```
Load Data -> Clean -> Handle Missing Values -> Remove Outliers -> Scale Features -> Train Models -> Evaluate -> Deploy
```

### 1. Data Loading & Cleaning

```python
df = pd.read_csv('data/city_day.csv', parse_dates=['Date'])
df = df.dropna(subset=['AQI'])  # Remove rows where target (AQI) is missing
```

- Load the CSV with date parsing enabled
- Drop rows where AQI is null (these cannot be used for training since we have no target value)
- Result: **24,850 usable rows** from 29,531 total

### 2. Handling Missing Values

```python
# Numeric columns: fill with median (robust to outliers)
df[col] = df[col].fillna(df[col].median())
```

**Why median instead of mean?**
- Mean is sensitive to extreme values (outliers pull it up/down)
- Median is the middle value, unaffected by outliers
- Example: Values [10, 12, 11, 13, **500**] → Mean = 109.2, Median = 12
- Median (12) is clearly more representative of the typical value

**Why mode for categorical columns?**
- Mode = most frequently occurring value
- For categories like AQI_Bucket, filling with the most common category is the safest assumption

### 3. Outlier Removal (IQR Method)

```
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
Remove any data point outside these bounds
```

**What is IQR?**
- Q1 (25th percentile): 25% of data falls below this value
- Q3 (75th percentile): 75% of data falls below this value
- IQR (Interquartile Range): The spread of the middle 50% of data
- The **1.5 multiplier** is a standard statistical convention (proposed by John Tukey)

**Why remove outliers?**
- Extreme pollution spikes (e.g., Diwali, industrial accidents) can distort model training
- Models learn the "normal" relationship between pollutants and AQI better without extreme cases
- Result: 24,850 rows → **7,232 rows** after outlier removal

### 4. Feature Engineering & Scaling

```python
# Features: Only the 12 pollutants (NOT AQI itself)
X = df_clean[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]
y = df_clean['AQI']  # Target

# StandardScaler: transforms each feature to mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why StandardScaler?**
- Different pollutants have vastly different scales (PM2.5: 0-300, CO: 0-5, Benzene: 0-30)
- Without scaling, features with larger values dominate the model
- StandardScaler normalizes all features to the same scale: `z = (x - mean) / std`
- After scaling, every feature has mean=0 and standard deviation=1

**Why NOT include AQI as a feature?**
- AQI is our **target** (what we're trying to predict)
- Including AQI as an input feature = **data leakage** (the model "sees the answer")
- This would give artificially perfect scores (~0.99 R2) but the model would be useless in practice

### 5. Model Training

Two models are trained and compared:

**Random Forest Regressor (200 trees)**
- An **ensemble** of 200 independent decision trees
- Each tree sees a random subset of data and features
- Final prediction = average of all 200 tree predictions
- **Strength:** Robust, hard to overfit, handles non-linear relationships
- **Analogy:** Like asking 200 different experts and averaging their opinions

**XGBoost Regressor (200 trees, learning rate 0.1)**
- Trees are built **sequentially** (not independently)
- Each new tree focuses on correcting the errors of previous trees
- Learning rate (0.1) controls how much each tree contributes
- **Strength:** Often higher accuracy than Random Forest
- **Analogy:** Like a student who reviews mistakes and improves with each attempt

```python
# Train-Test Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

**Why 80/20 split?** Standard practice. 80% is enough to learn patterns, 20% is enough to reliably test.

**Why random_state=42?** Ensures the same split every time, making results reproducible.

### 6. Model Evaluation

| Model          | R2 Score | RMSE   |
|---------------|----------|--------|
| Random Forest | 0.7858   | 16.76  |
| XGBoost       | 0.7828   | 16.88  |

**R2 Score (Coefficient of Determination)**
- Measures how much variance the model explains
- Range: 0 to 1 (1 = perfect prediction, 0 = predicting just the mean)
- **0.78 means the model explains 78% of the variation in AQI**
- Formula: `R2 = 1 - (sum of squared residuals / total sum of squares)`

**RMSE (Root Mean Squared Error)**
- Average prediction error in the **same units as AQI**
- **16.76 means predictions are off by ~17 AQI points on average**
- RMSE penalizes large errors more than small ones (due to squaring)
- Formula: `RMSE = sqrt(mean((actual - predicted)^2))`

---

## Project Structure

```
AeroVision/
├── app.py                 # Flask web application (routes & logic)
├── train_model.py         # ML training pipeline (run once)
├── config.py              # Shared constants & configuration
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── data/
│   └── city_day.csv       # Dataset (29,531 daily records)
│
├── models/                # Generated by train_model.py
│   ├── rf_model.pkl       # Trained Random Forest model
│   ├── xgb_model.pkl      # Trained XGBoost model
│   ├── scaler.pkl         # Fitted StandardScaler
│   ├── model_metrics.pkl  # R2 & RMSE scores
│   ├── feature_importance.pkl
│   ├── city_summary.pkl   # Pre-computed city stats
│   └── city_monthly.pkl   # Monthly AQI trends
│
├── templates/             # Jinja2 HTML templates
│   ├── base.html          # Base layout (navbar, footer)
│   ├── index.html         # Home page
│   ├── predict.html       # AQI prediction form
│   ├── dashboard.html     # City dashboard
│   └── visualizations.html # Charts & analysis
│
└── static/
    ├── css/style.css      # Green nature theme
    ├── js/charts.js       # Chart.js utilities
    └── images/            # Generated chart images
        ├── heatmap.png
        └── aqi_distribution.png
```

---

## How to Run

### Prerequisites
- Python 3.8 or higher

### Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the models (generates pickle files & chart images)
python train_model.py

# 3. Start the web application
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Features

### 1. AQI Prediction
- Enter 12 pollutant values through a form
- Get instant AQI prediction with color-coded health category
- Visual gauge bar showing where the prediction falls on the 0-500 scale
- Comparison between Random Forest and XGBoost predictions
- "Fill Sample Values" button for quick demo

### 2. City Dashboard
- Dropdown to select any of the 26 cities
- City statistics: Average, Maximum, Minimum AQI and record count
- Interactive line chart showing monthly AQI trends over time
- Horizontal bar chart comparing average AQI across all cities (color-coded by severity)

### 3. Data Visualizations
- **Correlation Heatmap:** Shows relationships between all pollutants and AQI
- **AQI Distribution:** Histogram of AQI values in the dataset
- **Model Comparison:** Side-by-side R2 and RMSE bar charts
- **Feature Importance:** Which pollutants matter most for prediction (RF vs XGBoost)
- **Key Findings:** Summary of insights from the analysis

---

## Key Findings

1. **PM2.5 is the strongest predictor of AQI** — Fine particulate matter has the highest feature importance in both models
2. **CO, SO2, PM10, NO2** are key secondary drivers of air quality
3. **O3, Benzene, Toluene** contribute less to overall AQI prediction
4. **Both models agree** on feature importance ranking, confirming the consistency of results
5. **Random Forest and XGBoost perform similarly** (R2 ~0.78), suggesting the relationship between pollutants and AQI is well-captured by tree-based models

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3** | Core programming language |
| **Flask** | Web framework for building the application |
| **pandas** | Data loading, cleaning, and manipulation |
| **NumPy** | Numerical computations |
| **scikit-learn** | StandardScaler, train_test_split, RandomForestRegressor, metrics |
| **XGBoost** | Gradient boosted tree model for AQI prediction |
| **matplotlib** | Generating static charts (heatmap, histogram) |
| **seaborn** | Statistical visualization (heatmap styling) |
| **Bootstrap 5** | Responsive UI framework |
| **Chart.js** | Interactive browser-based charts |
| **Jinja2** | HTML templating engine (built into Flask) |
| **pickle** | Saving/loading trained models and data artifacts |
