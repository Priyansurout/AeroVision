"""
AeroVision - Model Training Script
Run this once to train models and generate all artifacts.
Usage: python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

from config import CSV_PATH, MODELS_DIR, IMAGES_DIR, FEATURE_COLUMNS


def remove_outliers_iqr(dataframe, columns):
    """Remove outliers using IQR method."""
    for col in columns:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        dataframe = dataframe[(dataframe[col] >= lower) & (dataframe[col] <= upper)]
    return dataframe


def train():
    print("=" * 50)
    print("AeroVision - Training Pipeline")
    print("=" * 50)

    # 1. Load data
    print("\n[1/8] Loading dataset...")
    df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Drop rows where AQI is null (useless for training)
    df = df.dropna(subset=['AQI'])
    print(f"  After dropping null AQI: {df.shape[0]} rows")

    # 3. Fill remaining nulls in pollutant columns with median
    print("\n[2/8] Handling missing values...")
    for col in FEATURE_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  Filled {null_count} nulls in {col} with median")

    # 4. Remove outliers using IQR
    print("\n[3/8] Removing outliers...")
    df_clean = remove_outliers_iqr(df, FEATURE_COLUMNS + ['AQI'])
    print(f"  Before: {df.shape[0]} rows -> After: {df_clean.shape[0]} rows")

    # 5. Prepare features (12 pollutants only) and target (AQI)
    print("\n[4/8] Preparing features and target...")
    X = df_clean[FEATURE_COLUMNS].copy()
    y = df_clean['AQI'].copy()
    print(f"  Features: {list(X.columns)}")
    print(f"  Target: AQI (range: {y.min():.0f} - {y.max():.0f})")

    # 6. Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=FEATURE_COLUMNS,
        index=X.index
    )

    # 7. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

    # 8. Train models
    print("\n[5/8] Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    print("[5/8] Training XGBoost...")
    xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    # 9. Evaluate
    print("\n[6/8] Evaluating models...")
    metrics = {
        'rf_r2': round(r2_score(y_test, rf_preds), 4),
        'rf_rmse': round(np.sqrt(mean_squared_error(y_test, rf_preds)), 4),
        'xgb_r2': round(r2_score(y_test, xgb_preds), 4),
        'xgb_rmse': round(np.sqrt(mean_squared_error(y_test, xgb_preds)), 4),
    }
    print(f"  Random Forest  -> R2: {metrics['rf_r2']}, RMSE: {metrics['rf_rmse']}")
    print(f"  XGBoost        -> R2: {metrics['xgb_r2']}, RMSE: {metrics['xgb_rmse']}")

    # 10. Feature importances
    feature_importance = {
        'features': FEATURE_COLUMNS,
        'rf_importance': rf_model.feature_importances_.tolist(),
        'xgb_importance': xgb_model.feature_importances_.tolist(),
    }

    # 11. Save all model artifacts
    print("\n[7/8] Saving model artifacts...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    artifacts = {
        'rf_model.pkl': rf_model,
        'xgb_model.pkl': xgb_model,
        'scaler.pkl': scaler,
        'model_metrics.pkl': metrics,
        'feature_importance.pkl': feature_importance,
    }
    for filename, obj in artifacts.items():
        path = os.path.join(MODELS_DIR, filename)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  Saved {filename}")

    # 12. Pre-compute dashboard data
    # City-level summary stats
    city_summary = df.groupby('City').agg(
        avg_aqi=('AQI', 'mean'),
        max_aqi=('AQI', 'max'),
        min_aqi=('AQI', 'min'),
        avg_pm25=('PM2.5', 'mean'),
        avg_pm10=('PM10', 'mean'),
        avg_no2=('NO2', 'mean'),
        avg_co=('CO', 'mean'),
        avg_so2=('SO2', 'mean'),
        avg_o3=('O3', 'mean'),
        record_count=('AQI', 'count'),
    ).round(2).reset_index()

    with open(os.path.join(MODELS_DIR, 'city_summary.pkl'), 'wb') as f:
        pickle.dump(city_summary, f)
    print("  Saved city_summary.pkl")

    # Monthly AQI trends per city
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    city_monthly = df.groupby(['City', 'YearMonth'])['AQI'].mean().round(2).reset_index()

    with open(os.path.join(MODELS_DIR, 'city_monthly.pkl'), 'wb') as f:
        pickle.dump(city_monthly, f)
    print("  Saved city_monthly.pkl")

    # 13. Generate static EDA images
    print("\n[8/8] Generating charts...")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df_clean[FEATURE_COLUMNS + ['AQI']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Matrix of Air Quality Features')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'heatmap.png'), dpi=150)
    plt.close()
    print("  Saved heatmap.png")

    # AQI distribution histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df_clean['AQI'], bins=50, color='steelblue', edgecolor='black')
    plt.xlabel('AQI Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of AQI Values')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'aqi_distribution.png'), dpi=150)
    plt.close()
    print("  Saved aqi_distribution.png")

    print("\n" + "=" * 50)
    print("Training complete! All artifacts saved to models/")
    print("You can now run: python app.py")
    print("=" * 50)


if __name__ == '__main__':
    train()
