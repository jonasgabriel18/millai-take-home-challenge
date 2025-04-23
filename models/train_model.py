from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from model_utils import evaluate_model, roi_to_score

import pandas as pd

def train_xgboost(df: pd.DataFrame):
    other_targets = ['impressions', 'clicks', 'ctr', 'conversions', 'conversion_rate'] # I'll work only with ROI as my target for now
    df = df.drop(columns=other_targets)

    X = df.drop(columns=['roi'])
    y = df['roi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train_scaled, y_train)

    return xgb_model, scaler, X_test_scaled, y_test

def train_random_forest(df: pd.DataFrame):
    other_targets = ['impressions', 'clicks', 'ctr', 'conversions', 'conversion_rate'] # I'll work only with ROI as my target for now
    df = df.drop(columns=other_targets)

    X = df.drop(columns=['roi'])
    y = df['roi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train_scaled, y_train)

    return rf, scaler, X_test_scaled, y_test

if __name__ == "__main__":
    df = pd.read_csv('./data/processed/sample_feature_engineered.csv')

    xgb_model, xgb_scaler, xgb_X_test_scaled, xgb_y_test = train_xgboost(df)
    preds, metrics = evaluate_model(xgb_model, xgb_X_test_scaled, xgb_y_test)

    print("XGBoost Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    rf_model, rf_scaler, rf_X_test_scaled, rf_y_test = train_random_forest(df)
    rf_preds, rf_metrics = evaluate_model(rf_model, rf_X_test_scaled, rf_y_test)

    print("\nRandom Forest Model Evaluation Metrics:")
    for metric, value in rf_metrics.items():
        print(f"{metric}: {value:.4f}")

    ## Don't know how to use ROI Score 1-100 and the confidence intervals yet