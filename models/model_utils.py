from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import pandas as pd
import numpy as np

def evaluate_model(model, X_test, y_test):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    preds = cross_val_predict(model, X_test, y_test, cv=kf)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)

    return preds, {"R²": r2, "MAE": mae, "RMSE": rmse}

def roi_to_score(predicted_roi, min_roi, max_roi):
    score = 1 + 99 * ((predicted_roi - min_roi) / (max_roi - min_roi))
    return np.clip(np.round(score), 1, 100)