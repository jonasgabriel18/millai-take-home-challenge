from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import pandas as pd
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a regression model using K-Fold cross-validation.

    Parameters:
    -----------
    model : object
        A regression model that follows the scikit-learn API 
        (e.g., LinearRegression, RandomForestRegressor, etc.).
    X_test : array-like or DataFrame
        Feature set (independent variables) used for testing.
    y_test : array-like or Series
        True target values (dependent variable) corresponding to X_test.

    Returns:
    --------
    preds : ndarray
        Predictions made by the model using cross-validation.
    dict :
        A dictionary containing evaluation metrics: R² (coefficient of determination), 
        MAE (mean absolute error), and RMSE (root mean squared error).
    """

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    preds = cross_val_predict(model, X_test, y_test, cv=kf)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)

    return preds, {"R²": r2, "MAE": mae, "RMSE": rmse}

def roi_to_score(predicted_roi, min_roi, max_roi):
    """
    Converts a predicted ROI (Return on Investment) value into a score ranging from 1 to 100.

    Parameters:
    -----------
    predicted_roi : float or array-like
        Predicted ROI value(s).
    min_roi : float
        Minimum ROI value, used for normalization.
    max_roi : float
        Maximum ROI value, used for normalization.

    Returns:
    --------
    score : int
        A score from 1 to 100, where higher values indicate better ROI.
    """

    score = 1 + 99 * ((predicted_roi - min_roi) / (max_roi - min_roi))
    return np.clip(np.round(score), 1, 100)