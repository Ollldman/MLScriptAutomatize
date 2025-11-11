import logging
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, precision_score, recall_score
)
from xgboost import XGBRegressor, XGBClassifier

logger = logging.getLogger(__name__)

# Default parameter grids for XGBoost models
XGB_REGRESSOR_DEFAULT_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "random_state": [42]
}

XGB_CLASSIFIER_DEFAULT_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "random_state": [42]
}


def xgb_run_regression_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    param_grid: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    scoring: str = "r2",
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for XGBRegressor using GridSearchCV.

    Args:
        X_train (np.ndarray or pd.DataFrame): Training features.
        y_train (np.ndarray or pd.Series): Training target.
        X_test (np.ndarray or pd.DataFrame): Test features.
        y_test (np.ndarray or pd.Series): Test target.
        param_grid (Dict[str, Any], optional): Custom parameter grid to search.
            If None, uses default parameters for XGBRegressor.
        cv (int): Number of cross-validation folds. Default is 5.
        scoring (str): Scoring metric for GridSearchCV. Default is 'r2'.
        n_jobs (int): Number of jobs to run in parallel. Default is -1 (all CPUs).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'best_params': Best parameters found by GridSearchCV
            - 'best_score': Best cross-validation score
            - 'best_estimator': Fitted best estimator (XGBRegressor)
            - 'test_metrics': Dict with R2, MAE, MSE on test set
            - 'cv_results': Full cv_results_ from GridSearchCV (for plotting)
            - 'plot_data': Dict containing 'params', 'mean_test_score', 'std_test_score' for plotting

    Raises:
        TypeError: If data is not a numpy array or pandas DataFrame/Series.
    """
    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    logger.info("Starting GridSearchCV for XGBRegressor")

    if param_grid is None:
        param_grid = XGB_REGRESSOR_DEFAULT_PARAMS

    logger.info(f"Parameter grid: {param_grid}")

    estimator = XGBRegressor()
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    logger.info("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_estimator.predict(X_test)
    test_metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }

    # Prepare data for plotting
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    plot_data = {
        "params": cv_results_df["params"],
        "mean_test_score": cv_results_df["mean_test_score"],
        "std_test_score": cv_results_df["std_test_score"],
    }

    result = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": best_estimator,
        "test_metrics": test_metrics,
        "cv_results": grid_search.cv_results_,
        "plot_data": plot_data
    }

    logger.info(f"GridSearchCV completed. Best score: {grid_search.best_score_:.4f}")
    return result


def xgb_run_classification_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    param_grid: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for XGBClassifier using GridSearchCV.

    Args:
        X_train (np.ndarray or pd.DataFrame): Training features.
        y_train (np.ndarray or pd.Series): Training target.
        X_test (np.ndarray or pd.DataFrame): Test features.
        y_test (np.ndarray or pd.Series): Test target.
        param_grid (Dict[str, Any], optional): Custom parameter grid to search.
            If None, uses default parameters for XGBClassifier.
        cv (int): Number of cross-validation folds. Default is 5.
        scoring (str): Scoring metric for GridSearchCV. Default is 'accuracy'.
        n_jobs (int): Number of jobs to run in parallel. Default is -1 (all CPUs).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'best_params': Best parameters found by GridSearchCV
            - 'best_score': Best cross-validation score
            - 'best_estimator': Fitted best estimator (XGBClassifier)
            - 'test_metrics': Dict with Accuracy, F1, Precision, Recall on test set
            - 'cv_results': Full cv_results_ from GridSearchCV (for plotting)
            - 'plot_data': Dict containing 'params', 'mean_test_score', 'std_test_score' for plotting

    Raises:
        TypeError: If data is not a numpy array or pandas DataFrame/Series.
    """
    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    logger.info("Starting GridSearchCV for XGBClassifier")

    if param_grid is None:
        param_grid = XGB_CLASSIFIER_DEFAULT_PARAMS

    logger.info(f"Parameter grid: {param_grid}")

    estimator = XGBClassifier()
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    logger.info("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_estimator.predict(X_test)
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    # Prepare data for plotting
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    plot_data = {
        "params": cv_results_df["params"],
        "mean_test_score": cv_results_df["mean_test_score"],
        "std_test_score": cv_results_df["std_test_score"],
    }

    result = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": best_estimator,
        "test_metrics": test_metrics,
        "cv_results": grid_search.cv_results_,
        "plot_data": plot_data
    }

    logger.info(f"GridSearchCV completed. Best score: {grid_search.best_score_:.4f}")
    return result