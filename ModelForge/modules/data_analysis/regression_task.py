import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# Mapping from model name to sklearn class and default params
MODEL_REGISTRY = {
    "DecisionTreeRegressor": {
        "class": DecisionTreeRegressor,
        "default_params": {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "Ridge": {
        "class": Ridge,
        "default_params": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
            "solver": ["auto", "svd", "cholesky", "lsqr"]
        }
    }
}


def run_regression_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    param_grid: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    scoring: str = "r2",
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for a specified regression model using GridSearchCV.

    Args:
        X_train (np.ndarray or pd.DataFrame): Training features.
        y_train (np.ndarray or pd.Series): Training target.
        X_test (np.ndarray or pd.DataFrame): Test features.
        y_test (np.ndarray or pd.Series): Test target.
        model_name (str): Name of the model to use. Supported: 'DecisionTreeRegressor', 'Ridge'.
        param_grid (Dict[str, Any], optional): Custom parameter grid to search.
            If None, uses default parameters for the model.
        cv (int): Number of cross-validation folds. Default is 5.
        scoring (str): Scoring metric for GridSearchCV. Default is 'r2'.
        n_jobs (int): Number of jobs to run in parallel. Default is -1 (all CPUs).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'best_params': Best parameters found by GridSearchCV
            - 'best_score': Best cross-validation score
            - 'best_estimator': Fitted best estimator
            - 'test_metrics': Dict with R2, MAE, MSE on test set
            - 'cv_results': Full cv_results_ from GridSearchCV (for plotting)
            - 'plot_data': Dict containing 'params', 'mean_test_score', 'std_test_score' for plotting

    Raises:
        ValueError: If model_name is not supported.
        TypeError: If data is not a numpy array or pandas DataFrame/Series.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not supported. Available: {list(MODEL_REGISTRY.keys())}")

    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = np.asarray(X_train.values, dtype=np.float64)
    if isinstance(y_train, pd.Series):
        y_train = np.asarray(y_train.values, dtype=np.float64)
    if isinstance(X_test, pd.DataFrame):
        X_test = np.asarray(X_test.values, dtype=np.float64)
    if isinstance(y_test, pd.Series):
        y_test = np.asarray(y_test.values, dtype=np.float64)

    logger.info(f"Starting GridSearchCV for model: {model_name}")

    model_info = MODEL_REGISTRY[model_name]
    model_class = model_info["class"]
    default_params = model_info["default_params"]

    if param_grid is None:
        param_grid = default_params

    logger.info(f"Parameter grid: {param_grid}")

    estimator = model_class()
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