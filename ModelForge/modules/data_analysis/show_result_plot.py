import logging
import os
from typing import Dict, Any, Optional, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ModelForge.settings import settings

logger = logging.getLogger(__name__)


# Set global style
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'savefig.dpi': 150,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def plot_scatter_matrix_with_histograms(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    plot_kws: Optional[dict] = None,
    diag_kws: Optional[dict] = None,
    hue_column: Optional[str] = None,  # <-- Новый параметр
) -> None:
    """
    Plot a scatter matrix for numeric features with histograms on the diagonal using seaborn.pairplot.

    This function visualizes pairwise relationships between numeric columns.
    - Off-diagonal: scatter plots
    - Diagonal: histograms of feature distributions
    - If 'hue_column' is provided, points are colored by the class in that column (for classification).

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_columns (List[str], optional): Subset of numeric columns to plot.
            If None, all numeric columns are used.
        output_path (str, optional): Path to save the plot (e.g., 'plots/scatter_matrix.png').
            If None, the plot is not saved.
        figsize (tuple, optional): Figure size (width, height). seaborn will auto-scale,
            but you can override the default aspect ratio by setting plt.figure_size before calling.
        plot_kws (dict, optional): Keyword arguments for off-diagonal plots (e.g. {'alpha': 0.6}).
        diag_kws (dict, optional): Keyword arguments for diagonal plots (e.g. {'bins': 30}).
        hue_column (str, optional): Name of the column to use for coloring points (e.g., target class).
            If provided, it should be present in the DataFrame and suitable for classification (categorical/discrete).

    Returns:
        None

    Raises:
        ValueError: If no numeric columns are found or if hue_column is specified but missing.
        TypeError: If input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        # Validate that specified columns exist and are numeric
        missing = [col for col in numeric_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        non_numeric = [col for col in numeric_columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            logger.warning(f"Non-numeric columns will be skipped: {non_numeric}")
            numeric_columns = [col for col in numeric_columns if col not in non_numeric]

    if not numeric_columns:
        raise ValueError("No numeric columns available for scatter matrix.")

    # Validate hue_column if provided
    if hue_column is not None:
        if hue_column not in df.columns:
            raise ValueError(f"Hue column '{hue_column}' not found in DataFrame.")
        logger.info(f"Using '{hue_column}' for coloring points.")

    logger.info(f"Plotting scatter matrix for columns: {numeric_columns}")

    if figsize:
        plt.figure(figsize=figsize)

    # Use seaborn pairplot with optional hue
    g = sns.pairplot(
        df,
        vars=numeric_columns,  # Specify which variables to plot
        hue=hue_column,        # <-- This is the key addition
        diag_kind="hist",
        plot_kws=plot_kws or {},
        diag_kws=diag_kws or {}
    )

    if output_path:
        g.savefig(output_path, bbox_inches="tight")
        logger.info(f"Scatter matrix saved to {output_path}")
    else:
        plt.show()

    plt.close(g.figure)  # Close the figure to free memory


def plot_descriptive_statistics(
    stats_dict: Dict[str, Dict[str, Any]],
    output_dir: Optional[str] = None,
    save_format: str = "png",
) -> List[str]:
    """
    Generate boxplots or barplots for descriptive statistics (mean, median, std).

    Args:
        stats_dict (Dict[str, Dict]): Output from compute_descriptive_statistics.
        output_dir (str, optional): Directory to save plots. If None, plots are not saved.
        save_format (str): Format to save plots ('png', 'svg', etc.).

    Returns:
        List[str]: List of saved plot file paths (empty if not saved).
    """
    if not stats_dict:
        logger.warning("No statistics provided for plotting.")
        return []

    metrics = ["mean", "median", "std"]
    data = []
    for col, stats in stats_dict.items():
        for metric in metrics:
            if metric in stats:
                data.append({"Column": col, "Metric": metric, "Value": stats[metric]})

    if not data:
        logger.warning("No plottable metrics found in statistics.")
        return []

    df_plot = pd.DataFrame(data)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Column", y="Value", hue="Metric")
    plt.title("Descriptive Statistics by Column")
    plt.xticks(rotation=45)
    plt.tight_layout()

    saved_paths = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"descriptive_stats.{save_format}")
        plt.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
        logger.info(f"Descriptive statistics plot saved to {path}")

    plt.close()
    return saved_paths


def plot_time_series_decomposition(
    decomposition_result: Dict[str, Any],
    output_dir: Optional[str] = None,
    save_format: str = "png",
) -> List[str]:
    """
    Plot trend, seasonal, and residual components from seasonal decomposition.

    Args:
        decomposition_result (Dict): Result from analyze_time_series['decomposition'].
        output_dir (str, optional): Directory to save plots.
        save_format (str): Image format for saving.

    Returns:
        List[str]: List of saved plot file paths.
    """
    trend = decomposition_result.get("trend")
    seasonal = decomposition_result.get("seasonal")
    resid = decomposition_result.get("resid")

    if not (trend and seasonal and resid):
        logger.warning("Decomposition data incomplete. Skipping plot.")
        return []

    n = len(trend)
    time = range(n)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Original (reconstructed)
    original = np.array(trend) + np.array(seasonal) + np.array(resid)
    axes[0].plot(time, original, label="Reconstructed", color="black")
    axes[0].set_title("Original (Reconstructed)")
    axes[0].legend()

    axes[1].plot(time, trend, label="Trend", color="tab:blue")
    axes[1].set_title("Trend")
    axes[1].legend()

    axes[2].plot(time, seasonal, label="Seasonal", color="tab:orange")
    axes[2].set_title("Seasonal")
    axes[2].legend()

    axes[3].scatter(time, resid, label="Residuals", color="tab:red", s=5)
    axes[3].set_title("Residuals")
    axes[3].legend()

    plt.xlabel("Time")
    plt.tight_layout()

    saved_paths = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"time_series_decomposition.{save_format}")
        plt.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
        logger.info(f"Decomposition plot saved to {path}")

    plt.close()
    return saved_paths


def plot_grid_search_results(
    plot_data: Dict[str, Any],
    model_name: str,
    output_dir: Optional[str] = None,
    save_format: str = "png",
) -> List[str]:
    """
    Plot mean test scores with error bars from GridSearchCV results.

    Args:
        plot_data (Dict): From model_learning result['plot_data'].
        model_name (str): Name of the model (e.g., 'XGBRegressor').
        output_dir (str, optional): Directory to save plot.
        save_format (str): Image format.

    Returns:
        List[str]: List of saved plot file paths.
    """
    mean_scores = plot_data.get("mean_test_score")
    std_scores = plot_data.get("std_test_score")

    if mean_scores is None:
        logger.warning("No mean test scores found in plot_data.")
        return []

    x = range(len(mean_scores))
    plt.figure()
    plt.errorbar(x, mean_scores, yerr=std_scores, fmt='o-', capsize=5)
    plt.title(f"GridSearchCV Results — {model_name}")
    plt.xlabel("Parameter Combination Index")
    plt.ylabel("Mean Cross-Validation Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    saved_paths = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"grid_search_{model_name}.{save_format}")
        plt.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
        logger.info(f"GridSearch plot saved to {path}")
    else:
        plt.show()

    plt.close()
    return saved_paths


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Optional[str] = None,
    save_format: str = "png",
) -> List[str]:
    """
    Plot residuals vs predicted values and Q-Q plot.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted values.
        model_name (str): Model name for labeling.
        output_dir (str, optional): Directory to save plots.
        save_format (str): Image format.

    Returns:
        List[str]: List of saved plot file paths.
    """
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.7)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Predicted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title(f"Residuals vs Predicted — {model_name}")

    # Histogram of residuals
    ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residuals Distribution")

    plt.tight_layout()

    saved_paths = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"residuals_{model_name}.{save_format}")
        plt.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
        logger.info(f"Residuals plot saved to {path}")

    plt.close()
    return saved_paths


def generate_all_plots(
    analysis_results: Dict[str, Any],
    output_dir: str,
    save_format: str = "png",
) -> Dict[str, List[str]]:
    """
    High-level function to generate all relevant plots from a full pipeline result.

    Expected structure of analysis_results:
    {
        "statistics": {...},               # from compute_descriptive_statistics
        "trend_analysis": {...},          # from analyze_time_series
        "model_results": {...}            # from run_regression/classification_grid_search
    }

    Args:
        analysis_results (Dict): Combined results from all analysis stages.
        output_dir (str): Directory to save all plots.
        save_format (str): Image format.

    Returns:
        Dict[str, List[str]]: Mapping of plot type to saved file paths.
    """
    all_paths = {
        "descriptive_statistics": [],
        "time_series_decomposition": [],
        "grid_search": [],
        "residuals": []
    }

    # 1. Descriptive stats
    if "statistics" in analysis_results:
        all_paths["descriptive_statistics"] = plot_descriptive_statistics(
            analysis_results["statistics"], output_dir, save_format
        )

    # 2. Time series decomposition
    if "trend_analysis" in analysis_results and "decomposition" in analysis_results["trend_analysis"]:
        all_paths["time_series_decomposition"] = plot_time_series_decomposition(
            analysis_results["trend_analysis"]["decomposition"], output_dir, save_format
        )

    # 3. GridSearch results
    if "model_results" in analysis_results and "plot_data" in analysis_results["model_results"]:
        model_name = analysis_results["model_results"].get("model_name", "Model")
        all_paths["grid_search"] = plot_grid_search_results(
            analysis_results["model_results"]["plot_data"], model_name, output_dir, save_format
        )

    # 4. Residuals (only for regression)
    if (
        "model_results" in analysis_results
        and "y_true" in analysis_results["model_results"]
        and "y_pred" in analysis_results["model_results"]
    ):
        all_paths["residuals"] = plot_residuals(
            analysis_results["model_results"]["y_true"],
            analysis_results["model_results"]["y_pred"],
            model_name,  # type:ignore
            output_dir,
            save_format
        )

    logger.info("All visualizations completed.")
    return all_paths