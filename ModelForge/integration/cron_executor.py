import logging
import os
from typing import Dict, Any, Optional

from ModelForge.modules.data_loading_to_dataFrame.load_from_api import load_dataset
from ModelForge.modules.data_validation.handle_missing_values import handle_missing_values
from ModelForge.modules.data_validation.handle_outliers import handle_outliers
from ModelForge.modules.data_analysis.gboost_models import (
    xgb_run_regression_grid_search,
    xgb_run_classification_grid_search
)
from ModelForge.modules.data_analysis.data_statistics import compute_descriptive_statistics
from ModelForge.modules.report.report_generator import generate_automl_report
from ModelForge.modules.report.send_report_by_email import send_report_via_email

from ModelForge.integration.db_storage import save_results_to_db

from ModelForge.settings import settings

logger = logging.getLogger(__name__)


def run_full_pipeline(
    dataset_source: str,
    dataset_params: Dict[str, Any],
    target_column: str,
    report_recipients: Optional[list[str]] = None,
    save_to_db: bool = False,
    send_email: bool = True,
    output_dir: str = settings.reports_dir
) -> Optional[str]:
    """
    Executes a full ML pipeline: load -> clean -> model -> report.

    Args:
        dataset_source: Source for load_dataset (e.g., 'kaggle', 'sklearn', etc.)
        dataset_params: Parameters for the loader (e.g., {'dataset_id': ..., 'filename': ...})
        target_column: Name of the target column for modeling.
        report_recipients: List of emails to send the report to.
        save_to_db: If True, saves results to DB (to be implemented).
        send_email: If True, sends the generated report via email.
        output_dir: Directory to save the report.

    Returns:
        Path to the generated report, or None if failed.
    """
    try:
        logger.info("Starting full pipeline execution...")

        # 1. Load data
        logger.info("Step 1: Loading dataset")
        df = load_dataset(source=dataset_source, **dataset_params)
        if df is None:
            logger.error("Failed to load dataset.")
            return None

        logger.info(f"Dataset loaded with shape: {df.shape}")

        # 2. Compute statistics
        logger.info("Step 2: Compute statistics")
        stats = compute_descriptive_statistics(df)
        logger.info("Descriptive statistics computed.")

        # 3. Handle missing values
        logger.info("Step 3: Handle missing values")
        df = handle_missing_values(df, numeric_strategy="median", categorical_strategy="most_frequent")
        logger.info("Missing values handled.")

        # 4. Handle outliers (optional step)
        logger.info("Step 4: Handle outliers (optional step)")
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if target_column in numeric_cols:
            df = handle_outliers(df, column=target_column, method="iqr", handling="clip")
            logger.info("Outliers handled in target column.")

        # 5. Prepare features and target
        logger.info("Step 5: Prepare features and target")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Determine task type (regression/classification)
        logger.info("Step 6: Determine task type (regression/classification)")
        if y.dtype in ['object', 'category'] or y.nunique() <= 20:
            logger.info("Running classification pipeline...")
            model_results = xgb_run_classification_grid_search(
                X_train.values, y_train.values, X_test.values, y_test.values
            )
        else:
            logger.info("Running regression pipeline...")
            model_results = xgb_run_regression_grid_search(
                X_train.values, y_train.values, X_test.values, y_test.values
            )

        # 7. Prepare report data
        logger.info("Step 7: Prepare report data")
        report_data = {
            "data_understanding": {"num_rows": len(df), "num_cols": len(df.columns)},
            "modeling": {
                "model_name": model_results["best_estimator"].__class__.__name__,
                "best_params": model_results["best_params"]
            },
            "evaluation": {"metrics": model_results["test_metrics"]}
        }

        # 8. Generate report
        logger.info("Step 8: Generate report")
        report_path = generate_automl_report(
            results=report_data,
            output_dir=output_dir,
            project_name=f"AutoML Report - {dataset_source}",
            save_pdf=True
        )

        if report_path and send_email and report_recipients:
            success = send_report_via_email(
                recipient_emails=report_recipients,
                subject="AutoML Report Generated",
                body="<h1>Report is attached.</h1>",
                attachment_paths=[report_path]
            )
            if success:
                logger.info("Report sent via email successfully.")
            else:
                logger.error("Failed to send report via email.")

        # 9. Save to DB (to be implemented)
        logger.info("Step 9: Save to DB (to be implemented)")
        if save_to_db:
            logger.info("Saving results to DB...")
            save_results_to_db(
                model_name=model_results["best_estimator"].__class__.__name__,
                dataset_name="iris",  # или как-то динамически
                metrics=model_results["test_metrics"],
                best_params=model_results["best_params"]
            )

        logger.info("Pipeline completed successfully.")
        return report_path

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return None