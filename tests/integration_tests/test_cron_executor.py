import pytest
import tempfile
import os
from ModelForge.integration.cron_executor import run_full_pipeline


def test_run_full_pipeline_iris_xgb_classifier():
    """
    Test the full pipeline with the iris dataset and XGBoostClassifier.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = run_full_pipeline(
            dataset_source="sklearn",
            dataset_params={"name": "load_iris"},
            target_column="target",
            report_recipients=['dodrolla@gmail.com'],
            save_to_db=True,       
            send_email=True,       
            output_dir=tmpdir
        )

        # Проверяем, что отчёт был сгенерирован
        assert report_path is not None, "Pipeline failed, report_path is None"
        assert os.path.exists(report_path), f"Report file does not exist: {report_path}"
        assert report_path.endswith(".html"), "Generated report is not an HTML file"

        # Проверим, что файл не пустой
        assert os.path.getsize(report_path) > 0, "Generated report is empty"