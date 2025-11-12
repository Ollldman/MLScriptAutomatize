# tests/integration_tests/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from ModelForge.integration.api_integration import app, PipelineRequest
from unittest.mock import patch

client = TestClient(app)


def test_run_pipeline_success():
    """
    Тестирует успешный запуск пайплайна через API.
    """
    request_data = {
        "dataset_source": "sklearn",
        "dataset_params": {"name": "load_iris"},
        "target_column": "target",
        "report_recipients": ["test@example.com"],
        "save_to_db": False,
        "send_email": False
    }

    with patch("ModelForge.integration.api_integration.run_full_pipeline") as mock_pipeline:
        mock_pipeline.return_value = "/path/to/report.html"

        response = client.post("/run-pipeline", json=request_data)

        assert response.status_code == 200
        assert response.json() == {"status": "success", "report_path": "/path/to/report.html"}


def test_run_pipeline_failure():
    """
    Тестирует возврат ошибки, если пайплайн завершился с ошибкой.
    """
    request_data = {
        "dataset_source": "sklearn",
        "dataset_params": {"name": "load_iris"},
        "target_column": "target",
        "report_recipients": ["test@example.com"],
        "save_to_db": False,
        "send_email": False
    }

    with patch("ModelForge.integration.api_integration.run_full_pipeline") as mock_pipeline:
        mock_pipeline.return_value = None  # означает, что пайплайн не удался

        response = client.post("/run-pipeline", json=request_data)

        assert response.status_code == 500
        assert response.json() == {"detail": "Pipeline failed."}


def test_run_pipeline_validation_error():
    """
    Тестирует валидацию Pydantic-модели.
    """
    invalid_request_data = {
        "dataset_source": "sklearn",
        # Пропущен dataset_params
        "target_column": "target"
    }

    response = client.post("/run-pipeline", json=invalid_request_data)

    assert response.status_code == 422  # Unprocessable Entity
    # Проверим, что ошибка валидации возвращается
    assert "detail" in response.json()


def test_health_check():
    """
    Тестирует эндпоинт /health.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}