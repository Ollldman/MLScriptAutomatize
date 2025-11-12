import logging
from datetime import datetime
from typing import Dict, Any
from ModelForge.logger_config import setup_logging

logger = logging.getLogger(__name__)


def log_pipeline_start(pipeline_name: str, params: Dict[str, Any]) -> str:
    """Log the start of a pipeline execution."""
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Pipeline '{pipeline_name}' started at {start_time}")
    logger.info(f"Parameters: {params}")
    return start_time


def log_pipeline_step(step_name: str, status: str = "SUCCESS", details: str = "") -> None:
    """Log a specific step in the pipeline."""
    logger.info(f"[{status}] Step: {step_name}. Details: {details}")


def log_pipeline_end(pipeline_name: str, start_time: str, status: str = "SUCCESS") -> None:
    """Log the end of a pipeline execution."""
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Pipeline '{pipeline_name}' ended at {end_time} with status: {status}")
    logger.info(f"Total runtime: {end_time} - {start_time}")