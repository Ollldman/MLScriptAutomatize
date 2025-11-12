from typing import Dict, Tuple, Union, Any, Optional
import logging
from pydantic import ValidationError, field_validator

from report.report_data import ReportData

logger = logging.getLogger(__name__)

def validate_report_data(data: Dict[str, Any]) -> Tuple[bool, Optional[ReportData]]:
    """
    Validate the report data structure using Pydantic.

    Args:
        data (Dict[str, Any]): Raw dictionary of report data.

    Returns:
        Tuple[bool, Union[ReportData, Dict[str, Any]]]: 
            - bool: True if data is valid and complete enough for report generation, False otherwise.
            - ReportData: Validated and structured data if valid, else the original dict (or error info).
    """
    try:
        validated_data = ReportData(**data)
        is_valid_for_generation = validated_data.is_complete()
        logger.info(f"Report data validation passed. Ready for generation: {is_valid_for_generation}")
        return is_valid_for_generation, validated_data

    except ValidationError as e:
        logger.error(f"Validation error in report data: {e}")
        return False, None

    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        return False, None