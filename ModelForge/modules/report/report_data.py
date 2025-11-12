from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict
from typing import Literal, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# Pydantic Model for Report Data
class ReportData(BaseModel):
    """
    Pydantic model for validating report generation data following CRISP-DM structure.
    All fields are optional to allow incremental filling.
    """
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    project_name: Optional[str] = "AutoML Experiment"
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 1. Business Understanding
    business_understanding: Optional[Dict[str, str]] = None  # e.g. {"goal": "...", "success_metric": "..."}

    # 2. Data Understanding
    data_understanding: Optional[Dict[str, Any]] = None  # e.g. {"num_rows": int, "num_cols": int, "plot_paths": [str]}

    # 3. Data Preparation
    data_preparation: Optional[Dict[str, Any]] = None  # e.g. {"steps": [str], "plot_paths": [str]}

    # 4. Modeling
    modeling: Optional[Dict[str, Any]] = None  # e.g. {"model_name": str, "best_params": dict, "plot_paths": [str]}

    # 5. Evaluation
    evaluation: Optional[Dict[str, Any]] = None  # e.g. {"metrics": dict, "plot_paths": [str]}

    # 6. Deployment / Recommendations
    deployment: Optional[Dict[str, str]] = None  # e.g. {"recommendations": str}


    def is_complete(self) -> bool:
        """
        Check if the report has minimum required data to be considered complete for generation.
        This is a basic check — you can make it more sophisticated based on your needs.
        For example, you might require 'evaluation' and 'modeling' to be present.
        """
        # Example: Require that modeling and evaluation sections are filled
        # You can adjust this logic based on your business rules
        required_sections = ["modeling", "evaluation"]
        for section in required_sections:
            if getattr(self, section) is None:
                logger.warning(f"Required section '{section}' is missing for a complete report.")
                return False
        return True