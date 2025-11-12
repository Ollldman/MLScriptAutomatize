import logging
import os
import json
from typing import Dict, Any, Optional, List
from git import Repo
import pandas as pd
from jinja2 import Template
from xhtml2pdf import pisa
from datetime import datetime
from ModelForge.settings import settings
from ModelForge.modules.report.report_data import ReportData
from ModelForge.modules.report.validate_report_data import validate_report_data

logger = logging.getLogger(__name__)


def generate_automl_report(
    results: Dict[str, Any],
    output_dir: str = settings.reports_dir,
    project_name: str = "AutoML Experiment",
    save_pdf: bool = True,
) -> Optional[str]:
    """
    Generate a CRISP-DM compliant HTML report (with optional PDF) from pipeline results.

    The function dynamically includes only the sections that are present in `results`.

    Expected structure of `results` (all keys are optional):
    {
        "business_understanding": {"goal": "...", "success_metric": "..."},
        "data_understanding": {"num_rows": int, "num_cols": int, "plot_paths": [str]},
        "data_preparation": {"steps": [str], "plot_paths": [str]},
        "modeling": {"model_name": str, "best_params": dict, "plot_paths": [str]},
        "evaluation": {"metrics": dict, "plot_paths": [str]},
        "deployment": {"recommendations": str}
    }

    Args:
        results (Dict[str, Any]): Aggregated results from all pipeline stages.
        output_dir (str): Directory to save HTML and PDF reports.
        project_name (str): Project name for the report header.
        save_pdf (bool): If True, also generate a PDF version.

    Returns:
        str: Path to the generated HTML report.
    """
    is_valid, validated_data_obj = validate_report_data(results)

    if not is_valid:
        logger.error("Report data is not valid or complete. Report generation aborted.")
        return None
    if validated_data_obj:
        validated_data: ReportData = validated_data_obj
    else:
        logger.error("Validated data object is None")
        return None



    os.makedirs(output_dir, exist_ok=True)

    # Use the validated data's timestamp and project name, or defaults
    # Generate a timestamp for the filename (without seconds/microseconds)
    filename_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Use the project name for the report, sanitized for use in filenames
    sanitized_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    if not sanitized_project_name:
        sanitized_project_name = "Report"

    # Generate filenames with timestamp
    html_filename = f"automl_report_{sanitized_project_name}_{filename_timestamp}.html"
    pdf_filename = f"automl_report_{sanitized_project_name}_{filename_timestamp}.pdf"

    html_path = os.path.join(output_dir, html_filename)

    # Use the existing timestamp from validated_data for the *content* of the report
    timestamp = validated_data.timestamp
    # объединяем, получаем абсолютный путь до темплейта
    template_path = os.path.join(os.path.dirname(__file__), "template.html")

    template = Template(open(template_path).read())
    html_content = template.render(
        project_name=project_name,
        timestamp=timestamp,
        business_understanding=validated_data.business_understanding,
        data_understanding=validated_data.data_understanding,
        data_preparation=validated_data.data_preparation,
        modeling=validated_data.modeling,
        evaluation=validated_data.evaluation,
        deployment=validated_data.deployment,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML report saved to {html_path}")

    if save_pdf:
        pdf_path = os.path.join(output_dir, pdf_filename)
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                pisa.CreatePDF(f.read(), dest=open(pdf_path, "wb"))
            logger.info(f"PDF report saved to {pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to generate PDF: {e}")

    return html_path