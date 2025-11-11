# report_generation.py
import logging
import os
import json
from typing import Dict, Any, Optional, List
import pandas as pd
from jinja2 import Template
from xhtml2pdf import pisa
from datetime import datetime

logger = logging.getLogger(__name__)


# HTML Template as string (embedded for portability)
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <title>AutoML Report — {{ project_name }}</title>
#     <style>
#         body { font-family: Arial, sans-serif; margin: 40px; background: #f9f9f9; }
#         h1, h2 { color: #2c3e50; }
#         section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
#         table { width: 100%; border-collapse: collapse; margin: 15px 0; }
#         table, th, td { border: 1px solid #ddd; }
#         th, td { padding: 10px; text-align: left; }
#         th { background-color: #f2f2f2; }
#         img { max-width: 100%; height: auto; margin: 10px 0; }
#         ul { padding-left: 20px; }
#         .metric { font-weight: bold; color: #27ae60; }
#     </style>
# </head>
# <body>
#     <h1>📊 AutoML Experiment Report</h1>
#     <p><strong>Generated:</strong> {{ timestamp }}</p>
#     <p><strong>Project:</strong> {{ project_name }}</p>

#     <!-- 1. Business Understanding -->
#     {% if business_understanding %}
#     <section id="business-understanding">
#         <h2>1. Business Understanding</h2>
#         <p><strong>Goal:</strong> {{ business_understanding.goal }}</p>
#         {% if business_understanding.success_metric %}
#         <p><strong>Success Metric:</strong> {{ business_understanding.success_metric }}</p>
#         {% endif %}
#     </section>
#     {% endif %}

#     <!-- 2. Data Understanding -->
#     {% if data_understanding %}
#     <section id="data-understanding">
#         <h2>2. Data Understanding</h2>
#         <p>Loaded {{ data_understanding.num_rows }} rows and {{ data_understanding.num_cols }} columns.</p>
#         {% if data_understanding.plot_paths %}
#             {% for plot in data_understanding.plot_paths %}
#                 <img src="{{ plot }}" alt="Data distribution">
#             {% endfor %}
#         {% endif %}
#     </section>
#     {% endif %}

#     <!-- 3. Data Preparation -->
#     {% if data_preparation %}
#     <section id="data-preparation">
#         <h2>3. Data Preparation</h2>
#         <ul>
#             {% for step in data_preparation.steps %}
#             <li>{{ step }}</li>
#             {% endfor %}
#         </ul>
#         {% if data_preparation.plot_paths %}
#             {% for plot in data_preparation.plot_paths %}
#                 <img src="{{ plot }}" alt="Preprocessing visualization">
#             {% endfor %}
#         {% endif %}
#     </section>
#     {% endif %}

#     <!-- 4. Modeling -->
#     {% if modeling %}
#     <section id="modeling">
#         <h2>4. Modeling</h2>
#         <p><strong>Best Model:</strong> {{ modeling.model_name }}</p>
#         <p><strong>Best Parameters:</strong></p>
#         <pre>{{ modeling.best_params | tojson(indent=2) }}</pre>
#         {% if modeling.plot_paths %}
#             {% for plot in modeling.plot_paths %}
#                 <img src="{{ plot }}" alt="Model tuning">
#             {% endfor %}
#         {% endif %}
#     </section>
#     {% endif %}

#     <!-- 5. Evaluation -->
#     {% if evaluation %}
#     <section id="evaluation">
#         <h2>5. Evaluation</h2>
#         <table>
#             <tr><th>Metric</th><th>Value</th></tr>
#             {% for name, value in evaluation.metrics.items() %}
#             <tr><td>{{ name }}</td><td class="metric">{{ "%.4f" | format(value) }}</td></tr>
#             {% endfor %}
#         </table>
#         {% if evaluation.plot_paths %}
#             {% for plot in evaluation.plot_paths %}
#                 <img src="{{ plot }}" alt="Model performance">
#             {% endfor %}
#         {% endif %}
#     </section>
#     {% endif %}

#     <!-- 6. Deployment / Recommendations -->
#     {% if deployment %}
#     <section id="deployment">
#         <h2>6. Recommendations</h2>
#         <p>{{ deployment.recommendations }}</p>
#     </section>
#     {% endif %}

# </body>
# </html>
# """


def generate_automl_report(
    results: Dict[str, Any],
    output_dir: str,
    project_name: str = "AutoML Experiment",
    save_pdf: bool = True,
) -> str:
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
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    template = Template(open("./template.html").read())
    html_content = template.render(
        project_name=project_name,
        timestamp=timestamp,
        business_understanding=results.get("business_understanding"),
        data_understanding=results.get("data_understanding"),
        data_preparation=results.get("data_preparation"),
        modeling=results.get("modeling"),
        evaluation=results.get("evaluation"),
        deployment=results.get("deployment"),
    )

    html_path = os.path.join(output_dir, "automl_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML report saved to {html_path}")

    if save_pdf:
        pdf_path = os.path.join(output_dir, "automl_report.pdf")
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                pisa.CreatePDF(f.read(), dest=open(pdf_path, "wb"))
            logger.info(f"PDF report saved to {pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to generate PDF: {e}")

    return html_path