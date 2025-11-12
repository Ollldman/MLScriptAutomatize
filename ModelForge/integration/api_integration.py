from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from ModelForge.integration.cron_executor import run_full_pipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="ModelForge API", version="1.0.0")


class PipelineRequest(BaseModel):
    dataset_source: str
    dataset_params: Dict[str, Any]
    target_column: str
    report_recipients: Optional[list[str]] = None
    save_to_db: bool = False
    send_email: bool = True


@app.post("/run-pipeline")
def execute_pipeline(request: PipelineRequest):
    try:
        report_path = run_full_pipeline(
            dataset_source=request.dataset_source,
            dataset_params=request.dataset_params,
            target_column=request.target_column,
            report_recipients=request.report_recipients,
            save_to_db=request.save_to_db,
            send_email=request.send_email
        )
        if report_path:
            return {"status": "success", "report_path": report_path}
        else:
            raise HTTPException(status_code=500, detail="Pipeline failed.")
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail="Pipeline failed.")


@app.get("/health")
def health_check():
    return {"status": "ok"}