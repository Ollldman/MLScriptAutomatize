import logging
from typing import Dict, Any, Optional
from pytz import timezone
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime


from ModelForge.settings import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class MLResult(Base):
    __tablename__ = "ml_results"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    dataset_name = Column(String)
    metrics = Column(JSON)
    best_params = Column(JSON)
    timestamp = Column(DateTime, default=datetime.now(timezone('UTC')))


engine = create_engine(settings.DB_FOR_MODELS)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def save_results_to_db(
    model_name: str,
    dataset_name: str,
    metrics: Dict[str, Any],
    best_params: Dict[str, Any],
    db_session=None
) -> Optional[int]:
    """
    Save model results to the database.
    """
    if db_session is None:
        db = SessionLocal()
    else:
        db = db_session

    try:
        result = MLResult(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            best_params=best_params
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        logger.info(f"Results saved to DB with ID: {result.id}")
        return result.id # type:ignore
    except Exception as e:
        logger.error(f"Failed to save results to DB: {e}")
        db.rollback()
        return None
    finally:
        if db_session is None:
            db.close()