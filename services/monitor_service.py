# services/monitor_service.py
from __future__ import annotations
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import Dict, List
from models import MetricLog
from services.data_service import get_engine

engine = get_engine()

def get_metric_series(symbol: str, limit: int = 500) -> Dict[str, List]:
    """
    Return time series of MAE/RMSE/MAPE for a symbol, newest last (for plotting).
    """
    with Session(engine) as s:
        rows = (
            s.execute(
                select(MetricLog)
                .where(MetricLog.symbol == symbol)
                .order_by(MetricLog.target_ts.asc())
            )
            .scalars()
            .all()
        )

    if not rows:
        return {"x": [], "mae": [], "rmse": [], "mape": []}

    # limit to last N points
    rows = rows[-limit:]

    return {
        "x": [r.target_ts.isoformat() for r in rows],
        "mae": [r.mae for r in rows],
        "rmse": [r.rmse for r in rows],
        "mape": [r.mape for r in rows],
    }
