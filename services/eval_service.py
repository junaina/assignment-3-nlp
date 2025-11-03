# services/eval_service.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select

from services.data_service import load_series, get_engine  # <-- use this
from models import Instrument, Forecast, MetricLog, Base

engine = get_engine()  # <-- local engine

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return mae, rmse, mape

def evaluate_new_points(symbol: str, *, limit_pairs: int = 500) -> int:
    df = load_series(engine, symbol).dropna(subset=["Close"])
    if df.empty:
        return 0
    close = df["Close"]; last_actual_ts = close.index.max()

    with Session(engine) as s:
        inst_id = s.scalar(select(Instrument.id).where(Instrument.symbol == symbol))
        if inst_id is None:
            return 0

        q = s.execute(
            select(Forecast)
            .where(Forecast.instrument_id == inst_id, Forecast.target_ts <= last_actual_ts)
            .order_by(Forecast.target_ts.asc())
        ).scalars().all()
        if not q:
            return 0

        y_true, y_pred, tss = [], [], []
        actual_map = close.to_dict()
        for f in q[-limit_pairs:]:
            ts = pd.Timestamp(f.target_ts)
            if ts in actual_map:
                y_true.append(float(actual_map[ts]))
                y_pred.append(float(f.pred_close))
                tss.append(ts)
        if not tss:
            return 0

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mae, rmse, mape = _metrics(y_true, y_pred)

        s.add(MetricLog(symbol=symbol, target_ts=max(tss).to_pydatetime(),
                        mae=mae, rmse=rmse, mape=mape))
        s.commit()
        return len(tss)
