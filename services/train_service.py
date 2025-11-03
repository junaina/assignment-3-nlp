# services/train_service.py
from __future__ import annotations
import pandas as pd
from sqlalchemy.orm import Session

from services.data_service import load_series, get_engine  # <-- use this
from ml.online_updates import one_step_update
from ml.model_registry import register_version

engine = get_engine()   # <-- create a local engine for this module

def periodic_retrain(symbol: str, *, algo: str = "ARIMA", **kwargs) -> dict:
    df: pd.DataFrame = load_series(engine, symbol).dropna(subset=["Close"])
    if df.empty:
        return {"symbol": symbol, "status": "no-data"}

    close = df["Close"]
    yhat, meta = one_step_update(close, algo=algo, **kwargs)

    with Session(engine) as s:
        register_version(
            s,
            symbol=symbol,
            algo=algo,
            params=meta,
            train_start=close.index[0].to_pydatetime(),
            train_end=close.index[-1].to_pydatetime(),
            note="scheduled refit",
        )
        s.commit()

    return {"symbol": symbol, "algo": algo, "yhat": yhat, "meta": meta, "status": "ok"}
