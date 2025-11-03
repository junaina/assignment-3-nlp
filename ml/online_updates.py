# ml/online_updates.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import pandas as pd
from ml.arima_model import arima_forecast

_HAS_LSTM = False
try:
    from ml.lstm_model import lstm_forecast
    _HAS_LSTM = True
except Exception:
    _HAS_LSTM = False

def refit_arima(close_series: pd.Series, *, order=(5,1,0)) -> Tuple[float, Dict[str, Any]]:
    if len(close_series) < sum(order) + 5:
        last = float(close_series.iloc[-1])
        return last, {"fallback": "naive", "order": order}
    fc, meta = arima_forecast(close_series, steps=1, order=order)
    yhat = float(fc[-1])
    return yhat, {"order": meta.get("order", order), "aic": meta.get("aic")}

def refit_lstm(close_series: pd.Series, *, lookback=30) -> Tuple[float, Dict[str, Any]]:
    if not _HAS_LSTM:
        raise RuntimeError("LSTM not available")
    if len(close_series) <= lookback + 5:
        last = float(close_series.iloc[-1])
        return last, {"fallback": "naive", "lookback": lookback}
    fc, meta = lstm_forecast(close_series, steps=1, lookback=lookback)
    return float(fc[-1]), {"lookback": meta.get("lookback", lookback), **{k:v for k,v in meta.items() if k!="lookback"}}

def one_step_update(close_series: pd.Series, *, algo="ARIMA", **kwargs) -> Tuple[float, Dict[str, Any]]:
    algo = (algo or "ARIMA").upper()
    if algo == "ARIMA":
        return refit_arima(close_series, **kwargs)
    if algo == "LSTM":
        return refit_lstm(close_series, **kwargs)
    raise ValueError(f"Unsupported algo: {algo}")
