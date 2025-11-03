# services/portfolio_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select
from services.data_service import get_engine, load_series
from models import Instrument, Forecast

engine = get_engine()

@dataclass
class SimResult:
    x: List[str]
    equity: List[float]
    trades: List[Tuple[str, str, float, float]]  # (ts, side, qty, price)

def _join_actuals_and_forecasts(symbol: str) -> pd.DataFrame:
    """
    Return a DataFrame indexed by timestamp with columns:
      actual, pred
    Only includes rows where both exist.
    """
    prices = load_series(engine, symbol).dropna(subset=["Close"])
    close = prices["Close"]
    last_ts = close.index.max()

    with Session(engine) as s:
        inst_id = s.scalar(select(Instrument.id).where(Instrument.symbol == symbol))
        fcs = s.execute(
            select(Forecast)
            .where(Forecast.instrument_id == inst_id, Forecast.target_ts <= last_ts)
            .order_by(Forecast.target_ts.asc())
        ).scalars().all()

    if not fcs:
        return pd.DataFrame(columns=["actual", "pred"])

    df_pred = pd.DataFrame(
        {"x": [pd.Timestamp(f.target_ts) for f in fcs],
         "pred": [float(f.pred_close) for f in fcs]}
    ).set_index("x").sort_index()

    # exact align by timestamp
    df = pd.DataFrame({"actual": close}).join(df_pred, how="inner")
    return df.dropna(subset=["actual", "pred"])

def backtest_threshold(symbol: str, cash0: float = 10_000.0, thr_pct: float = 0.5) -> SimResult:
    """
    Very simple rule:
      if (pred/actual - 1)*100 > +thr -> go 100% long
      if < -thr -> go 0% (sell all to cash)
      else hold previous state
    """
    df = _join_actuals_and_forecasts(symbol)
    if df.empty:
        return SimResult(x=[], equity=[], trades=[])

    cash = cash0
    qty = 0.0
    trades: List[Tuple[str, str, float, float]] = []
    equity: List[float] = []
    xs: List[str] = []

    for ts, row in df.iterrows():
        px = float(row["actual"])
        ret_pct = (float(row["pred"]) / px - 1.0) * 100.0

        # decide action
        if ret_pct > thr_pct and cash > 0:          # BUY all-in
            qty = cash / px
            cash = 0.0
            trades.append((ts.isoformat(), "BUY", qty, px))
        elif ret_pct < -thr_pct and qty > 0:        # SELL all
            cash = qty * px
            trades.append((ts.isoformat(), "SELL", qty, px))
            qty = 0.0
        # else HOLD

        xs.append(ts.isoformat())
        equity.append(cash + qty * px)

    return SimResult(x=xs, equity=equity, trades=trades)
