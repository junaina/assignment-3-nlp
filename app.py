# app.py (clean)

import argparse
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sqlalchemy.orm import Session
from sqlalchemy import select

from services.data_service import (
    get_engine, init_db, seed_from_csvs, load_series, detect_frequency_hours
)
from models import Instrument, Forecast, Base
from ml.arima_model import arima_forecast

# Optional services/pages
from services.train_service import periodic_retrain
from services.eval_service import evaluate_new_points
from services.scheduler import start_scheduler
from services.monitor_service import get_metric_series
from services.portfolio_service import backtest_threshold

# ----- APP/DB INIT (only once) -----
app = Flask(__name__)
engine = get_engine()
Base.metadata.create_all(bind=engine)

# Start background jobs (you can comment this out during debugging if you want)
SCHED = start_scheduler(symbols=["AAPL", "MSFT", "BTC-USD"])

# Try to enable LSTM; fall back gracefully if not available
try:
    from ml.lstm_model import lstm_forecast
    HAS_LSTM = True
except Exception:
    HAS_LSTM = False


def ensure_seed():
    """Seed DB from local CSVs once."""
    csv_map = {
        "AAPL": "data/AAPL.csv",
        "MSFT": "data/MSFT.csv",
        "BTC-USD": "data/BTC-USD.csv",
    }
    with Session(engine) as s:
        has_any = s.query(Instrument).count() > 0
    if not has_any:
        seed_from_csvs(engine, csv_map)


@app.route("/")
def index():
    with Session(engine) as s:
        symbols = [x[0] for x in s.execute(select(Instrument.symbol)).all()]
    if not symbols:
        ensure_seed()
        with Session(engine) as s:
            symbols = [x[0] for x in s.execute(select(Instrument.symbol)).all()]
    return render_template("index.html", symbols=symbols, default_symbol=(symbols[0] if symbols else ""))


@app.route("/chart")
def chart():
    symbol = request.args.get("symbol", "")
    horizon_h = int(request.args.get("horizon", "24") or "24")
    if not symbol:
        return "Missing symbol", 400

    # Load & clean
    df = load_series(engine, symbol)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if df.empty:
        return f"No data for {symbol}", 404

    # Horizon -> steps
    freq_h = detect_frequency_hours(df)
    steps = max(1, horizon_h // max(1, freq_h))

    # Validation split
    split = max(1, int(len(df) * 0.85))
    train_close = df["Close"].iloc[:split]
    test_close = df["Close"].iloc[split:]

    # ---- ARIMA validation ----
    ar_val, ar_meta = arima_forecast(train_close, steps=len(test_close))
    ar_rmse = float(np.sqrt(np.mean((test_close.values[:len(ar_val)] - ar_val) ** 2))) if len(test_close) else None
    ar_mae  = float(np.mean(np.abs(test_close.values[:len(ar_val)] - ar_val)))       if len(test_close) else None

    # ---- LSTM validation (optional) ----
    ls_rmse = ls_mae = None
    ls_meta = {}
    can_use_lstm = HAS_LSTM and len(test_close) > 0
    if can_use_lstm:
        try:
            ls_val, ls_meta = lstm_forecast(train_close, steps=len(test_close))
            ls_rmse = float(np.sqrt(np.mean((test_close.values[:len(ls_val)] - ls_val) ** 2)))
            ls_mae  = float(np.mean(np.abs(test_close.values[:len(ls_val)] - ls_val)))
        except Exception as e:
            print("LSTM failed, falling back to ARIMA:", e)
            can_use_lstm = False

    # ---- Pick best by RMSE ----
    use_lstm = can_use_lstm and (ls_rmse is not None) and (ar_rmse is not None) and (ls_rmse < ar_rmse)
    if use_lstm:
        future_pred, _ = lstm_forecast(df["Close"], steps=steps)
        metrics = {"rmse": ls_rmse, "mae": ls_mae, "model": f"LSTM(L={ls_meta.get('lookback', 30)})"}
    else:
        future_pred, _ = arima_forecast(df["Close"], steps=steps)
        metrics = {"rmse": ar_rmse, "mae": ar_mae, "model": f"ARIMA{ar_meta.get('order')}"}

    # ---- Plotly data ----
    hist = {
        "x": [ts.isoformat() for ts in df.index],
        "open": df["Open"].tolist(),
        "high": df["High"].tolist(),
        "low":  df["Low"].tolist(),
        "close": df["Close"].tolist(),
    }
    last_ts = df.index[-1]
    future_index = [last_ts + (i + 1) * pd.Timedelta(hours=freq_h) for i in range(steps)]
    forecast = {"x": [ts.isoformat() for ts in future_index],
                "y": [float(v) for v in future_pred]}

    # ---- Persist forecasts (future) ----
    with Session(engine) as s:
        inst_id = s.scalar(select(Instrument.id).where(Instrument.symbol == symbol))
        s.add_all([
            Forecast(
                instrument_id=inst_id,
                horizon_hours=horizon_h,
                model_name=metrics["model"],
                target_ts=pd.Timestamp(ts).to_pydatetime(),
                pred_close=float(yhat),
                metric_rmse=float(metrics["rmse"]) if metrics["rmse"] is not None else 0.0,
                metric_mae=float(metrics["mae"]) if metrics["mae"] is not None else 0.0,
            )
            for ts, yhat in zip(future_index, future_pred)
        ])
        s.commit()

    # Debug sizes
    print("HIST lens:", len(hist["x"]), len(hist["open"]), len(hist["high"]), len(hist["low"]), len(hist["close"]))
    print("FC lens:", len(forecast["x"]), len(forecast["y"]), "| model:", metrics["model"])

    # ---- Error overlay from DB predictions that have actuals ----
    actual_map = {pd.Timestamp(ts): float(c) for ts, c in zip(df.index, df["Close"])}
    with Session(engine) as s:
        inst_id = s.scalar(select(Instrument.id).where(Instrument.symbol == symbol))
        past_fc = s.execute(
            select(Forecast)
            .where(Forecast.instrument_id == inst_id, Forecast.target_ts <= df.index.max())
            .order_by(Forecast.target_ts.asc())
        ).scalars().all()

    err_x, err_pct = [], []
    for f in past_fc:
        ts = pd.Timestamp(f.target_ts)
        if ts in actual_map:
            actual = actual_map[ts]
            pred = float(f.pred_close)
            pct = ((actual - pred) / abs(actual)) * 100.0 if actual != 0 else 0.0
            err_x.append(ts.isoformat())
            err_pct.append(float(pct))
    error = {"x": err_x, "pct": err_pct}

    # ---- Symbols for toolbar ----
    with Session(engine) as s:
        symbols_list = [x[0] for x in s.execute(select(Instrument.symbol)).all()]

    return render_template(
        "chart.html",
        symbol=symbol,
        horizon=horizon_h,
        hist=hist,
        forecast=forecast,
        metrics=metrics,
        error=error,
        symbols=symbols_list,
    )


@app.route("/api/forecasts/<symbol>")
def api_forecasts(symbol):
    with Session(engine) as s:
        inst = s.scalar(select(Instrument).where(Instrument.symbol == symbol))
        if not inst:
            return {"error": "unknown symbol"}, 404
        q = s.execute(
            select(Forecast)
            .where(Forecast.instrument_id == inst.id)
            .order_by(Forecast.created_at.desc())
            .limit(10)
        )
        items = [{
            "created_at": f.created_at.isoformat(),
            "horizon_h": f.horizon_hours,
            "model": f.model_name,
            "target_ts": f.target_ts.isoformat(),
            "pred_close": f.pred_close,
            "rmse": f.metric_rmse,
            "mae": f.metric_mae,
        } for f in q.scalars().all()]
    return {"symbol": symbol, "items": items}


@app.get("/admin/retrain/<symbol>")
def admin_retrain(symbol):
    out = periodic_retrain(symbol, algo="ARIMA", order=(5,1,0))
    return jsonify(out), 200

@app.get("/admin/evaluate/<symbol>")
def admin_evaluate(symbol):
    n = evaluate_new_points(symbol)
    return jsonify({"symbol": symbol, "evaluated_pairs": n}), 200

@app.get("/admin/backfill_forecasts/<symbol>")
def admin_backfill(symbol):
    """Walk forward through history: at each time t, forecast t+1 and store it."""
    df = load_series(engine, symbol).dropna(subset=["Close"])
    if df.empty or len(df) < 40:
        return jsonify({"error": "not enough data"}), 400

    lookback = 60
    saved = 0
    with Session(engine) as s:
        inst_id = s.scalar(select(Instrument.id).where(Instrument.symbol == symbol))
        if not inst_id:
            return jsonify({"error": "unknown symbol"}), 404

        close = df["Close"]
        for i in range(lookback, len(close) - 1):
            train = close.iloc[i - lookback:i]
            fc, meta = arima_forecast(train, steps=1, order=(5,1,0))
            target_ts = close.index[i + 1]
            s.add(Forecast(
                instrument_id=inst_id,
                horizon_hours=1,
                model_name=f"ARIMA{meta.get('order')}",
                target_ts=pd.Timestamp(target_ts).to_pydatetime(),
                pred_close=float(fc[-1]),
                metric_rmse=0.0,
                metric_mae=0.0,
            ))
            saved += 1
        s.commit()
    return jsonify({"symbol": symbol, "saved_forecasts": saved})

# -------- Monitor & Portfolio --------
@app.get("/api/metrics")
def api_metrics():
    symbol = request.args.get("symbol", "")
    if not symbol:
        return jsonify({"error": "missing symbol"}), 400
    return jsonify(get_metric_series(symbol))

@app.get("/monitor")
def monitor():
    with Session(engine) as s:
        symbols = [x[0] for x in s.execute(select(Instrument.symbol)).all()]
    default_symbol = symbols[0] if symbols else ""
    return render_template("monitor.html", default_symbol=default_symbol, symbols=symbols)

@app.get("/api/portfolio/backtest")
def api_portfolio_backtest():
    symbol = request.args.get("symbol", "")
    thr = float(request.args.get("thr", "0.5") or "0.5")
    if not symbol:
        return jsonify({"error": "missing symbol"}), 400
    res = backtest_threshold(symbol, cash0=10_000.0, thr_pct=thr)
    return jsonify({"x": res.x, "equity": res.equity, "trades": res.trades})

@app.get("/portfolio")
def portfolio():
    with Session(engine) as s:
        symbols = [x[0] for x in s.execute(select(Instrument.symbol)).all()]
    default_symbol = symbols[0] if symbols else ""
    return render_template("portfolio.html", symbols=symbols, default_symbol=default_symbol)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ----- MAIN -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_db(engine)
    if args.init_db:
        ensure_seed()
    app.run(debug=args.debug)
