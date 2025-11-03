import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict
from models import Base, Instrument, PriceOHLC

def get_engine(url: str = "sqlite:///finforecast.db"):
    return create_engine(url, echo=False, future=True)

def init_db(engine):
    Base.metadata.create_all(engine)

def seed_from_csvs(engine, csv_map: Optional[Dict[str, str]] = None):
    """
    csv_map = {"AAPL": "data/AAPL.csv", ...}
    CSV must have columns like: Date/Time, Open, High, Low, Close, [Volume]
    """
    if not csv_map:
        return
    with Session(engine) as s:
        for symbol, path in csv_map.items():
            df = pd.read_csv(path)
            lower = {c.lower(): c for c in df.columns}
            date_col = lower.get("date") or lower.get("time") or list(df.columns)[0]
            o = lower.get("open", "Open"); h = lower.get("high", "High")
            l = lower.get("low", "Low");  c = lower.get("close", "Close")
            v = lower.get("volume", None)

            keep = [date_col, o, h, l, c] + ([v] if v in df.columns and v else [])
            df = df[keep].copy()
            df.columns = ["Date", "Open", "High", "Low", "Close"] + (["Volume"] if v else [])
            df["Date"] = pd.to_datetime(df["Date"])

            inst = s.scalar(select(Instrument).where(Instrument.symbol == symbol))
            if not inst:
                inst = Instrument(symbol=symbol, name=symbol,
                                  asset_class="crypto" if "USD" in symbol else "equity")
                s.add(inst); s.flush()

            existing_ts = set(p.ts for p in inst.prices)
            rows = []
            for r in df.itertuples(index=False):
                ts = pd.Timestamp(getattr(r, "Date")).to_pydatetime()
                if ts in existing_ts:
                    continue
                rows.append(PriceOHLC(
                    instrument_id=inst.id, ts=ts,
                    open=float(getattr(r, "Open")),
                    high=float(getattr(r, "High")),
                    low=float(getattr(r, "Low")),
                    close=float(getattr(r, "Close")),
                    volume=float(getattr(r, "Volume", 0.0))
                ))
            s.add_all(rows)
        s.commit()

def load_series(engine, symbol: str) -> pd.DataFrame:
    with Session(engine) as s:
        inst = s.scalar(select(Instrument).where(Instrument.symbol == symbol))
        if not inst:
            raise ValueError(f"Unknown instrument: {symbol}")
        q = s.execute(
            select(PriceOHLC).where(PriceOHLC.instrument_id == inst.id).order_by(PriceOHLC.ts.asc())
        )
        rows = q.scalars().all()
    df = pd.DataFrame([{
        "Date": r.ts, "Open": r.open, "High": r.high,
        "Low": r.low, "Close": r.close, "Volume": r.volume
    } for r in rows])
    return df.set_index("Date").sort_index()

def detect_frequency_hours(df: pd.DataFrame) -> int:
    if len(df) < 3:
        return 24
    deltas = df.index.to_series().diff().dropna().dt.total_seconds() / 3600.0
    return int(round(deltas.mode().iloc[0])) if not deltas.empty else 24
