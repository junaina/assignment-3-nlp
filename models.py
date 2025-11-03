from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, UniqueConstraint, Index
from datetime import datetime
Base = declarative_base()

class Instrument(Base):
    __tablename__ = "instruments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    asset_class: Mapped[str] = mapped_column(String(16), default="equity")
    prices = relationship("PriceOHLC", back_populates="instrument", cascade="all, delete-orphan")

class PriceOHLC(Base):
    __tablename__ = "prices_ohlc"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), index=True)
    ts: Mapped[DateTime] = mapped_column(DateTime, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    instrument = relationship("Instrument", back_populates="prices")
    __table_args__ = (
        UniqueConstraint("instrument_id", "ts", name="uq_price_instrument_ts"),
        Index("ix_price_instrument_ts", "instrument_id", "ts"),
    )


class Forecast(Base):
    __tablename__ = "forecasts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    horizon_hours: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(64))
    target_ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    pred_close: Mapped[float] = mapped_column(Float)
    metric_rmse: Mapped[float] = mapped_column(Float, default=0.0)
    metric_mae: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (Index("ix_forecast_instrument_created", "instrument_id", "created_at"),)# ---- NEW TABLES FOR ASSIGNMENT 3 ----
class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    # keep symbol directly, consistent with your Forecast table style
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    algo: Mapped[str] = mapped_column(String(32))          # "ARIMA", "LSTM", "Ensemble"
    params: Mapped[str] = mapped_column(String(512))       # simple str/json-encoded params
    train_start: Mapped[datetime] = mapped_column(DateTime)
    train_end: Mapped[datetime] = mapped_column(DateTime)
    note: Mapped[str] = mapped_column(String(256), default="")

class MetricLog(Base):
    __tablename__ = "metric_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    target_ts: Mapped[datetime] = mapped_column(DateTime, index=True)  # timestamp being evaluated
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    mape: Mapped[float] = mapped_column(Float)

class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    side: Mapped[str] = mapped_column(String(4))           # "BUY" / "SELL"
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    cash_after: Mapped[float] = mapped_column(Float)       # portfolio cash after this trade
    holdings_after: Mapped[float] = mapped_column(Float)   # market value of holdings after trade

class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    equity: Mapped[float] = mapped_column(Float)           # cash + holdings value
    returns: Mapped[float] = mapped_column(Float)          # cumulative return (fraction, not %)
    volatility: Mapped[float] = mapped_column(Float)       # rolling std (pick a window later)
    sharpe: Mapped[float] = mapped_column(Float)           # simple annualized estimate
