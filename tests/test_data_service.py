from services.data_service import load_series, detect_frequency_hours
from sqlalchemy.orm import Session
from sqlalchemy import select
from models import Instrument
import app as app_mod



def test_load_series_returns_df(seeded_app):   # <-- add seeded_app here
    df = load_series(app_mod.engine, "TEST")
    assert not df.empty
    assert {"Open","High","Low","Close"}.issubset(df.columns)

def test_detect_frequency_hours_daily(seeded_app):  # <-- and here
    df = load_series(app_mod.engine, "TEST")
    h = detect_frequency_hours(df)
    assert h in (24, 23, 25)
