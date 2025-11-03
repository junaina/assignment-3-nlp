import os
import io
import pandas as pd
import pytest

# Import your app module and helpers
import app as app_mod
from services.data_service import get_engine, init_db, seed_from_csvs
from sqlalchemy.orm import Session
from sqlalchemy import select
from models import Instrument

@pytest.fixture(scope="session")
def tmp_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("db") / "test_finforecast.db"

@pytest.fixture(scope="session")
def seeded_app(tmp_db_path):
    # 1) point the app to a test DB
    test_engine = get_engine(f"sqlite:///{tmp_db_path}")
    app_mod.engine = test_engine  # override the global engine used by routes
    init_db(test_engine)

    # 2) make a tiny in-memory CSV for symbol TEST
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "Open": 100 + pd.Series(range(60)).astype(float),
        "High": 101 + pd.Series(range(60)).astype(float),
        "Low":  99  + pd.Series(range(60)).astype(float),
        "Close":100 + pd.Series(range(60)).astype(float),
        "Volume": 1000
    })
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    # 3) write to a temp file and seed
    os.makedirs("tests/_data", exist_ok=True)
    test_csv = "tests/_data/TEST.csv"
    with open(test_csv, "wb") as f:
        f.write(csv_bytes.read())

    seed_from_csvs(test_engine, {"TEST": test_csv})

    # sanity: symbol exists
    with Session(test_engine) as s:
        syms = [x[0] for x in s.execute(select(Instrument.symbol)).all()]
        assert "TEST" in syms

    app_mod.app.testing = True
    return app_mod.app

@pytest.fixture()
def client(seeded_app):
    with seeded_app.test_client() as c:
        yield c
