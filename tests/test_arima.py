import pandas as pd
from ml.arima_model import arima_forecast

def test_arima_forecast_length():
    s = pd.Series([i for i in range(100)], dtype=float)
    y, meta = arima_forecast(s, steps=5)
    assert len(y) == 5
    assert "order" in meta
