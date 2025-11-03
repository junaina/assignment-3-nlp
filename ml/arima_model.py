import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple

def arima_forecast(close_series: pd.Series, steps: int = 1,
                   order=(5,1,0)) -> Tuple[np.ndarray, dict]:
    model = ARIMA(close_series.astype(float), order=order)
    res = model.fit(method_kwargs={"warn_convergence": False})
    fc = res.forecast(steps=steps)
    return fc.values, {"order": order, "aic": getattr(res, "aic", None)}
