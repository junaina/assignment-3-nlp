# ml/lstm_model.py
import numpy as np
import pandas as pd
from typing import Tuple

# Prefer standalone Keras (v3); fall back to tf.keras if needed
try:
    import keras  # Keras 3 (recommended)
except Exception:
    try:
        from tensorflow import keras  # compatibility fallback
    except Exception as e:
        raise ImportError("Keras is not available. Install keras/tensorflow.") from e


def _to_supervised(arr: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(arr) - lookback):
        X.append(arr[i : i + lookback])
        y.append(arr[i + lookback])
    X = np.asarray(X)[:, :, None]  # (n, lookback, 1)
    y = np.asarray(y)
    return X, y


def lstm_forecast(close_series: pd.Series, steps: int, lookback: int = 30) -> Tuple[np.ndarray, dict]:
    data = close_series.astype(float).values
    if len(data) <= lookback + 5:
        last = float(data[-1])
        return np.array([last] * steps, dtype=float), {"fallback": "naive", "lookback": lookback}

    X, y = _to_supervised(data, lookback)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(lookback, 1)),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=6, batch_size=32, verbose=0)

    # roll-forward multi-step
    window = data[-lookback:].copy()
    preds = []
    for _ in range(steps):
        p = float(model.predict(window.reshape(1, lookback, 1), verbose=0)[0, 0])
        preds.append(p)
        window = np.roll(window, -1)
        window[-1] = p

    return np.array(preds, dtype=float), {"lookback": lookback, "epochs": 6}
