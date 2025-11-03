# ml/model_registry.py
from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING
from datetime import datetime

# Only for static type checkers; avoids runtime import errors/cycles
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
else:
    Session = Any  # at runtime we just accept any session-like object

try:
    from models import ModelVersion
    _HAS_MODEL_VERSION = True
except Exception:
    ModelVersion = None  # type: ignore
    _HAS_MODEL_VERSION = False


def register_version(
    session: "Session",
    *,
    symbol: str,
    algo: str,
    params: Dict[str, Any],
    train_start: datetime,
    train_end: datetime,
    note: str = "",
) -> Optional[int]:
    """Record a model version if the table exists; otherwise no-op."""
    if not _HAS_MODEL_VERSION or ModelVersion is None:
        # Graceful no-op so your app still runs before Step 5 is finished
        print("[model_registry] ModelVersion table not available; skipping")
        return None

    mv = ModelVersion(
        symbol=symbol,
        algo=algo,
        params=str(params),
        train_start=train_start,
        train_end=train_end,
        note=note,
    )
    session.add(mv)
    session.flush()  # assigns primary key
    return int(mv.id)
