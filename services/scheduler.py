# services/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from services.train_service import periodic_retrain
from services.eval_service import evaluate_new_points

def start_scheduler(symbols: list[str]) -> BackgroundScheduler:
    """
    Starts two jobs:
    1) Nightly model refit (adaptive learning)
    2) Frequent evaluation (continuous monitoring)
    """
    sch = BackgroundScheduler()

    # 1) Nightly refit at 03:00 local time
    sch.add_job(
        lambda: [periodic_retrain(sym, algo="ARIMA", order=(5,1,0)) for sym in symbols],
        trigger=CronTrigger(hour=3, minute=0),
        id="nightly_refit",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
        replace_existing=True,
    )

    # 2) Evaluate every 30 minutes
    sch.add_job(
        lambda: [evaluate_new_points(sym) for sym in symbols],
        trigger=IntervalTrigger(minutes=30),
        id="continuous_eval",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
        replace_existing=True,
    )

    sch.start()
    return sch
