"""
scheduler.py
------------
Runs batch_predict.py automatically every 5 minutes using the
APScheduler library (pure Python — no cron required).

Behaviour:
  - Executes one prediction run immediately on startup.
  - Repeats every INTERVAL_MINUTES minutes until you press Ctrl+C.
"""

import time
import signal
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler

from batch_predict import run_batch_prediction

# ── Configuration ─────────────────────────────────────────────────────────────
INTERVAL_MINUTES = 5   # How often the batch job runs


def job() -> None:
    """Scheduled job — wraps run_batch_prediction with error handling."""
    try:
        run_batch_prediction()
    except Exception as exc:
        print(f"[SCHEDULER] ERROR during batch run: {exc}")


def handle_shutdown(sig, frame) -> None:
    """Graceful shutdown on Ctrl+C."""
    print("\n[SCHEDULER] Shutdown signal received. Stopping scheduler …")
    sys.exit(0)


def main() -> None:
    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, handle_shutdown)

    print("=" * 60)
    print("  Batch Prediction Scheduler")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Interval : every {INTERVAL_MINUTES} minute(s)")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    # ── Run once immediately ──────────────────────────────────────────────────
    print("[SCHEDULER] Running initial batch prediction …")
    job()

    # ── Schedule recurring job ────────────────────────────────────────────────
    scheduler = BlockingScheduler()
    scheduler.add_job(
        func=job,
        trigger="interval",
        minutes=INTERVAL_MINUTES,
        id="batch_prediction_job",
        name="Titanic Batch Prediction",
        max_instances=1          # Prevent overlapping runs
    )

    print(f"[SCHEDULER] Next run in {INTERVAL_MINUTES} minute(s). Waiting …")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("[SCHEDULER] Scheduler stopped.")


if __name__ == "__main__":
    main()
