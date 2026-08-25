"""
event_logger.py

Logs transaction events to /app/logs/events_YYYY-MM-DD.jsonl (JSON Lines).
A new file is created automatically each day — nothing is ever deleted.
Each line is one event dict, written atomically under a threading lock.

Usage (from transaction_poller.py):
    from event_logger import EventLogger
    _event_logger = EventLogger()          # module-level singleton

    # after add_job() succeeds:
    _event_logger.log(
        tx_record     = tx_record,
        msg_id        = msg_id,
        job           = job,
        received_unix = received_unix,
        composition   = self._timeline.get_composition(),
    )

Loading in pandas:
    import pandas as pd, glob
    df = pd.concat([
        pd.read_json(f, lines=True)
        for f in sorted(glob.glob('/app/logs/events_*.jsonl'))
    ])
    df['tx_to_received_s'].describe()
    df.groupby('buyer_name')['running_total'].mean()
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from threading import Lock

logger = logging.getLogger(__name__)

LOG_DIR = "/app/logs"


class EventLogger:
    """
    Thread-safe JSON Lines event logger.
    Writes to /app/logs/events_YYYY-MM-DD.jsonl — one file per day,
    new file starts automatically at midnight, nothing is ever deleted.
    """

    def __init__(self, log_dir: str = LOG_DIR):
        self._log_dir   = log_dir
        self._lock      = Lock()
        self._cur_date  = None   # tracks which date file is open
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"EventLogger initialised — log dir: {self._log_dir}")

    # ── Public ────────────────────────────────────────────────────────────────

    def log(
        self,
        tx_record:     dict,
        msg_id:        str,
        job,                        # Job dataclass from timeline.py
        received_unix: float,
        composition:   dict,
    ) -> None:
        """
        Build and append one event record to today's log file.

        Parameters
        ----------
        tx_record     : full parsed JSON from the Redis stream message
        msg_id        : Redis stream message ID (e.g. "1750330694120-0")
        job           : Job object returned by timeline.add_job()
        received_unix : time.time() captured right after xreadgroup returned
        composition   : timeline.get_composition() called after add_job()
                        — reflects the new state including the just-added job
        """
        try:
            event = self._build_event(
                tx_record, msg_id, job, received_unix, composition
            )
            self._write(event)
        except Exception as e:
            logger.error(f"EventLogger.log failed: {e}", exc_info=True)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _today_path(self) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self._log_dir, f"events_{date_str}.jsonl")

    def _build_event(
        self,
        tx_record:     dict,
        msg_id:        str,
        job,
        received_unix: float,
        composition:   dict,
    ) -> dict:
        logged_unix   = time.time()
        tx            = tx_record.get("tx", {})
        buyer         = tx.get("buyer", {})
        seller        = tx.get("seller", {})
        resource      = buyer.get("resource", {})
        tx_start_str  = tx.get("tx_start_ts", "")
        tx_start_unix = _parse_iso_to_unix(tx_start_str)
        comp          = dict(composition) if composition else {}

        return {
            # Identity
            "event":   "transaction_received",
            "tx_hash": tx_record.get("tx_hash", ""),
            "job_id":  job.job_id,

            # Participants
            "buyer_name":  buyer.get("name", "unknown"),
            "buyer_ip":    buyer.get("ip", ""),
            "seller_name": seller.get("name", ""),
            "seller_ip":   seller.get("ip", ""),

            # Timing (Unix float)
            "tx_start_unix": tx_start_unix,
            "received_unix": received_unix,
            "logged_unix":   logged_unix,

            # Derived timing
            "tx_to_received_s": round(received_unix - tx_start_unix, 3)
                                 if tx_start_unix else None,

            # Job info
            "lease_duration_s": int(tx.get("lease_duration", 0)),
            "lifetime_s":       job.lifetime_seconds,
            "app_type":         job.app_type,

            # Running instances after this job was added
            # Flat fields — easy pandas use: df["running_hotel"].plot()
            "running_hotel": comp.get("hotel", 0),
            "running_sn":    comp.get("sn",    0),
            "running_sa":    comp.get("sa",    0),
            "running_es":    comp.get("es",    0),
            "running_total": sum(comp.values()),
            # Full dict — for completeness / future app types
            "composition_at_add": comp,

            # Stream info
            "redis_msg_id": msg_id,

            # Resource demand
            "demand_vcpu":    resource.get("cpu"),
            "demand_ram":     resource.get("ram"),
            "demand_storage": resource.get("storage"),
            "amount":         tx.get("amount"),
        }

    def _write(self, event: dict) -> None:
        """Append one JSON line to today's file. Thread-safe."""
        with self._lock:
            path = self._today_path()
            if path != self._cur_date:
                self._cur_date = path
                logger.info(f"EventLogger: writing to {path}")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_iso_to_unix(ts_str: str) -> float:
    """
    Parse an ISO-8601 timestamp string to a Unix float.
    Returns 0.0 on any failure.
    """
    if not ts_str:
        return 0.0
    try:
        ts = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0