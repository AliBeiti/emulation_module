"""
main.py

Entry point for the Emulation Module.
Starts two concurrent components:
  1. FastAPI server  — handles HTTP requests on port 8090
  2. Tick loop       — runs every 5s, drives the emulation

Flow per tick:
  [Calibration phase]
    replay calibration_pod.csv window by window
    serve metrics to AC via /usage/latest
    wait for POST /calibration/done {"signal": "FIXED"}

  [Normal phase]
    timeline.tick()
      → on job expiry: kwok_manager.remove_buyer_namespaces() for expired jobs
      → if composition changed:
          dataset_selector.select(composition) → new CSV path
          replay_engine.load(csv_path, preserve_window=True)
          build buyer namespace map from active jobs
          kwok_manager.sync(sample_pods with buyer namespaces)
      → pod_rows = replay_engine.get_current_window_pods()
      → kwok_manager.patch_annotations(pod_rows, window_index)
      → metrics = aggregator.compute(pod_rows)
      → state["latest_metrics"] = metrics
      → replay_engine.advance_window()

Usage:
  python main.py                    # normal mode
  python main.py --dry-run          # no Kubernetes API calls
  python main.py --no-kwok          # skip KWOK, only expose metrics API
"""

import argparse
import logging
import logging.handlers
import os
import threading
import time
import sys
from typing import Dict, List

import uvicorn

from config import API_HOST, API_PORT, TICK_INTERVAL_S, BASE_DIR
from timeline         import Timeline
from dataset_selector import DatasetSelector
from replay_engine    import ReplayEngine
from aggregator       import Aggregator
from kwok_manager     import KWOKManager
from api              import app, state
from transaction_poller import TransactionPoller



LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
root_logger.addHandler(stream_handler)

file_handler = logging.handlers.RotatingFileHandler(
    filename    = f"{LOG_DIR}/emulation.log",
    maxBytes    = 10 * 1024 * 1024,
    backupCount = 5,
    encoding    = "utf-8"
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

CALIBRATION_CSV = os.path.join(BASE_DIR, "calibration", "calibration_pod.csv")




# ── Buyer namespace helpers ────────────────────────────────────────────────────

def get_active_jobs_by_app(timeline: Timeline) -> Dict[str, List[str]]:
    """
    Return {app_type: [buyer_name, ...]} ordered by job start_tick (oldest first).
    Used to build buyer-named namespace mappings.
    """
    active_jobs = timeline.get_active_job_objects()
    active_jobs_sorted = sorted(active_jobs, key=lambda j: (j.app_type, j.start_tick))

    by_app: Dict[str, List[str]] = {}
    for job in active_jobs_sorted:
        if job.app_type not in by_app:
            by_app[job.app_type] = []
        by_app[job.app_type].append(job.buyer_name)

    return by_app

# ── snapshot loop ──────────────────────────────────────────────────────────────────

def snapshot_loop(timeline: Timeline, kwok_manager: KWOKManager):
    """Writes a JSON snapshot every 2 minutes to /app/logs/snapshots.jsonl"""
    import json
    from datetime import datetime

    snapshot_path = f"{LOG_DIR}/snapshots.jsonl"
    logger.info("Snapshot loop started")

    while True:
        time.sleep(120)  # every 2 minutes
        try:
            active_jobs = timeline.get_active_job_objects()
            namespaces  = kwok_manager.get_alive_namespaces()
            alive_pods  = kwok_manager._alive_pods

            snapshot = {
                "timestamp":       datetime.now().isoformat(),
                "active_job_count": len(active_jobs),
                "active_jobs": [
                    {
                        "job_id":    j.job_id,
                        "app_type":  j.app_type,
                        "buyer":     j.buyer_name,
                        "remaining": j.remaining_ticks,
                    }
                    for j in active_jobs
                ],
                "namespace_count": len(namespaces),
                "namespaces":      namespaces,
                "pod_counts": {
                    ns: len(pods) for ns, pods in alive_pods.items()
                },
                "total_pods": sum(len(p) for p in alive_pods.values()),
            }

            with open(snapshot_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot) + "\n")

        except Exception as e:
            logger.error(f"Snapshot error: {e}", exc_info=True)

# ── Tick loop ──────────────────────────────────────────────────────────────────

def tick_loop(
    timeline:      Timeline,
    selector:      DatasetSelector,
    replay_engine: ReplayEngine,
    aggregator:    Aggregator,
    kwok_manager:  KWOKManager,
    no_kwok:       bool = False,
):
    logger.info("Tick loop started")

    # ── Load calibration dataset at startup ───────────────────────────────────
    calibration_engine = ReplayEngine()
    if os.path.exists(CALIBRATION_CSV):
        calibration_engine.load(CALIBRATION_CSV, "calibration")
        cal_ns = calibration_engine.get_dataset_namespaces()
        calibration_engine.build_namespace_map(cal_ns)
        logger.info(f"Calibration dataset loaded: {CALIBRATION_CSV}")
    else:
        logger.warning(
            f"Calibration CSV not found: {CALIBRATION_CSV} — skipping calibration phase"
        )
        state["calibration_done"] = True

    while True:
        tick_start = time.time()

        try:
            # ── Calibration phase ─────────────────────────────────────────────
            if not state.get("calibration_done", False):
                if calibration_engine.is_loaded():
                    cal_pods = calibration_engine.get_current_window_pods()
                    metrics  = aggregator.compute(cal_pods)
                    state["latest_metrics"] = metrics
                    calibration_engine.advance_window()
                    logger.debug(
                        f"[CALIBRATION] window={calibration_engine.get_window_index()} | "
                        f"cpu={metrics.get('cpu_usage_pct',0):.1f}% | "
                        f"sched={metrics.get('sched_total_ms',0):.0f}ms"
                    )
                else:
                    state["latest_metrics"] = aggregator.compute([])
                _sleep_remaining(tick_start)
                continue

            # ── Normal operation ──────────────────────────────────────────────

            # ── 1. Advance timeline ───────────────────────────────────────────
            composition_changed = timeline.tick()
            composition         = timeline.get_composition()

            # ── 2. Handle expired jobs (remove only their namespaces) ─────────
            expired_jobs = timeline.get_expired_jobs()
            if expired_jobs and not no_kwok:
                for job in expired_jobs:
                    kwok_manager.remove_buyer_namespaces(
                        app_type   = job.app_type,
                        buyer_name = job.buyer_name,
                    )

            # ── 3. Handle composition change ──────────────────────────────────
            if composition_changed or not replay_engine.is_loaded():

                if not composition:
                    state["latest_metrics"] = aggregator.compute([])
                    if not no_kwok:
                        kwok_manager.sync([])
                    logger.info("No active jobs — serving baseline, KWOK pods removed")

                else:
                    csv_path, meta = selector.select(composition)

                    if csv_path is None:
                        logger.error(
                            f"No dataset found for {composition} — keeping old dataset"
                        )
                    else:
                        dataset_key  = selector._composition_to_key(composition)
                        is_first_load = not replay_engine.is_loaded()

                        # preserve window when switching, reset on first load
                        replay_engine.load(
                            csv_path,
                            dataset_key,
                            preserve_window = not is_first_load,
                        )

                        # build buyer-named namespace map
                        jobs_by_app = get_active_jobs_by_app(timeline)
                        replay_engine.build_buyer_namespace_map(jobs_by_app)

                        if not no_kwok:
                            sample_raw = replay_engine.get_current_window_pods()
                            kwok_manager.sync(sample_raw)
                        else:
                            dataset_ns = replay_engine.get_dataset_namespaces()
                            replay_engine.build_namespace_map(dataset_ns)

                        logger.info(
                            f"Dataset loaded: {dataset_key} | "
                            f"composition: {composition} | "
                            f"buyers: {jobs_by_app}"
                        )

            # ── 4. Skip tick if no active jobs ────────────────────────────────
            if not composition or not replay_engine.is_loaded():
                if not composition:
                    state["latest_metrics"] = aggregator.compute([])
                _sleep_remaining(tick_start)
                continue

            # ── 5. Get current window pods ────────────────────────────────────
            pod_rows = replay_engine.get_current_window_pods()

            # ── 6. Patch KWOK pod annotations ─────────────────────────────────
            if not no_kwok:
                kwok_manager.patch_annotations(
                    pod_rows, replay_engine.get_window_index()
                )

            # ── 7. Aggregate to node level ────────────────────────────────────
            metrics = aggregator.compute(pod_rows)

            # ── 8. Store for API ──────────────────────────────────────────────
            state["latest_metrics"] = metrics

            logger.info(
                f"Tick {timeline.get_current_tick()} | "
                f"window {replay_engine.get_window_index()} | "
                f"pods {metrics.get('_pod_count', 0)} | "
                f"cpu {metrics.get('cpu_usage_pct', 0):.1f}% | "
                f"psi {metrics.get('cpu_psi_some_pct', 0):.1f}% | "
                f"power {metrics.get('node_cpu_watts', 0):.1f}W"
            )

            # ── 9. Advance window ─────────────────────────────────────────────
            replay_engine.advance_window()

        except Exception as e:
            logger.error(f"Tick loop error: {e}", exc_info=True)

        _sleep_remaining(tick_start)


def _sleep_remaining(tick_start: float):
    elapsed   = time.time() - tick_start
    remaining = TICK_INTERVAL_S - elapsed
    if remaining > 0:
        time.sleep(remaining)
    else:
        logger.warning(
            f"Tick took {elapsed:.2f}s — longer than interval {TICK_INTERVAL_S}s"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Emulation Module")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--no-kwok",   action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 60)
    logger.info("EMULATION MODULE STARTING")
    logger.info("=" * 60)

    try:
        timeline      = Timeline()
        selector      = DatasetSelector()
        replay_engine = ReplayEngine()
        aggregator    = Aggregator()
        kwok_manager  = KWOKManager(dry_run=args.dry_run)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

    if not args.dry_run and not args.no_kwok:
        try:
            kwok_manager.ensure_node()
        except Exception as e:
            logger.warning(f"Could not ensure KWOK node: {e}")

    state["timeline"]      = timeline
    state["replay_engine"] = replay_engine

    logger.info(f"Dataset index loaded: {len(selector.list_available())} datasets")
    logger.info(f"API will be available at http://{API_HOST}:{API_PORT}")

    tick_thread = threading.Thread(
        target = tick_loop,
        args   = (timeline, selector, replay_engine, aggregator,
                  kwok_manager, args.no_kwok),
        daemon = True,
        name   = "tick-loop"
    )
    tick_thread.start()
    logger.info("Tick loop started in background thread")

    snapshot_thread = threading.Thread(
        target = snapshot_loop,
        args   = (timeline, kwok_manager),
        daemon = True,
        name   = "snapshot-loop"
    )
    snapshot_thread.start()
    logger.info("Snapshot loop started in background thread")

    poller = TransactionPoller(timeline=timeline)
    poller.start()

    uvicorn.run(
        app,
        host      = API_HOST,
        port      = API_PORT,
        log_level = args.log_level.lower(),
    )


if __name__ == "__main__":
    main()