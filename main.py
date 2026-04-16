"""
main.py

Entry point for the Emulation Module.
Starts two concurrent components:
  1. FastAPI server  — handles HTTP requests on port 8090
  2. Tick loop       — runs every 5s, drives the emulation

Flow per tick:
  timeline.tick()
    → if composition changed:
        dataset_selector.select(composition) → new CSV path
        replay_engine.load(csv_path)
        kwok_manager.sync(sample_pods)
        replay_engine.build_namespace_map(alive_namespaces)
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

import asyncio
import argparse
import logging
import threading
import time
import sys

import uvicorn

from config import API_HOST, API_PORT, TICK_INTERVAL_S, BASE_DIR
import os
CALIBRATION_CSV = os.path.join(BASE_DIR, "calibration", "calibration_pod.csv")
from timeline        import Timeline
from dataset_selector import DatasetSelector
from replay_engine   import ReplayEngine
from aggregator      import Aggregator
from kwok_manager    import KWOKManager
from api             import app, state
from transaction_poller import TransactionPoller

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── Tick loop ──────────────────────────────────────────────────────────────────

def tick_loop(
    timeline:         Timeline,
    selector:         DatasetSelector,
    replay_engine:    ReplayEngine,
    aggregator:       Aggregator,
    kwok_manager:     KWOKManager,
    no_kwok:          bool = False,
):
    """
    Main emulation loop. Runs every TICK_INTERVAL_S seconds.
    Drives dataset selection, KWOK sync, metric replay, and aggregation.
    """
    logger.info("Tick loop started")
    last_composition = {}

    # ── Load calibration dataset at startup ───────────────────────────────────
    calibration_engine = ReplayEngine()
    if os.path.exists(CALIBRATION_CSV):
        calibration_engine.load(CALIBRATION_CSV, "calibration")
        # build identity namespace map
        cal_ns = calibration_engine.get_dataset_namespaces()
        calibration_engine.build_namespace_map(cal_ns)
        logger.info(f"Calibration dataset loaded: {CALIBRATION_CSV}")
    else:
        logger.warning(f"Calibration CSV not found: {CALIBRATION_CSV} — skipping calibration phase")
        state["calibration_done"] = True   # skip calibration if no file

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
                    # no calibration data — serve baseline
                    state["latest_metrics"] = aggregator.compute([])
                _sleep_remaining(tick_start)
                continue

            # ── Normal operation (calibration done) ───────────────────────────

            # ── 1. Advance timeline ───────────────────────────────────────────
            composition_changed = timeline.tick()
            composition         = timeline.get_composition()

            # ── 2. Handle composition change ──────────────────────────────────
            if composition_changed or not replay_engine.is_loaded():

                if not composition:
                    # no active jobs — clear metrics and remove KWOK pods
                    state["latest_metrics"] = aggregator.compute([])
                    if not no_kwok:
                        kwok_manager.sync([])
                    logger.info("No active jobs — metrics cleared, KWOK pods removed")
                else:
                    # select new dataset
                    csv_path, meta = selector.select(composition)

                    if csv_path is None:
                        logger.error(f"No dataset found for {composition} — keeping old dataset")
                    else:
                        dataset_key = selector._composition_to_key(composition)
                        replay_engine.load(csv_path, dataset_key)

                        # sync KWOK pods (diff only)
                        if not no_kwok:
                            sample_raw = replay_engine._windows.get(0, [])
                            kwok_manager.sync(sample_raw)
                            alive_ns = kwok_manager.get_alive_namespaces()
                            replay_engine.build_namespace_map(alive_ns)
                        else:
                            # no-kwok mode: identity map using dataset namespaces
                            dataset_ns = replay_engine.get_dataset_namespaces()
                            replay_engine.build_namespace_map(dataset_ns)

                        logger.info(
                            f"Dataset loaded: {dataset_key} | "
                            f"composition: {composition}"
                        )

                last_composition = composition

            # ── 3. Skip tick if no active jobs ────────────────────────────────
            if not composition or not replay_engine.is_loaded():
                # keep serving fresh baseline rows every tick during idle
                if not composition:
                    state["latest_metrics"] = aggregator.compute([])
                    logger.info("No active jobs — serving baseline metrics")
                _sleep_remaining(tick_start)
                continue

            # ── 4. Get current window pods (with namespace mapping) ───────────
            pod_rows = replay_engine.get_current_window_pods()

            # ── 5. Patch KWOK pod annotations ─────────────────────────────────
            if not no_kwok:
                kwok_manager.patch_annotations(
                    pod_rows,
                    replay_engine.get_window_index()
                )

            # ── 6. Aggregate to node level ────────────────────────────────────
            metrics = aggregator.compute(pod_rows)

            # ── 7. Store for API ──────────────────────────────────────────────
            state["latest_metrics"] = metrics

            logger.info(
                f"Tick {timeline.get_current_tick()} | "
                f"window {replay_engine.get_window_index()} | "
                f"pods {metrics.get('_pod_count', 0)} | "
                f"cpu {metrics.get('cpu_usage_pct', 0):.1f}% | "
                f"psi {metrics.get('cpu_psi_some_pct', 0):.1f}% | "
                f"power {metrics.get('node_cpu_watts', 0):.1f}W"
            )

            # ── 8. Advance window ─────────────────────────────────────────────
            replay_engine.advance_window()

        except Exception as e:
            logger.error(f"Tick loop error: {e}", exc_info=True)

        _sleep_remaining(tick_start)


def _sleep_remaining(tick_start: float):
    """Sleep for the remainder of the tick interval."""
    elapsed = time.time() - tick_start
    remaining = TICK_INTERVAL_S - elapsed
    if remaining > 0:
        time.sleep(remaining)
    else:
        logger.warning(f"Tick took {elapsed:.2f}s — longer than interval {TICK_INTERVAL_S}s")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Emulation Module")
    parser.add_argument("--dry-run", action="store_true",
                        help="Disable Kubernetes API calls")
    parser.add_argument("--no-kwok", action="store_true",
                        help="Skip KWOK pod management (metrics API only)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("="*60)
    logger.info("EMULATION MODULE STARTING")
    logger.info("="*60)

    # ── Initialize components ─────────────────────────────────────────────────
    try:
        timeline      = Timeline()
        selector      = DatasetSelector()
        replay_engine = ReplayEngine()
        aggregator    = Aggregator()
        kwok_manager  = KWOKManager(dry_run=args.dry_run)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

    # ensure KWOK node exists
    if not args.dry_run and not args.no_kwok:
        try:
            kwok_manager.ensure_node()
        except Exception as e:
            logger.warning(f"Could not ensure KWOK node: {e}")

    # ── Share components with API ─────────────────────────────────────────────
    state["timeline"]      = timeline
    state["replay_engine"] = replay_engine

    logger.info(f"Dataset index loaded: {len(selector.list_available())} datasets")
    logger.info(f"API will be available at http://{API_HOST}:{API_PORT}")
    logger.info(f"Admission Control endpoint: GET /usage/latest")

    # ── Start tick loop in background thread ──────────────────────────────────
    tick_thread = threading.Thread(
        target   = tick_loop,
        args     = (timeline, selector, replay_engine, aggregator,
                    kwok_manager, args.no_kwok),
        daemon   = True,
        name     = "tick-loop"
    )
    tick_thread.start()
    logger.info("Tick loop started in background thread")

    # ── Start transaction poller ───────────────────────────────────────────────
    poller = TransactionPoller(timeline=timeline)
    poller.start()

    # ── Start FastAPI server (blocking) ───────────────────────────────────────
    uvicorn.run(
        app,
        host      = API_HOST,
        port      = API_PORT,
        log_level = args.log_level.lower(),
    )


if __name__ == "__main__":
    main()