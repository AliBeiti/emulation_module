"""
api.py

FastAPI application for the Emulation Module.
Exposes all REST endpoints consumed by Admission Control, Pricing,
and buyers.

Endpoints:
  GET  /usage/latest        ← Admission Control reads current node metrics
  GET  /usage/capacity      ← node total resources (constant)
  POST /calibration/done    ← Admission Control signals calibration complete
  GET  /calibration/status  ← whether calibration phase is done
  GET  /status              ← current jobs and composition
  GET  /healthz             ← Kubernetes liveness probe

Shared state (set by main.py tick loop):
  state["latest_metrics"]   ← dict updated every 5s by aggregator
  state["timeline"]         ← Timeline instance
  state["replay_engine"]    ← ReplayEngine instance
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import (
    NODE_CPU_MCORES, NODE_CPU_CORES,
    NODE_RAM_MI, NODE_DISK_GB,
    API_SERVICE_NAME
)

logger = logging.getLogger(__name__)

# ── Shared state (populated by main.py) ───────────────────────────────────────
# This dict is the bridge between the tick loop and the API
state: Dict = {
    "latest_metrics": None,   # dict from aggregator.compute()
    "timeline":       None,   # Timeline instance
    "replay_engine":  None,   # ReplayEngine instance
    "started_at":     datetime.now().isoformat(),
    "calibration_done": False,   # True after AC sends /calibration/done
}

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emulation Module API",
    description="Seller-side workload emulator for the decentralized resource trading platform",
    version="1.0.0"
)


# ── Request / Response models ─────────────────────────────────────────────────

class LatestMetrics(BaseModel):
    timestamp:        str
    cpu_usage_pct:    Optional[float]
    cpu_psi_some_pct: Optional[float]
    ram_usage_pct:    Optional[float]
    ram_usage_mi:     Optional[float]
    disk_used_pct:    Optional[float]
    sched_total_ms:   Optional[float]
    dstate_total_ms:  Optional[float]
    softirq_total_ms: Optional[float]
    node_cpu_watts:   Optional[float]


class CapacityResponse(BaseModel):
    cpu_total_mcores: int
    cpu_cores:        int
    ram_total_mi:     int
    disk_total_gb:    int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/usage/latest", response_model=LatestMetrics)
async def get_latest_metrics():
    """
    Returns current node-level resource usage.
    Updated every 5 seconds by the tick loop.
    Consumed by Admission Control (MZ_USAGE_API_URL).
    """
    metrics = state.get("latest_metrics")

    if metrics is None:
        # module started but no tick yet — return zeros
        return LatestMetrics(
            timestamp        = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cpu_usage_pct    = 0.0,
            cpu_psi_some_pct = 0.0,
            ram_usage_pct    = 0.0,
            ram_usage_mi     = 0.0,
            disk_used_pct    = 0.0,
            sched_total_ms   = 0.0,
            dstate_total_ms  = 0.0,
            softirq_total_ms = 0.0,
            node_cpu_watts   = 0.0,
        )

    return LatestMetrics(
        timestamp        = metrics.get("timestamp",        datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        cpu_usage_pct    = metrics.get("cpu_usage_pct"),
        cpu_psi_some_pct = metrics.get("cpu_psi_some_pct"),
        ram_usage_pct    = metrics.get("ram_usage_pct"),
        ram_usage_mi     = metrics.get("ram_usage_mi"),
        disk_used_pct    = metrics.get("disk_used_pct"),
        sched_total_ms   = metrics.get("sched_total_ms"),
        dstate_total_ms  = metrics.get("dstate_total_ms"),
        softirq_total_ms = metrics.get("softirq_total_ms"),
        node_cpu_watts   = metrics.get("node_cpu_watts"),
    )


@app.get("/usage/capacity", response_model=CapacityResponse)
async def get_capacity():
    """
    Returns node total resource capacity (constant values).
    Used by Admission Control to compute sellable resources.
    """
    return CapacityResponse(
        cpu_total_mcores = NODE_CPU_MCORES,
        cpu_cores        = NODE_CPU_CORES,
        ram_total_mi     = NODE_RAM_MI,
        disk_total_gb    = NODE_DISK_GB,
    )


@app.get("/status")
async def get_status():
    """
    Returns current emulation state:
      - active jobs
      - current composition
      - current window index
      - loaded dataset
    """
    timeline      = state.get("timeline")
    replay_engine = state.get("replay_engine")

    composition   = timeline.get_composition()    if timeline      else {}
    active_jobs   = timeline.get_active_jobs()    if timeline      else []
    all_jobs      = timeline.get_all_jobs()       if timeline      else []
    window_index  = replay_engine.get_window_index()  if replay_engine else 0
    total_windows = replay_engine.get_total_windows() if replay_engine else 0
    dataset_key   = replay_engine.get_dataset_key()   if replay_engine else None
    ns_map        = replay_engine.get_namespace_map() if replay_engine else {}

    return {
        "service":          API_SERVICE_NAME,
        "started_at":       state.get("started_at"),
        "composition":      composition,
        "active_jobs":      active_jobs,
        "all_jobs":         all_jobs,
        "window_index":     window_index,
        "total_windows":    total_windows,
        "dataset_key":      dataset_key,
        "namespace_map":    ns_map,
    }


@app.get("/healthz")
async def healthz():
    """Kubernetes liveness probe."""
    return {"status": "ok", "service": API_SERVICE_NAME}


@app.post("/calibration/done")
async def calibration_done(payload: dict = None):
    """
    AC posts to this endpoint when calibration is complete.
    Expected body: {"signal": "FIXED"}
    Switches emulation module from calibration phase to normal operation.
    """
    signal = (payload or {}).get("signal", "")
    if signal != "FIXED":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid signal '{signal}'. Expected 'FIXED'."
        )

    if state.get("calibration_done"):
        return {"status": "already_done", "message": "Calibration was already completed"}

    state["calibration_done"] = True
    logger.info("Calibration done signal received — switching to normal operation")

    return {
        "status": "ok",
        "message": "Calibration complete. Emulation module switching to normal mode."
    }


@app.get("/calibration/status")
async def calibration_status():
    """Returns whether calibration phase is complete."""
    done = state.get("calibration_done", False)
    return {
        "calibration_done": done,
        "phase": "normal" if done else "calibration",
    }