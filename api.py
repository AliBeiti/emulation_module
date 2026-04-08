"""
api.py

FastAPI application for the Emulation Module.
Exposes all REST endpoints consumed by Admission Control, Pricing,
and buyers.

Endpoints:
  POST /submit              ← buyer submits a job
  GET  /usage/latest        ← Admission Control reads current node metrics
  GET  /usage/capacity      ← node total resources (constant)
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
from pydantic import BaseModel, Field

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
}

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emulation Module API",
    description="Seller-side workload emulator for the decentralized resource trading platform",
    version="1.0.0"
)


# ── Request / Response models ─────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    app_type:         str = Field(..., description="Application type: hotel, sn, or sa")
    lifetime_seconds: int = Field(..., gt=0, description="Job lifetime in seconds")


class SubmitResponse(BaseModel):
    status:   str
    job_id:   str
    message:  str


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


class TransactionRequest(BaseModel):
    type:            str = Field(..., description="Transaction type, e.g. 'transfer'")
    buyer:           dict = Field(..., description="Buyer object from trading module")
    seller:          dict = Field(..., description="Seller object from trading module")
    amount:          float = Field(..., description="Agreed transaction amount")
    tx_start_ts:     str = Field(..., description="Transaction start timestamp")
    lease_duration:  int = Field(..., gt=0, description="Lease duration in seconds")


class TransactionResponse(BaseModel):
    status:   str
    job_id:   str
    message:  str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/submit", response_model=SubmitResponse, status_code=202)
async def submit_job(request: SubmitRequest):
    """
    Buyer submits a job request.
    The job becomes active at the next 5s tick boundary.
    Returns immediately with job_id and status.
    """
    valid_apps = {"hotel", "sn", "sa"}
    if request.app_type not in valid_apps:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid app_type '{request.app_type}'. Must be one of {valid_apps}"
        )

    timeline = state.get("timeline")
    if timeline is None:
        raise HTTPException(status_code=503, detail="Emulation module not ready")

    job = timeline.add_job(
        app_type         = request.app_type,
        lifetime_seconds = request.lifetime_seconds
    )

    logger.info(f"Job accepted: {job.job_id} | {request.app_type} | {request.lifetime_seconds}s")

    return SubmitResponse(
        status  = "accepted",
        job_id  = job.job_id,
        message = (
            f"Job accepted. Will start at next tick. "
            f"Active for {request.lifetime_seconds}s "
            f"(~{request.lifetime_seconds // 5} windows)."
        )
    )


@app.post("/transaction", response_model=TransactionResponse, status_code=202)
async def handle_transaction(request: TransactionRequest):
    """
    Receives a confirmed transaction from the trading module.
    Immediately adds a hotel workload for the duration of the lease.

    The emulation module picks the correct dataset automatically
    based on current composition + 1 new hotel instance.
    """
    if request.type != "transfer":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported transaction type '{request.type}'. Only 'transfer' is supported."
        )

    timeline = state.get("timeline")
    if timeline is None:
        raise HTTPException(status_code=503, detail="Emulation module not ready")

    job = timeline.add_job(
        app_type         = "hotel",
        lifetime_seconds = request.lease_duration
    )

    logger.info(
        f"Transaction accepted: job={job.job_id} | "
        f"buyer={request.buyer} | amount={request.amount} | "
        f"lease={request.lease_duration}s"
    )

    return TransactionResponse(
        status  = "accepted",
        job_id  = job.job_id,
        message = (
            f"Hotel workload scheduled for {request.lease_duration}s "
            f"(~{request.lease_duration // 5} windows). "
            f"Job ID: {job.job_id}"
        )
    )


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