"""
timeline.py

Tracks all active emulation jobs and detects composition changes.
Each job has a start_tick and end_tick derived from lifetime_seconds.
Every tick, expired jobs are removed and the current composition is recomputed.

Key concepts:
  - tick: one 5-second interval (tick 0, 1, 2, ...)
  - composition: dict of {app_type: count} for all currently running jobs
  - composition_changed: True if the composition is different from last tick
"""

import uuid
import logging
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config import TICK_INTERVAL_S

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents one buyer's emulation request."""
    job_id:           str
    app_type:         str          # "hotel", "sn", or "sa"
    lifetime_seconds: int
    submitted_at:     datetime
    start_tick:       int          # tick at which this job becomes active
    end_tick:         int          # tick at which this job expires
    buyer_name:       str = "unknown"  # buyer identity for namespace naming
    status:           str = "queued"   # queued → running → done

    def is_active(self, current_tick: int) -> bool:
        return self.start_tick <= current_tick < self.end_tick

    def is_expired(self, current_tick: int) -> bool:
        return current_tick >= self.end_tick

    def to_dict(self) -> Dict:
        return {
            "job_id":           self.job_id,
            "app_type":         self.app_type,
            "lifetime_seconds": self.lifetime_seconds,
            "submitted_at":     self.submitted_at.isoformat(),
            "start_tick":       self.start_tick,
            "end_tick":         self.end_tick,
            "buyer_name":       self.buyer_name,
            "status":           self.status,
        }


class Timeline:
    """
    Manages the lifecycle of all active emulation jobs.

    Thread-safe: all public methods acquire a lock before modifying state.

    How it works:
      - Jobs are submitted via add_job() and become active at the next tick
      - Every tick() call advances the clock by one and removes expired jobs
      - get_composition() returns current {app_type: count}
      - composition_changed() returns True if last tick caused a change
    """

    def __init__(self):
        self._lock             = threading.Lock()
        self._jobs: List[Job]  = []          # all jobs (active + queued)
        self._current_tick     = 0
        self._last_composition: Dict[str, int] = {}
        self._composition_changed = False
        self._expired_jobs: List[Job] = []   # jobs that expired this tick

    # ── Public API ────────────────────────────────────────────────────────────

    def add_job(
        self,
        app_type:         str,
        lifetime_seconds: int,
        buyer_name:       str = "unknown",
    ) -> Job:
        """
        Register a new job. It becomes active at the next tick boundary.
        Returns the created Job object.
        """
        with self._lock:
            job_id     = str(uuid.uuid4())[:8]
            start_tick = self._current_tick + 1   # active from next tick
            end_tick   = start_tick + max(1, lifetime_seconds // TICK_INTERVAL_S)

            # sanitize buyer_name for use in namespace (lowercase, no spaces)
            safe_buyer = buyer_name.strip().lower().replace(" ", "_") or "unknown"

            job = Job(
                job_id           = job_id,
                app_type         = app_type,
                lifetime_seconds = lifetime_seconds,
                submitted_at     = datetime.now(),
                start_tick       = start_tick,
                end_tick         = end_tick,
                buyer_name       = safe_buyer,
                status           = "queued",
            )
            self._jobs.append(job)

            logger.info(
                f"Job added: {job_id} | app={app_type} | "
                f"buyer={safe_buyer} | lifetime={lifetime_seconds}s | "
                f"ticks [{start_tick}, {end_tick})"
            )
            return job

    def tick(self) -> bool:
        """
        Advance the clock by one tick.
        - Activates jobs whose start_tick == current_tick
        - Removes jobs whose end_tick <= current_tick
        - Detects composition changes

        Returns True if composition changed this tick.
        """
        with self._lock:
            self._current_tick += 1
            tick = self._current_tick

            # activate queued jobs that start this tick
            for job in self._jobs:
                if job.status == "queued" and job.start_tick <= tick:
                    job.status = "running"
                    logger.info(
                        f"Job started: {job.job_id} | "
                        f"app={job.app_type} | buyer={job.buyer_name}"
                    )

            # collect expired jobs before removing
            self._expired_jobs = [j for j in self._jobs if j.is_expired(tick)]
            for job in self._expired_jobs:
                job.status = "done"
                logger.info(
                    f"Job expired: {job.job_id} | "
                    f"app={job.app_type} | buyer={job.buyer_name}"
                )
            self._jobs = [j for j in self._jobs if not j.is_expired(tick)]

            # compute new composition
            new_composition = self._compute_composition(tick)

            # detect change
            self._composition_changed = (new_composition != self._last_composition)
            if self._composition_changed:
                logger.info(
                    f"Composition changed: {self._last_composition} → {new_composition}"
                )
            self._last_composition = new_composition

            return self._composition_changed

    def get_composition(self) -> Dict[str, int]:
        """Return current {app_type: count} for all active jobs."""
        with self._lock:
            return dict(self._last_composition)

    def composition_changed(self) -> bool:
        """True if the last tick caused a composition change."""
        with self._lock:
            return self._composition_changed

    def get_active_jobs(self) -> List[Dict]:
        """Return list of all active job dicts."""
        with self._lock:
            tick = self._current_tick
            return [j.to_dict() for j in self._jobs if j.is_active(tick)]

    def get_all_jobs(self) -> List[Dict]:
        """Return list of all jobs including queued."""
        with self._lock:
            return [j.to_dict() for j in self._jobs]

    def get_expired_jobs(self) -> List[Job]:
        """Return jobs that expired during the last tick."""
        with self._lock:
            return list(self._expired_jobs)

    def get_active_job_objects(self) -> List[Job]:
        """Return active Job objects (not dicts) — for namespace mapping."""
        with self._lock:
            tick = self._current_tick
            return [j for j in self._jobs if j.is_active(tick)]

    def get_current_tick(self) -> int:
        with self._lock:
            return self._current_tick

    def is_empty(self) -> bool:
        """True if no active or queued jobs."""
        with self._lock:
            return len(self._jobs) == 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_composition(self, tick: int) -> Dict[str, int]:
        """Count active jobs per app type at the given tick."""
        composition: Dict[str, int] = {}
        for job in self._jobs:
            if job.is_active(tick):
                composition[job.app_type] = composition.get(job.app_type, 0) + 1
        return composition