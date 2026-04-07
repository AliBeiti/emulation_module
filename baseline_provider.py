"""
baseline_provider.py

Loads real baseline (idle) node metrics from synced_node.csv and serves
them sequentially, one row per tick. Cycles through all clean rows.

Used by Aggregator when no pods are active — replaces the zeroed
empty_snapshot() with real idle node behavior.

Output format matches aggregator.compute() exactly so the API sees
no difference between idle and loaded states.
"""

import logging
import os
import pandas as pd
from datetime import datetime
from typing import Optional, Dict

from config import (
    NODE_CPU_MCORES, NODE_RAM_MI, NODE_DISK_MB, NODE_PSI_MAX_US,
    BASE_DIR,
)

logger = logging.getLogger(__name__)

# Expected baseline CSV path (alongside the emulation module)
BASELINE_CSV = os.path.join(BASE_DIR, "baseline", "synced_node.csv")

# PSI sanity cap — anything above this is a bad row
PSI_SANITY_CAP = NODE_PSI_MAX_US  # 5,000,000 us


class BaselineProvider:
    """
    Serves real idle node metrics tick by tick.
    Cycles through clean baseline rows indefinitely.
    """

    def __init__(self, csv_path: Optional[str] = None):
        self._rows = []
        self._index = 0
        path = csv_path or BASELINE_CSV
        self._load(path)

    def _load(self, path: str):
        if not os.path.exists(path):
            logger.warning(
                f"Baseline CSV not found at {path}. "
                f"Idle metrics will use static fallback values."
            )
            return

        df = pd.read_csv(path)

        # keep only BASELINE rows if app_name column exists
        if "app_name" in df.columns:
            df = df[df["app_name"] == "BASELINE"].copy()

        # drop bad PSI rows (sensor glitches / warmup artifacts)
        if "cpu_psi_some_us" in df.columns:
            before = len(df)
            df = df[df["cpu_psi_some_us"] <= PSI_SANITY_CAP].copy()
            dropped = before - len(df)
            if dropped:
                logger.warning(f"Dropped {dropped} baseline rows with PSI > {PSI_SANITY_CAP}")

        df = df.reset_index(drop=True)
        self._rows = df.to_dict(orient="records")
        logger.info(f"Baseline loaded: {len(self._rows)} clean rows from {path}")

    def next(self) -> Dict:
        """
        Return the next baseline row formatted as aggregator output.
        Cycles back to start after exhausting all rows.
        """
        if not self._rows:
            return self._static_fallback()

        row = self._rows[self._index % len(self._rows)]
        self._index += 1
        return self._to_metrics(row)

    def peek(self) -> Dict:
        """Return current row without advancing the index."""
        if not self._rows:
            return self._static_fallback()
        row = self._rows[self._index % len(self._rows)]
        return self._to_metrics(row)

    def is_loaded(self) -> bool:
        return len(self._rows) > 0

    def _to_metrics(self, row: dict) -> Dict:
        """Convert a raw baseline CSV row to aggregator-compatible output."""
        cpu_mcores   = float(row.get("cpu_usage_mcores", 2400))
        ram_mi       = float(row.get("ram_usage_mi",     25480))
        psi_us       = float(row.get("cpu_psi_some_us",  1_000_000))
        sched_ms     = float(row.get("sched_total_ms",   15000))
        dstate_ms    = float(row.get("dstate_total_ms",  80))
        softirq_ms   = float(row.get("softirq_total_ms", 64))
        power_w      = float(row.get("node_cpu_watts",   218))
        disk_used_gb = float(row.get("disk_used_gb",     128))

        disk_used_mb = disk_used_gb * 1024

        cpu_pct      = min((cpu_mcores  / NODE_CPU_MCORES) * 100, 100.0)
        ram_pct      = min((ram_mi      / NODE_RAM_MI)     * 100, 100.0)
        psi_pct      = min((psi_us      / NODE_PSI_MAX_US) * 100, 100.0)
        disk_pct     = min((disk_used_mb / max(NODE_DISK_MB, 1)) * 100, 100.0)

        return {
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_usage_pct":    round(cpu_pct,    2),
            "cpu_psi_some_pct": round(psi_pct,    2),
            "ram_usage_pct":    round(ram_pct,    2),
            "ram_usage_mi":     round(ram_mi,     2),
            "disk_used_pct":    round(disk_pct,   2),
            "sched_total_ms":   round(sched_ms,   2),
            "dstate_total_ms":  round(dstate_ms,  2),
            "softirq_total_ms": round(softirq_ms, 2),
            "node_cpu_watts":   round(power_w,    2),
            # internal fields
            "_cpu_mcores":      round(cpu_mcores,   2),
            "_ram_mi":          round(ram_mi,       2),
            "_psi_us":          round(psi_us,       2),
            "_disk_space_mb":   round(disk_used_mb, 2),
            "_disk_usage_mb":   round(disk_used_mb, 2),
            "_pod_count":       0,
        }

    def _static_fallback(self) -> Dict:
        """Used when no baseline CSV is available — realistic idle constants."""
        return {
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_usage_pct":    0.94,
            "cpu_psi_some_pct": 20.0,
            "ram_usage_pct":    1.22,
            "ram_usage_mi":     25480.0,
            "disk_used_pct":    14.63,
            "sched_total_ms":   15000.0,
            "dstate_total_ms":  80.0,
            "softirq_total_ms": 64.0,
            "node_cpu_watts":   218.0,
            "_cpu_mcores":      2400.0,
            "_ram_mi":          25480.0,
            "_psi_us":          1_000_000.0,
            "_disk_space_mb":   131072.0,
            "_disk_usage_mb":   131072.0,
            "_pod_count":       0,
        }