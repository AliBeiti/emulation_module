"""
aggregator.py

Aggregates pod-level metrics to node-level metrics every tick.

Rules:
  CPU/RAM/disk/power → simple sum across all pods
  PSI               → Ridge regression model (no simple sum — double counting)
  sched/dstate/softirq → sum (already distributed top-down, guaranteed = node value)

When no pods are active (idle), returns real baseline node metrics
from BaselineProvider instead of zeroed values.

Output format matches /usage/latest API response:
  timestamp, cpu_usage_pct, cpu_psi_some_pct, ram_usage_pct, ram_usage_mi,
  disk_used_pct, sched_total_ms, dstate_total_ms, softirq_total_ms, node_cpu_watts
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from config import (
    NODE_CPU_MCORES, NODE_RAM_MI, NODE_DISK_MB,
    NODE_PSI_MAX_US,
    RIDGE_INTERCEPT, RIDGE_PSI_SUM, RIDGE_PSI_WAV,
    RIDGE_NUM_PODS, RIDGE_ACT_PODS, ACTIVE_CPU_THRESH,
)
from baseline_provider import BaselineProvider

logger = logging.getLogger(__name__)


class Aggregator:
    """
    Computes node-level metrics from a list of pod row dicts.
    When pod_rows is empty, serves real idle metrics from BaselineProvider
    instead of returning zeros.
    """

    def __init__(self, baseline_provider: Optional[BaselineProvider] = None):
        """
        Args:
            baseline_provider: optional BaselineProvider instance.
                               If None, a new one is created automatically.
        """
        self._baseline = baseline_provider or BaselineProvider()

    def compute(self, pod_rows: List[Dict]) -> Optional[Dict]:
        """
        Aggregate pod rows to node-level metrics.

        Args:
            pod_rows: list of pod dicts from ReplayEngine.get_current_window_pods()

        Returns:
            Node-level metrics dict ready for API exposure.
            When empty → real baseline metrics (not zeros).
        """
        if not pod_rows:
            # hard switch: serve next baseline row instead of zeros
            return self._baseline.next()

        # ── Simple sums ───────────────────────────────────────────────────────
        cpu_mcores     = sum(r.get("cpu_usage_mcores", 0) for r in pod_rows)
        ram_mi         = sum(r.get("ram_usage_mi",     0) for r in pod_rows)
        disk_space_mb  = sum(r.get("disk_space_mb",    0) for r in pod_rows)
        disk_usage_mb  = sum(r.get("disk_usage_mb",    0) for r in pod_rows)
        sched_ms       = sum(r.get("sched_total_ms",   0) for r in pod_rows)
        dstate_ms      = sum(r.get("dstate_total_ms",  0) for r in pod_rows)
        softirq_ms     = sum(r.get("softirq_total_ms", 0) for r in pod_rows)
        power_watts    = sum(r.get("pod_cpu_watts",    0) for r in pod_rows)

        # ── PSI via Ridge model ───────────────────────────────────────────────
        psi_us = self._estimate_psi(pod_rows)

        # ── Percentages ───────────────────────────────────────────────────────
        cpu_pct       = min((cpu_mcores   / NODE_CPU_MCORES) * 100, 100.0)
        ram_pct       = min((ram_mi       / NODE_RAM_MI)     * 100, 100.0)
        psi_pct       = min((psi_us       / NODE_PSI_MAX_US) * 100, 100.0)
        disk_used_pct = min((disk_space_mb / max(NODE_DISK_MB, 1)) * 100, 100.0)

        return {
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_usage_pct":    round(cpu_pct,       2),
            "cpu_psi_some_pct": round(psi_pct,       2),
            "ram_usage_pct":    round(ram_pct,       2),
            "ram_usage_mi":     round(ram_mi,        2),
            "disk_used_pct":    round(disk_used_pct, 2),
            "sched_total_ms":   round(sched_ms,      2),
            "dstate_total_ms":  round(dstate_ms,     2),
            "softirq_total_ms": round(softirq_ms,    2),
            "node_cpu_watts":   round(power_watts,   2),
            # internal extras (not in API but useful for status endpoint)
            "_cpu_mcores":      round(cpu_mcores,    2),
            "_ram_mi":          round(ram_mi,        2),
            "_psi_us":          round(psi_us,        2),
            "_disk_space_mb":   round(disk_space_mb, 2),
            "_disk_usage_mb":   round(disk_usage_mb, 2),
            "_pod_count":       len(pod_rows),
        }

    def _estimate_psi(self, pod_rows: List[Dict]) -> float:
        """
        Apply Ridge regression model to estimate node PSI from pod pool.

        Model: log(node_psi) = 0.463×log(psi_sum) + 0.484×log(psi_wav)
                               − 0.003×num_pods + 0.008×active_pods + 0.854
        node_psi_us = exp(result) − 1
        """
        psi_values  = [r.get("cpu_psi_some_us",  0) for r in pod_rows]
        cpu_values  = [r.get("cpu_usage_mcores", 0) for r in pod_rows]

        psi_sum     = sum(psi_values)
        total_cpu   = sum(cpu_values)
        num_pods    = len(pod_rows)
        active_pods = sum(1 for c in cpu_values if c > ACTIVE_CPU_THRESH)

        # CPU-weighted average PSI
        if total_cpu > 0:
            psi_wav = sum(p * c for p, c in zip(psi_values, cpu_values)) / total_cpu
        else:
            psi_wav = psi_sum / max(num_pods, 1)

        log_result = (
            RIDGE_PSI_SUM  * np.log1p(psi_sum) +
            RIDGE_PSI_WAV  * np.log1p(psi_wav) +
            RIDGE_NUM_PODS * num_pods +
            RIDGE_ACT_PODS * active_pods +
            RIDGE_INTERCEPT
        )
        raw_psi = float(np.expm1(log_result))

        # PSI/CPU logical constraint: PSI cannot exceed cpu_utilization × max
        cpu_util = min(total_cpu / NODE_CPU_MCORES, 1.0)
        psi_cap  = cpu_util * NODE_PSI_MAX_US
        return max(0.0, min(raw_psi, psi_cap, NODE_PSI_MAX_US))