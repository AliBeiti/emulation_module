"""
dry_run_testbench.py

Standalone dry run of the Emulation Module.
Simulates realistic scenarios (idle → load → idle → mixed load → ...)
using real module components but without FastAPI, Kubernetes, or real time.

Output: a multi-panel plot showing node-level metrics across the full timeline
with vertical lines marking every transition event.

Usage:
    python dry_run_testbench.py --datasets ./datasets --baseline ./baseline/synced_node.csv
    python dry_run_testbench.py --seed 42        # reproducible scenario
    python dry_run_testbench.py --events 8       # number of job events
"""

import os
import sys
import argparse
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── allow running from any directory ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("testbench")

# ── constants ─────────────────────────────────────────────────────────────────
NODE_CPU_MCORES = 256_000
NODE_RAM_MI     = 2 * 1024 * 1024   # 2,097,152
NODE_DISK_MB    = 878 * 1024        # 899,072
NODE_PSI_MAX_US = 5_000_000
EBPF_MAX_MS     = 256 * 5_000       # 1,280,000 ms
TICK_S          = 5
PSI_SANITY_CAP  = NODE_PSI_MAX_US

# Ridge PSI model coefficients
RIDGE_INTERCEPT = 0.854
RIDGE_PSI_SUM   = 0.463
RIDGE_PSI_WAV   = 0.484
RIDGE_NUM_PODS  = -0.003
RIDGE_ACT_PODS  = 0.008
ACTIVE_CPU_THRESH = 10


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Provider
# ─────────────────────────────────────────────────────────────────────────────

class BaselineProvider:
    def __init__(self, csv_path: str):
        self._rows = []
        self._idx  = 0
        self._load(csv_path)

    def _load(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Baseline CSV not found: {path} — using static fallback")
            return
        df = pd.read_csv(path)
        if "app_name" in df.columns:
            df = df[df["app_name"] == "BASELINE"].copy()
        if "cpu_psi_some_us" in df.columns:
            df = df[df["cpu_psi_some_us"] <= PSI_SANITY_CAP].copy()
        self._rows = df.reset_index(drop=True).to_dict(orient="records")
        logger.info(f"Baseline: {len(self._rows)} clean rows loaded")

    def next(self) -> Dict:
        if not self._rows:
            return self._fallback()
        row = self._rows[self._idx % len(self._rows)]
        self._idx += 1
        return self._convert(row)

    def _convert(self, row: dict) -> Dict:
        cpu   = float(row.get("cpu_usage_mcores", 2400))
        ram   = float(row.get("ram_usage_mi",     25480))
        psi   = float(row.get("cpu_psi_some_us",  1_000_000))
        sched = float(row.get("sched_total_ms",   15000))
        dst   = float(row.get("dstate_total_ms",  80))
        sirq  = float(row.get("softirq_total_ms", 64))
        pwr   = float(row.get("node_cpu_watts",   218))
        disk_gb = float(row.get("disk_used_gb",   128))
        disk_mb = disk_gb * 1024
        return {
            "cpu_usage_pct":    min(cpu  / NODE_CPU_MCORES * 100, 100),
            "cpu_psi_some_pct": min(psi  / NODE_PSI_MAX_US * 100, 100),
            "ram_usage_pct":    min(ram  / NODE_RAM_MI     * 100, 100),
            "ram_usage_mi":     ram,
            "disk_used_pct":    min(disk_mb / NODE_DISK_MB * 100, 100),
            "sched_total_ms":   sched,
            "dstate_total_ms":  dst,
            "softirq_total_ms": sirq,
            "node_cpu_watts":   pwr,
            "_cpu_mcores":      cpu,
            "_psi_us":          psi,
            "_pod_count":       0,
            "_phase":           "idle",
        }

    def _fallback(self) -> Dict:
        return {
            "cpu_usage_pct": 0.94, "cpu_psi_some_pct": 20.0,
            "ram_usage_pct": 1.22, "ram_usage_mi": 25480.0,
            "disk_used_pct": 14.63, "sched_total_ms": 15000.0,
            "dstate_total_ms": 80.0, "softirq_total_ms": 64.0,
            "node_cpu_watts": 218.0, "_cpu_mcores": 2400.0,
            "_psi_us": 1_000_000.0, "_pod_count": 0, "_phase": "idle",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────────────────────

class DatasetLoader:
    def __init__(self, datasets_dir: str, index_path: str):
        self._dir   = datasets_dir
        self._index = {}
        self._windows: Dict[int, List[Dict]] = {}
        self._total  = 0
        self._key    = None
        self._cursor = 0

        if os.path.exists(index_path):
            import json
            with open(index_path) as f:
                self._index = json.load(f)
            logger.info(f"Dataset index: {len(self._index)} entries")
        else:
            logger.error(f"Dataset index not found: {index_path}")

    def available_keys(self) -> List[str]:
        return list(self._index.keys())

    def select(self, composition: Dict[str, int]) -> Optional[str]:
        """Return the best matching dataset key for a composition."""
        h = composition.get("hotel", 0)
        s = composition.get("sn",    0)
        a = composition.get("sa",    0)
        key = f"h{h}s{s}a{a}"
        if key in self._index:
            return key
        # find closest without exceeding counts
        best, best_score = None, float("inf")
        for k, entry in self._index.items():
            bh = entry.get("hotel", 0)
            bs = entry.get("sn",    0)
            ba = entry.get("sa",    0)
            if bh > h or bs > s or ba > a:
                continue
            gap = (h - bh) + (s - bs) + (a - ba)
            miss = sum([h > 0 and bh == 0, s > 0 and bs == 0, a > 0 and ba == 0])
            score = gap + miss * 100
            if score < best_score:
                best_score, best = score, k
        return best

    def load(self, key: str) -> bool:
        entry = self._index.get(key)
        if not entry:
            return False
        path = os.path.join(os.path.dirname(self._dir), entry["file"])
        if not os.path.exists(path):
            logger.error(f"Dataset file missing: {path}")
            return False
        df = pd.read_csv(path)
        self._windows = {}
        for win_idx, grp in df.groupby("window_index"):
            self._windows[int(win_idx)] = grp.to_dict(orient="records")
        self._total  = len(self._windows)
        self._key    = key
        self._cursor = 0
        logger.info(f"Loaded dataset {key}: {self._total} windows")
        return True

    def current_pods(self) -> List[Dict]:
        if not self._windows:
            return []
        win = self._cursor % self._total
        return list(self._windows.get(win, []))

    def advance(self):
        if self._total:
            self._cursor = (self._cursor + 1) % self._total

    @property
    def key(self) -> Optional[str]:
        return self._key


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator (inline — no imports from module)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(pod_rows: List[Dict], baseline: BaselineProvider, composition: Dict) -> Dict:
    if not pod_rows:
        snap = baseline.next()
        snap["_phase"] = "idle"
        return snap

    cpu   = sum(r.get("cpu_usage_mcores", 0) for r in pod_rows)
    ram   = sum(r.get("ram_usage_mi",     0) for r in pod_rows)
    disk  = sum(r.get("disk_space_mb",    0) for r in pod_rows)
    sched = sum(r.get("sched_total_ms",   0) for r in pod_rows)
    dst   = sum(r.get("dstate_total_ms",  0) for r in pod_rows)
    sirq  = sum(r.get("softirq_total_ms", 0) for r in pod_rows)
    pwr   = sum(r.get("pod_cpu_watts",    0) for r in pod_rows)

    # PSI ridge model
    psi_vals = [r.get("cpu_psi_some_us", 0) for r in pod_rows]
    cpu_vals = [r.get("cpu_usage_mcores", 0) for r in pod_rows]
    psi_sum  = sum(psi_vals)
    total_cpu = sum(cpu_vals)
    num_pods  = len(pod_rows)
    act_pods  = sum(1 for c in cpu_vals if c > ACTIVE_CPU_THRESH)
    psi_wav   = (sum(p*c for p,c in zip(psi_vals, cpu_vals)) / total_cpu
                 if total_cpu > 0 else psi_sum / max(num_pods, 1))
    log_res  = (RIDGE_PSI_SUM  * np.log1p(psi_sum) +
                RIDGE_PSI_WAV  * np.log1p(psi_wav)  +
                RIDGE_NUM_PODS * num_pods +
                RIDGE_ACT_PODS * act_pods +
                RIDGE_INTERCEPT)
    raw_psi  = float(np.expm1(log_res))
    cpu_util = min(total_cpu / NODE_CPU_MCORES, 1.0)
    psi_us   = max(0.0, min(raw_psi, cpu_util * NODE_PSI_MAX_US, NODE_PSI_MAX_US))

    label = "+".join(f"{v}{k}" for k, v in sorted(composition.items()) if v > 0)

    return {
        "cpu_usage_pct":    min(cpu  / NODE_CPU_MCORES * 100, 100),
        "cpu_psi_some_pct": min(psi_us / NODE_PSI_MAX_US * 100, 100),
        "ram_usage_pct":    min(ram  / NODE_RAM_MI      * 100, 100),
        "ram_usage_mi":     ram,
        "disk_used_pct":    min(disk / max(NODE_DISK_MB, 1) * 100, 100),
        "sched_total_ms":   sched,
        "dstate_total_ms":  dst,
        "softirq_total_ms": sirq,
        "node_cpu_watts":   pwr,
        "_cpu_mcores":      cpu,
        "_psi_us":          psi_us,
        "_pod_count":       num_pods,
        "_phase":           label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario generator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioEvent:
    kind:         str           # "idle" or "load"
    duration_ticks: int
    composition:  Dict[str, int] = field(default_factory=dict)
    label:        str = ""


def generate_scenario(n_events: int, rng: random.Random) -> List[ScenarioEvent]:
    """
    Generate a diverse sequence of scenario events.
    Ensures variety: single apps, mixed apps, varying loads, idle gaps.
    """
    app_types = ["hotel", "sn", "sa"]
    events: List[ScenarioEvent] = []

    # always start with idle
    events.append(ScenarioEvent(
        kind="idle",
        duration_ticks=rng.randint(6, 14),
        label="idle"
    ))

    for i in range(n_events):
        # random idle gap between loads
        idle_ticks = rng.randint(4, 12)
        events.append(ScenarioEvent(
            kind="idle",
            duration_ticks=idle_ticks,
            label="idle"
        ))

        # pick a random composition
        comp = {}
        n_app_types = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        chosen_apps = rng.sample(app_types, n_app_types)
        for app in chosen_apps:
            max_inst = {"hotel": 8, "sn": 7, "sa": 6}[app]
            comp[app] = rng.randint(1, max_inst)

        load_ticks = rng.randint(10, 30)
        label = "+".join(f"{v}{k}" for k, v in sorted(comp.items()))
        events.append(ScenarioEvent(
            kind="load",
            duration_ticks=load_ticks,
            composition=comp,
            label=label,
        ))

    # end with idle
    events.append(ScenarioEvent(kind="idle", duration_ticks=10, label="idle"))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    events: List[ScenarioEvent],
    loader: DatasetLoader,
    baseline: BaselineProvider,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run the scenario tick by tick.
    Returns:
        records    — list of metric snapshots (one per tick)
        transitions — list of {tick, label, kind} for plotting
    """
    records:     List[Dict] = []
    transitions: List[Dict] = []
    tick = 0

    for event in events:
        logger.info(
            f"Event: [{event.kind}] {event.label} "
            f"({event.duration_ticks} ticks = {event.duration_ticks*TICK_S}s)"
        )

        transitions.append({"tick": tick, "label": event.label, "kind": event.kind})

        if event.kind == "idle":
            for _ in range(event.duration_ticks):
                snap = baseline.next()
                snap["_phase"] = "idle"
                snap["_tick"]  = tick
                records.append(snap)
                tick += 1

        else:  # load
            # select and load dataset
            key = loader.select(event.composition)
            if key is None:
                logger.warning(f"No dataset for {event.composition} — skipping event")
                continue
            if key != loader.key:
                loader.load(key)

            for _ in range(event.duration_ticks):
                pods = loader.current_pods()
                snap = aggregate(pods, baseline, event.composition)
                snap["_phase"] = event.label
                snap["_tick"]  = tick
                records.append(snap)
                loader.advance()
                tick += 1

    return records, transitions


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

# color palette per phase
PHASE_COLORS = {
    "idle":   "#90A4AE",
    "hotel":  "#1E88E5",
    "sn":     "#43A047",
    "sa":     "#E53935",
    "mixed":  "#FB8C00",
}

def phase_color(label: str) -> str:
    if label == "idle":
        return PHASE_COLORS["idle"]
    apps = [a for a in ["hotel", "sn", "sa"] if a in label]
    if len(apps) == 1:
        return PHASE_COLORS[apps[0]]
    return PHASE_COLORS["mixed"]


def plot_results(
    records:     List[Dict],
    transitions: List[Dict],
    output_path: str,
):
    df = pd.DataFrame(records)
    ticks = df["_tick"].values
    time_s = ticks * TICK_S   # x-axis in seconds

    metrics = [
        ("cpu_usage_pct",    "CPU Usage (%)",       "steelblue",   (0, 100)),
        ("ram_usage_pct",    "RAM Usage (%)",        "darkorange",  (0, None)),
        ("cpu_psi_some_pct", "PSI Some (%)",         "crimson",     (0, 100)),
        ("disk_used_pct",    "Disk Used (%)",        "mediumpurple",(0, None)),
        ("sched_total_ms",   "Sched (ms)",           "teal",        (0, None)),
        ("dstate_total_ms",  "Dstate (ms)",          "sienna",      (0, None)),
        ("softirq_total_ms", "Softirq (ms)",         "olive",       (0, None)),
        ("node_cpu_watts",   "Power (W)",            "darkgreen",   (0, None)),
    ]

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(18, n * 2.8), sharex=True)
    fig.suptitle("Emulation Module Dry Run — Node-Level Metrics Timeline",
                 fontsize=15, fontweight="bold", y=1.001)

    for ax, (col, ylabel, color, ylim) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        vals = df[col].values
        ax.plot(time_s, vals, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(time_s, vals, alpha=0.12, color=color)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

        if ylim[0] is not None:
            ax.set_ylim(bottom=ylim[0])
        if ylim[1] is not None:
            ax.set_ylim(top=ylim[1])

        # draw transition lines
        for tr in transitions:
            t_s = tr["tick"] * TICK_S
            c   = phase_color(tr["label"])
            ax.axvline(x=t_s, color=c, linewidth=1.5,
                       linestyle="--" if tr["kind"] == "idle" else "-",
                       alpha=0.75)

        # shade background by phase
        for i, tr in enumerate(transitions):
            t_start = tr["tick"] * TICK_S
            t_end   = (transitions[i+1]["tick"] * TICK_S
                       if i+1 < len(transitions)
                       else time_s[-1])
            c = phase_color(tr["label"])
            ax.axvspan(t_start, t_end, alpha=0.06, color=c)

    # x-axis label only on bottom
    axes[-1].set_xlabel("Time (seconds)", fontsize=10)

    # ── legend ────────────────────────────────────────────────────────────────
    seen_labels = {}
    for tr in transitions:
        lbl = tr["label"]
        if lbl not in seen_labels:
            seen_labels[lbl] = phase_color(lbl)

    legend_handles = [
        mpatches.Patch(color=c, label=lbl, alpha=0.7)
        for lbl, c in seen_labels.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8,
        title="Phase",
        framealpha=0.85,
    )

    # ── transition annotations on top axes ───────────────────────────────────
    ax_top = axes[0]
    for tr in transitions:
        t_s = tr["tick"] * TICK_S
        ax_top.annotate(
            tr["label"],
            xy=(t_s, ax_top.get_ylim()[1]),
            xytext=(t_s + 1, ax_top.get_ylim()[1] * 0.92),
            fontsize=6.5,
            color=phase_color(tr["label"]),
            rotation=45,
            ha="left",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(records: List[Dict], transitions: List[Dict]):
    df = pd.DataFrame(records)
    print("\n" + "="*60)
    print("DRY RUN SUMMARY")
    print("="*60)
    print(f"Total ticks simulated : {len(records)}")
    print(f"Total time simulated  : {len(records) * TICK_S}s "
          f"({len(records) * TICK_S / 60:.1f} min)")
    print(f"Transition events     : {len(transitions)}")
    print()

    print("Phase breakdown:")
    for tr in transitions:
        print(f"  tick {tr['tick']:>4} | {tr['kind']:<4} | {tr['label']}")

    print()
    print("Metric ranges (min / mean / max):")
    cols = ["cpu_usage_pct", "ram_usage_pct", "cpu_psi_some_pct",
            "disk_used_pct", "node_cpu_watts", "sched_total_ms"]
    for c in cols:
        if c in df.columns:
            print(f"  {c:<22} {df[c].min():8.2f} / {df[c].mean():8.2f} / {df[c].max():8.2f}")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Emulation Module Dry Run Test Bench")
    parser.add_argument("--datasets",  default="./datasets",
                        help="Path to datasets directory")
    parser.add_argument("--baseline",  default="./baseline/synced_node.csv",
                        help="Path to baseline synced_node.csv")
    parser.add_argument("--index",     default=None,
                        help="Path to dataset_index.json (default: datasets/dataset_index.json)")
    parser.add_argument("--events",    type=int, default=6,
                        help="Number of load events in the scenario (default: 6)")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output",    default="testbench_output.png",
                        help="Output plot filename")
    parser.add_argument("--csv",       default=None,
                        help="Also save raw tick data to this CSV path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.seed is not None:
        np.random.seed(args.seed)
        logger.info(f"Random seed: {args.seed}")

    index_path = args.index or os.path.join(args.datasets, "dataset_index.json")

    # ── initialize components ─────────────────────────────────────────────────
    baseline = BaselineProvider(args.baseline)
    loader   = DatasetLoader(args.datasets, index_path)

    if not loader.available_keys():
        logger.error("No datasets found. Check --datasets and --index paths.")
        sys.exit(1)

    # ── generate scenario ─────────────────────────────────────────────────────
    events = generate_scenario(args.events, rng)

    print("\n" + "="*60)
    print("SCENARIO PLAN")
    print("="*60)
    for i, ev in enumerate(events):
        print(f"  {i:>2}. [{ev.kind:<4}] {ev.label:<25} "
              f"{ev.duration_ticks} ticks ({ev.duration_ticks*TICK_S}s)")
    print("="*60 + "\n")

    # ── run simulation ────────────────────────────────────────────────────────
    records, transitions = run_simulation(events, loader, baseline)

    # ── optional CSV export ───────────────────────────────────────────────────
    if args.csv:
        pd.DataFrame(records).to_csv(args.csv, index=False)
        logger.info(f"Raw tick data saved: {args.csv}")

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(records, transitions)

    # ── plot ──────────────────────────────────────────────────────────────────
    plot_results(records, transitions, args.output)
    print(f"\nPlot saved to: {args.output}")


if __name__ == "__main__":
    main()