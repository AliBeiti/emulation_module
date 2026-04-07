"""
analyze_testbench.py

Reads the CSV output from dry_run_testbench.py and automatically
detects problems across three categories:

  1. Oscillation      — values jumping too much within a stable phase
  2. Transition       — unrealistic jumps or wrong direction at boundaries
  3. Physical validity — values outside allowed bounds or broken correlations

Outputs:
  - Console report with flagged issues
  - analysis_output.png — diagnostic plots highlighting problems

Usage:
    python analyze_testbench.py --csv run1.csv
    python analyze_testbench.py --csv run1.csv --output my_analysis.png
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# ── physical bounds ────────────────────────────────────────────────────────────
BOUNDS = {
    "cpu_usage_pct":    (0,   100),
    "ram_usage_pct":    (0,   100),
    "cpu_psi_some_pct": (0,   100),
    "disk_used_pct":    (0,   100),
    "sched_total_ms":   (0,   1_280_000),
    "dstate_total_ms":  (0,   1_280_000),
    "softirq_total_ms": (0,   1_280_000),
    "node_cpu_watts":   (150, 700),
}

# ── expected idle ranges (from real baseline stats) ────────────────────────────
IDLE_EXPECTED = {
    "cpu_usage_pct":    (0.0,  2.5),
    "ram_usage_pct":    (1.2,  1.3),
    "cpu_psi_some_pct": (0.0,  50.0),
    "node_cpu_watts":   (210,  240),
    "sched_total_ms":   (1000, 60000),
}

# ── thresholds ─────────────────────────────────────────────────────────────────
# CV (std/mean) above this within a phase = oscillation
OSCILLATION_CV_THRESHOLD = 0.5

# max tick-to-tick change within a stable phase (as fraction of mean)
SPIKE_THRESHOLD = 1.5

# minimum expected jump at idle→load transition (cpu_usage_pct)
MIN_LOAD_JUMP_CPU = 0.5   # at least 0.5% cpu increase expected

ISSUES = []   # global collector


def flag(category: str, metric: str, detail: str, severity: str = "WARN"):
    ISSUES.append({
        "category": category,
        "metric":   metric,
        "detail":   detail,
        "severity": severity,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. Physical validity checks
# ─────────────────────────────────────────────────────────────────────────────

def check_bounds(df: pd.DataFrame):
    print("\n[1] Checking physical bounds...")
    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        n_low  = (df[col] < lo).sum()
        n_high = (df[col] > hi).sum()
        if n_low > 0:
            flag("Physical", col,
                 f"{n_low} ticks below minimum ({lo}). "
                 f"Min seen: {df[col].min():.2f}", "ERROR")
        if n_high > 0:
            flag("Physical", col,
                 f"{n_high} ticks above maximum ({hi}). "
                 f"Max seen: {df[col].max():.2f}", "ERROR")
    print(f"   Done. Issues so far: {len(ISSUES)}")


def check_correlations(df: pd.DataFrame):
    """
    Check expected metric relationships:
    - high CPU → high PSI (during load)
    - high CPU → high sched
    - PSI and sched should correlate
    """
    print("\n[2] Checking metric correlations...")
    load_df = df[df["_phase"] != "idle"]
    if len(load_df) < 5:
        print("   Not enough load ticks to check correlations.")
        return

    pairs = [
        ("cpu_usage_pct",    "cpu_psi_some_pct", 0.3),
        ("cpu_usage_pct",    "sched_total_ms",   0.3),
        ("cpu_psi_some_pct", "sched_total_ms",   0.4),
    ]
    for a, b, min_corr in pairs:
        if a not in load_df.columns or b not in load_df.columns:
            continue
        corr = load_df[a].corr(load_df[b])
        if corr < min_corr:
            flag("Correlation", f"{a} vs {b}",
                 f"Pearson r={corr:.3f} (expected >= {min_corr}). "
                 f"Metrics may be decoupled.",
                 "WARN" if corr > 0 else "ERROR")
        else:
            print(f"   {a} vs {b}: r={corr:.3f} OK")


def check_idle_values(df: pd.DataFrame):
    """Idle metrics should match real baseline ranges."""
    print("\n[3] Checking idle metric ranges...")
    idle_df = df[df["_phase"] == "idle"]
    if len(idle_df) == 0:
        print("   No idle ticks found.")
        return
    for col, (lo, hi) in IDLE_EXPECTED.items():
        if col not in idle_df.columns:
            continue
        mean_val = idle_df[col].mean()
        if not (lo <= mean_val <= hi):
            flag("IdleRange", col,
                 f"Idle mean={mean_val:.3f} outside expected [{lo}, {hi}]",
                 "WARN")
        else:
            print(f"   {col}: idle mean={mean_val:.3f} in [{lo}, {hi}] OK")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Oscillation checks
# ─────────────────────────────────────────────────────────────────────────────

def check_oscillation(df: pd.DataFrame):
    """
    Within each contiguous phase segment, check:
    - CV (std/mean) — high CV = oscillation
    - tick-to-tick spikes
    """
    print("\n[4] Checking oscillation within phases...")
    metrics = ["cpu_usage_pct", "cpu_psi_some_pct", "sched_total_ms",
               "node_cpu_watts", "ram_usage_pct"]

    # identify contiguous phase segments
    segments = _get_segments(df)

    for seg in segments:
        phase = seg["phase"]
        seg_df = seg["data"]
        if len(seg_df) < 3:
            continue

        for col in metrics:
            if col not in seg_df.columns:
                continue
            vals = seg_df[col].values
            mean = np.mean(vals)
            std  = np.std(vals)
            if mean < 0.01:
                continue

            cv = std / mean
            if cv > OSCILLATION_CV_THRESHOLD:
                flag("Oscillation", col,
                     f"Phase '{phase}': CV={cv:.2f} (>{OSCILLATION_CV_THRESHOLD}). "
                     f"Mean={mean:.2f}, Std={std:.2f}. High variability.",
                     "WARN")

            # tick-to-tick spikes
            diffs = np.abs(np.diff(vals))
            if mean > 0:
                max_jump = diffs.max() / mean
                if max_jump > SPIKE_THRESHOLD:
                    tick_idx = seg_df.index[np.argmax(diffs) + 1]
                    flag("Oscillation", col,
                         f"Phase '{phase}': spike at tick {tick_idx}. "
                         f"Jump={diffs.max():.2f} ({max_jump*100:.0f}% of mean).",
                         "WARN")

    print(f"   Done. Issues so far: {len(ISSUES)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transition checks
# ─────────────────────────────────────────────────────────────────────────────

def check_transitions(df: pd.DataFrame):
    """
    At each idle→load and load→idle boundary:
    - idle→load: CPU should increase
    - load→idle: CPU should drop back toward baseline
    - load→load: check direction makes sense
    """
    print("\n[5] Checking transitions...")
    segments = _get_segments(df)
    if len(segments) < 2:
        print("   Not enough segments to check transitions.")
        return

    for i in range(len(segments) - 1):
        curr = segments[i]
        nxt  = segments[i + 1]

        curr_phase = curr["phase"]
        nxt_phase  = nxt["phase"]

        # last few ticks of current, first few of next
        curr_tail = curr["data"]["cpu_usage_pct"].tail(3).mean()
        nxt_head  = nxt["data"]["cpu_usage_pct"].head(3).mean()
        delta     = nxt_head - curr_tail

        if curr_phase == "idle" and nxt_phase != "idle":
            # expect increase
            if delta < MIN_LOAD_JUMP_CPU:
                flag("Transition", "cpu_usage_pct",
                     f"idle→{nxt_phase}: CPU barely changed "
                     f"(Δ={delta:.2f}%). Load may not be applied.",
                     "WARN")
            else:
                print(f"   idle→{nxt_phase}: CPU +{delta:.2f}% OK")

        elif curr_phase != "idle" and nxt_phase == "idle":
            # expect decrease
            if delta > 0:
                flag("Transition", "cpu_usage_pct",
                     f"{curr_phase}→idle: CPU increased after job ended "
                     f"(Δ=+{delta:.2f}%). Expected decrease.",
                     "ERROR")
            else:
                print(f"   {curr_phase}→idle: CPU {delta:.2f}% OK")

        elif curr_phase != "idle" and nxt_phase != "idle":
            # load→load: just report, no strict rule
            print(f"   {curr_phase}→{nxt_phase}: load-to-load Δcpu={delta:+.2f}%")

    # check PSI at transitions too
    for i in range(len(segments) - 1):
        curr = segments[i]
        nxt  = segments[i + 1]
        if curr["phase"] == "idle" and nxt["phase"] != "idle":
            curr_psi = curr["data"]["cpu_psi_some_pct"].tail(3).mean()
            nxt_psi  = nxt["data"]["cpu_psi_some_pct"].head(3).mean()
            if nxt_psi < curr_psi * 0.5:
                flag("Transition", "cpu_psi_some_pct",
                     f"idle→{nxt['phase']}: PSI dropped at load start "
                     f"({curr_psi:.2f}% → {nxt_psi:.2f}%). Unexpected.",
                     "WARN")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_analysis(df: pd.DataFrame, output_path: str):
    segments = _get_segments(df)
    time_s   = df["_tick"].values * 5

    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    axes = axes.flatten()
    fig.suptitle("Testbench Analysis — Problem Detection", fontsize=14, fontweight="bold")

    plot_configs = [
        ("cpu_usage_pct",    "CPU Usage (%)",       "steelblue"),
        ("cpu_psi_some_pct", "PSI Some (%)",         "crimson"),
        ("ram_usage_pct",    "RAM Usage (%)",        "darkorange"),
        ("node_cpu_watts",   "Power (W)",            "darkgreen"),
        ("sched_total_ms",   "Sched (ms)",           "teal"),
        ("dstate_total_ms",  "Dstate (ms)",          "sienna"),
        ("softirq_total_ms", "Softirq (ms)",         "olive"),
        ("disk_used_pct",    "Disk Used (%)",        "mediumpurple"),
    ]

    for ax, (col, label, color) in zip(axes, plot_configs):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        vals = df[col].values
        ax.plot(time_s, vals, color=color, linewidth=1.0, alpha=0.85)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        # shade phases
        for seg in segments:
            t0 = seg["data"]["_tick"].iloc[0]  * 5
            t1 = seg["data"]["_tick"].iloc[-1] * 5
            c  = "#CCCCCC" if seg["phase"] == "idle" else "#FFE0B2"
            ax.axvspan(t0, t1, alpha=0.2, color=c)

        # highlight flagged ticks for this metric
        for issue in ISSUES:
            if issue["metric"] == col and "tick" in issue["detail"]:
                try:
                    tok  = [t for t in issue["detail"].split() if t.startswith("tick")]
                    tnum = int(tok[0].replace("tick", "").strip(".").strip(":"))
                    ax.axvline(x=tnum * 5, color="red", linewidth=1.5,
                               linestyle=":", alpha=0.9, label="spike")
                except Exception:
                    pass

        # mark oscillation segments
        for seg in segments:
            seg_df = seg["data"]
            if col not in seg_df.columns or len(seg_df) < 3:
                continue
            v = seg_df[col].values
            m = np.mean(v)
            if m < 0.01:
                continue
            cv = np.std(v) / m
            if cv > OSCILLATION_CV_THRESHOLD:
                t0 = seg_df["_tick"].iloc[0]  * 5
                t1 = seg_df["_tick"].iloc[-1] * 5
                ax.axvspan(t0, t1, alpha=0.15, color="red",
                           label=f"oscillation CV={cv:.2f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nAnalysis plot saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_segments(df: pd.DataFrame) -> List[Dict]:
    """Split dataframe into contiguous same-phase segments."""
    segments = []
    current_phase = None
    start_idx = 0
    for i, row in df.iterrows():
        phase = row["_phase"]
        if phase != current_phase:
            if current_phase is not None:
                segments.append({
                    "phase": current_phase,
                    "data":  df.loc[start_idx:i-1],
                })
            current_phase = phase
            start_idx = i
    if current_phase is not None:
        segments.append({"phase": current_phase, "data": df.loc[start_idx:]})
    return segments


def print_report():
    print("\n" + "="*65)
    print("ANALYSIS REPORT")
    print("="*65)
    if not ISSUES:
        print("  No issues found. Data looks healthy.")
    else:
        errors = [i for i in ISSUES if i["severity"] == "ERROR"]
        warns  = [i for i in ISSUES if i["severity"] == "WARN"]
        print(f"  Total issues: {len(ISSUES)}  "
              f"(ERRORs: {len(errors)}, WARNs: {len(warns)})\n")

        for sev, items in [("ERROR", errors), ("WARN", warns)]:
            if not items:
                continue
            print(f"  {'─'*60}")
            print(f"  [{sev}]")
            for iss in items:
                print(f"    [{iss['category']}] {iss['metric']}")
                print(f"      → {iss['detail']}")
    print("="*65)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Testbench Output Analyzer")
    parser.add_argument("--csv",    required=True, help="CSV output from dry_run_testbench.py")
    parser.add_argument("--output", default="analysis_output.png", help="Output plot path")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} ticks from {args.csv}")
    print(f"Phases found: {df['_phase'].unique().tolist()}")

    # run all checks
    check_bounds(df)
    check_correlations(df)
    check_idle_values(df)
    check_oscillation(df)
    check_transitions(df)

    # report and plot
    print_report()
    plot_analysis(df, args.output)


if __name__ == "__main__":
    main()