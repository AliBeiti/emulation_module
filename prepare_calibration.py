"""
prepare_calibration.py

Selects low-load experiment windows from real data and creates a
calibration dataset for the AC baseline calibration phase.

Target characteristics:
  - sched in 100,000–500,000ms range (6-digit, low load)
  - CPU < 50%
  - PSI > 15% (real pressure, not idle)
  - Mix of different experiments for variety

Output: calibration/calibration_pod.csv
        calibration/calibration_meta.json

Usage:
    python prepare_calibration.py --raw ./all_data --out ./calibration
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
NODE_CPU       = 256_000
NODE_PSI       = 5_000_000
PSI_SANITY_CAP = NODE_PSI

# Low-load target range
SCHED_MIN  = 100_000   # ms — above idle (5-digit)
SCHED_MAX  = 500_000   # ms — below high load
CPU_MAX    = 50.0      # % — not saturated
PSI_MIN    = 15.0      # % — real pressure, not idle

# Calibration window count — 5 minutes at 5s per window = 60 windows
# We use 120 to give more variety and allow looping
TARGET_WINDOWS = 120

# System pod prefixes to filter
SYS_PREFIXES = ('kube-system', 'monitoring', 'kepler', 'kube-flannel', 'kepler-old')

# Output columns
OUTPUT_COLS = [
    'window_index', 'pod_name',
    'cpu_usage_mcores', 'ram_usage_mi',
    'disk_space_mb', 'disk_usage_mb', 'disk_ios',
    'cpu_psi_some_us',
    'sched_total_ms', 'dstate_total_ms', 'softirq_total_ms',
    'pod_cpu_watts',
]

POWER_CPU_COEF  = 0.002412766401392309
POWER_RAM_COEF  = 3.124235379731155e-05
POWER_INTERCEPT = 0.07331256227741711


def is_low_load(node_df: pd.DataFrame) -> bool:
    """Check if experiment is in the low-load target range."""
    node_df = node_df[node_df['cpu_psi_some_us'] <= PSI_SANITY_CAP]
    sched_med = node_df['sched_total_ms'].median()
    cpu_pct   = node_df['cpu_usage_mcores'].median() / NODE_CPU * 100
    psi_pct   = node_df['cpu_psi_some_us'].median()  / NODE_PSI * 100
    return (SCHED_MIN <= sched_med <= SCHED_MAX and
            cpu_pct <= CPU_MAX and
            psi_pct >= PSI_MIN)


def load_corrected_pod(raw_dir: str, name: str) -> pd.DataFrame:
    """
    Load raw pod CSV, filter system pods, consolidate disk,
    distribute eBPF top-down from node CSV.
    """
    node_path = os.path.join(raw_dir, f"{name}_node.csv")
    pod_path  = os.path.join(raw_dir, f"{name}_pod.csv")

    if not os.path.exists(node_path) or not os.path.exists(pod_path):
        return pd.DataFrame()

    node_df = pd.read_csv(node_path)
    pod_df  = pd.read_csv(pod_path)

    # filter bad PSI
    node_df = node_df[node_df['cpu_psi_some_us'] <= PSI_SANITY_CAP].copy()

    # filter system pods
    app_pods = pod_df[~pod_df['pod_name'].apply(
        lambda x: any(str(x).startswith(p) for p in SYS_PREFIXES)
    )].copy()

    # align timestamps
    valid_ts = set(node_df['timestamp'])
    app_pods = app_pods[app_pods['timestamp'].isin(valid_ts)].copy()

    if len(app_pods) == 0:
        return pd.DataFrame()

    # consolidate disk
    if 'disk_read_mb' in app_pods.columns:
        app_pods['disk_usage_mb'] = (app_pods['disk_read_mb'].fillna(0) +
                                     app_pods['disk_write_mb'].fillna(0))
    if 'disk_read_ios' in app_pods.columns:
        app_pods['disk_ios'] = (app_pods['disk_read_ios'].fillna(0) +
                                app_pods['disk_write_ios'].fillna(0))
    if 'disk_space_mb' not in app_pods.columns:
        app_pods['disk_space_mb'] = 0.0
    if 'disk_usage_mb' not in app_pods.columns:
        app_pods['disk_usage_mb'] = 0.0
    if 'disk_ios' not in app_pods.columns:
        app_pods['disk_ios'] = 0.0

    # assign window_index
    ts_order = {ts: i for i, ts in
                enumerate(sorted(node_df['timestamp'].unique()))}
    app_pods['window_index'] = app_pods['timestamp'].map(ts_order)

    # distribute eBPF top-down per window
    results = []
    for ts, pod_grp in app_pods.groupby('timestamp'):
        node_row = node_df[node_df['timestamp'] == ts]
        if len(node_row) == 0:
            continue
        nr = node_row.iloc[0]

        grp = pod_grp.copy()
        n   = len(grp)

        psi_vals = grp['cpu_psi_some_us'].values.astype(float)
        cpu_vals = grp['cpu_usage_mcores'].values.astype(float)
        active   = cpu_vals > 10

        # sched: PSI-weighted
        psi_sum = psi_vals.sum()
        if psi_sum > 0:
            grp['sched_total_ms'] = (psi_vals / psi_sum) * float(nr['sched_total_ms'])
        else:
            cpu_sum = cpu_vals.sum()
            w = cpu_vals / cpu_sum if cpu_sum > 0 else np.ones(n) / n
            grp['sched_total_ms'] = w * float(nr['sched_total_ms'])

        # dstate: CPU-weighted
        cpu_sum = cpu_vals.sum()
        if cpu_sum > 0:
            grp['dstate_total_ms'] = (cpu_vals / cpu_sum) * float(nr['dstate_total_ms'])
        else:
            grp['dstate_total_ms'] = float(nr['dstate_total_ms']) / n

        # softirq: uniform across active pods
        n_active = active.sum()
        if n_active > 0:
            grp['softirq_total_ms'] = np.where(
                active, float(nr['softirq_total_ms']) / n_active, 0.0
            )
        else:
            grp['softirq_total_ms'] = float(nr['softirq_total_ms']) / n

        # power
        grp['pod_cpu_watts'] = (
            POWER_CPU_COEF * grp['cpu_usage_mcores'] +
            POWER_RAM_COEF * grp['ram_usage_mi'] +
            POWER_INTERCEPT
        ).clip(lower=0)

        results.append(grp)

    if not results:
        return pd.DataFrame()

    result = pd.concat(results, ignore_index=True)

    # ensure output columns exist
    for col in OUTPUT_COLS:
        if col not in result.columns:
            result[col] = 0.0

    return result[OUTPUT_COLS].copy()


def build_calibration_dataset(
    candidates: List[Dict],
    raw_dir: str,
    target_windows: int,
) -> pd.DataFrame:
    """
    Mix windows from candidate experiments into a calibration dataset.
    Distributes windows evenly across candidates.
    """
    n_candidates = len(candidates)
    windows_per_exp = max(target_windows // n_candidates, 10)

    all_parts = []
    global_window = 0

    for cand in candidates:
        name = cand['name']
        logger.info(f"  Loading {name} ({windows_per_exp} windows)...")

        pod_df = load_corrected_pod(raw_dir, name)
        if pod_df.empty:
            logger.warning(f"  {name}: no pod data, skipping")
            continue

        available_windows = sorted(pod_df['window_index'].unique())

        # sample windows_per_exp windows evenly
        if len(available_windows) <= windows_per_exp:
            selected = available_windows
        else:
            indices = np.linspace(0, len(available_windows)-1,
                                  windows_per_exp, dtype=int)
            selected = [available_windows[i] for i in indices]

        for w in selected:
            part = pod_df[pod_df['window_index'] == w].copy()
            part['window_index'] = global_window
            all_parts.append(part)
            global_window += 1

    if not all_parts:
        raise ValueError("No calibration data could be loaded")

    result = pd.concat(all_parts, ignore_index=True)
    result = result[OUTPUT_COLS].round(4)
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare AC calibration dataset")
    parser.add_argument('--raw',     default='./all_data',
                        help='Raw experiment data directory')
    parser.add_argument('--out',     default='./calibration',
                        help='Output directory')
    parser.add_argument('--windows', type=int, default=TARGET_WINDOWS,
                        help='Target number of calibration windows')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # find all low-load experiments
    logger.info("Scanning for low-load experiments...")
    raw_files = [f.replace('_node.csv', '')
                 for f in os.listdir(args.raw)
                 if f.endswith('_node.csv') and f != 'baseline_node.csv']

    candidates = []
    for name in sorted(raw_files):
        node_path = os.path.join(args.raw, f"{name}_node.csv")
        pod_path  = os.path.join(args.raw, f"{name}_pod.csv")
        if not os.path.exists(pod_path):
            continue
        node_df = pd.read_csv(node_path)
        if is_low_load(node_df):
            node_df = node_df[node_df['cpu_psi_some_us'] <= PSI_SANITY_CAP]
            sched = node_df['sched_total_ms'].median()
            cpu   = node_df['cpu_usage_mcores'].median() / NODE_CPU * 100
            psi   = node_df['cpu_psi_some_us'].median()  / NODE_PSI * 100
            candidates.append({
                'name':  name,
                'sched': round(sched, 1),
                'cpu':   round(cpu, 1),
                'psi':   round(psi, 1),
            })
            logger.info(f"  {name:<25} cpu={cpu:.1f}%  psi={psi:.1f}%  sched={sched:.0f}ms")

    if not candidates:
        raise ValueError("No low-load experiments found in raw data directory")

    logger.info(f"\nFound {len(candidates)} low-load candidates")
    logger.info(f"Building calibration dataset ({args.windows} windows)...")

    cal_df = build_calibration_dataset(candidates, args.raw, args.windows)

    # save
    out_csv  = os.path.join(args.out, 'calibration_pod.csv')
    out_meta = os.path.join(args.out, 'calibration_meta.json')

    cal_df.to_csv(out_csv, index=False)

    meta = {
        'windows':    cal_df['window_index'].nunique(),
        'pods':       cal_df['pod_name'].nunique(),
        'rows':       len(cal_df),
        'sources':    candidates,
        'sched_mean': round(cal_df.groupby('window_index')['sched_total_ms'].sum().mean(), 1),
        'sched_std':  round(cal_df.groupby('window_index')['sched_total_ms'].sum().std(), 1),
        'cpu_mean':   round(cal_df.groupby('window_index')['cpu_usage_mcores'].sum().mean() / NODE_CPU * 100, 1),
        'psi_mean':   round(cal_df.groupby('window_index')['cpu_psi_some_us'].sum().mean()  / NODE_PSI * 100, 1),
    }
    with open(out_meta, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\nCalibration dataset saved: {out_csv}")
    logger.info(f"  Windows: {meta['windows']}")
    logger.info(f"  Pods:    {meta['pods']}")
    logger.info(f"  sched:   mean={meta['sched_mean']}ms  std={meta['sched_std']}ms")
    logger.info(f"  cpu:     {meta['cpu_mean']}%")
    logger.info(f"  psi:     {meta['psi_mean']}%")
    logger.info(f"Meta saved: {out_meta}")


if __name__ == '__main__':
    main()