"""
dataset_generator.py

Callable, on-demand version of prepare_datasets.py's batch-generation logic.
Instead of precomputing the full hotel x sn x sa x es grid up front, exposes
get_or_generate() so a caller (dataset_selector.py) can generate — and cache —
exactly the composition it needs, the moment it needs it.

Strategy per composition (h, s, a, e):
  Exact Tier A    -> copy corrected pod CSV directly
  No exact match  -> build from closest Tier A base (or idle) + extra scaled replicas
  Tier B only     -> use CPU/RAM from Tier B, replace PSI/eBPF from nearest Tier A

ES support:
  - ES scales exactly like hotel/sn/sa: es{N} = N separate namespaces
    (es-stress-d, es-stress-d2, ... es-stress-d{N}), each with one pod.
  - No standalone ES experiment exists in the raw data, so h0s0a0eN targets
    fall back to the idle baseline as their base, then add N ES replicas.
  - ES is RAM-dominant: a generated combo's total RAM is checked against
    NODE_RAM_MI after assembly and scaled down proportionally if it would
    exceed physical node capacity.

PSI:  Ridge model (pod inputs -> node estimate), floor at BASELINE_PSI_US
eBPF: top-down distribution with two-component sched (OS + app portions)
Power: pod_cpu_watts = 0.00241xcpu + 0.0000312xram + 0.073
Disk: pod disk_space_mb + OS baseline floor
Output: Parquet, metric columns downcast to float32

Usage:
    from dataset_generator import load_templates, load_experiment_meta, get_or_generate

    templates = load_templates(CORRECTED_DIR)
    meta      = load_experiment_meta(CORRECTED_DIR)
    cache     = {}
    entry = get_or_generate(2, 0, 1, 0, cache, templates, meta["tier_A"],
                             available, CORRECTED_DIR, BASELINE_PATH)
"""

import os
import re
import json
import gc
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

# ── Node constants ─────────────────────────────────────────────────────────────
NODE_CPU_MCORES  = 256_000
NODE_RAM_MI      = 2 * 1024 * 1024   # 2,097,152
NODE_DISK_MB     = 878 * 1024        # 899,072
NODE_PSI_MAX_US  = 5_000_000
NODE_EBPF_MAX_MS = 256 * 5_000      # 1,280,000 ms
ACTIVE_CPU_THRESH = 10

# ── RAM cap enforcement for generated combos ────────────────────────────────────
# Real es5-inclusive experiments already reach ~82% of node RAM capacity
# (e.g. sn5_es5 = 1,721,796 Mi). Generated combos that stack ES with other
# app replicas can exceed the physical 2 TB node capacity — scale RAM down
# proportionally when that happens, rather than emitting an unrealistic
# dataset. RAM_CAP_MARGIN leaves a small buffer below the hard node limit.
RAM_CAP_MARGIN = 0.97

# ── Baseline floors ────────────────────────────────────────────────────────────
BASELINE_SCHED_MS   = 13_631.0
BASELINE_DSTATE_MS  = 59.9
BASELINE_SOFTIRQ_MS = 63.8
BASELINE_PSI_US     = 1_088_511.0
BASELINE_CPU_MCORES = 2_304.0
BASELINE_RAM_MI     = 25_485.0
BASELINE_DISK_GB    = 128.51
BASELINE_DISK_MB    = BASELINE_DISK_GB * 1024

OS_SCHED_FRAC   = 0.982   # fraction of baseline sched from OS/kernel
OS_DSTATE_FRAC  = 0.982
OS_SOFTIRQ_FRAC = 0.980
OS_SCHED_MS     = BASELINE_SCHED_MS  * OS_SCHED_FRAC    # ~13,386ms
OS_DSTATE_MS    = BASELINE_DSTATE_MS * OS_DSTATE_FRAC   # ~58.8ms
OS_SOFTIRQ_MS   = BASELINE_SOFTIRQ_MS * OS_SOFTIRQ_FRAC # ~62.5ms

# ── Ridge PSI model ────────────────────────────────────────────────────────────
RIDGE_INTERCEPT = 0.854
RIDGE_PSI_SUM   = 0.463
RIDGE_PSI_WAV   = 0.484
RIDGE_NUM_PODS  = -0.003
RIDGE_ACT_PODS  = 0.008

# ── Power model ────────────────────────────────────────────────────────────────
POWER_CPU_COEF = 0.002412766401392309
POWER_RAM_COEF = 3.124235379731155e-05
POWER_INTERCEPT = 0.07331256227741711

# ── Scale factor bounds for extra replicas ─────────────────────────────────────
SCALE_MIN = 0.5
SCALE_MAX = 3.0

# ── Output columns ─────────────────────────────────────────────────────────────
OUTPUT_COLS = [
    'window_index', 'pod_name',
    'cpu_usage_mcores', 'ram_usage_mi',
    'disk_space_mb', 'disk_usage_mb', 'disk_ios',
    'cpu_psi_some_us',
    'sched_total_ms', 'dstate_total_ms', 'softirq_total_ms',
    'pod_cpu_watts',
]

# Metric columns downcast to float32 before writing Parquet — everything in
# OUTPUT_COLS except the index/identity columns.
METRIC_COLS = [c for c in OUTPUT_COLS if c not in ('window_index', 'pod_name')]

# Sentinel name for the idle baseline used as a fallback base when no real
# Tier A experiment overlaps the target composition (e.g. pure ES targets).
IDLE_BASE = 'IDLE'


# ─────────────────────────────────────────────────────────────────────────────
# PSI aggregation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_node_psi(pod_df: pd.DataFrame) -> pd.Series:
    """
    Apply Ridge model per window to estimate node-level PSI.
    Returns Series indexed by window_index.
    Floor enforced at BASELINE_PSI_US.
    Cap enforced at cpu_utilization x NODE_PSI_MAX_US.
    """
    results = {}
    for win, grp in pod_df.groupby('window_index'):
        psi_vals = grp['cpu_psi_some_us'].values.astype(float)
        cpu_vals = grp['cpu_usage_mcores'].values.astype(float)

        psi_sum   = psi_vals.sum()
        total_cpu = cpu_vals.sum()
        num_pods  = len(grp)
        act_pods  = (cpu_vals > ACTIVE_CPU_THRESH).sum()

        if total_cpu > 0:
            psi_wav = np.sum(psi_vals * cpu_vals) / total_cpu
        else:
            psi_wav = psi_sum / max(num_pods, 1)

        log_res = (
            RIDGE_PSI_SUM  * np.log1p(psi_sum) +
            RIDGE_PSI_WAV  * np.log1p(psi_wav) +
            RIDGE_NUM_PODS * num_pods +
            RIDGE_ACT_PODS * act_pods +
            RIDGE_INTERCEPT
        )
        raw_psi  = float(np.expm1(log_res))
        cpu_util = min(total_cpu / NODE_CPU_MCORES, 1.0)
        psi_cap  = cpu_util * NODE_PSI_MAX_US

        # real-world floor: node PSI never below OS baseline
        # Apply floor AFTER cap so baseline always wins over low-load cap
        psi_final = min(raw_psi, NODE_PSI_MAX_US)   # hard physical cap first
        psi_final = max(psi_final, BASELINE_PSI_US)  # then enforce floor
        results[win] = psi_final

    return pd.Series(results)


# ─────────────────────────────────────────────────────────────────────────────
# eBPF distribution
# ─────────────────────────────────────────────────────────────────────────────

def distribute_ebpf_to_pods(
    pod_df: pd.DataFrame,
    node_sched_by_win: pd.Series,
    node_dstate_by_win: pd.Series,
    node_softirq_by_win: pd.Series,
) -> pd.DataFrame:
    """
    Distribute node-level eBPF top-down to pods per window.
    sched   -> PSI-weighted
    dstate  -> blended CPU+RAM weighted (RAM-heavy, low-CPU pods like ES
              still get a fair share instead of being zeroed out)
    softirq -> uniform across active pods (cpu > 10 mcores OR ram > 50 MiB)
    """
    df = pod_df.copy()
    sched_out   = np.zeros(len(df))
    dstate_out  = np.zeros(len(df))
    softirq_out = np.zeros(len(df))

    for win, grp in df.groupby('window_index'):
        idx      = grp.index
        psi_vals = grp['cpu_psi_some_us'].values.astype(float)
        cpu_vals = grp['cpu_usage_mcores'].values.astype(float)
        ram_vals = grp['ram_usage_mi'].values.astype(float) if 'ram_usage_mi' in grp.columns else np.zeros(len(grp))
        active   = (cpu_vals > ACTIVE_CPU_THRESH) | (ram_vals > 50)
        n        = len(grp)

        node_sched   = float(node_sched_by_win.get(win, BASELINE_SCHED_MS))
        node_dstate  = float(node_dstate_by_win.get(win, BASELINE_DSTATE_MS))
        node_softirq = float(node_softirq_by_win.get(win, BASELINE_SOFTIRQ_MS))

        # sched: PSI-weighted
        psi_sum = psi_vals.sum()
        if psi_sum > 0:
            w = psi_vals / psi_sum
        else:
            cpu_sum = cpu_vals.sum()
            w = cpu_vals / cpu_sum if cpu_sum > 0 else np.ones(n) / n
        sched_out[idx] = w * node_sched

        # dstate: blended CPU+RAM weighted
        cpu_sum = cpu_vals.sum()
        ram_sum = ram_vals.sum()
        cpu_frac = cpu_vals / cpu_sum if cpu_sum > 0 else np.zeros(n)
        ram_frac = ram_vals / ram_sum if ram_sum > 0 else np.zeros(n)
        if cpu_sum > 0 and ram_sum > 0:
            w = 0.5 * cpu_frac + 0.5 * ram_frac
        elif ram_sum > 0:
            w = ram_frac
        elif cpu_sum > 0:
            w = cpu_frac
        else:
            w = np.ones(n) / n
        dstate_out[idx] = w * node_dstate

        # softirq: uniform across active pods
        n_active = active.sum()
        if n_active > 0:
            softirq_out[idx] = np.where(active, node_softirq / n_active, 0.0)
        else:
            softirq_out[idx] = node_softirq / n

    df['sched_total_ms']   = sched_out
    df['dstate_total_ms']  = dstate_out
    df['softirq_total_ms'] = softirq_out
    return df


def estimate_node_sched(
    node_psi_by_win: pd.Series,
    base_sched: float,
    base_psi: float,
) -> pd.Series:
    """
    Two-component sched estimation:
      node_sched = OS_sched + app_sched x (new_psi / base_psi)
    Preserves OS baseline and only scales the app-driven portion.
    Floor at BASELINE_SCHED_MS.
    Cap at NODE_EBPF_MAX_MS.
    """
    app_sched_base = max(base_sched - OS_SCHED_MS, 0.0)
    results = {}
    for win, new_psi in node_psi_by_win.items():
        ratio     = (new_psi / base_psi) if base_psi > 0 else 1.0
        ratio     = np.clip(ratio, SCALE_MIN, SCALE_MAX)
        sched     = OS_SCHED_MS + app_sched_base * ratio
        sched     = max(sched, BASELINE_SCHED_MS)
        sched     = min(sched, NODE_EBPF_MAX_MS)
        results[win] = sched
    return pd.Series(results)


def estimate_node_dstate(
    pod_df: pd.DataFrame,
    base_dstate: float,
    base_cpu: float,
) -> pd.Series:
    """
    CPU-ratio dstate estimation per window.
    Floor at BASELINE_DSTATE_MS.
    """
    results = {}
    for win, grp in pod_df.groupby('window_index'):
        new_cpu = grp['cpu_usage_mcores'].sum()
        ratio   = (new_cpu / base_cpu) if base_cpu > 0 else 1.0
        ratio   = np.clip(ratio, SCALE_MIN, SCALE_MAX)
        dstate  = max(base_dstate * ratio, BASELINE_DSTATE_MS)
        dstate  = min(dstate, NODE_EBPF_MAX_MS)
        results[win] = dstate
    return pd.Series(results)


def estimate_node_softirq(
    pod_df: pd.DataFrame,
    base_softirq: float,
    base_active_pods: float,
) -> pd.Series:
    """
    Softirq scales weakly with active pod count.
    Floor at BASELINE_SOFTIRQ_MS.
    """
    results = {}
    for win, grp in pod_df.groupby('window_index'):
        n_active   = (grp['cpu_usage_mcores'].values > ACTIVE_CPU_THRESH).sum()
        ratio      = (n_active / base_active_pods) if base_active_pods > 0 else 1.0
        ratio      = np.clip(ratio, SCALE_MIN, SCALE_MAX)
        softirq    = max(base_softirq * ratio, BASELINE_SOFTIRQ_MS)
        softirq    = min(softirq, NODE_EBPF_MAX_MS)
        results[win] = softirq
    return pd.Series(results)


# ─────────────────────────────────────────────────────────────────────────────
# Power model
# ─────────────────────────────────────────────────────────────────────────────

def compute_pod_power(df: pd.DataFrame) -> pd.DataFrame:
    """Apply power model per pod row."""
    df = df.copy()
    df['pod_cpu_watts'] = (
        POWER_CPU_COEF * df['cpu_usage_mcores'] +
        POWER_RAM_COEF * df['ram_usage_mi'] +
        POWER_INTERCEPT
    ).clip(lower=0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RAM cap enforcement
# ─────────────────────────────────────────────────────────────────────────────

def enforce_ram_cap(df: pd.DataFrame, key: str) -> Tuple[pd.DataFrame, bool, float]:
    """
    Check total RAM per window against node capacity. If the max exceeds
    the cap (with margin), scale ALL pods' RAM down proportionally so the
    combo stays physically realistic. Returns (df, was_capped, scale_factor).
    """
    ram_by_win = df.groupby('window_index')['ram_usage_mi'].sum()
    max_ram = ram_by_win.max()
    cap = NODE_RAM_MI * RAM_CAP_MARGIN

    if max_ram <= cap:
        return df, False, 1.0

    scale_factor = cap / max_ram
    df = df.copy()
    df['ram_usage_mi'] = df['ram_usage_mi'] * scale_factor
    logger.warning(
        f"  {key}: RAM cap applied — projected max {max_ram:,.0f} Mi "
        f"exceeds {cap:,.0f} Mi cap, scaled by {scale_factor:.3f}"
    )
    return df, True, scale_factor


# ─────────────────────────────────────────────────────────────────────────────
# Extra replica scaling
# ─────────────────────────────────────────────────────────────────────────────

def scale_replica_pods(
    template_df: pd.DataFrame,
    new_ns: str,
    base_df: pd.DataFrame,
    n_base_windows: int,
) -> pd.DataFrame:
    """
    Scale a single-instance template pod pool to match base experiment windows.
    - Renames namespace to new_ns (e.g. hotel2/, es-stress-d2/)
    - Scales CPU/RAM by load ratio between base and template
    - Aligns to same window count as base
    - PSI kept from template (will be re-aggregated by Ridge model)
    """
    template_windows = template_df['window_index'].unique()
    base_windows     = sorted(base_df['window_index'].unique())
    n_out            = min(n_base_windows, len(template_windows))

    # compute per-window load ratio from base
    base_cpu_by_win = base_df.groupby('window_index')['cpu_usage_mcores'].sum()
    tmpl_cpu_by_win = template_df.groupby('window_index')['cpu_usage_mcores'].sum()
    global_base_cpu = base_cpu_by_win.mean()
    global_tmpl_cpu = tmpl_cpu_by_win.mean()
    load_ratio      = np.clip(
        global_base_cpu / global_tmpl_cpu if global_tmpl_cpu > 0 else 1.0,
        SCALE_MIN, SCALE_MAX
    )

    # reassign window indices to match base
    tmpl_sorted = sorted(template_windows)
    win_map     = {old: base_windows[i % len(base_windows)]
                   for i, old in enumerate(tmpl_sorted)}

    scaled_parts = []
    for old_win, new_win in win_map.items():
        grp = template_df[template_df['window_index'] == old_win].copy()
        grp['window_index']      = new_win
        grp['cpu_usage_mcores']  = (grp['cpu_usage_mcores'] * load_ratio).clip(lower=0)
        grp['ram_usage_mi']      = grp['ram_usage_mi']   # RAM doesn't scale with load
        grp['disk_space_mb']     = grp['disk_space_mb']
        grp['disk_usage_mb']     = grp['disk_usage_mb']
        grp['disk_ios']          = grp['disk_ios']

        # rename namespace
        grp['pod_name'] = grp['pod_name'].apply(
            lambda x: re.sub(r'^[^/]+/', new_ns + '/', str(x))
        )
        scaled_parts.append(grp)

    result = pd.concat(scaled_parts, ignore_index=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dataset selector
# ─────────────────────────────────────────────────────────────────────────────

def composition_to_key(h: int, s: int, a: int, e: int) -> str:
    return f"h{h}s{s}a{a}e{e}"


def parse_composition(name: str) -> Tuple[int, int, int, int]:
    if name == IDLE_BASE:
        return 0, 0, 0, 0
    bh = int(m.group(1)) if (m := re.search(r'hotel(\d+)', name)) else 0
    bs = int(m.group(1)) if (m := re.search(r'sn(\d+)',   name)) else 0
    ba = int(m.group(1)) if (m := re.search(r'sa(\d+)',   name)) else 0
    be = int(m.group(1)) if (m := re.search(r'es(\d+)',   name)) else 0
    return bh, bs, ba, be


def find_best_base(
    h: int, s: int, a: int, e: int,
    tier_A: List[str],
) -> Optional[str]:
    """
    Find closest Tier A experiment for the target composition (or the idle
    baseline as a universal fallback with bh=bs=ba=be=0).

    Normal case: base must not exceed requested counts (scale up).
    Special case: pure SA targets (h=0, s=0, a<5) — no Tier A base with
      fewer SA exists, so allow sa5/sa6(+ES) as base and subset pods down.
    """
    candidates = list(tier_A) + [IDLE_BASE]
    best, best_score = None, float('inf')

    for exp in candidates:
        bh, bs, ba, be = parse_composition(exp)

        # Special: pure SA low-count — allow base with more SA (will subset),
        # also require be <= e so we don't inherit more ES than requested
        if h == 0 and s == 0 and a > 0 and a < 5:
            if bh == 0 and bs == 0 and ba >= a and be <= e:
                score = (ba - a) + (e - be)   # prefer closest sa AND closest es
                if score < best_score:
                    best_score, best = score, exp
            continue

        # Normal: base must not exceed requested counts
        if bh > h or bs > s or ba > a or be > e:
            continue

        gap  = (h - bh) + (s - bs) + (a - ba) + (e - be)
        miss = (
            (1 if h > 0 and bh == 0 else 0) +
            (1 if s > 0 and bs == 0 else 0) +
            (1 if a > 0 and ba == 0 else 0) +
            (1 if e > 0 and be == 0 else 0)
        )
        score = gap + miss * 100
        if score < best_score:
            best_score, best = score, exp

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Baseline idle dataset (h0s0a0e0)
# ─────────────────────────────────────────────────────────────────────────────

def load_idle_base_df(baseline_path: str) -> pd.DataFrame:
    """
    Build a base_df from baseline_node.csv, formatted like a corrected pod
    CSV (single synthetic 'os/baseline' pod per window). Used both to write
    the h0s0a0e0 dataset and as a fallback base for targets with no
    overlapping real Tier A experiment (e.g. pure ES targets).
    """
    df = pd.read_csv(baseline_path)
    df = df[df['cpu_psi_some_us'] <= NODE_PSI_MAX_US].copy()
    df = df.reset_index(drop=True)

    rows = []
    for i, row in df.iterrows():
        cpu = float(row.get('cpu_usage_mcores', BASELINE_CPU_MCORES))
        ram = float(row.get('ram_usage_mi',     BASELINE_RAM_MI))
        pwr = POWER_CPU_COEF * cpu + POWER_RAM_COEF * ram + POWER_INTERCEPT
        rows.append({
            'window_index':    i,
            'pod_name':        'os/baseline',
            'cpu_usage_mcores': cpu,
            'ram_usage_mi':    ram,
            'disk_space_mb':   float(row.get('disk_used_gb', BASELINE_DISK_GB)) * 1024,
            'disk_usage_mb':   float(row.get('disk_read_mb', 0)) + float(row.get('disk_write_mb', 0)),
            'disk_ios':        float(row.get('disk_read_ios', 0)) + float(row.get('disk_write_ios', 0)),
            'cpu_psi_some_us': float(row.get('cpu_psi_some_us', BASELINE_PSI_US)),
            'sched_total_ms':  float(row.get('sched_total_ms',  BASELINE_SCHED_MS)),
            'dstate_total_ms': float(row.get('dstate_total_ms', BASELINE_DSTATE_MS)),
            'softirq_total_ms':float(row.get('softirq_total_ms',BASELINE_SOFTIRQ_MS)),
            'pod_cpu_watts':   round(pwr, 4),
        })

    return pd.DataFrame(rows)[OUTPUT_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Parquet / dtype output helper
# ─────────────────────────────────────────────────────────────────────────────

def _downcast_metric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast metric columns to float32 to shrink Parquet output size."""
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Direct copy strategy (exact Tier A match)
# ─────────────────────────────────────────────────────────────────────────────

def make_direct_dataset(
    exp_name: str,
    h: int, s: int, a: int, e: int,
    corrected_dir: str,
    out_dir: str,
) -> Optional[Dict]:
    """Copy corrected pod CSV, recompute PSI and power, enforce floors."""
    path = os.path.join(corrected_dir, f"{exp_name}_pod_corrected.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df = df.drop(columns=['tier'], errors='ignore')

    # recompute PSI per window via Ridge (with floor)
    psi_by_win = estimate_node_psi(df)

    # recompute node eBPF using base experiment node stats
    base_psi     = psi_by_win.mean()
    base_sched   = df.groupby('window_index')['sched_total_ms'].sum().mean()
    base_dstate  = df.groupby('window_index')['dstate_total_ms'].sum().mean()
    base_softirq = df.groupby('window_index')['softirq_total_ms'].sum().mean()
    base_cpu     = df.groupby('window_index')['cpu_usage_mcores'].sum().mean()
    base_active  = df.groupby('window_index')['cpu_usage_mcores'].apply(
        lambda s: (s > ACTIVE_CPU_THRESH).sum()
    ).mean()

    sched_by_win   = estimate_node_sched(psi_by_win, base_sched, base_psi)
    dstate_by_win  = estimate_node_dstate(df, base_dstate, base_cpu)
    softirq_by_win = estimate_node_softirq(df, base_softirq, base_active)

    df = distribute_ebpf_to_pods(df, sched_by_win, dstate_by_win, softirq_by_win)

    # distribute PSI to pods (PSI-weighted)
    df = _distribute_psi_to_pods(df, psi_by_win)

    # recompute power
    df = compute_pod_power(df)

    # add OS disk floor at row level
    df['disk_space_mb'] = df['disk_space_mb'] + (BASELINE_DISK_MB / max(df['pod_name'].nunique(), 1))

    key = composition_to_key(h, s, a, e)

    # RAM cap check — real experiments shouldn't need this (max observed is
    # ~82% of capacity), but check defensively rather than assume.
    df, ram_capped, ram_scale = enforce_ram_cap(df, key)

    df = df[OUTPUT_COLS].round(4)
    windows = df['window_index'].nunique()
    pods    = df['pod_name'].nunique()

    # ── Parquet output, metric columns downcast to float32 ────────────────────
    df = _downcast_metric_cols(df)
    out_path = os.path.join(out_dir, f"{key}_pod.parquet")
    df.to_parquet(out_path, index=False)

    result = {
        'file':    f'datasets/{key}_pod.parquet',
        'source':  'real',
        'hotel':   h, 'sn': s, 'sa': a, 'es': e,
        'windows': windows,
        'pods':    pods,
        'tier':    'A',
        'ram_capped': ram_capped,
        'ram_scale_factor': round(ram_scale, 4),
    }

    del df
    gc.collect()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Build from base + extra replicas strategy
# ─────────────────────────────────────────────────────────────────────────────

def make_generated_dataset(
    h: int, s: int, a: int, e: int,
    base_exp: str,
    corrected_dir: str,
    out_dir: str,
    baseline_path: str,
    templates: Dict[str, pd.DataFrame],
) -> Optional[Dict]:
    """
    Build unseen combination from closest Tier A base (or idle) + extra
    replica pods.
    """
    if base_exp == IDLE_BASE:
        base_df = load_idle_base_df(baseline_path)
    else:
        base_path = os.path.join(corrected_dir, f"{base_exp}_pod_corrected.csv")
        if not os.path.exists(base_path):
            return None
        base_df = pd.read_csv(base_path).drop(columns=['tier'], errors='ignore')

    base_windows = sorted(base_df['window_index'].unique())
    n_windows    = len(base_windows)

    # parse base composition
    bh, bs, ba, be = parse_composition(base_exp)

    # Special: pure SA low-count target (a<5) — subset base SA pods down to `a`
    if h == 0 and s == 0 and a > 0 and a < 5 and ba > a:
        all_sa_ns = sorted(
            base_df['pod_name'].apply(lambda x: str(x).split('/')[0]).unique(),
            key=lambda x: (len(x), x)
        )
        keep_ns  = set(all_sa_ns[:a])
        base_df  = base_df[base_df['pod_name'].apply(
            lambda x: str(x).split('/')[0] in keep_ns
        )].copy()
        base_windows = sorted(base_df['window_index'].unique())
        n_windows    = len(base_windows)
        bh, bs, ba   = 0, 0, a
        # be unchanged — any ES already in the base stays; extra ES added below

    pod_pool = [base_df]

    # ── add extra hotel replicas ───────────────────────────────────────────────
    hotel_tmpl = templates.get('hotel')
    if hotel_tmpl is not None:
        for i in range(bh + 1, h + 1):
            ns      = f"hotel{i}" if i > 1 else "hotel"
            extra   = scale_replica_pods(hotel_tmpl, ns, base_df, n_windows)
            pod_pool.append(extra)

    # ── add extra SN replicas (CPU/RAM only — PSI from sn1 template scaled) ───
    sn_tmpl = templates.get('sn')
    if sn_tmpl is not None:
        for i in range(bs + 1, s + 1):
            ns    = f"sn{i}" if i > 1 else "sn"
            extra = scale_replica_pods(sn_tmpl, ns, base_df, n_windows)
            # zero out PSI for extra SN pods (load gen bug reality)
            extra['cpu_psi_some_us'] = 0.0
            pod_pool.append(extra)

    # ── add extra SA replicas (always high-load template sa5) ─────────────────
    sa_tmpl = templates.get('sa')
    if sa_tmpl is not None:
        for i in range(ba + 1, a + 1):
            ns    = f"sa{i}" if i > 1 else "sa"
            extra = scale_replica_pods(sa_tmpl, ns, base_df, n_windows)
            pod_pool.append(extra)

    # ── add extra ES replicas (namespace convention: es-stress-d, -d2, -d3...) ─
    es_tmpl = templates.get('es')
    if es_tmpl is not None:
        for i in range(be + 1, e + 1):
            ns    = f"es-stress-d{i}" if i > 1 else "es-stress-d"
            extra = scale_replica_pods(es_tmpl, ns, base_df, n_windows)
            pod_pool.append(extra)

    merged = pd.concat(pod_pool, ignore_index=True)

    # recompute PSI, eBPF, power on merged pool
    psi_by_win = estimate_node_psi(merged)

    base_psi     = estimate_node_psi(base_df).mean()
    base_sched   = base_df.groupby('window_index')['sched_total_ms'].sum().mean()
    base_dstate  = base_df.groupby('window_index')['dstate_total_ms'].sum().mean()
    base_softirq = base_df.groupby('window_index')['softirq_total_ms'].sum().mean()
    base_cpu     = base_df.groupby('window_index')['cpu_usage_mcores'].sum().mean()
    base_active  = base_df.groupby('window_index')['cpu_usage_mcores'].apply(
        lambda s: (s > ACTIVE_CPU_THRESH).sum()
    ).mean()

    sched_by_win   = estimate_node_sched(psi_by_win, base_sched, base_psi)
    dstate_by_win  = estimate_node_dstate(merged, base_dstate, base_cpu)
    softirq_by_win = estimate_node_softirq(merged, base_softirq, base_active)

    merged = distribute_ebpf_to_pods(merged, sched_by_win, dstate_by_win, softirq_by_win)
    merged = _distribute_psi_to_pods(merged, psi_by_win)
    merged = compute_pod_power(merged)

    # OS disk floor per pod
    n_unique_pods = max(merged['pod_name'].nunique(), 1)
    merged['disk_space_mb'] = merged['disk_space_mb'] + (BASELINE_DISK_MB / n_unique_pods)

    # physical caps
    merged['cpu_usage_mcores']  = merged['cpu_usage_mcores'].clip(upper=NODE_CPU_MCORES)
    merged['cpu_psi_some_us']   = merged['cpu_psi_some_us'].clip(upper=NODE_PSI_MAX_US)
    merged['sched_total_ms']    = merged['sched_total_ms'].clip(upper=NODE_EBPF_MAX_MS)
    merged['dstate_total_ms']   = merged['dstate_total_ms'].clip(upper=NODE_EBPF_MAX_MS)
    merged['softirq_total_ms']  = merged['softirq_total_ms'].clip(upper=NODE_EBPF_MAX_MS)

    key = composition_to_key(h, s, a, e)

    # RAM cap check — this is the main place it matters: stacking ES
    # replicas with other apps can exceed node RAM capacity.
    merged, ram_capped, ram_scale = enforce_ram_cap(merged, key)

    merged = merged[OUTPUT_COLS].round(4)
    windows = merged['window_index'].nunique()
    pods    = merged['pod_name'].nunique()

    # ── Parquet output, metric columns downcast to float32 ────────────────────
    merged = _downcast_metric_cols(merged)
    out_path = os.path.join(out_dir, f"{key}_pod.parquet")
    merged.to_parquet(out_path, index=False)

    result = {
        'file':    f'datasets/{key}_pod.parquet',
        'source':  'generated',
        'base':    base_exp,
        'hotel':   h, 'sn': s, 'sa': a, 'es': e,
        'windows': windows,
        'pods':    pods,
        'tier':    'A',
        'ram_capped': ram_capped,
        'ram_scale_factor': round(ram_scale, 4),
    }

    del base_df, pod_pool, merged
    gc.collect()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PSI pod distribution helper
# ─────────────────────────────────────────────────────────────────────────────

def _distribute_psi_to_pods(
    df: pd.DataFrame,
    node_psi_by_win: pd.Series,
) -> pd.DataFrame:
    """
    Redistribute node PSI back to pods proportionally to original pod PSI.
    Ensures pod PSI values are consistent with Ridge model output.
    """
    df = df.copy()
    new_psi = np.zeros(len(df))

    for win, grp in df.groupby('window_index'):
        idx      = grp.index
        psi_vals = grp['cpu_psi_some_us'].values.astype(float)
        node_psi = float(node_psi_by_win.get(win, BASELINE_PSI_US))
        psi_sum  = psi_vals.sum()

        if psi_sum > 0:
            weights = psi_vals / psi_sum
        else:
            weights = np.ones(len(grp)) / len(grp)

        new_psi[idx] = weights * node_psi

    df['cpu_psi_some_us'] = new_psi
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Experiment name candidate generation (exact-match lookup)
# ─────────────────────────────────────────────────────────────────────────────

def build_exp_candidates(h: int, s: int, a: int, e: int) -> List[str]:
    """
    Try known real-data naming patterns for this composition, both the
    original CPU-only naming (hotel_sn_sa) and the ES-inclusive naming
    (hotel_sa_sn_es) actually used on disk. Order matters for es==0 legacy
    combos, otherwise all plausible patterns are tried.
    """
    cands = []

    if e == 0:
        # legacy CPU-only naming (unchanged from the original script)
        if h > 0 and s > 0 and a >= 0:
            cands.append(f"hotel{h}_sn{s}_sa{a}")
        if h > 0 and s == 0 and a >= 0:
            cands.append(f"hotel{h}_sa{a}")
        if h == 0 and s > 0 and a >= 0:
            cands.append(f"sn{s}_sa{a}")
        if h > 0 and s == 0 and a == 0:
            cands.append(f"hotel{h}")
        if h == 0 and s > 0 and a == 0:
            cands.append(f"sn{s}")
        if h == 0 and s == 0 and a > 0:
            cands.append(f"sa{a}")
    else:
        # ES-inclusive naming as actually produced by the collection pipeline
        if h > 0 and s > 0 and a > 0:
            cands.append(f"hotel{h}_sa{a}_sn{s}_es{e}")
        if h > 0 and s == 0 and a > 0:
            cands.append(f"hotel{h}_sa{a}_es{e}")
        if h > 0 and s == 0 and a == 0:
            cands.append(f"hotel{h}_es{e}")
        if h == 0 and s > 0 and a == 0:
            cands.append(f"sn{s}_es{e}")
        if h == 0 and s == 0 and a > 0:
            cands.append(f"sa{a}_es{e}")
        if h == 0 and s == 0 and a == 0:
            cands.append(f"es{e}")
        # future-proofing for naming patterns not yet collected
        if h == 0 and s > 0 and a > 0:
            cands.append(f"sn{s}_sa{a}_es{e}")
        if h > 0 and s > 0 and a == 0:
            cands.append(f"hotel{h}_sn{s}_es{e}")

    return cands


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_templates(corrected_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load single-instance replica templates used to scale extra hotel/sn/sa/es
    pods onto a base composition. Only includes keys for templates actually
    found on disk (missing ones are logged and skipped, not raised).
    """
    templates = {}
    for app, tmpl_name in [('hotel', 'hotel1'), ('sn', 'sn1'), ('sa', 'sa5')]:
        path = os.path.join(corrected_dir, f"{tmpl_name}_pod_corrected.csv")
        if os.path.exists(path):
            templates[app] = pd.read_csv(path).drop(columns=['tier'], errors='ignore')
            logger.info(f"  Template '{app}' loaded from {tmpl_name}")
        else:
            logger.warning(f"  Template '{app}' not found: {path}")

    # ES template: extract just the ES pod rows from hotel1_es1 (lowest-noise
    # combo, single ES namespace). No standalone ES experiment exists.
    es_source_path = os.path.join(corrected_dir, "hotel1_es1_pod_corrected.csv")
    if os.path.exists(es_source_path):
        es_full = pd.read_csv(es_source_path).drop(columns=['tier'], errors='ignore')
        es_tmpl = es_full[es_full['pod_name'].str.startswith('es-stress-d/')].copy()
        if len(es_tmpl) > 0:
            templates['es'] = es_tmpl
            logger.info(f"  Template 'es' extracted from hotel1_es1 ({es_tmpl['pod_name'].nunique()} pod)")
        else:
            logger.warning("  Template 'es' — no es-stress-d/ pods found in hotel1_es1")
    else:
        logger.warning(f"  Template 'es' not found: {es_source_path}")

    return templates


def load_experiment_meta(corrected_dir: str) -> Dict:
    """Load experiment_meta.json (tier_A, tier_B, experiments, tier_B_usable_fields)."""
    meta_path = os.path.join(corrected_dir, 'experiment_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    logger.info(
        f"Tier A: {len(meta.get('tier_A', []))} | "
        f"Tier B: {len(meta.get('tier_B', []))}"
    )
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_or_generate(
    h: int, s: int, a: int, e: int,
    cache: Dict[str, Dict],
    templates: Dict[str, pd.DataFrame],
    tier_A: List[str],
    available: set,
    corrected_dir: str,
    baseline_path: str,
    out_dir: Optional[str] = None,
) -> Optional[Dict]:
    """
    Return the dataset index entry for composition (h, s, a, e), generating
    and writing it (as Parquet, under out_dir) on a cache miss.

    out_dir defaults to config.DATASETS_DIR, read at call time (not at
    import time) so tests that monkeypatch config.DATASETS_DIR still work.

    Lookup order, mirroring the old batch loop's per-cell logic:
      1. cache hit                          -> return immediately
      2. exact Tier A match on disk         -> make_direct_dataset
      3. closest Tier A base (or idle)      -> make_generated_dataset
      4. no viable base                     -> None (not cached)
    """
    if out_dir is None:
        out_dir = config.DATASETS_DIR

    key = composition_to_key(h, s, a, e)
    if key in cache:
        return cache[key]

    exp_candidates = build_exp_candidates(h, s, a, e)

    exact_A = None
    for cand in exp_candidates:
        if cand in tier_A and cand in available:
            exact_A = cand
            break

    if exact_A:
        entry = make_direct_dataset(exact_A, h, s, a, e, corrected_dir, out_dir)
        if entry:
            cache[key] = entry
            return entry

    base_exp = find_best_base(h, s, a, e, tier_A)
    if base_exp is None:
        logger.warning(f"  {key}: no Tier A base found — cannot generate")
        return None

    entry = make_generated_dataset(
        h, s, a, e, base_exp, corrected_dir, out_dir, baseline_path, templates
    )
    if entry:
        cache[key] = entry
        return entry

    return None
