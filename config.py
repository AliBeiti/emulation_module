"""
config.py

Single source of truth for all constants used by the Emulation Module.
No logic — only values. Edit this file to adapt to different hardware.
"""

import os

# ─── Node Hardware Specs ──────────────────────────────────────────────────────
# Physical specs of the seller node being emulated
NODE_CPU_CORES    = 256                      # physical cores
NODE_CPU_MCORES   = NODE_CPU_CORES * 1000    # 256,000 millicores
NODE_RAM_MI       = 2 * 1024 * 1024          # 2 TB in Mi = 2,097,152
NODE_DISK_GB      = 878                      # total disk in GB
NODE_DISK_MB      = NODE_DISK_GB * 1024      # in MB

# ─── Physical Limit Caps ──────────────────────────────────────────────────────
# Maximum physically possible values per 5-second window
WINDOW_SECONDS    = 5
WINDOW_MS         = WINDOW_SECONDS * 1000    # 5,000 ms

# PSI: max = 100% of 5s window in microseconds
NODE_PSI_MAX_US   = WINDOW_SECONDS * 1_000_000   # 5,000,000 us

# eBPF: max = all cores × full window
NODE_EBPF_MAX_MS  = NODE_CPU_CORES * WINDOW_MS   # 256 × 5000 = 1,280,000 ms

# Pod power: reasonable upper bound per pod
POD_POWER_MAX_W   = 500.0   # watts

# ─── Replay Settings ──────────────────────────────────────────────────────────
TICK_INTERVAL_S   = 5        # seconds between each replay tick
MAX_WINDOWS       = 120      # number of windows per dataset loop (10 minutes)

# ─── Ridge PSI Model Coefficients ─────────────────────────────────────────────
# Validated model: log(node_psi) = coefs × features + intercept
# node_psi_us = exp(result) - 1
RIDGE_INTERCEPT   =  0.854
RIDGE_PSI_SUM     =  0.463   # coefficient for log(sum of pod PSI)
RIDGE_PSI_WAV     =  0.484   # coefficient for log(CPU-weighted avg PSI)
RIDGE_NUM_PODS    = -0.003   # coefficient for total pod count
RIDGE_ACT_PODS    =  0.008   # coefficient for active pod count (cpu > 10 mcores)
ACTIVE_CPU_THRESH =  10      # mcores — threshold to consider a pod "active"

# ─── Power Model Coefficients ─────────────────────────────────────────────────
# Fitted from real Kepler measurements (R²=0.87)
# pod_cpu_watts = CPU_COEF × cpu_mcores + RAM_COEF × ram_mi + INTERCEPT
POWER_CPU_COEF    =  0.002412766401392309
POWER_RAM_COEF    =  3.124235379731155e-05
POWER_INTERCEPT   =  0.07331256227741711

# ─── Dataset Paths ────────────────────────────────────────────────────────────
# All paths relative to the emulation module root directory
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR      = os.path.join(BASE_DIR, "datasets")
DATASET_INDEX     = os.path.join(DATASETS_DIR, "dataset_index.json")

# ─── API Settings ─────────────────────────────────────────────────────────────
API_HOST          = "0.0.0.0"
API_PORT          = 8090
API_NAMESPACE     = "ksense"           # Kubernetes namespace
API_SERVICE_NAME  = "ksense-usage-api" # Kubernetes service name

# ─── Kubernetes / KWOK Settings ───────────────────────────────────────────────
KWOK_NODE_NAME    = "emulation-node"   # name of the fake KWOK node
KWOK_NODE_CPU     = f"{NODE_CPU_CORES}"           # e.g. "256"
KWOK_NODE_MEMORY  = f"{NODE_RAM_MI}Mi"            # e.g. "2097152Mi"

# Annotation keys used to store metrics on KWOK pods
ANNOTATION_PREFIX = "emulation.metrics.k8s.io"
ANNOT_CPU         = f"{ANNOTATION_PREFIX}/cpu_mcores"
ANNOT_RAM         = f"{ANNOTATION_PREFIX}/ram_mi"
ANNOT_PSI         = f"{ANNOTATION_PREFIX}/psi_us"
ANNOT_SCHED       = f"{ANNOTATION_PREFIX}/sched_ms"
ANNOT_DSTATE      = f"{ANNOTATION_PREFIX}/dstate_ms"
ANNOT_SOFTIRQ     = f"{ANNOTATION_PREFIX}/softirq_ms"
ANNOT_POWER       = f"{ANNOTATION_PREFIX}/power_watts"
ANNOT_DISK_SPACE  = f"{ANNOTATION_PREFIX}/disk_space_mb"
ANNOT_DISK_USAGE  = f"{ANNOTATION_PREFIX}/disk_usage_mb"
ANNOT_WINDOW      = f"{ANNOTATION_PREFIX}/window_index"
ANNOT_TIMESTAMP   = f"{ANNOTATION_PREFIX}/timestamp"

# ─── Emulation Output Columns ─────────────────────────────────────────────────
# Expected columns in every emulation-ready pod CSV
POD_CSV_COLS = [
    "window_index",
    "pod_name",
    "cpu_usage_mcores",
    "ram_usage_mi",
    "disk_space_mb",
    "disk_usage_mb",
    "disk_ios",
    "cpu_psi_some_us",
    "sched_total_ms",
    "dstate_total_ms",
    "softirq_total_ms",
    "pod_cpu_watts",
]

# This node's IP — injected by Kubernetes Downward API at runtime
# Used to filter transactions belonging to this seller node
SELLER_NODE_IP = os.environ.get("SELLER_NODE_IP", "")

# ─── Transaction Poller Settings ──────────────────────────────────────────────
# URL of the transaction module ABCI query endpoint
TRANSACTION_API_URL = os.environ.get(
    "TRANSACTION_API_URL",
    f"http://{os.environ.get('SELLER_NODE_IP', 'localhost')}:26657/abci_query?data=%22tx%22"
)


# How often to poll for new transactions (seconds)
POLL_INTERVAL_S = 5