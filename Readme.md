# Emulation Module

Seller-side workload emulator for the decentralized resource trading platform. Replays pre-recorded Kubernetes workload metrics on fake KWOK pods and exposes aggregated node-level metrics via REST API.

## Overview

When a buyer's transaction is confirmed, the Emulation Module:
1. Selects (generating on demand if needed) the dataset matching the current workload composition
2. Replays pod-level metrics every 5 seconds on KWOK (fake) pods
3. Aggregates metrics to node level using validated models
4. Exposes results via REST API for Admission Control and Pricing modules

## Repository Structure

```
emulation_module/
├── main.py                 # Entry point — starts API + tick loop
├── api.py                  # FastAPI endpoints
├── timeline.py             # Job lifecycle and composition tracking
├── transaction_poller.py   # Redis stream consumer -> Timeline.add_job()
├── event_logger.py         # Per-transaction JSONL event log
├── dataset_selector.py     # Selects dataset for a given workload composition
├── dataset_generator.py    # On-demand dataset generation (Tier A copy / generated-from-base)
├── dataset_cache.py        # Size-bound LRU cache in front of dataset_generator
├── replay_engine.py        # Window-by-window dataset replay + namespace mapping
├── aggregator.py           # Pod → node level metric aggregation
├── baseline_provider.py    # Real idle metrics served when no jobs are active
├── kwok_manager.py         # KWOK pod/namespace lifecycle + annotation patching
├── config.py               # All constants (node capacity, model coefficients)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image (datasets baked in)
├── build.sh                # Build and push to DockerHub
├── k8s/
│   ├── namespace.yaml      # ksense namespace
│   ├── rbac.yaml           # ServiceAccount + ClusterRole + ClusterRoleBinding
│   └── deployment.yaml     # Deployment + Service
└── test_emulation_module.py # Unit tests (41 tests, no Kubernetes required)
```

## API Endpoints

| Method | Endpoint | Consumer | Description |
|---|---|---|---|
| `GET` | `/usage/latest` | Admission Control | Current node metrics (updates every 5s) |
| `GET` | `/usage/capacity` | Admission Control | Node total resources (constant) |
| `POST` | `/calibration/done` | Admission Control | Signal calibration phase complete |
| `GET` | `/calibration/status` | Admission Control / Monitoring | Whether calibration phase is done |
| `GET` | `/status` | Monitoring | Active jobs, composition, dataset info |
| `GET` | `/healthz` | Kubernetes | Liveness probe |

### GET /usage/latest
```json
{
  "timestamp": "2026-03-31 12:30:05",
  "cpu_usage_pct": 42.3,
  "cpu_psi_some_pct": 18.7,
  "ram_usage_pct": 63.2,
  "ram_usage_mi": 54321.0,
  "disk_used_pct": 61.2,
  "sched_total_ms": 12345.6,
  "dstate_total_ms": 120.4,
  "softirq_total_ms": 45.8,
  "node_cpu_watts": 487.3
}
```

## Transaction Ingestion

Jobs enter the Emulation Module through a Redis stream, not an HTTP call. `transaction_poller.py` runs as a background thread inside the module and is the only ingestion path:

1. **Poll**: every `POLL_INTERVAL_S` (5s), reads new messages from the Redis stream `emulate` (field `ongoingtx`) via a consumer group (`XREADGROUP ... ">"`), so each message is delivered exactly once — even across pod restarts, since messages are only acknowledged (`XACK`) after successful processing.
2. **Filter**: a message is only processed if its `status` is `"ongoing"`, its `tx.type` is `"transfer"`, and — if this seller node's IP was resolved (env var or auto-detected via a ContainerLab UDP probe) — `tx.seller.ip` matches it. Anything else is acknowledged and skipped so it doesn't accumulate in the pending list.
3. **Map app type**: `tx.buyer.app_type` (one of `HR`, `SN`, `SA`, `ES`) is mapped to the internal short form (`hotel`, `sn`, `sa`, `es`). A missing or unrecognized value is logged as a warning, acknowledged, and skipped — no job is created and nothing is silently defaulted.
4. **Compute remaining lifetime**: `tx.lease_duration` minus elapsed time since `tx.tx_start_ts`, so a transaction picked up late doesn't run longer than its original lease.
5. **Create the job**: `Timeline.add_job(app_type, lifetime_seconds, buyer_name)` — the job becomes active at the next 5s tick.
6. **Log the event**: `event_logger.py` appends one JSON line per processed transaction to `/app/logs/events_YYYY-MM-DD.jsonl`, including the resulting composition and timing.

Processing is gated by calibration: no messages are handled until `calibration_done` is `True` (see `POST /calibration/done` above) — they simply accumulate unread on the stream until then.

## Supported Workloads

| App | Key | vCPU | RAM | Disk |
|---|---|---|---|---|
| Hotel Reservation | `hotel` | ~20 | ~1.2 GB | ~12 GB |
| Social Network | `sn` | ~40 | ~54 GB | ~14 GB |
| Sentiment Analysis | `sa` | ~13 | ~14 GB | ~17 GB |
| Elasticsearch | `es` | ~0 | ~304 GB | ~1.4 GB |

## Datasets

Datasets are generated on demand, not pre-computed as a fixed batch. The theoretical composition space is hotel 0–8 × sn 0–7 × sa 0–6 × es 0–5 (3,024 combinations); only the ones actually requested at runtime are ever materialized.

**Generation strategy** (`dataset_generator.get_or_generate()`), per requested composition:
- **Exact Tier A match** — a real experiment exists for this exact composition: its corrected pod CSV is copied and PSI/eBPF/power are recomputed directly.
- **No exact match** — built from the closest Tier A experiment (or the idle baseline, for compositions with no overlapping real data) plus extra scaled replica pods for the missing instances. ES replicas are RAM-dominant enough that a generated combination's total RAM is checked against node capacity and scaled down if it would exceed it (`enforce_ram_cap`).

Output is Parquet, with metric columns downcast to `float32`.

**Caching**: `DatasetSelector` sits in front of generation with a size-bound LRU cache (`dataset_cache.py`), bounded by total bytes on disk rather than entry count — `config.DATASET_CACHE_MAX_BYTES` (500 MiB by default). At startup it:
1. Sweeps any stray `.parquet` files left in `datasets/` from a previous run whose composition isn't a Tier A entry (cheap to regenerate on demand, so nothing needs to persist across restarts).
2. Pre-warms and permanently pins all 62 Tier A compositions, so the real, non-generated data is always instantly available and never evicted.

Anything generated beyond the pinned Tier A set is subject to normal least-recently-used eviction once the byte budget is reached.

## Building and Pushing

```bash
# Login to DockerHub
docker login

# Build and push (datasets must be in datasets/ folder)
chmod +x build.sh
./build.sh

# With version tag
./build.sh v1.0
```

Image: `alibeiti/emulation-module:latest`

## Kubernetes Deployment

Apply in order on the seller node:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/deployment.yaml
```

The module will be reachable at:
```
http://ksense-usage-api.ksense.svc:8090
```

## Running Locally (without Kubernetes)

```bash
pip install -r requirements.txt

# Run without KWOK (metrics API only)
python main.py --no-kwok

# Dry run (no Kubernetes API calls)
python main.py --dry-run
```

## Running Tests

```bash
pip install pandas numpy
python test_emulation_module.py -v
```

Expected: 41/41 tests passing.

## Data Preparation Scripts

These scripts prepare the corrected source data and models `dataset_generator.py` builds on (not needed at runtime):

| Script | Purpose |
|---|---|
| `generate_pod_corrected_v2.py` | Distribute eBPF metrics to pod level |
| `fit_power_model.py` | Fit power consumption regression model |
| `prepare_datasets.py` | Batch-regenerate the full composition grid ahead of time, if ever needed — not required at runtime; `dataset_generator.py` generates on demand instead |
| `assess_dataset_quality.py` | Quality assessment of generated datasets |

## Metric Aggregation Methods

| Metric | Method |
|---|---|
| CPU, RAM, Disk, Power | Simple sum across pods |
| PSI | Ridge regression model (R²=0.88) |
| sched, dstate, softirq | Sum (pre-distributed top-down in datasets) |