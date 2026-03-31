# Emulation Module

Seller-side workload emulator for the decentralized resource trading platform. Replays pre-recorded Kubernetes workload metrics on fake KWOK pods and exposes aggregated node-level metrics via REST API.

## Overview

When a buyer submits a job request, the Emulation Module:
1. Selects the appropriate pre-generated dataset matching the workload composition
2. Replays pod-level metrics every 5 seconds on KWOK (fake) pods
3. Aggregates metrics to node level using validated models
4. Exposes results via REST API for Admission Control and Pricing modules

## Repository Structure

```
emulation_module/
├── main.py                 # Entry point — starts API + tick loop
├── api.py                  # FastAPI endpoints
├── timeline.py             # Job lifecycle and composition tracking
├── dataset_selector.py     # Selects dataset for a given workload composition
├── replay_engine.py        # Window-by-window dataset replay + namespace mapping
├── aggregator.py           # Pod → node level metric aggregation
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
| `POST` | `/submit` | Buyer | Submit a workload job |
| `GET` | `/usage/latest` | Admission Control | Current node metrics (updates every 5s) |
| `GET` | `/usage/capacity` | Admission Control | Node total resources (constant) |
| `GET` | `/status` | Monitoring | Active jobs, composition, dataset info |
| `GET` | `/healthz` | Kubernetes | Liveness probe |

### POST /submit
```json
{
  "app_type": "hotel",
  "lifetime_seconds": 300
}
```

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

## Supported Workloads

| App | Key | vCPU | RAM | Disk |
|---|---|---|---|---|
| Hotel Reservation | `hotel` | ~20 | ~1.2 GB | ~12 GB |
| Social Network | `sn` | ~40 | ~54 GB | ~14 GB |
| Sentiment Analysis | `sa` | ~13 | ~14 GB | ~17 GB |

## Datasets

503 pre-generated datasets covering all combinations of:
- Hotel: 0–8 instances
- Social Network: 0–7 instances  
- Sentiment Analysis: 0–6 instances

Real experiments are used directly. Missing combinations are generated using a validated superposition method with physical constraint enforcement.

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

These scripts are run once to prepare the datasets (not needed at runtime):

| Script | Purpose |
|---|---|
| `generate_pod_corrected_v2.py` | Distribute eBPF metrics to pod level |
| `fit_power_model.py` | Fit power consumption regression model |
| `prepare_datasets.py` | Generate all 503 emulation-ready datasets |
| `assess_dataset_quality.py` | Quality assessment of generated datasets |

## Metric Aggregation Methods

| Metric | Method |
|---|---|
| CPU, RAM, Disk, Power | Simple sum across pods |
| PSI | Ridge regression model (R²=0.88) |
| sched, dstate, softirq | Sum (pre-distributed top-down in datasets) |