# ── Emulation Module Dockerfile ───────────────────────────────────────────────
# Builds a self-contained image for the seller-side emulation module.
# Datasets and baseline are baked into the image at build time.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/kubectl \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    "numpy==1.24.4" \
    "pandas==2.0.3" \
    "fastapi>=0.110.0" \
    "uvicorn>=0.29.0" \
    "kubernetes>=29.0.0" \
    "pydantic>=2.0.0"

# Copy module source files
COPY config.py .
COPY timeline.py .
COPY dataset_selector.py .
COPY replay_engine.py .
COPY aggregator.py .
COPY baseline_provider.py .
COPY transaction_poller.py .
COPY prepare_calibration.py .
COPY kwok_manager.py .
COPY api.py .
COPY main.py .

# Copy datasets and baseline into the image (baked in at build time)
# Run prepare_datasets_v2.py first to generate datasets/ folder
COPY datasets/ /app/datasets/
COPY baseline/ /app/baseline/
COPY calibration/ /app/calibration/

# Expose API port
EXPOSE 8090

# Health check for Kubernetes liveness probe
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8090/healthz || exit 1

# Entry point
ENTRYPOINT ["python", "main.py"]
CMD []