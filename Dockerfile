# ── Emulation Module Dockerfile ───────────────────────────────────────────────
# Builds a self-contained image for the seller-side emulation module.
# Corrected source data and baseline are baked into the image at build time;
# datasets/ itself starts empty — it's populated on container startup by
# DatasetSelector's sweep + Tier A pre-warm (dataset_generator.py generates
# on demand, dataset_cache.py caches the result).

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
RUN mkdir -p /app/logs
RUN mkdir -p /app/datasets
# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy module source files
COPY config.py .
COPY timeline.py .
COPY dataset_selector.py .
COPY dataset_generator.py .
COPY dataset_cache.py .
COPY replay_engine.py .
COPY aggregator.py .
COPY baseline_provider.py .
COPY transaction_poller.py .
COPY event_logger.py .
COPY prepare_calibration.py .
COPY kwok_manager.py .
COPY api.py .
COPY main.py .

# Corrected source data + baseline, baked in at build time.
# datasets/ is intentionally NOT copied — it's generated on demand at
# startup (see comment at the top of this file).
COPY corrected_full/ /app/corrected_full/
COPY all_data_full/baseline_node.csv /app/all_data_full/baseline_node.csv
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