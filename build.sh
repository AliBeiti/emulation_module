#!/bin/bash
# build.sh
# Builds and pushes the emulation module image to DockerHub.
# Run from the emulation_module/ directory.
#
# Usage:
#   ./build.sh          # build and push with tag latest
#   ./build.sh v1.0     # build and push with custom tag

set -e

DOCKERHUB_USER="alibeiti"
IMAGE_NAME="${DOCKERHUB_USER}/emulation-module"
# NOTE: default tag here is "latest", but k8s/deployment.yaml currently pins
# "v2" — building with no argument does NOT update what's actually deployed.
# Not fixing that behavior here, just flagging it.
TAG="${1:-latest}"

# Check the runtime data dataset_generator.py/main.py actually need exists
# (datasets/ itself is generated on demand at container startup, not
# pre-built, so it is intentionally not checked here)
for required in \
    "corrected_full/experiment_meta.json" \
    "all_data_full/baseline_node.csv" \
    "calibration/calibration_pod.csv"
do
    if [ ! -f "$required" ]; then
        echo "ERROR: $required not found"
        exit 1
    fi
done

echo "Building ${IMAGE_NAME}:${TAG}..."
docker build -t "${IMAGE_NAME}:${TAG}" .

echo "Pushing ${IMAGE_NAME}:${TAG} to DockerHub..."
docker push "${IMAGE_NAME}:${TAG}"

echo "Done: ${IMAGE_NAME}:${TAG}"