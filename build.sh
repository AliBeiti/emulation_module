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
TAG="${1:-latest}"

# Check datasets exist
if [ ! -f "datasets/dataset_index.json" ]; then
    echo "ERROR: datasets/dataset_index.json not found"
    echo "Run prepare_datasets.py first"
    exit 1
fi

echo "Building ${IMAGE_NAME}:${TAG}..."
docker build -t "${IMAGE_NAME}:${TAG}" .

echo "Pushing ${IMAGE_NAME}:${TAG} to DockerHub..."
docker push "${IMAGE_NAME}:${TAG}"

echo "Done: ${IMAGE_NAME}:${TAG}"