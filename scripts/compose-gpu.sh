#!/usr/bin/env bash
set -euo pipefail

# GPU-enabled counterpart to compose.sh. Requires the NVIDIA Container Toolkit
# on the host so the container can access the GPU(s).

# Build docker image (fast if already completed)
docker compose build yolov5-rknn-gpu

# Start a container with GPU passthrough. The entrypoint
# (scripts/docker-entrypoint.sh) creates a user and group matching these
# UID/GID, so files written to the mounted /workspace volume are owned by us on
# the host rather than by root.
HOST_UID="$(id -u)" HOST_GID="$(id -g)" \
  docker compose run --rm yolov5-rknn-gpu "$@"
