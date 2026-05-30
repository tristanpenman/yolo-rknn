#!/usr/bin/env bash
set -euo pipefail

# Build docker image (fast if already completed)
docker compose build yolov5-rknn

# Start a container. The entrypoint (scripts/docker-entrypoint.sh) creates a
# user and group matching these UID/GID, so files written to the mounted
# /workspace volume are owned by us on the host rather than by root.
HOST_UID="$(id -u)" HOST_GID="$(id -g)" \
  docker compose run --rm yolov5-rknn "$@"
