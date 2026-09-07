#!/usr/bin/env bash
set -e

# Convenience wrapper to execute commands within the custom PyTorch 2.14 Docker container
exec docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  pytorch:2.14.0-cuda11.8-py312-universal \
  "$@"
