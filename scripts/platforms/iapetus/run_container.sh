#!/usr/bin/env bash
set -e

# Convenience wrapper to execute commands within the custom PyTorch 2.14 Docker container
EXTRA_ARGS=()
if [ -n "$WANDB_API_KEY" ]; then
  EXTRA_ARGS+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
fi
if [ -n "$WANDB_ENTITY" ]; then
  EXTRA_ARGS+=(-e "WANDB_ENTITY=$WANDB_ENTITY")
fi
if [ -f "$HOME/.netrc" ]; then
  EXTRA_ARGS+=(-v "$HOME/.netrc:/home/kna/.netrc:ro" -v "$HOME/.netrc:/root/.netrc:ro")
fi
if [ -d "$HOME/.config/wandb" ]; then
  EXTRA_ARGS+=(-v "$HOME/.config/wandb:/home/kna/.config/wandb" -v "$HOME/.config/wandb:/root/.config/wandb")
fi
if [ -d "$HOME/.cache" ]; then
  EXTRA_ARGS+=(-v "$HOME/.cache:/home/kna/.cache" -v "$HOME/.cache:/root/.cache")
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

exec docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v "$repo_root:/workspace" \
  -w /workspace \
  "${EXTRA_ARGS[@]}" \
  pytorch:2.14.0-cuda11.8-py312-universal \
  "$@"

