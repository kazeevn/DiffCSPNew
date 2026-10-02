#!/usr/bin/env bash
# Run a command against DiffCSPNew's container-only venv inside the stock PyTorch image.
#
#   scripts/platforms/aspire2a/run_in_singularity.sh python -m diffcsp.cli.train --help
#   scripts/platforms/aspire2a/run_in_singularity.sh uv run --no-sync pytest
#
# The venv (built by build_venv.sh) inherits torch 2.14.0+cu126 from the image through
# --system-site-packages, so it is valid only inside the image. Never `uv sync` / `uv run`
# from the host: the host has no python, and uv would delete the venv.
#
# REPO_DIR defaults to the checkout this script lives in and goes first on PYTHONPATH,
# so a worktree imports its own `diffcsp` rather than the main checkout's editable install.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
MAIN_REPO=/home/project/11001786/WyFormer/DiffCSPNew
SIF=${SIF:-/home/users/nus/kna/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif}
if [ -z "${VENV_DIR:-}" ]; then
    if [ -d "$REPO_DIR/.venv" ]; then VENV_DIR="$REPO_DIR/.venv"; else VENV_DIR="$MAIN_REPO/.venv"; fi
fi

export SINGULARITY_NO_EVAL=1
if ! command -v singularity >/dev/null 2>&1; then
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
    module load singularity
fi

# /home/project and /data/projects are not bound automatically on ASPIRE 2A.
BINDS="$REPO_DIR"
for d in /home/project /data/projects /raid; do [ -d "$d" ] && BINDS="$BINDS,$d"; done
[ -n "${EXTRA_BIND:-}" ] && BINDS="$BINDS,$EXTRA_BIND"

exec singularity run --nv \
    --bind "$BINDS" \
    --env "VIRTUAL_ENV=$VENV_DIR" \
    --env "UV_PROJECT_ENVIRONMENT=$VENV_DIR" \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/users/nus/kna/WyFormer/uv-cache}" \
    --env "PATH=$VENV_DIR/bin:$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env "PYTHONPATH=$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$SIF" "$@"
