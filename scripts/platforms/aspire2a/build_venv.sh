#!/usr/bin/env bash
# Build DiffCSPNew's venv for ASPIRE 2A, from a login or compute node (it downloads):
#
#   bash scripts/platforms/aspire2a/build_venv.sh
#
# uv.lock pins torch 2.14.0+cu133 from zeus's local wheel index, which does not exist
# here, so `uv sync` cannot be used. Instead, inside the image:
#   1. `uv pip compile pyproject.toml` (+ extras) -> requirements-aspire2a.txt, committed,
#      so the dependency set of every run is recorded in git;
#   2. strip torch / nvidia-* / triton -- the image provides torch 2.14.0+cu126;
#   3. `uv venv --system-site-packages` and install the closure with --no-deps;
#   4. install the project itself, editable.
# RESOLVE=0 skips step 1 and reinstalls the committed requirements verbatim.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
EXTRAS=${EXTRAS:-"wandb orb dev"}
RESOLVE=${RESOLVE:-1}
REQ="scripts/platforms/aspire2a/requirements-aspire2a.txt"

extra_args=""
for e in $EXTRAS; do extra_args="$extra_args --extra $e"; done

REPO_DIR="$REPO_DIR" VENV_DIR="$REPO_DIR/.venv" bash "$SCRIPT_DIR/run_in_singularity.sh" bash -c "
set -euo pipefail
cd '$REPO_DIR'
export UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
UV=\$HOME/.local/bin/uv
if [ '$RESOLVE' = 1 ]; then
    \$UV pip compile --python /usr/bin/python3.12 --no-annotate --no-header $extra_args \
        -o $REQ.full pyproject.toml
    grep -viE '^(torch|nvidia-[a-z0-9-]+|pytorch-triton|triton|triton-[a-z]+)([[:space:]=<>!~;]|\$)' \
        $REQ.full > $REQ
    rm $REQ.full
fi
\$UV venv --clear --system-site-packages --python /usr/bin/python3.12 .venv
\$UV pip install --python .venv/bin/python --no-deps -r $REQ
\$UV pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c '
import torch, torch_geometric, pymatgen.core, pyxtal, wandb, diffcsp
from importlib.metadata import version
print(\"torch\", torch.__version__, \"cuda_ok=\", torch.cuda.is_available(), torch.__file__)
print(\"torch_geometric\", torch_geometric.__version__, \"| pymatgen\", version(\"pymatgen\"), \"| pyxtal\", version(\"pyxtal\"))
print(\"diffcsp from\", diffcsp.__file__)
'
"
