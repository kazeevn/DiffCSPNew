# Environment Management
- **Always use `uv`** for Python environment management and running commands.
- Run tests with `uv run pytest`.
- Execute CLI tools and Python scripts with `uv run ...` (for example, `uv run python -m diffcsp.cli.train ...` or `uv run diffcsp-train ...`).
- Manage dependencies with `uv` (`uv add`, `uv sync`, etc.). Do not use raw system python or bare `pytest`.

# Experiments
- The experiments must be reproducible
- The experiments should log into WanDB
- If a textual write up of an experiment is done, it should clearly reference the date, git commit hash and the WanDB run ID, so that the report won't be confusing when the code base changes.
