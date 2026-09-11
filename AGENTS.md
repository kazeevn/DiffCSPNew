# Environment Management
- **Always use `uv`** for Python environment management and running commands.
- Run tests with `uv run pytest`.
- Execute CLI tools and Python scripts with `uv run ...` (for example, `uv run python -m diffcsp.cli.train ...` or `uv run diffcsp-train ...`).
- Manage dependencies with `uv` (`uv add`, `uv sync`, etc.). Do not use raw system python or bare `pytest`.
