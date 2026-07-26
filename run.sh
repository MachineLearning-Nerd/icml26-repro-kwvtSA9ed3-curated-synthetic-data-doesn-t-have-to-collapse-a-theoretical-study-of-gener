#!/usr/bin/env bash
# The fixed reproduction command for EVERY node in this project: `bash run.sh`.
# What a node does is decided by the committed file repro/config/active.json,
# never by this script, the command line, or environment variables.
set -euo pipefail

echo "== run.sh: pinned environment via uv (no conda, no unmanaged pip) =="

if ! command -v uv >/dev/null 2>&1; then
  echo "-- installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv --version

# One repository-level .venv, reused by every stage and every node.
uv sync --frozen --no-progress

echo "== resolved environment =="
uv run python -c "import sys, numpy, scipy, sympy, torch; \
print('python  ', sys.version.split()[0]); \
print('numpy   ', numpy.__version__); \
print('scipy   ', scipy.__version__); \
print('sympy   ', sympy.__version__); \
print('torch   ', torch.__version__, '| cuda available:', torch.cuda.is_available())"

# Hard guard: this campaign is authorised for CPU only.
uv run python -c "import torch, sys; \
sys.exit('REFUSING TO RUN: GPU visible but this campaign is CPU-only') if torch.cuda.is_available() else None"

echo "== reproduction =="
uv run python -m repro.main
