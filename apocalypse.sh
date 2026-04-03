#!/bin/bash
set -e
# SOTA PATH: Ensure uv and python are found
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/minh/.local/bin"

PROJECT_DIR=~/projects/turboquant_pp
MNT_DIR=/mnt/c/Users/ADMIN/OneDrive/Desktop/turboquant_pp

echo "--- 1. SMART SYNC (Preserving .venv) ---"
# Create project dir if not exists
mkdir -p $PROJECT_DIR

# Sync code excluding virtual env and pycache
rsync -a --delete --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' $MNT_DIR/ $PROJECT_DIR/

echo "--- 2. UV SYNC (Module Registration) ---"
cd $PROJECT_DIR
# SOTA: Force editable installation to ensure local source is used
uv sync
uv pip install -e .

# Kill ghost pycache in source
find src -name "__pycache__" -exec rm -rf {} +

echo "--- 4. RUNNING FINAL VERIFICATION ---"
export PYTHONPATH=$PROJECT_DIR/src
echo "[Status] Running debug_hybrid.py..."
uv run python debug_hybrid.py

echo "[Status] Running Pytest Suite..."
uv run python -m pytest -v
