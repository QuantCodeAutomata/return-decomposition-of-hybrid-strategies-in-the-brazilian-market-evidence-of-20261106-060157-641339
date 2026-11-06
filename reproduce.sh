#!/usr/bin/env bash
# Runs the unit test suite and all three experiments in order, then leaves
# their outputs under results/. Market-data tests and the experiments
# themselves require DATALOADER_API_KEY and MASSIVE_API_KEY (see
# .env.example); set them in the environment or in a local .env file before
# running this script.
set -uo pipefail

cd "$(dirname "$0")"

echo "== unit tests (synthetic fixtures) =="
python -m pytest -m "not market_data"

echo "== experiment 1: backtest and descriptive performance =="
python -m exp.exp1

echo "== experiment 2: factor decomposition =="
python -m exp.exp2

echo "== experiment 3: SMA sensitivity and sub-period robustness =="
python -m exp.exp3

echo "Done. See results/RESULTS.md for the comparison against the paper."
