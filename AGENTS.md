# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A replication of Freitas (2025), *"Return Decomposition of Hybrid Strategies
in the Brazilian Market: Evidence of Genuine Alpha and Factor Exposure."* It
backtests a hybrid strategy combining a fundamentalist 13-asset B3 stock
screen with an SMA(9)/SMA(46) moving-average crossover system (Section 3.2),
computes descriptive risk-return metrics and a Jobson-Korkie/Memmel test
against the Ibovespa (Section 3.3), decomposes the hybrid portfolio's monthly
returns with CAPM, Fama-French three-factor and Carhart four-factor models
against NEFIN/FEA-USP risk factors (Section 3.4), and runs robustness checks
over alternative SMA configurations and three sub-periods (Section 3.5).

## Layout

```
src/     backtest engine, performance metrics, factor-model regressions,
         Jobson-Korkie test, plotting, config.py (all paper parameters)
exp/     exp1 (descriptive performance + Jobson-Korkie test)
         exp2 (CAPM / FF3 / Carhart4 factor decomposition)
         exp3 (SMA-parameter robustness + sub-period Adaptive-Markets analysis)
tests/   pytest unit tests (synthetic fixtures) and market-data integration tests
data/    data_loader.py: OHLCV via the QuantX Dataloader (with Yahoo backstop)
         and the NEFIN/FEA-USP monthly risk-factor download
results/ all experiment outputs (CSV/PNG) and RESULTS.md
```

Library code lives only in flat modules under `src/`; experiment scripts only
in `exp/`; all tests only in `tests/`. Experiments run as `python -m
exp.<name>` and tests as `python -m pytest` from the repository root; no
`sys.path` edits.

## Conventions

- Every paper-derived parameter (tickers, SMA windows, sample dates, initial
  capital, NEFIN URL, OLS covariance type, SMA robustness configurations,
  sub-period boundaries) is a module-level `UPPER_CASE` variable in
  `src/config.py`, each with a comment naming its meaning, unit, and the
  paper section it comes from, or stating that the paper does not specify it.
  No parameters are hardcoded inside `src/` or `exp/` functions.
- Market data is loaded only through `data/data_loader.py`
  (`load_experiment_universe`, OHLCV via the QuantX Dataloader REST API, and
  the NEFIN/FEA-USP CSV for risk factors). No module outside `data/` calls an
  external market-data endpoint directly, and downloaded data is never
  committed (see `.gitignore`).
- API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from the
  environment via `python-dotenv` and only used in requests to the QuantX and
  Massive APIs; they are kept in an uncommitted local `.env` file, never in
  source, logs, or results. See `.env.example` for the two placeholders.
- Deviations from the paper and any unresolved gaps are recorded only in
  `results/RESULTS.md`, not in code comments or config values.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests on synthetic fixtures (no API keys needed)
python -m pytest                        # full suite, including market-data integration tests
                                         # (requires DATALOADER_API_KEY and MASSIVE_API_KEY)
```

Market-data tests are marked `market_data` in `tests/conftest.py` and are
skipped automatically when `DATALOADER_API_KEY` is not set. They load a
reduced slice of the universe once per test session and cache it, so repeated
runs do not re-download it.

## Experiments

```bash
python -m exp.exp1   # Backtesting and descriptive performance analysis
python -m exp.exp2   # Factor decomposition and Jensen alpha generation
python -m exp.exp3   # SMA parameter sensitivity and sub-period robustness

./reproduce.sh       # runs the unit tests and all three experiments in order
```

Each experiment run overwrites its own output files directly under
`results/` (no per-experiment subdirectories). The numbers and verdicts in
`results/RESULTS.md` are **not** regenerated automatically: after a rerun,
check them against the new files under `results/` and update
`results/RESULTS.md` by hand.
