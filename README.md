# Return Decomposition of Hybrid Strategies in the Brazilian Market: Evidence of Genuine Alpha and Factor Exposure

Replication of Freitas (2025), *"Return Decomposition of Hybrid Strategies in
the Brazilian Market: Evidence of Genuine Alpha and Factor Exposure."*

The paper backtests a hybrid strategy that combines a fundamentalist stock
screen (13 B3-listed assets) with a technical moving-average crossover (MAC)
system, and decomposes the resulting return series with CAPM, Fama-French
three-factor and Carhart four-factor models to test for genuine (alpha)
return beyond systematic risk-factor exposure.

## Repository layout

```
src/     backtest engine, performance metrics, factor-model regressions, plotting
exp/     exp1 (descriptive performance + Jobson-Korkie test)
         exp2 (CAPM / FF3 / Carhart4 factor decomposition)
         exp3 (SMA-parameter robustness + sub-period Adaptive-Markets analysis)
tests/   pytest unit tests (synthetic fixtures) and market-data integration tests
data/    data_loader.py: OHLCV via QuantX Dataloader (with Yahoo backstop) and
         NEFIN/FEA-USP monthly risk-factor download
results/ all experiment outputs (CSV/PNG) and RESULTS.md
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"      # unit tests on synthetic fixtures
python -m pytest                           # full suite, including market-data tests
                                            # (requires DATALOADER_API_KEY/MASSIVE_API_KEY)
```

## Experiments

```bash
python -m exp.exp1   # Backtesting and descriptive performance analysis
python -m exp.exp2   # Factor decomposition and Jensen alpha generation
python -m exp.exp3   # SMA parameter sensitivity and sub-period robustness
```

Each run overwrites its output files directly under `results/`. The reported
numbers and verdicts in `results/RESULTS.md` are updated by hand after a rerun.

```bash
./reproduce.sh       # runs the unit tests and all three experiments in order
```

## Data

Daily OHLCV bars for the 13 B3 tickers (CEBR3, CMIG4, CSMG3, CYRE3, EALT3,
GRND3, ITSA3, JHSF3, MDNE3, PRIO3, MRSA3B, SYNE3, TECN3) are loaded through the
QuantX Dataloader REST API (`POST /fetch-ohlcvs`, dividend-adjusted), which
falls back to Yahoo Finance when Massive has no coverage for a ticker over the
sample period (see "Deviations from the paper" in `results/RESULTS.md`). The
Ibovespa index level (`^BVSP`) is loaded directly from Yahoo Finance, since it
is an index level rather than an OHLCV security and Massive has no entitlement
for it under any tested symbol convention. Monthly risk factors (market
Rm-Rf, SMB, HML, WML, IML) and the risk-free rate are downloaded from
NEFIN/FEA-USP. See `.env.example` for the required API keys.

Sample period: January 2000 to December 2025, as specified in the paper.
