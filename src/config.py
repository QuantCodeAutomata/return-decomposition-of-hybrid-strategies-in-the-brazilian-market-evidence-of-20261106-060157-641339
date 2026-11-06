"""Parameters for the replication of Freitas (2025), "Return Decomposition of
Hybrid Strategies in the Brazilian Market: Evidence of Genuine Alpha and
Factor Exposure".

Every module-level variable below is a plain value (no classes, no nested
dicts) with a comment stating its meaning, unit and source (paper section or
equation, or "not specified in the paper").
"""

# ---------------------------------------------------------------------------
# Universe (Section 3.1, Table 01): fundamentalist screen, fixed 13-asset list
# ---------------------------------------------------------------------------
TICKERS = [
    "CEBR3", "CMIG4", "CSMG3", "CYRE3", "EALT3", "GRND3", "ITSA3",
    "JHSF3", "MDNE3", "PRIO3", "MRSA3B", "SYNE3", "TECN3",
]  # B3 tickers of the 13 assets selected once by the fundamentalist screen (Table 01)

DATALOADER_TICKER_SUFFIX = ".SA"  # Yahoo/Dataloader exchange suffix for B3-listed stocks

# Ibovespa benchmark (Section 3.3): the paper uses the Ibovespa index itself.
# Massive has no entitlement for the Brazilian index under any tested ticker
# convention (I:IBOV, IBOV, I:BVSP, I:BVSP.SA all return NOT_AUTHORIZED /
# "no data available"), so the ETF that tracks the index is used instead, as
# instructed for an unentitled index. BOVA11 only trades from 2008-11-28.
IBOVESPA_TICKER = "BOVA11"  # B3 ETF tracking the Ibovespa index (substitute, see RESULTS.md)
IBOVESPA_AVAILABLE_FROM = "2008-11-28"  # first trading date of BOVA11 on B3

# ---------------------------------------------------------------------------
# Sample period (Section 3, paper's time horizon)
# ---------------------------------------------------------------------------
START_DATE = "2000-01-01"  # paper's sample start (Section 3)
END_DATE = "2025-12-31"  # paper's sample end (Section 3)
IDENTITY_DATE = END_DATE  # point-in-time ticker identity resolved at the sample end

# ---------------------------------------------------------------------------
# Backtest mechanics (Section 3.1-3.2)
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 1_000_000.0  # BRL, theoretical initial capital per asset (Section 3.1)
SMA_SHORT_BASELINE = 9  # periods, short SMA of the baseline MAC system (Section 3.2)
SMA_LONG_BASELINE = 46  # periods, long SMA of the baseline MAC system (Section 3.2)
USE_STOP_LOSS = False  # paper: "No fixed stop-losses were used" (Section 3.2)

# ---------------------------------------------------------------------------
# Risk-free rate (Section 3.3-3.4)
# ---------------------------------------------------------------------------
# "The risk-free rate corresponds to the monthly average of the SELIC rate."
# NEFIN/FEA-USP publishes a monthly Risk_Free series built from SELIC, which
# is used directly for both the Sharpe ratio (Section 3.3) and the factor
# models (Section 3.4) instead of re-deriving it from BCB SELIC.
NEFIN_FACTORS_URL = "https://nefin.com.br/nefindata/risk-factors/nefin_factors_monthly.csv"  # NEFIN/FEA-USP monthly risk-factor export (Section 3.4)

TRADING_DAYS_PER_YEAR = 252  # calendar convention for annualizing daily series (not specified in the paper; standard Brazilian B3 convention)
MONTHS_PER_YEAR = 12  # used to annualize monthly Sharpe ratios (Section 3.3, "annualized Sharpe ratio of the aggregate portfolio is 0.443")

# ---------------------------------------------------------------------------
# Profit Factor definition (Section 3.3, uncertainty flagged in the task)
# ---------------------------------------------------------------------------
# The paper defines Profit Factor as "the ratio of total gross profit to total
# gross loss across all operations" (Section 3.3, Table 04 note). "Operations"
# is interpreted as completed round-trip trades for the MAC system (which has
# discrete entries/exits), and as periods of positive/negative return for the
# buy-and-hold and Ibovespa series (which have no discrete trades).
PROFIT_FACTOR_BASIS = "trade"  # "trade" for the MAC system; period-based for buy-and-hold/Ibovespa

# ---------------------------------------------------------------------------
# Experiment 1: descriptive performance / Jobson-Korkie test (Section 3.3)
# ---------------------------------------------------------------------------
JK_CONFIDENCE_LEVEL = 0.05  # 5% significance level used in the paper's JK test (Section 4.3)

# ---------------------------------------------------------------------------
# Experiment 2: factor decomposition (Section 3.4)
# ---------------------------------------------------------------------------
FACTOR_SAMPLE_START = "2000-01-01"  # Section 3.4: "January 2000 to December 2025 (T = 300 observations)"
FACTOR_SAMPLE_END = "2025-12-31"
OLS_COV_TYPE = "HC0"  # White (1980) heteroskedasticity-robust standard errors (Section 3.4)

# ---------------------------------------------------------------------------
# Experiment 3: robustness (Section 3.5)
# ---------------------------------------------------------------------------
SMA_CONFIGS = [
    (5, 20, "SMA 5-20 (reactive)"),
    (9, 46, "SMA 9-46 (baseline)"),
    (12, 26, "SMA 12-26 (MACD)"),
    (20, 50, "SMA 20-50 (conservative)"),
]  # Section 3.5: four SMA configurations tested for robustness

SUB_PERIODS = [
    ("2000-01-01", "2008-12-31", "2000-2008"),
    ("2009-01-01", "2016-12-31", "2009-2016"),
    ("2017-01-01", "2025-12-31", "2017-2025"),
]  # Section 3.5: three sub-periods evaluating the Adaptive Markets Hypothesis

# ---------------------------------------------------------------------------
# Universe size / runtime management
# ---------------------------------------------------------------------------
# The full universe is fixed at 13 assets plus the benchmark (Table 01): it is
# small enough that no download-time reduction is needed for Experiments 1-3.
UNIVERSE_SIZE_PAPER = 13
UNIVERSE_SIZE_RUN = 13
