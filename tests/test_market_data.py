"""Integration tests on a reduced real-data slice (Sections 3.1-3.2).

These tests exercise the shared data-loading path and the signal/backtest
pipeline on a few real instruments instead of synthetic fixtures, checking
the paper's structural properties (no look-ahead, correct window lengths,
monotone price index) rather than its reported numbers.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import asset_daily_returns, hybrid_portfolio_returns
from src.signals import sma_crossover_signals

pytestmark = pytest.mark.market_data


def test_market_data_slice_loads_expected_shape(market_data_slice):
    panel = market_data_slice
    assert panel.shape[0] > 100
    assert panel.shape[1] == 3
    assert panel.index.is_monotonic_increasing
    assert (panel.dropna(how="all") > 0).all().all()


def test_sma_signals_on_real_slice_have_no_lookahead(market_data_slice):
    panel = market_data_slice
    ticker = panel.columns[0]
    prices = panel[ticker].dropna()
    entries, exits = sma_crossover_signals(prices, short_window=9, long_window=46)
    # A crossover cannot be detected before the long SMA's window is full:
    # the first long_window-1 observations must carry no entry/exit signal.
    assert not entries.iloc[: 46 - 1].any()
    assert not exits.iloc[: 46 - 1].any()
    assert entries.index.equals(prices.index)
    assert exits.index.equals(prices.index)


def test_asset_returns_and_hybrid_portfolio_on_real_slice(market_data_slice):
    panel = market_data_slice
    asset_returns = asset_daily_returns(panel, short_window=9, long_window=46)
    hybrid = hybrid_portfolio_returns(asset_returns)
    assert isinstance(hybrid, pd.Series)
    assert hybrid.index.is_monotonic_increasing
    # The hybrid portfolio return on any day is the simple cross-sectional
    # average of the (at most 3) asset returns available that day (Section 3.1).
    sample_day = asset_returns.dropna(how="all").index[-1]
    expected = asset_returns.loc[sample_day].mean()
    assert np.isclose(hybrid.loc[sample_day], expected, equal_nan=True)
