"""Unit tests of the per-asset backtest simulation (src/backtest.py)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    asset_daily_returns,
    buy_and_hold_daily_returns,
    hybrid_portfolio_returns,
    pooled_trade_pnls,
    simulate_buy_and_hold,
    simulate_mac_portfolio,
)


def test_mac_portfolio_no_look_ahead(synthetic_price_series):
    """Truncating the price series must not change past simulated returns."""
    pf_full = simulate_mac_portfolio(synthetic_price_series, 9, 46, 1_000_000.0)
    pf_trunc = simulate_mac_portfolio(synthetic_price_series.iloc[:200], 9, 46, 1_000_000.0)
    full_returns = pf_full.returns()
    trunc_returns = pf_trunc.returns()
    common = full_returns.index.intersection(trunc_returns.index)
    pd.testing.assert_series_equal(
        full_returns.loc[common].iloc[:190], trunc_returns.loc[common].iloc[:190], check_names=False
    )


def test_buy_and_hold_matches_price_return(synthetic_price_series):
    pf = simulate_buy_and_hold(synthetic_price_series, 1_000_000.0)
    total_return = pf.final_value() / 1_000_000.0 - 1.0
    expected = synthetic_price_series.iloc[-1] / synthetic_price_series.iloc[0] - 1.0
    assert total_return == pytest.approx(expected, rel=1e-6)


def test_asset_daily_returns_one_column_per_asset(synthetic_price_panel):
    returns = asset_daily_returns(synthetic_price_panel, 9, 46)
    assert set(returns.columns) == set(synthetic_price_panel.columns)
    assert len(returns) > 0


def test_hybrid_portfolio_returns_is_simple_average(synthetic_price_panel):
    """Section 3.1 step 4: the hybrid return is the simple cross-sectional
    average of the individually simulated asset returns on each date."""
    asset_returns = pd.DataFrame({
        "A": [0.01, 0.02, np.nan],
        "B": [0.03, -0.01, 0.02],
        "C": [np.nan, 0.04, 0.00],
    })
    hybrid = hybrid_portfolio_returns(asset_returns)
    assert hybrid.iloc[0] == pytest.approx((0.01 + 0.03) / 2)
    assert hybrid.iloc[1] == pytest.approx((0.02 - 0.01 + 0.04) / 3)
    assert hybrid.iloc[2] == pytest.approx((0.02 + 0.00) / 2)


def test_buy_and_hold_daily_returns_handles_multiple_assets(synthetic_price_panel):
    returns = buy_and_hold_daily_returns(synthetic_price_panel)
    assert returns.shape[1] == synthetic_price_panel.shape[1]
    assert not returns.isna().all().any()


def test_pooled_trade_pnls_keys_match_tickers(synthetic_price_panel):
    pnls = pooled_trade_pnls(synthetic_price_panel, 9, 46, 1_000_000.0)
    assert set(pnls.keys()).issubset(set(synthetic_price_panel.columns))
    for arr in pnls.values():
        assert isinstance(arr, np.ndarray)


def test_short_window_not_shorter_than_series_is_skipped():
    short_price = pd.DataFrame({"X": pd.Series(np.linspace(10, 11, 10))})
    returns = asset_daily_returns(short_price, 9, 46)
    assert "X" not in returns.columns
