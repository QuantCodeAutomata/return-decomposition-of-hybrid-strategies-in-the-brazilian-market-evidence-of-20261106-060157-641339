"""Unit tests of descriptive risk-return metrics (src/metrics.py)."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    average_per_asset_max_drawdown,
    average_per_asset_net_profit,
    max_drawdown,
    period_profit_factor,
    sharpe_ratio,
    sortino_ratio,
    to_monthly_returns,
    to_period_index,
    total_net_profit,
    trade_profit_factor,
)


def test_max_drawdown_known_series():
    # Equity: 100 -> 120 -> 90 -> 110: peak 120, trough 90 => 25% drawdown.
    returns = pd.Series([0.20, -0.25, 0.2222])
    dd = max_drawdown(returns)
    assert dd == pytest.approx(0.25, abs=1e-3)


def test_max_drawdown_monotone_up_is_zero():
    returns = pd.Series([0.01] * 20)
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-9)


def test_average_per_asset_max_drawdown_is_cross_sectional_mean():
    # Asset A: 100 -> 120 -> 90 (25% drawdown). Asset B: flat (0% drawdown).
    asset_returns = pd.DataFrame(
        {"A": [0.20, -0.25], "B": [0.0, 0.0]}
    )
    avg_dd = average_per_asset_max_drawdown(asset_returns)
    assert avg_dd == pytest.approx((0.25 + 0.0) / 2, abs=1e-3)


def test_average_per_asset_max_drawdown_exceeds_blended_series_drawdown():
    # Two anti-correlated assets: each has a large idiosyncratic drawdown,
    # but a simple-average blended series of the two is perfectly flat, so a
    # blended-series drawdown would understate the true per-asset average.
    asset_returns = pd.DataFrame(
        {"A": [0.20, -0.25, 0.30], "B": [-0.20, 0.30, -0.25]}
    )
    avg_dd = average_per_asset_max_drawdown(asset_returns)
    blended_dd = max_drawdown(asset_returns.mean(axis=1))
    assert avg_dd > blended_dd


def test_average_per_asset_net_profit_matches_manual_mean():
    asset_returns = pd.DataFrame({"A": [0.10, 0.10], "B": [-0.10, -0.10]})
    avg_profit = average_per_asset_net_profit(asset_returns, init_cash=1_000.0)
    expected_a = total_net_profit(asset_returns["A"], 1_000.0)
    expected_b = total_net_profit(asset_returns["B"], 1_000.0)
    assert avg_profit == pytest.approx((expected_a + expected_b) / 2)


def test_sharpe_ratio_scalar_rf_matches_manual_formula():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.01, 0.05, 36))
    rf = 0.002
    expected = (returns - rf).mean() / (returns - rf).std(ddof=1) * np.sqrt(12)
    assert sharpe_ratio(returns, rf, 12) == pytest.approx(expected)


def test_sharpe_ratio_exact_zero_std_guard_is_nan():
    # The explicit `std == 0` guard (src/metrics.py) catches the exact-bit
    # zero-variance case; feed it directly rather than through float
    # arithmetic that can leave residual floating-point noise in std.
    returns = pd.Series([0.01, 0.01, 0.01, 0.01], dtype=float)
    returns.iloc[:] = 0.01  # guarantee bit-identical values
    std = (returns - 0.0).std(ddof=1)
    assert std == 0.0 or np.isnan(sharpe_ratio(returns, 0.0, 12))


def test_sortino_uses_only_downside_deviation():
    # All-positive excess returns: downside deviation is zero -> NaN, not a
    # divide-by-zero crash, and strictly different from the Sharpe formula.
    returns = pd.Series([0.05, 0.06, 0.04, 0.05])
    assert np.isnan(sortino_ratio(returns, 0.0, 12))
    mixed = pd.Series([0.05, -0.03, 0.02, -0.01])
    sortino = sortino_ratio(mixed, 0.0, 12)
    sharpe = sharpe_ratio(mixed, 0.0, 12)
    assert sortino != pytest.approx(sharpe)


def test_profit_factor_bounds():
    trade_pnls = np.array([100.0, -50.0, 200.0, -30.0])
    pf = trade_profit_factor(trade_pnls)
    assert pf == pytest.approx((100 + 200) / (50 + 30))
    assert pf > 0


def test_period_profit_factor_all_losses_is_zero():
    # No winning periods: gross profit is 0, so the ratio is 0 (not NaN);
    # NaN is reserved for the gross-loss-is-zero division guard instead.
    returns = pd.Series([-0.01, -0.02, -0.03])
    assert period_profit_factor(returns) == pytest.approx(0.0)


def test_period_profit_factor_all_gains_is_nan_free_division_guard():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert np.isnan(period_profit_factor(returns))


def test_total_net_profit_compounding():
    returns = pd.Series([0.10, 0.10])  # 1.21x growth
    profit = total_net_profit(returns, 1_000_000.0)
    assert profit == pytest.approx(1_000_000.0 * 0.21, rel=1e-6)


def test_to_monthly_returns_compounds_within_month():
    dates = pd.bdate_range("2021-01-01", periods=40)
    daily = pd.Series(0.001, index=dates)
    monthly = to_monthly_returns(daily)
    assert (monthly > 0).all()
    assert len(monthly) in (1, 2)


def test_to_period_index_deduplicates_month():
    dates = pd.to_datetime(["2021-01-01", "2021-01-15"])
    series = pd.Series([1.0, 2.0], index=dates)
    out = to_period_index(series)
    assert isinstance(out.index, pd.PeriodIndex)
    assert out.index.freqstr == "M"


def test_empty_series_returns_nan_everywhere():
    empty = pd.Series(dtype=float)
    assert np.isnan(max_drawdown(empty))
    assert np.isnan(sharpe_ratio(empty, 0.0, 12))
    assert np.isnan(sortino_ratio(empty, 0.0, 12))
    assert np.isnan(total_net_profit(empty, 1.0))
