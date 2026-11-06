"""Unit tests of the Jobson-Korkie / Memmel equal-Sharpe test (Section 3.3)."""

import numpy as np
import pandas as pd
import pytest

from src.stats_tests import jobson_korkie_memmel_test


def test_identical_series_gives_undefined_statistic():
    # Exactly identical series make the Memmel (2003) asymptotic variance
    # (theta, Eq. 2) analytically zero (equal mean/std/covariance terms
    # cancel), so the z-statistic is the undefined 0/0 -> NaN.
    rng = np.random.default_rng(5)
    series = pd.Series(rng.normal(0.01, 0.04, 60))
    result = jobson_korkie_memmel_test(series, series.copy())
    assert np.isnan(result.z_statistic)


def test_statistic_is_antisymmetric_under_swap():
    # Swapping the two series must flip the sign of the z-statistic, since
    # the Jobson-Korkie (1981) numerator (Eq. 10) is antisymmetric in a, b.
    rng = np.random.default_rng(7)
    series_a = pd.Series(rng.normal(0.015, 0.04, 60))
    series_b = pd.Series(rng.normal(0.005, 0.05, 60))
    forward = jobson_korkie_memmel_test(series_a, series_b)
    backward = jobson_korkie_memmel_test(series_b, series_a)
    assert forward.z_statistic == pytest.approx(-backward.z_statistic, rel=1e-9)


def test_higher_mean_same_vol_gives_positive_statistic_for_higher_sharpe_series():
    rng = np.random.default_rng(6)
    n = 400
    noise = rng.normal(0.0, 0.03, n)
    low = pd.Series(0.001 + noise)
    high = pd.Series(0.02 + noise)
    result = jobson_korkie_memmel_test(high, low)
    assert result.sharpe_a > result.sharpe_b
    assert result.z_statistic > 0


def test_raises_on_too_few_observations():
    with pytest.raises(ValueError):
        jobson_korkie_memmel_test(pd.Series([0.01, 0.02]), pd.Series([0.01, 0.02]))


def test_misaligned_index_is_inner_joined():
    idx_a = pd.period_range("2020-01", periods=10, freq="M")
    idx_b = pd.period_range("2020-03", periods=10, freq="M")
    series_a = pd.Series(np.linspace(0.01, 0.05, 10), index=idx_a)
    series_b = pd.Series(np.linspace(0.02, 0.06, 10), index=idx_b)
    result = jobson_korkie_memmel_test(series_a, series_b)
    assert result.n_obs == 8  # 2020-03..2020-10 overlap


def test_p_value_within_unit_interval():
    rng = np.random.default_rng(8)
    a = pd.Series(rng.normal(0.01, 0.05, 50))
    b = pd.Series(rng.normal(0.015, 0.04, 50))
    result = jobson_korkie_memmel_test(a, b)
    assert 0.0 <= result.p_value <= 1.0
