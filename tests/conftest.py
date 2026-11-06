"""Shared fixtures for unit tests and market-data integration tests."""

import os

import numpy as np
import pandas as pd
import pytest

from src import config as paper_config


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: tests that load a reduced slice of real market data"
    )


@pytest.fixture
def synthetic_price_series() -> pd.Series:
    """A deterministic synthetic daily price series (random walk with drift)."""
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = rng.normal(loc=0.0005, scale=0.02, size=n)
    price = 100.0 * (1.0 + pd.Series(returns, index=dates)).cumprod()
    return price


@pytest.fixture
def synthetic_price_panel() -> pd.DataFrame:
    """A small synthetic multi-asset price panel for backtest-pipeline tests."""
    rng = np.random.default_rng(7)
    n = 300
    dates = pd.bdate_range("2019-01-01", periods=n)
    panel = {}
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        returns = rng.normal(loc=0.0003 + 0.0001 * i, scale=0.015, size=n)
        panel[ticker] = 50.0 * (1.0 + pd.Series(returns, index=dates)).cumprod()
    return pd.DataFrame(panel)


@pytest.fixture
def synthetic_monthly_factors() -> pd.DataFrame:
    """A small synthetic monthly factor panel mimicking the NEFIN layout."""
    rng = np.random.default_rng(123)
    n = 60
    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    data = {
        "Rm_minus_Rf": rng.normal(0.01, 0.05, n),
        "SMB": rng.normal(0.002, 0.03, n),
        "HML": rng.normal(0.001, 0.03, n),
        "WML": rng.normal(0.003, 0.03, n),
        "IML": rng.normal(0.0, 0.03, n),
        "Risk_Free": np.full(n, 0.008),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture(scope="session")
def market_data_slice(pytestconfig):
    """A reduced real-data slice: 3 tickers, 2 years, cached across test runs."""
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set: market_data tests skipped")
    from data.data_loader import load_close_price_panel

    cache_dir = pytestconfig.cache.mkdir("market_data_slice")
    cache_file = cache_dir / "prices.csv"
    tickers = paper_config.TICKERS[:3]
    start, end = "2022-01-01", "2023-12-31"
    if cache_file.exists():
        panel = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        panel, _ = load_close_price_panel(tickers, start, end, use_cache=True)
        panel.to_csv(cache_file)
    return panel
