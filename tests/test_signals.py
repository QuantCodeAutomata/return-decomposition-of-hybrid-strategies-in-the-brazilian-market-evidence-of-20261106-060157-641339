"""Unit tests of the SMA crossover signal generation (src/signals.py)."""

import numpy as np
import pandas as pd
import pytest

from src.signals import sma_crossover_signals


def test_invalid_window_order_raises():
    price = pd.Series(np.linspace(1, 2, 60))
    with pytest.raises(ValueError):
        sma_crossover_signals(price, 46, 9)


def test_entries_and_exits_are_mutually_exclusive(synthetic_price_series):
    entries, exits = sma_crossover_signals(synthetic_price_series, 9, 46)
    assert entries.dtype == bool
    assert exits.dtype == bool
    assert not (entries & exits).any()


def test_signal_fires_on_known_crossover():
    # A W-shaped price series: down-up-down-up, forces at least one
    # predictable crossover of SMA(3) above and one below SMA(6).
    leg = lambda start, end: np.linspace(start, end, 30)
    price = pd.Series(np.concatenate([
        leg(100, 50), leg(50, 150), leg(150, 70), leg(70, 180),
    ]))
    entries, exits = sma_crossover_signals(price, 3, 6)
    assert entries.sum() >= 1
    assert exits.sum() >= 1
    # No look-ahead: a signal at time t must depend only on data up to t,
    # i.e. truncating the series and recomputing must not change past signals.
    truncated_entries, _ = sma_crossover_signals(price.iloc[:90], 3, 6)
    pd.testing.assert_series_equal(
        entries.iloc[:90], truncated_entries, check_names=False
    )


def test_short_longer_than_data_handles_gracefully():
    price = pd.Series(np.linspace(1, 2, 5))
    entries, exits = sma_crossover_signals(price, 2, 4)
    assert len(entries) == len(price)
