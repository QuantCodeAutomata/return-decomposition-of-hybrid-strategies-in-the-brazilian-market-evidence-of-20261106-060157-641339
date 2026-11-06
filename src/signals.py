"""Moving-average crossover (MAC) technical signals (Section 3.2).

# vectorbt MA: used for the rolling SMA and the crossing-point detection
# instead of a hand-rolled rolling-window loop (Context7 "vectorbt.dev"
# documentation: ``vbt.MA.run`` + ``ma_crossed_above``/``ma_crossed_below``).
"""

import pandas as pd
import vectorbt as vbt


def sma_crossover_signals(
    price: pd.Series, short_window: int, long_window: int
) -> tuple[pd.Series, pd.Series]:
    """Buy/sell signals of the MAC system (Section 3.2, Eq. "SMA(9) x SMA(46)").

    A Buy signal fires on the bar where the short SMA crosses above the long
    SMA; a Sell signal fires where it crosses below. No fixed stop-loss is
    used: exits are determined exclusively by the opposite technical signal.

    Parameters
    ----------
    price: daily close price series of one asset, dividend-adjusted.
    short_window: period of the short SMA (9 in the baseline system).
    long_window: period of the long SMA (46 in the baseline system).

    Returns
    -------
    (entries, exits): boolean series aligned to ``price.index``.
    """
    if short_window >= long_window:
        raise ValueError("short_window must be strictly shorter than long_window")
    short_ma = vbt.MA.run(price, short_window)
    long_ma = vbt.MA.run(price, long_window)
    entries = short_ma.ma_crossed_above(long_ma)
    exits = short_ma.ma_crossed_below(long_ma)
    return entries, exits
