"""Per-asset backtest simulation (Sections 3.1-3.2): MAC system vs buy-and-hold.

# vectorbt Portfolio.from_signals / from_holding: used to simulate the trading
# account instead of a hand-rolled equity-curve loop (Context7 "vectorbt.dev"
# documentation on signal-based portfolios).
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.signals import sma_crossover_signals


def simulate_mac_portfolio(
    price: pd.Series, short_window: int, long_window: int, init_cash: float
) -> vbt.Portfolio:
    """Simulate the MAC trading account for one asset (Section 3.2).

    Whole trading account simulation, long-only, reinvesting the full
    position on each signal (no leverage, no fixed stop-loss: exits are
    exclusively the opposite crossover signal).
    """
    entries, exits = sma_crossover_signals(price, short_window, long_window)
    return vbt.Portfolio.from_signals(
        price, entries, exits, init_cash=init_cash, freq="D", accumulate=False
    )


def simulate_buy_and_hold(price: pd.Series, init_cash: float) -> vbt.Portfolio:
    """Simulate a passive buy-and-hold account on the same asset (Section 3.3)."""
    return vbt.Portfolio.from_holding(price, init_cash=init_cash, freq="D")


def trade_pnls(portfolio: vbt.Portfolio) -> np.ndarray:
    """Realized profit/loss (BRL) of every closed round-trip trade."""
    records = portfolio.trades.records_readable
    return records["PnL"].to_numpy() if len(records) else np.array([])


def asset_daily_returns(prices: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Daily simple returns of the MAC strategy, one column per asset.

    Each column is ``portfolio.returns()`` of :func:`simulate_mac_portfolio`
    applied independently to that asset's own price history (Section 3.1:
    "simulate a trading account... for each asset").
    """
    returns = {}
    for ticker in prices.columns:
        price = prices[ticker].dropna()
        if len(price) <= long_window:
            continue
        pf = simulate_mac_portfolio(price, short_window, long_window, init_cash=1.0)
        returns[ticker] = pf.returns()
    return pd.DataFrame(returns)


def buy_and_hold_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns of a buy-and-hold position, one column per asset."""
    returns = {}
    for ticker in prices.columns:
        price = prices[ticker].dropna()
        if len(price) < 2:
            continue
        pf = simulate_buy_and_hold(price, init_cash=1.0)
        returns[ticker] = pf.returns()
    return pd.DataFrame(returns)


def hybrid_portfolio_returns(asset_returns: pd.DataFrame) -> pd.Series:
    """Simple average across assets of their individually simulated returns.

    Implements Section 3.1 step 4: "Calculate the simple average of the 13
    individually simulated assets to represent the hybrid portfolio return."
    Assets without a return on a given day (not yet listed, or no trade
    signal reached) are excluded from that day's average rather than treated
    as a zero return.
    """
    return asset_returns.mean(axis=1, skipna=True)


def pooled_trade_pnls(
    prices: pd.DataFrame, short_window: int, long_window: int, init_cash: float
) -> dict[str, np.ndarray]:
    """Per-asset realized trade PnL (BRL) of the MAC system, one array per ticker.

    Used for the Profit Factor of "all operations" (Section 3.3, Table 04
    note): pooling every asset's completed round-trip trades together.
    """
    pnls = {}
    for ticker in prices.columns:
        price = prices[ticker].dropna()
        if len(price) <= long_window:
            continue
        pf = simulate_mac_portfolio(price, short_window, long_window, init_cash)
        pnls[ticker] = trade_pnls(pf)
    return pnls
