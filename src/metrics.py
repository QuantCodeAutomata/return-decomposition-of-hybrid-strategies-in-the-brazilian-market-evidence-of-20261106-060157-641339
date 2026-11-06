"""Descriptive risk-return metrics (Section 3.3).

# Custom implementation (paper Section 3.3: Sharpe ratio, Sortino ratio,
# Maximum Drawdown, Profit Factor, Total Net Profit). quantstats.stats was
# checked first (Context7 "ranaroussi/quantstats" docs): its ``sharpe`` and
# ``sortino`` only accept a single constant annual risk-free rate, but the
# paper's risk-free rate is the *monthly average SELIC rate*, a genuine time
# series that must be subtracted period-by-period before annualizing, so the
# formulas are implemented directly from their standard definitions
# (Sharpe, 1966; Sortino & van der Meer, 1991) instead of forcing a scalar rf
# through the library.
"""

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    """Largest percentage loss from peak to trough of the equity curve.

    Returns a positive fraction (e.g. 0.2549 for a 25.49% drawdown), matching
    the paper's "Maximum Drawdown (%)" metric (Section 3.3).
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(-drawdown.min())


def _align_rf(returns: pd.Series, rf: pd.Series | float) -> pd.Series:
    if np.isscalar(rf):
        return pd.Series(float(rf), index=returns.index)
    return rf.reindex(returns.index).ffill()


def sharpe_ratio(returns: pd.Series, rf: pd.Series | float, periods_per_year: int) -> float:
    """Annualized Sharpe ratio with a (possibly time-varying) risk-free rate.

    Sharpe (1966): mean(excess return) / std(excess return) * sqrt(periods).
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    excess = returns - _align_rf(returns, rf)
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf: pd.Series | float, periods_per_year: int) -> float:
    """Annualized Sortino ratio (Sortino & van der Meer, 1991).

    Downside deviation uses only the negative excess-return observations,
    squared and averaged over the full sample length (not just the count of
    negative observations), matching the standard academic definition.
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    excess = returns - _align_rf(returns, rf)
    downside = excess.copy()
    downside[downside > 0] = 0.0
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def trade_profit_factor(trade_pnls: np.ndarray) -> float:
    """Profit Factor from discrete round-trip trades (Section 3.3, Table 04).

    Ratio of total gross profit to total gross loss across all completed
    operations of the MAC system.
    """
    trade_pnls = np.asarray(trade_pnls, dtype=float)
    if trade_pnls.size == 0:
        return np.nan
    gross_profit = trade_pnls[trade_pnls > 0].sum()
    gross_loss = -trade_pnls[trade_pnls < 0].sum()
    if gross_loss == 0:
        return np.nan
    return float(gross_profit / gross_loss)


def period_profit_factor(returns: pd.Series) -> float:
    """Profit Factor for strategies without discrete trades (buy-and-hold,
    Ibovespa): ratio of the sum of positive period returns to the absolute
    sum of negative period returns (Section 3.3, Table 04 note; see the
    "Profit Factor" entry in ``src/config.py`` for the basis-of-computation
    assumption)."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()
    if gross_loss == 0:
        return np.nan
    return float(gross_profit / gross_loss)


def total_net_profit(returns: pd.Series, init_cash: float) -> float:
    """Final capital minus initial capital of a compounded return series."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    final_value = init_cash * (1.0 + returns).prod()
    return float(final_value - init_cash)


def average_per_asset_net_profit(asset_returns: pd.DataFrame, init_cash: float) -> float:
    """Cross-sectional average of the 13 individually simulated accounts' own
    Total Net Profit (Section 3.1 step 2: each asset is its own trading
    account starting at ``init_cash``; Section 3.3 reports the "average net
    profit" across these 13 accounts, not the profit of a single blended
    account). Used for the Hybrid Portfolio (MAC) and Buy-and-Hold rows of
    Table 04; the Ibovespa row has only one series and needs no averaging."""
    per_asset = asset_returns.apply(lambda col: total_net_profit(col, init_cash))
    return float(per_asset.mean())


def average_per_asset_max_drawdown(asset_returns: pd.DataFrame) -> float:
    """Cross-sectional average of the 13 individually simulated accounts' own
    Maximum Drawdown (Section 3.3: "average drawdown"; Section 3.5/Table 06
    reports the same cross-sectional average for the robustness tests). Used
    for the Hybrid Portfolio (MAC) and Buy-and-Hold rows of Table 04, paired
    with ``average_per_asset_net_profit``: both are averages across the 13
    independently simulated accounts, not metrics of a single blended series.
    A blended (simple-average-of-returns) equity curve diversifies away each
    asset's own idiosyncratic drawdown, which is why it is not used for this
    metric."""
    per_asset = asset_returns.apply(max_drawdown)
    return float(per_asset.mean())


def to_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """Compound daily simple returns into monthly simple returns.

    Used to bring the daily MAC/buy-and-hold/Ibovespa return series to the
    monthly frequency of the NEFIN risk-free rate and factor series before
    computing the Sharpe/Sortino ratios (Section 3.3) or the factor
    regressions (Section 3.4).
    """
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return daily_returns
    return (1.0 + daily_returns).resample("ME").prod() - 1.0


def per_asset_descriptive_table(
    asset_returns: pd.DataFrame,
    rf_monthly: pd.Series,
    init_cash: float,
    profit_factor_by_ticker: dict,
) -> pd.DataFrame:
    """Per-asset descriptive metrics table with an "Asset Average" row.

    Matches the paper's reporting convention (Section 3.3, Tables 02-04):
    Total Net Profit and Maximum Drawdown are computed on each asset's own
    daily equity curve; Sharpe and Sortino are computed on each asset's own
    monthly excess-return series (left at the monthly, non-annualized scale
    of a single observation, since the "Asset Average" is itself a
    cross-sectional average across the 13 individually simulated accounts,
    not a metric of one combined/rebalanced return series).
    """
    rows = {}
    for ticker in asset_returns.columns:
        daily = asset_returns[ticker].dropna()
        if daily.empty:
            continue
        monthly = to_period_index(to_monthly_returns(daily))
        paired_rf = rf_monthly.reindex(monthly.index).ffill()
        rows[ticker] = {
            "net_profit_brl": total_net_profit(daily, init_cash),
            "max_drawdown_pct": max_drawdown(daily) * 100,
            "sharpe_monthly": sharpe_ratio(monthly, paired_rf, 1),
            "sortino_monthly": sortino_ratio(monthly, paired_rf, 1),
            "profit_factor": profit_factor_by_ticker.get(ticker, np.nan),
        }
    table = pd.DataFrame(rows).T
    table.loc["Asset Average"] = table.mean(numeric_only=True)
    return table


def count_negative_assets(per_asset_table: pd.DataFrame) -> int:
    """Count of individually simulated assets with a negative Total Net Profit,
    excluding the summary "Asset Average" row (Section 4.4, Table 06)."""
    per_asset = per_asset_table.drop(index="Asset Average", errors="ignore")
    return int((per_asset["net_profit_brl"] < 0).sum())


def to_period_index(series: pd.Series) -> pd.Series:
    """Re-index a monthly series by a calendar-month :class:`pandas.Period`.

    Both the strategy's monthly returns (month-end timestamps from
    :func:`to_monthly_returns`) and the NEFIN factor series (first-of-month
    timestamps) are normalized this way before being merged on a common
    calendar month.
    """
    out = series.copy()
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out
