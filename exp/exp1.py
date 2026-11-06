"""Experiment 1: Backtesting and Descriptive Performance Analysis of the
Hybrid Portfolio (Sections 3.1-3.3).

Builds the 13-asset hybrid portfolio, applies the SMA(9)/SMA(46) moving
average crossover (MAC) system, and compares its descriptive risk-return
metrics against a buy-and-hold strategy on the same assets and the Ibovespa
benchmark. Also runs the Jobson-Korkie (1981)/Memmel (2003) test of equal
Sharpe ratios between the hybrid portfolio and the Ibovespa (Section 3.3).
"""

import json

import numpy as np
import pandas as pd

from data.data_loader import load_experiment_universe
from src import config as cfg
from src.backtest import (
    asset_daily_returns,
    buy_and_hold_daily_returns,
    hybrid_portfolio_returns,
    pooled_trade_pnls,
)
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
)
from src.plotting import plot_drawdowns, plot_equity_curves
from src.stats_tests import jobson_korkie_memmel_test

RESULTS_DIR = "results"


def _monthly_rf(nefin: pd.DataFrame) -> pd.Series:
    rf = to_period_index(nefin["Risk_Free"])
    return rf[~rf.index.duplicated()]


def _descriptive_row(
    label: str,
    daily_returns: pd.Series,
    monthly_returns: pd.Series,
    rf_monthly: pd.Series,
    profit_factor: float,
    net_profit_brl: float,
    max_drawdown_pct: float,
) -> dict:
    """One row of Table 04-style descriptive metrics (Section 3.3).

    Total Net Profit and Maximum Drawdown are passed in explicitly: for the
    Hybrid Portfolio and Buy-and-Hold rows they are the cross-sectional
    average across the 13 individually simulated R$1,000,000 accounts
    (Section 3.1 step 2; Section 3.3 reports an "average net profit" and
    "average drawdown" across assets, the same cross-sectional averaging
    Section 3.5/Table 06 applies to the robustness tests). Sharpe and Sortino
    instead use the single blended daily-return series Rp (Section 3.1 step
    4: simple average across assets' own returns; Section 3.3 explicitly
    calls this the "annualized Sharpe ratio of the aggregate portfolio").
    """
    paired_rf = rf_monthly.reindex(monthly_returns.index).ffill()
    return {
        "label": label,
        "total_net_profit_brl": net_profit_brl,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio(monthly_returns, paired_rf, cfg.MONTHS_PER_YEAR),
        "sortino_ratio": sortino_ratio(monthly_returns, paired_rf, cfg.MONTHS_PER_YEAR),
        "profit_factor": profit_factor,
        "n_daily_obs": int(daily_returns.dropna().shape[0]),
        "n_monthly_obs": int(monthly_returns.dropna().shape[0]),
    }


def run(use_cache: bool = True) -> dict:
    data = load_experiment_universe(
        cfg.TICKERS, cfg.IBOVESPA_TICKER, cfg.START_DATE, cfg.END_DATE,
        cfg.IBOVESPA_AVAILABLE_FROM, cfg.NEFIN_FACTORS_URL,
        ticker_suffix=cfg.DATALOADER_TICKER_SUFFIX, identity_date=cfg.IDENTITY_DATE,
        use_cache=use_cache,
    )
    prices, price_meta = data["prices"], data["price_meta"]
    ibov_prices, ibov_meta = data["ibov_prices"], data["ibov_meta"]
    nefin = data["nefin"]
    rf_monthly = _monthly_rf(nefin)

    print(
        f"[exp1] prices: {prices.shape[0]} rows x {prices.shape[1]} tickers, "
        f"{prices.index.min().date()}..{prices.index.max().date()}; "
        f"failed={price_meta['failed']}; non_massive_source={price_meta['non_massive_source']}"
    )
    print(
        f"[exp1] Ibovespa (substitute {cfg.IBOVESPA_TICKER}): {ibov_prices.shape[0]} rows, "
        f"{ibov_prices.index.min().date()}..{ibov_prices.index.max().date()}; "
        f"non_massive_source={ibov_meta['non_massive_source']}"
    )
    print(f"[exp1] NEFIN factors: {nefin.shape[0]} monthly rows, {nefin.index.min()}..{nefin.index.max()}")

    mac_asset_returns = asset_daily_returns(prices, cfg.SMA_SHORT_BASELINE, cfg.SMA_LONG_BASELINE)
    bh_asset_returns = buy_and_hold_daily_returns(prices)

    hybrid_mac_daily = hybrid_portfolio_returns(mac_asset_returns)
    hybrid_bh_daily = hybrid_portfolio_returns(bh_asset_returns)
    ibov_daily = ibov_prices[cfg.IBOVESPA_TICKER].pct_change().dropna()

    hybrid_mac_monthly = to_period_index(to_monthly_returns(hybrid_mac_daily))
    hybrid_bh_monthly = to_period_index(to_monthly_returns(hybrid_bh_daily))
    ibov_monthly = to_period_index(to_monthly_returns(ibov_daily))

    # Profit Factor: pooled trade-by-trade PnL across all 13 assets for the
    # MAC system ("all operations", Section 3.3, Table 04 note); period-based
    # for the buy-and-hold and Ibovespa series, which have no discrete trades.
    pooled_pnls = pooled_trade_pnls(prices, cfg.SMA_SHORT_BASELINE, cfg.SMA_LONG_BASELINE, cfg.INITIAL_CAPITAL)
    from src.metrics import trade_profit_factor
    all_pnls = np.concatenate(list(pooled_pnls.values())) if pooled_pnls else np.array([])
    mac_profit_factor = trade_profit_factor(all_pnls)
    bh_profit_factor = period_profit_factor(hybrid_bh_daily)
    ibov_profit_factor = period_profit_factor(ibov_daily)

    mac_net_profit = average_per_asset_net_profit(mac_asset_returns, cfg.INITIAL_CAPITAL)
    bh_net_profit = average_per_asset_net_profit(bh_asset_returns, cfg.INITIAL_CAPITAL)
    ibov_net_profit = total_net_profit(ibov_daily, cfg.INITIAL_CAPITAL)

    mac_avg_drawdown = average_per_asset_max_drawdown(mac_asset_returns) * 100
    bh_avg_drawdown = average_per_asset_max_drawdown(bh_asset_returns) * 100
    ibov_drawdown = max_drawdown(ibov_daily) * 100

    rows = [
        _descriptive_row("Hybrid Portfolio (SMA 9-46)", hybrid_mac_daily, hybrid_mac_monthly, rf_monthly, mac_profit_factor, mac_net_profit, mac_avg_drawdown),
        _descriptive_row("Buy-and-Hold (13 assets)", hybrid_bh_daily, hybrid_bh_monthly, rf_monthly, bh_profit_factor, bh_net_profit, bh_avg_drawdown),
        _descriptive_row(f"Ibovespa ({cfg.IBOVESPA_TICKER})", ibov_daily, ibov_monthly, rf_monthly, ibov_profit_factor, ibov_net_profit, ibov_drawdown),
    ]
    table = pd.DataFrame(rows).set_index("label")
    table.to_csv(f"{RESULTS_DIR}/exp1_descriptive_metrics.csv")

    # The paper's Sharpe ratio is defined on excess (risk-free-subtracted)
    # returns (Section 3.3), so the Jobson-Korkie/Memmel test of equal Sharpe
    # ratios (Section 3.3) is run on the same excess-return series, paired on
    # the common calendar months.
    hybrid_excess = hybrid_mac_monthly - rf_monthly.reindex(hybrid_mac_monthly.index).ffill()
    ibov_excess = ibov_monthly - rf_monthly.reindex(ibov_monthly.index).ffill()
    jk = jobson_korkie_memmel_test(hybrid_excess, ibov_excess)
    jk_row = {
        "sharpe_hybrid_monthly": jk.sharpe_a,
        "sharpe_ibovespa_monthly": jk.sharpe_b,
        "sharpe_hybrid_annualized": jk.sharpe_a * np.sqrt(cfg.MONTHS_PER_YEAR),
        "sharpe_ibovespa_annualized": jk.sharpe_b * np.sqrt(cfg.MONTHS_PER_YEAR),
        "z_statistic": jk.z_statistic,
        "p_value": jk.p_value,
        "n_obs": jk.n_obs,
        "significance_level": cfg.JK_CONFIDENCE_LEVEL,
        "reject_null_equal_sharpe": bool(jk.p_value < cfg.JK_CONFIDENCE_LEVEL),
    }
    pd.Series(jk_row).to_csv(f"{RESULTS_DIR}/exp1_jobson_korkie_test.csv")

    plot_equity_curves(
        {"Hybrid Portfolio": hybrid_mac_daily, "Buy-and-Hold": hybrid_bh_daily, "Ibovespa": ibov_daily},
        cfg.INITIAL_CAPITAL, f"{RESULTS_DIR}/exp1_equity_curves.png",
    )
    plot_drawdowns(
        {"Hybrid Portfolio": hybrid_mac_daily, "Buy-and-Hold": hybrid_bh_daily, "Ibovespa": ibov_daily},
        f"{RESULTS_DIR}/exp1_drawdowns.png",
    )

    log = {
        "price_rows": int(prices.shape[0]),
        "price_tickers": int(prices.shape[1]),
        "date_range": [str(prices.index.min().date()), str(prices.index.max().date())],
        "failed_tickers": price_meta["failed"],
        "non_massive_source": price_meta["non_massive_source"],
        "ibovespa_non_massive_source": ibov_meta["non_massive_source"],
        "nefin_rows": int(nefin.shape[0]),
    }
    with open(f"{RESULTS_DIR}/exp1_data_log.json", "w") as handle:
        json.dump(log, handle, indent=2, default=str)

    print(table)
    print(jk_row)
    return {"table": table, "jobson_korkie": jk_row, "log": log}


if __name__ == "__main__":
    run()
