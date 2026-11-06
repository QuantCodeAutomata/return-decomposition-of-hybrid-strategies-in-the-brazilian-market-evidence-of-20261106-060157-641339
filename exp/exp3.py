"""Experiment 3: Robustness Analysis: SMA Parameter Sensitivity and
Sub-Period Evaluation (Section 3.5).

Part A tests the baseline SMA(9)/SMA(46) system against three alternative
crossover configurations and reports the average Sharpe ratio, average
maximum drawdown, average profit and the count of assets with negative
results, for each configuration (data-snooping robustness check).

Part B segments the baseline hybrid-portfolio monthly return series into
three sub-periods (2000-2008, 2009-2016, 2017-2025) and re-estimates the
Carhart four-factor model (Section 3.4) within each one, to evaluate the
Adaptive Markets Hypothesis (Section 3.5).
"""

import json

import pandas as pd

from data.data_loader import load_experiment_universe
from exp.exp2 import hybrid_monthly_excess_returns
from src import config as cfg
from src.backtest import asset_daily_returns, buy_and_hold_daily_returns
from src.factor_models import carhart_4f_model, result_to_row
from src.metrics import max_drawdown, sharpe_ratio, to_monthly_returns, to_period_index, total_net_profit
from src.plotting import plot_sma_sensitivity, plot_subperiod_alpha

RESULTS_DIR = "results"


def _monthly_rf(nefin: pd.DataFrame) -> pd.Series:
    rf = to_period_index(nefin["Risk_Free"])
    return rf[~rf.index.duplicated()]


def sma_sensitivity_table(data: dict, use_cache: bool = True) -> pd.DataFrame:
    """Part A (Section 3.5): average Sharpe/drawdown/profit per SMA config.

    Each asset is simulated independently under every configuration listed
    in ``cfg.SMA_CONFIGS``; the table reports, per configuration, the cross-
    sectional average Sharpe ratio, average maximum drawdown, average net
    profit (BRL) and the number of assets ending with a negative net profit.
    """
    prices, nefin = data["prices"], data["nefin"]
    rf_monthly = _monthly_rf(nefin)
    rows = []
    for short_w, long_w, label in cfg.SMA_CONFIGS:
        asset_returns = asset_daily_returns(prices, short_w, long_w)
        sharpes, drawdowns, profits, negative = [], [], [], 0
        for ticker in asset_returns.columns:
            daily = asset_returns[ticker].dropna()
            if daily.empty:
                continue
            monthly = to_period_index(to_monthly_returns(daily))
            paired_rf = rf_monthly.reindex(monthly.index).ffill()
            sharpes.append(sharpe_ratio(monthly, paired_rf, cfg.MONTHS_PER_YEAR))
            drawdowns.append(max_drawdown(daily))
            profit = total_net_profit(daily, cfg.INITIAL_CAPITAL)
            profits.append(profit)
            if profit < 0:
                negative += 1
        rows.append({
            "config": label,
            "short_window": short_w,
            "long_window": long_w,
            "avg_sharpe": pd.Series(sharpes).mean(),
            "avg_max_drawdown": pd.Series(drawdowns).mean(),
            "avg_profit_brl": pd.Series(profits).mean(),
            "n_negative_assets": negative,
            "n_assets": len(profits),
        })
    return pd.DataFrame(rows)


def buy_and_hold_benchmark_row(data: dict) -> dict:
    """Passive buy-and-hold row added to the sensitivity table for comparison
    (Section 3.5: "none of the configurations should outperform buy-and-hold")."""
    prices, nefin = data["prices"], data["nefin"]
    rf_monthly = _monthly_rf(nefin)
    bh_returns = buy_and_hold_daily_returns(prices)
    sharpes, drawdowns, profits, negative = [], [], [], 0
    for ticker in bh_returns.columns:
        daily = bh_returns[ticker].dropna()
        if daily.empty:
            continue
        monthly = to_period_index(to_monthly_returns(daily))
        paired_rf = rf_monthly.reindex(monthly.index).ffill()
        sharpes.append(sharpe_ratio(monthly, paired_rf, cfg.MONTHS_PER_YEAR))
        drawdowns.append(max_drawdown(daily))
        profit = total_net_profit(daily, cfg.INITIAL_CAPITAL)
        profits.append(profit)
        if profit < 0:
            negative += 1
    return {
        "config": "Buy-and-Hold (passive)",
        "short_window": None,
        "long_window": None,
        "avg_sharpe": pd.Series(sharpes).mean(),
        "avg_max_drawdown": pd.Series(drawdowns).mean(),
        "avg_profit_brl": pd.Series(profits).mean(),
        "n_negative_assets": negative,
        "n_assets": len(profits),
    }


def subperiod_alpha_table(use_cache: bool = True) -> pd.DataFrame:
    """Part B (Section 3.5): Carhart four-factor alpha within each sub-period.

    Reuses the single baseline hybrid-portfolio monthly excess-return series
    and NEFIN factor panel built by :func:`exp.exp2.hybrid_monthly_excess_returns`,
    sliced by the calendar sub-periods of ``cfg.SUB_PERIODS``.
    """
    excess_return, panel, _ = hybrid_monthly_excess_returns(use_cache)
    rows = []
    for start, end, label in cfg.SUB_PERIODS:
        start_p, end_p = pd.Period(start, freq="M"), pd.Period(end, freq="M")
        mask = (panel.index >= start_p) & (panel.index <= end_p)
        sub_excess = excess_return[mask]
        sub_panel = panel.loc[mask]
        if len(sub_excess) < 6:
            continue
        result = carhart_4f_model(
            sub_excess, sub_panel["Rm_minus_Rf"], sub_panel["SMB"], sub_panel["HML"],
            sub_panel["WML"], cfg.OLS_COV_TYPE,
        )
        row = result_to_row(result)
        row["sub_period"] = label
        rows.append(row)
    return pd.DataFrame(rows).set_index("sub_period")


def run(use_cache: bool = True) -> dict:
    data = load_experiment_universe(
        cfg.TICKERS, cfg.IBOVESPA_TICKER, cfg.START_DATE, cfg.END_DATE,
        cfg.IBOVESPA_AVAILABLE_FROM, cfg.NEFIN_FACTORS_URL,
        ticker_suffix=cfg.DATALOADER_TICKER_SUFFIX, identity_date=cfg.IDENTITY_DATE,
        use_cache=use_cache,
    )
    print(
        f"[exp3] prices: {data['prices'].shape[0]} rows x {data['prices'].shape[1]} tickers, "
        f"{data['prices'].index.min().date()}..{data['prices'].index.max().date()}"
    )

    sensitivity = sma_sensitivity_table(data, use_cache)
    sensitivity = pd.concat([sensitivity, pd.DataFrame([buy_and_hold_benchmark_row(data)])], ignore_index=True)
    sensitivity.to_csv(f"{RESULTS_DIR}/exp3_sma_sensitivity.csv", index=False)
    plot_sma_sensitivity(sensitivity[sensitivity["short_window"].notna()], f"{RESULTS_DIR}/exp3_sma_sensitivity.png")

    subperiod_alphas = subperiod_alpha_table(use_cache)
    subperiod_alphas.to_csv(f"{RESULTS_DIR}/exp3_subperiod_alpha.csv")
    plot_subperiod_alpha(subperiod_alphas.reset_index(), f"{RESULTS_DIR}/exp3_subperiod_alpha.png")

    log = {
        "sma_configs": cfg.SMA_CONFIGS,
        "sub_periods": cfg.SUB_PERIODS,
        "price_failed_tickers": data["price_meta"]["failed"],
    }
    with open(f"{RESULTS_DIR}/exp3_data_log.json", "w") as handle:
        json.dump(log, handle, indent=2, default=str)

    print(sensitivity)
    print(subperiod_alphas)
    return {"sensitivity": sensitivity, "subperiod_alphas": subperiod_alphas, "log": log}


if __name__ == "__main__":
    run()
