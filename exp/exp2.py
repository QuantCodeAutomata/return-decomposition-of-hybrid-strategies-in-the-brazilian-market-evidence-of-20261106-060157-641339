"""Experiment 2: Factor Decomposition and Jensen Alpha Generation (Section 3.4).

Decomposes the hybrid portfolio's monthly returns (SMA(9)/SMA(46) crossover
system, Section 3.2) using the CAPM, Fama-French three-factor, and Carhart
four-factor models, estimated by OLS with White (1980) heteroskedasticity-
robust standard errors, and reports the Jensen alpha and factor loadings.
"""

import json

import pandas as pd

from data.data_loader import load_experiment_universe
from src import config as cfg
from src.backtest import asset_daily_returns, hybrid_portfolio_returns
from src.factor_models import capm_model, carhart_4f_model, fama_french_3f_model, result_to_row
from src.metrics import to_monthly_returns, to_period_index

RESULTS_DIR = "results"


def hybrid_monthly_excess_returns(use_cache: bool = True) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Shared Rp monthly series and NEFIN factor panel aligned on calendar months.

    Reused by the sub-period analysis of Experiment 3 so that both
    experiments regress the identical baseline hybrid-portfolio return series
    against the identical factor data.
    """
    data = load_experiment_universe(
        cfg.TICKERS, cfg.IBOVESPA_TICKER, cfg.FACTOR_SAMPLE_START, cfg.FACTOR_SAMPLE_END,
        cfg.IBOVESPA_AVAILABLE_FROM, cfg.NEFIN_FACTORS_URL,
        ticker_suffix=cfg.DATALOADER_TICKER_SUFFIX, identity_date=cfg.IDENTITY_DATE,
        use_cache=use_cache,
    )
    prices, nefin = data["prices"], data["nefin"]
    mac_asset_returns = asset_daily_returns(prices, cfg.SMA_SHORT_BASELINE, cfg.SMA_LONG_BASELINE)
    hybrid_daily = hybrid_portfolio_returns(mac_asset_returns)
    hybrid_monthly = to_period_index(to_monthly_returns(hybrid_daily))

    nefin_period = nefin.copy()
    nefin_period.index = pd.PeriodIndex(nefin_period.index, freq="M")
    nefin_period = nefin_period[~nefin_period.index.duplicated()]

    panel = pd.concat([hybrid_monthly.rename("Rp"), nefin_period], axis=1, join="inner").dropna()
    panel["excess_return"] = panel["Rp"] - panel["Risk_Free"]
    return panel["excess_return"], panel, data


def run(use_cache: bool = True) -> dict:
    excess_return, panel, data = hybrid_monthly_excess_returns(use_cache)
    print(
        f"[exp2] hybrid monthly excess-return series: {len(excess_return)} observations, "
        f"{panel.index.min()}..{panel.index.max()}"
    )

    capm = capm_model(excess_return, panel["Rm_minus_Rf"], cfg.OLS_COV_TYPE)
    ff3 = fama_french_3f_model(excess_return, panel["Rm_minus_Rf"], panel["SMB"], panel["HML"], cfg.OLS_COV_TYPE)
    carhart4 = carhart_4f_model(
        excess_return, panel["Rm_minus_Rf"], panel["SMB"], panel["HML"], panel["WML"], cfg.OLS_COV_TYPE
    )

    table = pd.DataFrame([result_to_row(capm), result_to_row(ff3), result_to_row(carhart4)]).set_index("model")
    table.to_csv(f"{RESULTS_DIR}/exp2_factor_models.csv")

    log = {
        "n_obs": int(len(excess_return)),
        "date_range": [str(panel.index.min()), str(panel.index.max())],
        "price_failed_tickers": data["price_meta"]["failed"],
        "non_massive_source": data["price_meta"]["non_massive_source"],
    }
    with open(f"{RESULTS_DIR}/exp2_data_log.json", "w") as handle:
        json.dump(log, handle, indent=2, default=str)

    print(table)
    return {"table": table, "log": log}


if __name__ == "__main__":
    run()
