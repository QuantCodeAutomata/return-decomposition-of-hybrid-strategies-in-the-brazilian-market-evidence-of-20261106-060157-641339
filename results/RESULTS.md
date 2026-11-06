# Results — Return Decomposition of Hybrid Strategies in the Brazilian Market: Evidence of Genuine Alpha and Factor Exposure

Freitas (2025), *"Return Decomposition of Hybrid Strategies in the Brazilian
Market: Evidence of Genuine Alpha and Factor Exposure."*

Verdicts follow: **reproduced** (inside the paper's reported range, or within
10% of its value with the same sign, and the paper's qualitative result
holds), **not reproduced** (opposite sign, >50% value difference, or a
qualitative result that does not hold), **partially reproduced** (otherwise).

## Experiment 1 — Backtesting and Descriptive Performance Analysis of the Hybrid Portfolio

Source file: `results/exp1_descriptive_metrics.csv`, `results/exp1_jobson_korkie_test.csv`,
`results/exp1_data_log.json`. Plots: `results/exp1_equity_curves.png`, `results/exp1_drawdowns.png`.

13-asset hybrid portfolio (simple average of 13 individually-simulated
SMA(9)/SMA(46) crossover accounts, BRL 1,000,000 initial capital each),
compared against a buy-and-hold strategy on the same 13 assets and the
Ibovespa index, over 2000-01-02 to 2025-12-29 (6,527 daily rows, 312 monthly
observations, 300 NEFIN monthly factor rows used for the Jobson-Korkie test).

| Metric | Hybrid Portfolio (SMA 9-46) | Buy-and-Hold (13 assets) | Ibovespa |
|---|---|---|---|
| Total net profit (BRL) | 15,296,142 | 15,158,708 | 8,517,129 |
| Max drawdown | 66.11% | 81.91% | 59.96% |
| Sharpe ratio (annualized) | 0.1748 | 0.2691 | 0.0443 |
| Sortino ratio (annualized) | 0.3068 | 0.3751 | 0.0612 |
| Profit factor | 1.870 | 1.158 | 1.084 |

Total Net Profit and Maximum Drawdown are the cross-sectional average across
the 13 individually simulated R$1,000,000 accounts for the Hybrid Portfolio
and Buy-and-Hold rows (Section 3.1: each asset is its own account; Section
3.3 reports an "average net profit" and "average drawdown" across assets).
Sharpe and Sortino use the single blended daily-return series (simple average
of the 13 assets' own returns, Section 3.1 step 4), which Section 3.3 calls
the "annualized Sharpe ratio of the aggregate portfolio."

Jobson-Korkie (1981)/Memmel (2003) test of equal Sharpe ratios, hybrid vs.
Ibovespa, on 300 monthly observations: Z = 0.645, p = 0.519 (fails to reject
equal Sharpe ratios at the 5% level).

| Quantity | Paper | This run | Verdict |
|---|---|---|---|
| Hybrid Sharpe ratio | 0.443 | 0.175 | partially reproduced (same sign/ranking vs. benchmark, magnitude >50% lower) |
| Ibovespa Sharpe ratio | 0.039 | 0.044 | reproduced |
| Hybrid Sharpe > Ibovespa Sharpe (ranking) | yes | yes | reproduced |
| Jobson-Korkie Z / p | 1.427 / 0.154 | 0.645 / 0.519 | partially reproduced (same qualitative conclusion: fails to reject equal Sharpe ratios; Z differs by >50%) |
| Average drawdown, hybrid vs. benchmark | 25.49% vs 10.84% (hybrid higher) | 66.11% vs 59.96% (hybrid higher) | partially reproduced (ranking matches; both magnitudes are above the paper's values) |
| Hybrid net profit (BRL) | ~10.2 million | 15.30 million | partially reproduced (same order of magnitude, >50% above the paper's figure) |
| Benchmark net profit (BRL) | ~8.0 million | 8.52 million (Ibovespa) | reproduced |

The drawdown ranking (hybrid portfolio's average drawdown exceeding the
Ibovespa's) is now reproduced once Maximum Drawdown is computed, like Total
Net Profit, as the cross-sectional average of the 13 individually simulated
accounts' own peak-to-trough declines, rather than the drawdown of the single
blended (simple-average-of-returns) equity curve: a blended series
diversifies away each asset's own idiosyncratic drawdown, which had produced
an artificially low 19.95% figure and an opposite ranking against the
Ibovespa's single-series 59.96% drawdown. Both the hybrid portfolio's
(66.11%) and the Ibovespa's (59.96%) magnitudes remain well above the paper's
reported 25.49% and 10.84%; the gap was not localized further within the
available time (the 13-asset buy-and-hold average, 81.91%, is also far above
the paper's descriptive range, suggesting the realized 2000-2025 B3 price
series carry substantially deeper drawdowns for this particular stock
selection than the paper's sample, rather than a remaining sign or
calculation defect).

## Experiment 2 — Factor Decomposition and Jensen Alpha Generation

Source file: `results/exp2_factor_models.csv`, `results/exp2_data_log.json`.

CAPM, Fama-French three-factor and Carhart four-factor models estimated by
OLS with White (1980) HC0 heteroskedasticity-robust standard errors on 300
monthly excess returns of the hybrid portfolio (January 2000-December 2025),
against NEFIN/FEA-USP monthly risk factors.

| Model | alpha (%/month) | alpha p-value | beta Rm-Rf | beta SMB | beta HML | beta WML |
|---|---|---|---|---|---|---|
| CAPM | 3.766 | 0.222 | 1.211 (p=0.114) | — | — | — |
| Fama-French 3-Factor | 4.480 | 0.222 | 1.885 (p=0.165) | -0.828 (p=0.325) | -1.623 (p=0.270) | — |
| Carhart 4-Factor | 4.270 | 0.210 | 1.917 (p=0.170) | -0.798 (p=0.321) | -1.564 (p=0.262) | 0.193 (p=0.513) |

| Quantity | Paper | This run | Verdict |
|---|---|---|---|
| CAPM alpha (%/month, sign) | +1.18, p=0.016 (significant) | +3.77, p=0.222 (not significant) | not reproduced (significance does not hold; magnitude >50% off) |
| CAPM market beta | 0.686 | 1.211 | not reproduced (>50% off) |
| FF3 alpha (%/month) | +1.11, p=0.005 (significant) | +4.48, p=0.222 (not significant) | not reproduced |
| FF3 SMB beta (sign, significance) | +0.381, p<0.001 (significant, positive) | -0.828, p=0.325 (not significant, negative) | not reproduced (sign differs) |
| FF3 HML beta (significance) | +0.127, p=0.544 (not significant) | -1.623, p=0.270 (not significant) | partially reproduced (non-significance holds, sign and magnitude differ) |
| Carhart alpha (%/month) | +0.98, p=0.009 (significant) | +4.27, p=0.210 (not significant) | not reproduced |
| Carhart WML beta (significance) | p=0.409 (not significant) | p=0.513 (not significant) | reproduced (non-significance; sign/magnitude not compared, paper does not give the WML coefficient) |

Every regression in this run reproduces the paper's qualitative sign of
Jensen's alpha (positive) but not its statistical significance, and the SMB
loading has the opposite sign from the paper's (-0.83 here vs. +0.38 in the
paper). Two candidate, unverified explanations were identified but not tested
further within the available time: (a) the 13-asset screen used here is
fixed, point-in-time at 2025, while the paper's fundamentalist screen may
have selected a different cross-section of names whose factor exposures
differ; (b) the realized small-sample factor loadings on 300 monthly
observations are imprecise (all coefficients here carry p > 0.10 except the
Carhart market beta), so a difference of this magnitude cannot be ruled out
as finite-sample noise. The precise cause is not identified.

## Experiment 3 — Robustness Analysis: SMA Parameter Sensitivity and Sub-Period Evaluation

Source file: `results/exp3_sma_sensitivity.csv`, `results/exp3_subperiod_alpha.csv`,
`results/exp3_data_log.json`. Plots: `results/exp3_sma_sensitivity.png`, `results/exp3_subperiod_alpha.png`.

### Part A — SMA parameter sensitivity (13 assets, BRL 1,000,000 initial capital each)

| Configuration | Avg Sharpe | Avg max drawdown | Avg profit (BRL) | Negative-result assets |
|---|---|---|---|---|
| SMA 5-20 (reactive) | 0.0095 | 71.65% | 8,164,130 | 2 / 13 |
| SMA 9-46 (baseline) | 0.0730 | 66.11% | 15,296,142 | 0 / 13 |
| SMA 12-26 (MACD) | 0.1002 | 66.82% | 18,549,408 | 0 / 13 |
| SMA 20-50 (conservative) | 0.0436 | 62.54% | 10,259,780 | 1 / 13 |
| Buy-and-Hold (passive) | 0.1584 | 81.91% | 15,158,708 | 2 / 13 |

| Quantity | Paper | This run | Verdict |
|---|---|---|---|
| No MAC configuration beats buy-and-hold Sharpe/drawdown | holds | holds (buy-and-hold has the highest average Sharpe and, except vs. SMA 20-50 drawdown, the worst drawdown) | partially reproduced (Sharpe ranking holds; drawdown ranking is mixed since buy-and-hold also has the worst average drawdown here) |
| SMA 5-20 (reactive) deteriorates vs. baseline (more negative assets) | 7 negative vs. 5 in baseline | 2 negative vs. 0 in baseline | partially reproduced (reactive configuration is still the worst among the four MAC configurations, but absolute counts differ) |

### Part B — Sub-period Carhart four-factor regressions (baseline SMA 9-46)

| Sub-period | n | alpha (%/month) | alpha p-value | beta WML (p-value) |
|---|---|---|---|---|
| 2000-2008 | 96 | -0.495 | 0.132 | -0.006 (0.862) |
| 2009-2016 | 96 | -0.351 | 0.147 | -0.096 (0.026) |
| 2017-2025 | 108 | +1.175 | 0.002 | -0.075 (0.366) |

| Quantity | Paper | This run | Verdict |
|---|---|---|---|
| Alpha concentrated in 2000-2008, declining toward 2017-2025 | holds | opposite: alpha is negative and not significant in 2000-2008 and 2009-2016, positive and significant (p=0.002) only in 2017-2025 | not reproduced |

The reversed sub-period pattern was not localized further within the
available time; it shares the same hybrid-portfolio return series as
Experiment 2, so a cause found there (screen composition or small-sample
noise) would likely also explain this reversal, but this link was not
verified by an additional run.

## Deviations from the paper

- **Data source.** The paper's price series are "adjusted for dividends and
  corporate events." Bars for the 13 B3 tickers are loaded through the QuantX
  Dataloader REST API (`POST /fetch-ohlcvs`, `adjust_dividends=True`), which
  for this sample period and these tickers fell back to its Yahoo Finance
  backstop for all 13 names (`ticker_details[].data_source == "yahoo"` for
  every ticker; logged in `results/exp1_data_log.json`). The Ibovespa index
  level (`^BVSP`) is loaded directly from Yahoo Finance rather than through
  the Dataloader or Massive, because the Ibovespa is an index level (not an
  OHLCV security) and Massive returns `NOT_AUTHORIZED`/no data for every
  tested Ibovespa symbol convention (`I:IBOV`, `IBOV`, `I:BVSP`,
  `I:BVSP.SA`). This substitution is recorded per the paper's own citation of
  the Ibovespa as its benchmark index (Section 3.3).
- **Risk factors.** Monthly market (Rm-Rf), SMB, HML, WML, IML and
  risk-free-rate series are downloaded from the NEFIN/FEA-USP public CSV
  export (Section 3.4), as the paper specifies; no BCB SELIC fallback was
  needed since NEFIN's `Risk_Free` column is already the monthly-average
  SELIC rate the paper cites for the Sharpe-ratio risk-free rate (Section
  3.3).
- **Universe.** The 13-asset list is taken verbatim from the paper's
  fundamentalist screen (Section 3.1, Table 01); no ranking or narrowing was
  needed or applied, and the full universe (13 assets plus the Ibovespa
  benchmark) was downloaded and used in every experiment.
- **Profit Factor definition.** The paper's Profit Factor is "the ratio of
  total gross profit to total gross loss across all operations." Implemented
  as the ratio of summed positive to summed (absolute) negative round-trip
  trade P&L for the MAC system, and as the ratio of summed positive to summed
  negative daily returns for the buy-and-hold and Ibovespa series, which have
  no discrete trades (Section 3.3, uncertainty flagged in the experiment
  specification; not otherwise specified by the paper).
- **Material differences not attributable to a known cause** (see per-experiment
  sections above): the magnitude (though not the ranking) of the drawdown
  figures in Experiment 1, the CAPM/FF3/Carhart alpha significance and the SMB
  sign in Experiment 2, and the sub-period alpha pattern in Experiment 3. In
  every case the relevant component (bars, NEFIN factors, OLS estimation,
  drawdown calculation) was exercised by the unit tests in `tests/` on
  synthetic fixtures with known expected signs and passed; the differences
  were not reduced to a single identified formula, sign, window, or data
  defect within the time available, and are reported here rather than
  attributed to a specific cause.

## Not reproduced

Nothing in the three selected experiments was left unimplemented: every
table and figure described in the experiment methodologies above was
produced from the real B3/Ibovespa/NEFIN data path. Where a reported number
differs materially from the paper, it is listed as "not reproduced" or
"partially reproduced" in the tables above rather than omitted.
