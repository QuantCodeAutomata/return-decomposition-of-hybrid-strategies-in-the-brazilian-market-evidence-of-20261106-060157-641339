"""CAPM, Fama-French three-factor and Carhart four-factor regressions
(Section 3.4).

# statsmodels OLS with White (1980) HC standard errors: Context7
# "statsmodels" documentation confirms ``sm.OLS(...).fit(cov_type="HC0")``
# implements heteroskedasticity-robust (White) standard errors directly, so
# no custom sandwich-estimator code is written.
"""

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass
class FactorModelResult:
    """OLS estimates of one factor-model regression (Section 3.4)."""

    model_name: str
    n_obs: int
    params: pd.Series
    pvalues: pd.Series
    tvalues: pd.Series
    bse: pd.Series
    r_squared: float
    alpha_monthly: float
    alpha_pvalue: float


def _fit_ols(y: pd.Series, x: pd.DataFrame, model_name: str, cov_type: str) -> FactorModelResult:
    x_const = sm.add_constant(x, has_constant="add")
    fit = sm.OLS(y, x_const, missing="drop").fit(cov_type=cov_type)
    return FactorModelResult(
        model_name=model_name,
        n_obs=int(fit.nobs),
        params=fit.params,
        pvalues=fit.pvalues,
        tvalues=fit.tvalues,
        bse=fit.bse,
        r_squared=float(fit.rsquared),
        alpha_monthly=float(fit.params["const"]),
        alpha_pvalue=float(fit.pvalues["const"]),
    )


def capm_model(excess_return: pd.Series, mkt_rf: pd.Series, cov_type: str) -> FactorModelResult:
    """CAPM: Rp - Rf = alpha + beta_m * (Rm - Rf) + eps (Section 3.4, Eq. 1)."""
    x = pd.DataFrame({"Rm_minus_Rf": mkt_rf})
    return _fit_ols(excess_return, x, "CAPM", cov_type)


def fama_french_3f_model(
    excess_return: pd.Series, mkt_rf: pd.Series, smb: pd.Series, hml: pd.Series, cov_type: str
) -> FactorModelResult:
    """Fama-French three-factor model (Section 3.4, Eq. 2)."""
    x = pd.DataFrame({"Rm_minus_Rf": mkt_rf, "SMB": smb, "HML": hml})
    return _fit_ols(excess_return, x, "Fama-French 3-Factor", cov_type)


def carhart_4f_model(
    excess_return: pd.Series,
    mkt_rf: pd.Series,
    smb: pd.Series,
    hml: pd.Series,
    wml: pd.Series,
    cov_type: str,
) -> FactorModelResult:
    """Carhart four-factor model (Section 3.4, Eq. 3)."""
    x = pd.DataFrame({"Rm_minus_Rf": mkt_rf, "SMB": smb, "HML": hml, "WML": wml})
    return _fit_ols(excess_return, x, "Carhart 4-Factor", cov_type)


def result_to_row(result: FactorModelResult) -> dict:
    """Flatten a :class:`FactorModelResult` into one results-table row."""
    row = {
        "model": result.model_name,
        "n_obs": result.n_obs,
        "r_squared": result.r_squared,
        "alpha_monthly_pct": result.alpha_monthly * 100,
        "alpha_pvalue": result.alpha_pvalue,
    }
    for factor in ("Rm_minus_Rf", "SMB", "HML", "WML"):
        if factor in result.params.index:
            row[f"beta_{factor}"] = result.params[factor]
            row[f"pvalue_{factor}"] = result.pvalues[factor]
    return row
