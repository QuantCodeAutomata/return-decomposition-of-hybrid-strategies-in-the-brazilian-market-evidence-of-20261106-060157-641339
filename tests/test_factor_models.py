"""Unit tests of the CAPM / Fama-French / Carhart regressions (Section 3.4)."""

import numpy as np
import pandas as pd
import pytest

from src.factor_models import capm_model, carhart_4f_model, fama_french_3f_model, result_to_row


def test_capm_recovers_known_beta_and_zero_alpha():
    rng = np.random.default_rng(1)
    n = 200
    mkt_rf = pd.Series(rng.normal(0.01, 0.05, n))
    true_beta = 0.7
    noise = rng.normal(0, 0.001, n)
    excess_return = true_beta * mkt_rf + noise  # zero true alpha by construction
    result = capm_model(excess_return, mkt_rf, "HC0")
    assert result.params["Rm_minus_Rf"] == pytest.approx(true_beta, abs=0.05)
    assert abs(result.alpha_monthly) < 0.01
    assert result.n_obs == n


def test_ff3_alpha_recovers_injected_alpha():
    rng = np.random.default_rng(2)
    n = 240
    mkt_rf = pd.Series(rng.normal(0.01, 0.05, n))
    smb = pd.Series(rng.normal(0.0, 0.03, n))
    hml = pd.Series(rng.normal(0.0, 0.03, n))
    true_alpha, true_betas = 0.01, (0.6, 0.3, -0.1)
    noise = rng.normal(0, 0.001, n)
    excess_return = (
        true_alpha + true_betas[0] * mkt_rf + true_betas[1] * smb + true_betas[2] * hml + noise
    )
    result = fama_french_3f_model(excess_return, mkt_rf, smb, hml, "HC0")
    assert result.alpha_monthly == pytest.approx(true_alpha, abs=0.003)
    assert result.params["SMB"] == pytest.approx(true_betas[1], abs=0.05)
    assert result.params["HML"] == pytest.approx(true_betas[2], abs=0.05)


def test_carhart_includes_wml_and_result_to_row_has_all_betas():
    rng = np.random.default_rng(3)
    n = 180
    factors = {name: pd.Series(rng.normal(0.0, 0.03, n)) for name in ["Rm_minus_Rf", "SMB", "HML", "WML"]}
    excess_return = pd.Series(rng.normal(0.005, 0.02, n))
    result = carhart_4f_model(
        excess_return, factors["Rm_minus_Rf"], factors["SMB"], factors["HML"], factors["WML"], "HC0"
    )
    row = result_to_row(result)
    for factor in ("Rm_minus_Rf", "SMB", "HML", "WML"):
        assert f"beta_{factor}" in row
        assert f"pvalue_{factor}" in row
    assert 0.0 <= row["pvalue_WML"] <= 1.0


def test_robust_se_differs_from_default_under_heteroskedasticity():
    """HC0 standard errors (White, 1980) must differ from homoskedastic OLS SEs
    when the regression residuals are heteroskedastic."""
    rng = np.random.default_rng(4)
    n = 300
    mkt_rf = pd.Series(rng.normal(0.01, 0.05, n))
    hetero_noise = rng.normal(0, 1, n) * (0.001 + 0.01 * np.abs(mkt_rf))
    excess_return = 0.5 * mkt_rf + hetero_noise
    robust = capm_model(excess_return, mkt_rf, "HC0")

    import statsmodels.api as sm
    x_const = sm.add_constant(mkt_rf.to_frame("Rm_minus_Rf"))
    classical = sm.OLS(excess_return, x_const).fit()
    assert robust.bse["Rm_minus_Rf"] != pytest.approx(classical.bse["Rm_minus_Rf"], rel=1e-6)
