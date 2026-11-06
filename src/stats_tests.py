"""Jobson-Korkie (1981) test with the Memmel (2003) correction (Section 3.3).

# Custom implementation (paper Section 3.3: Jobson-Korkie test, Memmel
# correction). Context7 "scipy.stats"/"statsmodels" documentation covers
# generic z-tests but neither ships the Jobson-Korkie Sharpe-ratio-difference
# test, so its asymptotic variance formula is implemented directly from
# Jobson & Korkie (1981) Eq. 10 and the Memmel (2003) small-sample
# correction, Eq. 2.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class JobsonKorkieResult:
    """Result of the Jobson-Korkie/Memmel test of equal Sharpe ratios."""

    sharpe_a: float
    sharpe_b: float
    z_statistic: float
    p_value: float
    n_obs: int


def jobson_korkie_memmel_test(
    returns_a: pd.Series, returns_b: pd.Series
) -> JobsonKorkieResult:
    """Test H0: Sharpe(a) = Sharpe(b) using the same (unannualized) frequency.

    Jobson & Korkie (1981) derive the asymptotic variance of the difference
    of two sample Sharpe ratios computed on the same periods; Memmel (2003)
    corrects an error in their original formula. Both series must be the
    returns of the *same* sampling frequency (e.g. both monthly), aligned on
    a common date index, since the test statistic uses their sample
    covariance.
    """
    paired = pd.concat([returns_a, returns_b], axis=1, join="inner").dropna()
    paired.columns = ["a", "b"]
    n = len(paired)
    if n < 3:
        raise ValueError("Jobson-Korkie test requires at least 3 paired observations")

    mean_a, mean_b = paired["a"].mean(), paired["b"].mean()
    std_a, std_b = paired["a"].std(ddof=1), paired["b"].std(ddof=1)
    sigma_ab = paired["a"].cov(paired["b"])

    sharpe_a = mean_a / std_a
    sharpe_b = mean_b / std_b

    # Memmel (2003) correction of the Jobson-Korkie (1981) asymptotic variance:
    theta = (1.0 / n) * (
        2 * std_a ** 2 * std_b ** 2
        - 2 * std_a * std_b * sigma_ab
        + 0.5 * mean_a ** 2 * std_b ** 2
        + 0.5 * mean_b ** 2 * std_a ** 2
        - (mean_a * mean_b / (std_a * std_b)) * sigma_ab ** 2
    )
    se = np.sqrt(theta)
    z = (std_b * mean_a - std_a * mean_b) / se
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return JobsonKorkieResult(
        sharpe_a=float(sharpe_a),
        sharpe_b=float(sharpe_b),
        z_statistic=float(z),
        p_value=float(p_value),
        n_obs=n,
    )
