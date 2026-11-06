"""Plots for the descriptive, factor-decomposition and robustness experiments.

# matplotlib: standard plotting library used throughout (no specialized
# quantitative-plotting library is required for these static figures).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(returns_by_label: dict[str, pd.Series], init_cash: float, path: str) -> None:
    """Equity curves of several return series compounded from ``init_cash``."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, returns in returns_by_label.items():
        equity = init_cash * (1.0 + returns.dropna()).cumprod()
        ax.plot(equity.index, equity.values, label=label)
    ax.set_title("Equity Curves: Hybrid Portfolio vs Buy-and-Hold vs Ibovespa")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (BRL)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drawdowns(returns_by_label: dict[str, pd.Series], path: str) -> None:
    """Drawdown (peak-to-trough) curves, expressed as negative percentages."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, returns in returns_by_label.items():
        wealth = (1.0 + returns.dropna()).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        ax.plot(drawdown.index, drawdown.values * 100, label=label)
    ax.set_title("Drawdown: Hybrid Portfolio vs Buy-and-Hold vs Ibovespa")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sma_sensitivity(summary: pd.DataFrame, path: str) -> None:
    """Bar chart of average Sharpe ratio and drawdown across SMA configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(summary["config"], summary["avg_sharpe"], color="steelblue")
    axes[0].set_title("Average Sharpe Ratio by SMA Configuration")
    axes[0].set_ylabel("Average Sharpe Ratio")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(summary["config"], summary["avg_max_drawdown"] * 100, color="indianred")
    axes[1].set_title("Average Maximum Drawdown by SMA Configuration")
    axes[1].set_ylabel("Average Max Drawdown (%)")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_subperiod_alpha(subperiod_alphas: pd.DataFrame, path: str) -> None:
    """Bar chart of the Carhart four-factor monthly alpha across sub-periods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(subperiod_alphas["sub_period"], subperiod_alphas["alpha_monthly_pct"], color="seagreen")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Monthly Alpha (Carhart 4-Factor) by Sub-Period")
    ax.set_ylabel("Monthly Alpha (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
