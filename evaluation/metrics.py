"""
Scoring: turn the raw predicted-vs-actual records into honest per-horizon
metrics and a blunt verdict.

These functions are moved essentially verbatim from the original
run_backtest.py -- they were already model-agnostic (they only ever look at
`actual_ret` / `pred_ret` columns), which is exactly why they needed no
rewriting to become part of the framework. The honesty guardrails live here:
directional accuracy is always shown next to the naive baseline it must beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def binom_p_value(successes: int, n: int, baseline_p: float) -> float:
    """One-sided P(X >= successes) under Binomial(n, baseline_p).

    Answers: is the directional accuracy significantly ABOVE the naive baseline,
    or is it just noise? A small p-value (<0.05) means the edge is unlikely to
    be luck.
    """
    if n == 0:
        return float("nan")
    return float(stats.binom.sf(successes - 1, n, max(min(baseline_p, 1.0), 1e-9)))


def spearman_ic(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Information Coefficient + its p-value.

    IC is the rank correlation of predicted vs actual returns; ~0 means no
    signal. The p-value answers the question the raw IC can't: "if there were
    truly ZERO relationship, how often would chance alone produce an IC this
    far from zero?" (two-sided). With n~150, noise routinely produces
    |IC| ~ 0.16, so a positive IC with a large p-value is unremarkable --
    and the table should say so rather than let the number sit there.

    Note: a constant forecast (e.g. random-walk's zero return) has no rank
    variation, so IC is undefined (nan) for it -- that's correct, not a bug.
    """
    if len(pred) < 3:
        return float("nan"), float("nan")
    ic, p = stats.spearmanr(pred, actual)
    return float(ic), float(p)


def summarize(res: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per-horizon metrics table from the raw walk-forward records."""
    rows = []
    for h in range(1, horizon + 1):
        sub = res[res["horizon"] == h]
        n = len(sub)
        if n == 0:
            continue
        dir_acc = sub["correct_dir"].mean()
        up_rate = sub["actual_up"].mean()
        # Best naive constant predictor: "always up", "always down", or coin flip.
        baseline = max(0.5, up_rate, 1 - up_rate)
        successes = int(sub["correct_dir"].sum())
        p_val = binom_p_value(successes, n, baseline)

        rw_rmse = np.sqrt(np.mean(sub["actual_ret"] ** 2))          # predict zero
        model_rmse = np.sqrt(np.mean((sub["pred_ret"] - sub["actual_ret"]) ** 2))
        rmse_ratio = model_rmse / rw_rmse if rw_rmse > 0 else float("nan")
        ic, ic_p = spearman_ic(sub["pred_ret"].values, sub["actual_ret"].values)

        rows.append({
            "horizon": h, "n": n,
            "dir_acc": dir_acc, "naive_baseline": baseline,
            "edge_vs_baseline": dir_acc - baseline, "p_value": p_val,
            "model_rmse": model_rmse, "rw_rmse": rw_rmse,
            "rmse_ratio": rmse_ratio, "IC": ic, "IC_p": ic_p,
        })
    return pd.DataFrame(rows)


def verdict(summary: pd.DataFrame) -> str:
    """Blunt, honest read of whether there's detectable signal at horizon 1."""
    h1 = summary[summary["horizon"] == 1]
    if h1.empty:
        return "No horizon-1 results to judge."
    r = h1.iloc[0]
    beats_dir = (r["p_value"] < 0.05) and (r["edge_vs_baseline"] > 0)
    # Require a NON-TRIVIAL error beat: a ratio of 0.997 is a rounding fluke,
    # not a signal, so demand at least a 2% improvement over random walk.
    beats_err = r["rmse_ratio"] < 0.98
    # IC counts as signal only if it's positive AND statistically significant --
    # magnitude alone (the old |IC|>=0.05 rule) mistakes noise for signal at
    # small n.
    strong_ic = (r["IC"] > 0) and (r.get("IC_p", float("nan")) < 0.05)

    ic_note = "significant" if r.get("IC_p", 1.0) < 0.05 else "not significant"
    lines = [
        f"At 1 step: {r['dir_acc']:.1%} directional accuracy vs a "
        f"{r['naive_baseline']:.1%} naive baseline (p={r['p_value']:.3f}); "
        f"error is {r['rmse_ratio']:.2f}x random-walk; "
        f"IC={r['IC']:+.3f} (p={r.get('IC_p', float('nan')):.3f}, {ic_note})."
    ]
    signals = sum([beats_dir, beats_err, strong_ic])
    if signals == 0:
        lines.append(
            "VERDICT: No detectable edge on this ticker/period. Does not beat "
            "'always predict the majority direction' significantly, does not beat "
            "a random walk on error, and shows ~zero rank correlation. This is the "
            "common and expected result -- treat it as the null until proven "
            "otherwise across MANY tickers and regimes."
        )
    elif signals == 3:
        lines.append(
            "VERDICT: Consistent with (not proof of) real signal -- beats the naive "
            "direction baseline significantly, beats random-walk error, and has a "
            "positive IC. Before believing it: re-run on other tickers, other time "
            "windows, and check the edge survives realistic costs in a separate test."
        )
    else:
        lines.append(
            "VERDICT: Mixed. Some metrics lean positive but not all agree, so this "
            "is most likely noise. Expand the sample (more eval points, more "
            "tickers, more regimes) before reading anything into it."
        )
    return "\n".join(lines)
