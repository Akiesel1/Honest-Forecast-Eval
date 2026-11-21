#!/usr/bin/env python3
"""
fx_study.py -- runner for the preregistered FX studies (see PREREG_FX.md).

    python fx_study.py --print-config-hash     # no data touched
    python fx_study.py --study both            # refuses to run until approved

Study A: uncovered interest parity vs random walk and drift, 6 pairs x 3 horizons.
Study B: triangular-deviation mean reversion on 3 crosses, gross and net of cost.

This file adds NO evaluation logic of its own for forecast skill: every
forecast goes through evaluation.walk_forward and every skill metric comes from
evaluation.summarize, unchanged. What lives here is only study plumbing --
which pairs, which horizons, the cost arithmetic, and the preregistered
pass/fail rules.

Two locks keep the run honest:
  1. It will not touch real data unless PREREG_FX.md says "STATUS: APPROVED".
  2. It will not run if CONFIG below no longer hashes to the value recorded in
     PREREG_FX.md -- so the configuration cannot drift after approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from data.loader import align_on_common_dates, load_fred_series, load_ohlcv
from evaluation import summarize, walk_forward
from forecasters import DriftForecaster, RandomWalkForecaster
from forecasters.triangular import (
    TRIANGLES,
    TriangularForecaster,
    round_trip_cost,
    triangle_residual,
)
from forecasters.uip import UIPForecaster

PREREG_PATH = "PREREG_FX.md"

# Everything that could change a result is in this one dict. It is hashed, and
# the hash is written into PREREG_FX.md at approval time.
CONFIG = {
    "alpha": 0.05,
    "rmse_beat_threshold": 0.98,
    "lookback": 252,
    "anomaly_abs_log_return": 0.10,
    "study_a": {
        "start": "2004-01-01",
        "end_exclusive": "2026-04-01",
        "pairs": {
            "EURUSD": {"ticker": "EURUSD=X", "base": "EUR", "quote": "USD"},
            "USDJPY": {"ticker": "USDJPY=X", "base": "USD", "quote": "JPY"},
            "GBPUSD": {"ticker": "GBPUSD=X", "base": "GBP", "quote": "USD"},
            "AUDUSD": {"ticker": "AUDUSD=X", "base": "AUD", "quote": "USD"},
            "USDCAD": {"ticker": "USDCAD=X", "base": "USD", "quote": "CAD"},
            "USDCHF": {"ticker": "USDCHF=X", "base": "USD", "quote": "CHF"},
        },
        "rate_series": {
            "USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N",
            "GBP": "IR3TIB01GBM156N", "AUD": "IR3TIB01AUM156N", "CAD": "IR3TIB01CAM156N",
            "CHF": "IR3TIB01CHM156N",
        },
        "rate_lag_months": 2,
        "rate_max_stale_months": 3,
        "periods_per_year": 252,
        "horizons": [1, 21, 63],
        "step_equals_horizon": True,
        "min_cells_passing": 2,
    },
    "study_b": {
        "start": "2004-01-01",
        "end_exclusive": "2026-11-01",
        "triangles": TRIANGLES,
        "horizon": 1,
        "step": 1,
        "pips_per_leg": 1.5,
        "min_trades": 30,
        "min_triangles_passing": 2,
    },
}
NO_CAP = 10**9   # max_evals: never subsample; every eval point is used


def config_hash() -> str:
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]


def require_approval() -> None:
    """Refuse to touch real data until the preregistration is approved and unchanged."""
    if not os.path.exists(PREREG_PATH):
        raise SystemExit(f"{PREREG_PATH} not found. Write and approve it before running.")
    text = open(PREREG_PATH, encoding="utf-8").read()
    if not re.search(r"^STATUS:\s*APPROVED", text, flags=re.MULTILINE):
        raise SystemExit(f"{PREREG_PATH} is not marked 'STATUS: APPROVED'. Nothing was run.")
    m = re.search(r"^CONFIG HASH:\s*`?([0-9a-f]{16})`?", text, flags=re.MULTILINE)
    if not m or m.group(1) != config_hash():
        raise SystemExit(
            f"CONFIG hash is {config_hash()} but {PREREG_PATH} records "
            f"{m.group(1) if m else 'none'}. The configuration changed after "
            "preregistration. Nothing was run."
        )


# ------------------------------------------------------------------ shared bits

def final_step(res: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Keep only the rows scored at the full horizon (one per eval point)."""
    return res[res["horizon"] == horizon]


def score(res: pd.DataFrame, horizon: int) -> dict:
    """The framework's own metrics for the full-horizon step, plus pass flags."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # constant forecasts -> undefined IC
        r = summarize(final_step(res, horizon), horizon).iloc[0]
    a = CONFIG["alpha"]
    beat_dir = bool(r["p_value"] < a and r["edge_vs_baseline"] > 0)
    beat_err = bool(r["rmse_ratio"] < CONFIG["rmse_beat_threshold"])
    ic_sig = bool(r["IC"] > 0 and r["IC_p"] < a)
    return {
        "n": int(r["n"]), "dir_acc": r["dir_acc"], "naive_baseline": r["naive_baseline"],
        "edge_vs_baseline": r["edge_vs_baseline"], "dir_p": r["p_value"],
        "model_rmse": r["model_rmse"], "rmse_ratio_vs_rw": r["rmse_ratio"],
        "IC": r["IC"], "IC_p": r["IC_p"],
        "beat_dir": beat_dir, "beat_err": beat_err, "ic_sig": ic_sig,
        "pass_all": beat_dir and beat_err and ic_sig,
    }


def anomaly_bars(df: pd.DataFrame) -> np.ndarray:
    """Positions of bars whose daily |log return| exceeds the declared threshold."""
    lr = np.abs(np.diff(np.log(df["close"].to_numpy(dtype=float))))
    return np.where(lr > CONFIG["anomaly_abs_log_return"])[0] + 1


def drop_anomalous_points(res: pd.DataFrame, horizon: int, bad: np.ndarray) -> pd.DataFrame:
    """Remove eval points whose scored bars t..t+horizon-1 include an anomalous bar."""
    if len(bad) == 0:
        return res
    ts = res["t"].unique()
    hit = [t for t in ts if np.any((bad >= t) & (bad <= t + horizon - 1))]
    return res[~res["t"].isin(hit)]


# ---------------------------------------------------------------------- study A

def run_study_a(outdir: str, stamp: str) -> pd.DataFrame:
    cfg = CONFIG["study_a"]
    rates = {ccy: load_fred_series(sid) for ccy, sid in cfg["rate_series"].items()}
    for ccy, s in rates.items():
        s.to_csv(os.path.join(outdir, "data", f"{stamp}_rate_{ccy}.csv"))

    rows = []
    for pair, p in cfg["pairs"].items():
        df = load_ohlcv(p["ticker"], start=cfg["start"], end=cfg["end_exclusive"])
        df.to_csv(os.path.join(outdir, "data", f"{stamp}_A_{pair}.csv"))
        bad = anomaly_bars(df)
        models = {
            "uip": lambda: UIPForecaster(
                rates[p["base"]], rates[p["quote"]], lag_months=cfg["rate_lag_months"],
                max_stale_months=cfg["rate_max_stale_months"],
                periods_per_year=cfg["periods_per_year"]),
            "random-walk": RandomWalkForecaster,
            "drift": DriftForecaster,
        }
        for h in cfg["horizons"]:
            expected = len(range(CONFIG["lookback"], len(df) - h, h))
            finals = {}
            for model, build in models.items():
                res = walk_forward(df, build(), CONFIG["lookback"], h, step=h,
                                   max_evals=NO_CAP, verbose=False)
                fin = final_step(res, h)
                fin.to_csv(os.path.join(outdir, "raw", f"{stamp}_A_{pair}_h{h}_{model}.csv"),
                           index=False)
                finals[model] = fin
                row = {"pair": pair, "horizon": h, "model": model, **score(res, h),
                       "n_expected": expected, "n_skipped": expected - fin["t"].nunique(),
                       "n_anomalous_bars": int(len(bad))}
                clean = drop_anomalous_points(fin, h, bad)
                if len(clean) < len(fin) and len(clean) >= 3:
                    ex = score(clean, h)
                    row.update({"n_ex_anom": ex["n"], "rmse_ratio_ex_anom": ex["rmse_ratio_vs_rw"],
                                "pass_all_ex_anom": ex["pass_all"]})
                rows.append(row)

            # UIP error relative to DRIFT, on the eval points both produced.
            both = finals["uip"].merge(finals["drift"], on="t", suffixes=("_u", "_d"))
            ru = np.sqrt(np.mean((both["pred_ret_u"] - both["actual_ret_u"]) ** 2))
            rd = np.sqrt(np.mean((both["pred_ret_d"] - both["actual_ret_d"]) ** 2))
            rows[-3]["rmse_ratio_vs_drift"] = ru / rd if rd > 0 else float("nan")
            print(f"  A  {pair} h={h}: done")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(outdir, f"{stamp}_study_a_table.csv"), index=False)
    return table


def study_a_verdict(table: pd.DataFrame) -> str:
    """Preregistered family-level rule for Study A."""
    cfg = CONFIG["study_a"]
    uip = table[table["model"] == "uip"]
    n = len(uip)
    n_pass = int(uip["pass_all"].sum())
    med = float(uip["rmse_ratio_vs_rw"].median())
    neg_ic = int(((uip["IC"] < 0) & (uip["IC_p"] < CONFIG["alpha"])).sum())
    detail = (f"{n_pass}/{n} pair-horizon cells pass all three metrics "
              f"(dir {int(uip['beat_dir'].sum())}, error {int(uip['beat_err'].sum())}, "
              f"IC {int(uip['ic_sig'].sum())}; chance alone predicts about "
              f"{CONFIG['alpha'] * n:.1f} per single metric); median RMSE ratio vs random walk "
              f"{med:.4f}; {neg_ic} cells have a significantly NEGATIVE IC.")
    if n_pass >= cfg["min_cells_passing"] and med < 1.0:
        return "STUDY A VERDICT: meets the preregistered pass rule. " + detail
    return ("STUDY A VERDICT: NULL -- UIP does not meet the preregistered pass rule "
            "against a random walk. " + detail)


# ---------------------------------------------------------------------- study B

def residual_diagnostics(name: str, resid: pd.Series, cost: pd.Series) -> dict:
    a = resid.abs()
    bp = 1e4
    q = resid.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) * bp
    return {
        "triangle": name, "n_days": int(len(resid)),
        "mean_bp": resid.mean() * bp, "std_bp": resid.std() * bp,
        "p01_bp": q[0.01], "p05_bp": q[0.05], "p25_bp": q[0.25], "median_bp": q[0.5],
        "p75_bp": q[0.75], "p95_bp": q[0.95], "p99_bp": q[0.99],
        "mean_abs_bp": a.mean() * bp, "median_abs_bp": a.median() * bp,
        "mean_cost_bp": cost.mean() * bp,
        "frac_abs_resid_gt_cost": float((a > cost).mean()),
        "lag1_autocorr": float(resid.autocorr(1)),
    }


def one_sided_mean_test(x: np.ndarray) -> tuple[float, float]:
    """Mean and one-sided p-value for H1: mean > 0."""
    if len(x) < 3:
        return (float(np.mean(x)) if len(x) else float("nan")), float("nan")
    t = stats.ttest_1samp(x, 0.0, alternative="greater")
    return float(np.mean(x)), float(t.pvalue)


def run_study_b(outdir: str, stamp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = CONFIG["study_b"]
    tickers = sorted({t[k] for t in cfg["triangles"].values() for k in ("cross", "leg1", "leg2")})
    raw = {tk: load_ohlcv(tk, start=cfg["start"], end=cfg["end_exclusive"]) for tk in tickers}
    for tk, df in raw.items():
        df.to_csv(os.path.join(outdir, "data", f"{stamp}_B_{tk.replace('=X', '')}.csv"))

    # ---- Stage 1: residuals vs cost, printed and saved BEFORE any evaluation.
    prepared, diag_rows = {}, []
    for name, tri in cfg["triangles"].items():
        al = align_on_common_dates({k: raw[tri[k]] for k in ("cross", "leg1", "leg2")})
        cross, leg1, leg2 = (al[k]["close"] for k in ("cross", "leg1", "leg2"))
        resid = triangle_residual(cross, leg1, leg2, tri["sign2"])
        cost = round_trip_cost({tri["cross"]: cross, tri["leg1"]: leg1, tri["leg2"]: leg2},
                               cfg["pips_per_leg"])
        pd.DataFrame({"residual": resid, "cost": cost}).to_csv(
            os.path.join(outdir, f"{stamp}_study_b_residuals_{name}.csv"))
        diag_rows.append(residual_diagnostics(name, resid, cost))
        prepared[name] = (al["cross"], leg1, leg2, resid, cost, tri["sign2"])
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(os.path.join(outdir, f"{stamp}_study_b_residual_diagnostics.csv"), index=False)
    print("\nSTUDY B, STAGE 1 -- residual distribution vs round-trip cost (before evaluation)")
    print(diag.to_string(index=False))
    print("Daily closes are not synchronised across pairs; a residual can be timestamp "
          "noise rather than a tradable gap.\n")

    # ---- Stage 2: walk-forward evaluation, then gross and net of cost.
    rows = []
    for name, (df, leg1, leg2, resid, cost, sign2) in prepared.items():
        bad = anomaly_bars(df)
        models = {"triangular": lambda: TriangularForecaster(leg1, leg2, sign2),
                  "random-walk": RandomWalkForecaster, "drift": DriftForecaster}
        for model, build in models.items():
            res = walk_forward(df, build(), CONFIG["lookback"], cfg["horizon"], step=cfg["step"],
                               max_evals=NO_CAP, verbose=False)
            res.to_csv(os.path.join(outdir, "raw", f"{stamp}_B_{name}_{model}.csv"), index=False)
            row = {"triangle": name, "model": model, **score(res, cfg["horizon"]),
                   "n_anomalous_bars": int(len(bad))}
            if model == "triangular":
                asof = df.index[res["t"].to_numpy() - 1]
                r_asof = resid.loc[asof].to_numpy()
                c_asof = cost.loc[asof].to_numpy()
                gross = np.sign(res["pred_ret"].to_numpy()) * res["actual_ret"].to_numpy()
                net = gross - c_asof
                trade = np.abs(r_asof) > c_asof
                row["gross_mean_all_bp"], row["gross_p_all"] = one_sided_mean_test(gross)
                row["net_mean_all_bp"], row["net_p_all"] = one_sided_mean_test(net)
                row["n_trades"] = int(trade.sum())
                row["gross_mean_trades_bp"], row["gross_p_trades"] = one_sided_mean_test(gross[trade])
                row["net_mean_trades_bp"], row["net_p_trades"] = one_sided_mean_test(net[trade])
                for k in ("gross_mean_all_bp", "net_mean_all_bp",
                          "gross_mean_trades_bp", "net_mean_trades_bp"):
                    row[k] *= 1e4
                gross_edge = row["pass_all"]
                net_ok = (row["n_trades"] >= cfg["min_trades"] and row["net_mean_trades_bp"] > 0
                          and row["net_p_trades"] < CONFIG["alpha"])
                if gross_edge and net_ok:
                    row["outcome"] = "PASS (gross and net)"
                elif gross_edge and row["n_trades"] < cfg["min_trades"]:
                    row["outcome"] = "NULL (gross pass; too few residuals exceed cost to test net)"
                elif gross_edge:
                    row["outcome"] = "NULL (gross pass vanishes net of cost)"
                else:
                    row["outcome"] = "NULL (no gross pass)"
            rows.append(row)
        print(f"  B  {name}: done")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(outdir, f"{stamp}_study_b_table.csv"), index=False)
    return diag, table


def study_b_verdict(table: pd.DataFrame) -> str:
    """Preregistered rule for Study B."""
    cfg = CONFIG["study_b"]
    tri = table[table["model"] == "triangular"]
    n_pass = int(tri["outcome"].str.startswith("PASS").sum())
    n_gross = int(tri["pass_all"].sum())
    detail = (f"{n_gross}/{len(tri)} triangles pass the forecast metrics gross; "
              f"{n_pass}/{len(tri)} also pass net of cost.")
    if n_pass >= cfg["min_triangles_passing"]:
        return "STUDY B VERDICT: meets the preregistered pass rule. " + detail
    return ("STUDY B VERDICT: NULL -- triangular mean reversion does not meet the "
            "preregistered pass rule net of cost. " + detail)


# ------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Preregistered FX studies (see PREREG_FX.md)")
    p.add_argument("--study", choices=["A", "B", "both"], default="both")
    p.add_argument("--output-dir", default=os.path.join("results", "fx_study"))
    p.add_argument("--print-config-hash", action="store_true",
                   help="print the CONFIG hash and exit (touches no data)")
    args = p.parse_args()

    if args.print_config_hash:
        print(config_hash())
        return

    require_approval()
    for sub in ("", "data", "raw"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    meta = {"stamp": stamp, "git_commit": commit, "config_hash": config_hash(),
            "config": CONFIG, "study": args.study,
            "versions": {"numpy": np.__version__, "pandas": pd.__version__}}
    with open(os.path.join(args.output_dir, f"{stamp}_run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 250)
    verdicts = []
    if args.study in ("A", "both"):
        ta = run_study_a(args.output_dir, stamp)
        print("\nSTUDY A -- UIP vs random walk and drift (scored at the full horizon)")
        print(ta.to_string(index=False))
        verdicts.append(study_a_verdict(ta))
    if args.study in ("B", "both"):
        _, tb = run_study_b(args.output_dir, stamp)
        print("\nSTUDY B -- triangular mean reversion")
        print(tb.to_string(index=False))
        verdicts.append(study_b_verdict(tb))

    print("\n" + "\n".join(verdicts))
    with open(os.path.join(args.output_dir, f"{stamp}_verdicts.txt"), "w") as f:
        f.write("\n".join(verdicts) + "\n")
    print(f"\nSaved under {args.output_dir} with stamp {stamp}.")


if __name__ == "__main__":
    main()
