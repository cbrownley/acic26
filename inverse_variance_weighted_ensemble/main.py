"""
main.py
Entry point.  Orchestrates the full pipeline for one or many datasets.

Timing and progress
-------------------
Every numbered step inside process_dataset is wrapped in a _timed() context
manager that prints "  [step]  Done in X.Xs" immediately on exit.  The
overall dataset wall time is printed at the end of each run.

run_batch uses tqdm for a dataset-level progress bar showing:
  - completed / total count
  - elapsed time
  - estimated time remaining
  - last dataset ID processed

run_batch_parallel shows the same tqdm bar, updated as futures complete.

Usage
-----
    # Single dataset (CLI):
    python -m acic2026.main 1 XXXX 1

    # Curated 18-dataset track (parallel):
    from acic2026.main import run_batch_parallel, CURATED_IDS
    results = run_batch_parallel(CURATED_IDS, team_id="XXXX", subm_id="1",
                                 data_dir="path/to/data",
                                 out_dir="submissions",
                                 plot_dir="plots")

    # Multi-dataset summary plots after batch:
    from acic2026 import visualizations as viz
    viz.plot_auc_heatmap(results["auc_records"], save_dir="plots")
    viz.plot_scate_across_datasets(results["scate_records"], save_dir="plots")
    viz.plot_icate_violin_grid(results["icate_dfs"], save_dir="plots")
    viz.plot_pate_vs_scate(results["pate_vs_scate_records"], save_dir="plots")
"""

import sys
import os
import time
import traceback
import warnings
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import config as config
from data import load_and_split, build_preprocessor
from evaluate import cv_evaluate, compute_gain_curves, evaluate_aucs, compute_rate
from estimators import (
    fit_drlearner,
    fit_linear_drlearner,
    fit_causal_forest,
    fit_forest_drlearner,
)
from inference import (
    get_icates,
    compute_scate,
    compute_subcate,
    compute_pate,
)
from outputs import (
    build_icate_df,
    best_icate,
    best_scalar,
    best_subcate,
    save,
)
import visualizations as viz

# Suppress EconML/sklearn/LightGBM interop warnings that are cosmetic only.
# LightGBM stores feature names when fitted on a pandas DataFrame; EconML's
# cross-fitting then calls predict() with a plain numpy array, triggering
# sklearn's feature-name mismatch check.  The predictions are numerically
# identical regardless — these are pure bookkeeping warnings, not errors.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="feature_name keyword",
    category=UserWarning,
)


# ─────────────────────────────────────────────────────────────────────────────
# Timing utilities
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(seconds: float) -> str:
    """Human-readable duration string: '2m 13.4s' or '45.2s'."""
    if seconds >= 60:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"
    return f"{seconds:.1f}s"


@contextmanager
def _timed(label: str):
    """
    Context manager that prints elapsed time for a named pipeline step.

    Usage:
        with _timed("DRLearner fit"):
            est = fit_drlearner(...)

    Prints on exit:
        ── DRLearner fit  ···  done in 34.2s
    """
    print(f"  ── {label} …", flush=True)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"  ── {label}  ···  done in {_fmt(elapsed)}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Curated 18-dataset IDs
# ─────────────────────────────────────────────────────────────────────────────
CURATED_IDS = [str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]


# ─────────────────────────────────────────────────────────────────────────────
# Single-dataset pipeline
# ─────────────────────────────────────────────────────────────────────────────


def process_dataset(
    data_id: str,
    team_id: str = config.TEAM_ID,
    subm_id: str = config.SUBM_ID,
    run_timestamp: str = config.TIMESTAMP,
    data_dir=config.DATA_DIR,
    out_dir=config.OUT_DIR,
    plot_dir=None,
    inner_n_jobs: int = config.INNER_N_JOBS,
) -> dict:
    """
    Run the complete pipeline for one dataset and write all output files.

    Produces
    --------
    8 competition CSV files  (iCATE, sCATE, subCATE, PATE, BEST_*)
    1 AURC CSV               (aucs_*.csv)
    1 RATE CSV               (rate_*.csv)
    4 PNG plots              (if plot_dir is set)

    Returns
    -------
    dict with keys:
        'icate', 'scate', 'subcate', 'pate'  — estimand DataFrames
        'aucs'                                — {z: float} AURC per arm
        'curves'                              — {z: fklearn DataFrame}
        'timing'                              — {step: seconds} wall times
    """
    config.INNER_N_JOBS = inner_n_jobs

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    auc_dir = Path("AUCs")
    rate_dir = Path("RATEs")
    plot_dir = Path(plot_dir) if plot_dir else None
    # Ensure they exist
    for p in [data_dir, out_dir, auc_dir, rate_dir, plot_dir]:
        p.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    timing = {}  # step_name -> elapsed seconds

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Dataset {data_id}  |  team {team_id}  |  submission {subm_id}")
    print(
        f"  AutoML={config.USE_AUTOML}  IF_CI={config.USE_IF_CI}  "
        f"ForestDRL={config.USE_FOREST_DRL}  "
        f"inner_n_jobs={inner_n_jobs}"
    )
    print(sep)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    path = data_dir / f"data_{data_id}.csv"
    ID, Y, T, X_raw, x12 = load_and_split(path)
    n = len(Y)
    preprocessor = build_preprocessor(X_raw)
    X_feat = preprocessor.transform(X_raw)
    timing["load+preprocess"] = time.perf_counter() - t0
    print(
        f"  [1] Load & preprocess  ···  n={n}  "
        f"p_raw={X_raw.shape[1]}  p_enc={X_feat.shape[1]}  "
        f"arms={sorted(set(T))}  "
        f"({_fmt(timing['load+preprocess'])})"
    )

    # ── 2. CV quality check ───────────────────────────────────────────────────
    with _timed("[2] CV outcome model"):
        t0 = time.perf_counter()
        cv_evaluate(X_feat, Y, T)
        timing["cv_evaluate"] = time.perf_counter() - t0

    # ── 3. Fit CATE estimators ────────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print(f"  [3] Fitting CATE estimators")
    print(f"  {'─'*56}")

    t0 = time.perf_counter()
    est_drl, drl_has_ci = fit_drlearner(X_feat, Y, T)
    timing["DRLearner"] = time.perf_counter() - t0
    print(f"      DRLearner          {_fmt(timing['DRLearner']):>10}")

    t0 = time.perf_counter()
    est_lin = fit_linear_drlearner(X_feat, Y, T) if config.USE_IF_CI else None
    timing["LinearDRLearner"] = time.perf_counter() - t0
    if config.USE_IF_CI:
        print(f"      LinearDRLearner    {_fmt(timing['LinearDRLearner']):>10}")

    t0 = time.perf_counter()
    forests = fit_causal_forest(X_feat, Y, T)
    timing["CausalForest"] = time.perf_counter() - t0
    print(f"      CausalForest       {_fmt(timing['CausalForest']):>10}  " f"({len(TREATMENTS)} arms × binary)")

    t0 = time.perf_counter()
    forest_drl = fit_forest_drlearner(X_feat, Y, T) if config.USE_FOREST_DRL else None
    timing["ForestDRLearner"] = time.perf_counter() - t0
    if config.USE_FOREST_DRL:
        print(f"      ForestDRLearner    {_fmt(timing['ForestDRLearner']):>10}")

    t_est_total = sum(
        timing[k] for k in ["DRLearner", "LinearDRLearner", "CausalForest", "ForestDRLearner"] if k in timing
    )
    print(f"  {'─'*56}")
    print(f"  [3] Estimators total   {_fmt(t_est_total):>10}")

    # ── 4. Variance-weighted ensemble iCATE ───────────────────────────────────
    with _timed("[4] Ensemble iCATEs"):
        t0 = time.perf_counter()
        icates, lowers, uppers = get_icates(
            est_drl,
            drl_has_ci,
            est_lin,
            forests,
            X_feat,
            n,
            forest_drl=forest_drl,
        )
        timing["ensemble"] = time.perf_counter() - t0

    # ── 5. Aggregate estimands ────────────────────────────────────────────────
    with _timed("[5] Aggregate estimands"):
        t0 = time.perf_counter()
        icate_df = build_icate_df(icates, lowers, uppers, ID)
        scate_df = compute_scate(icates, lowers, uppers, n)
        subcate_df = compute_subcate(icates, lowers, uppers, x12, n)
        pate_df = compute_pate(est_drl, drl_has_ci, est_lin, X_feat)
        timing["aggregate"] = time.perf_counter() - t0

    # ── 6. Gain curves, AUCs, RATE ────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    curves = {}
    aucs = {}
    t0 = time.perf_counter()
    try:
        curves = compute_gain_curves(icates, T, Y)
        aucs = evaluate_aucs(icates, T, Y)
    except ImportError as e:
        print(f"  [AUC] fklearn not installed — skipping: {e}")
    timing["gain_curves"] = time.perf_counter() - t0

    rates = {}
    t0 = time.perf_counter()
    try:
        rates = compute_rate(icates, T, Y)
    except Exception as e:
        print(f"  [RATE] Failed: {e}")
    timing["rate"] = time.perf_counter() - t0

    # ── 7. Best-treatment derivations ─────────────────────────────────────────
    best_icate_df = best_icate(icates, ID)
    best_scate_df = best_scalar(scate_df)
    best_subcate_df = best_subcate(subcate_df)
    best_pate_df = best_scalar(pate_df)

    # ── 8. Save all CSVs ──────────────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    d, t, s = data_id, team_id, subm_id
    ts = run_timestamp
    t0 = time.perf_counter()

    save(icate_df, f"iCATE_{d}_{t}_{s}_{ts}.csv", out_dir)
    save(scate_df, f"sCATE_{d}_{t}_{s}_{ts}.csv", out_dir)
    save(subcate_df, f"subCATE_{d}_{t}_{s}_{ts}.csv", out_dir)
    save(pate_df, f"PATE_{d}_{t}_{s}_{ts}.csv", out_dir)
    save(best_icate_df, f"BEST_iCATE_{d}_{t}_{s}_{ts}.csv", out_dir)
    save(best_scate_df, f"BEST_sCATE_{d}_{t}_{s}_{ts}.csv", out_dir)
    save(best_subcate_df, f"BEST_subCATE_{d}_{t}_{s}_{ts}.csv", out_dir)
    save(best_pate_df, f"BEST_PATE_{d}_{t}_{s}_{ts}.csv", out_dir)

    # Diagnostic files → AUCs/ and RATEs/
    if aucs:
        aucs_df = pd.DataFrame([{"z": z, "AURC": v} for z, v in aucs.items()])
        # Updated filename: aucs_[dataID]_[teamID]_[submID]_[timestamp].csv
        save(aucs_df, f"aucs_{d}_{t}_{s}_{ts}.csv", auc_dir)

    if rates:
        rate_df = pd.DataFrame([{"z": z, **v} for z, v in rates.items()])
        # Updated filename: rates_[dataID]_[teamID]_[submID]_[timestamp].csv
        save(rate_df, f"rates_{d}_{t}_{s}_{ts}.csv", rate_dir)

    timing["save_csvs"] = time.perf_counter() - t0

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    if plot_dir is not None:
        t0 = time.perf_counter()
        print(f"\n  [VIZ] Generating plots → {plot_dir}")
        viz.plot_all_single_dataset(
            data_id=data_id,
            team_id=team_id,
            subm_id=subm_id,
            icate_df=icate_df,
            scate_df=scate_df,
            subcate_df=subcate_df,
            pate_df=pate_df,
            curves_dict=curves,
            aucs_dict=aucs,
            save_dir=plot_dir,
        )
        timing["plots"] = time.perf_counter() - t0

    # ── 10. Summary ───────────────────────────────────────────────────────────
    wall_total = time.perf_counter() - wall_start
    timing["total"] = wall_total

    print(f"\n  {'═'*56}")
    print(f"  Dataset {data_id} — timing summary")
    print(f"  {'─'*56}")
    step_rows = [
        ("load + preprocess", timing["load+preprocess"]),
        ("cv evaluate", timing["cv_evaluate"]),
        ("DRLearner", timing["DRLearner"]),
    ]
    if config.USE_IF_CI:
        step_rows.append(("LinearDRLearner", timing["LinearDRLearner"]))
    step_rows.append(("CausalForest", timing["CausalForest"]))
    if config.USE_FOREST_DRL:
        step_rows.append(("ForestDRLearner", timing["ForestDRLearner"]))
    step_rows += [
        ("ensemble + aggregate", timing["ensemble"] + timing["aggregate"]),
        ("gain curves + RATE", timing["gain_curves"] + timing["rate"]),
        ("save CSVs", timing["save_csvs"]),
    ]
    if "plots" in timing:
        step_rows.append(("plots", timing["plots"]))

    for label, secs in step_rows:
        pct = 100 * secs / wall_total
        bar = "█" * int(pct / 4)
        print(f"    {label:<26}  {_fmt(secs):>8}  {pct:5.1f}%  {bar}")

    print(f"  {'─'*56}")
    print(f"    {'TOTAL':<26}  {_fmt(wall_total):>8}")
    print(f"  {'═'*56}")

    print(f"\n  sCATE:  " + "  ".join(f"{r['z']}={r['Estimate']:+.3f}" for _, r in scate_df.iterrows()))
    print(f"  best sCATE={best_scate_df['best_z'].iloc[0]}" f"  best PATE={best_pate_df['best_z'].iloc[0]}")
    if aucs:
        print("  AURC:   " + "  ".join(f"{z}={v:.4f}" for z, v in aucs.items()))

    return {
        "icate": icate_df,
        "scate": scate_df,
        "subcate": subcate_df,
        "pate": pate_df,
        "aucs": aucs,
        "curves": curves,
        "timing": timing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for accumulating multi-dataset summary records
# ─────────────────────────────────────────────────────────────────────────────


def _make_auc_record(data_id, aucs):
    rec = {"data_id": data_id}
    rec.update({z: aucs.get(z, float("nan")) for z in config.TREATMENTS})
    return rec


def _make_scate_records(data_id, scate_df):
    rows = scate_df.copy()
    rows["data_id"] = data_id
    return rows.to_dict("records")


def _make_pate_vs_scate_records(data_id, scate_df, pate_df):
    rows = []
    for z in config.TREATMENTS:
        s = scate_df.loc[scate_df["z"] == z].iloc[0]
        p = pate_df.loc[pate_df["z"] == z].iloc[0]
        rows.append(
            {
                "data_id": data_id,
                "z": z,
                "sCATE": s["Estimate"],
                "sCATE_L95": s["L95"],
                "sCATE_U95": s["U95"],
                "PATE": p["Estimate"],
                "PATE_L95": p["L95"],
                "PATE_U95": p["U95"],
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sequential batch  (tqdm progress bar)
# ─────────────────────────────────────────────────────────────────────────────


def run_batch(
    data_ids,
    team_id: str = config.TEAM_ID,
    subm_id: str = config.SUBM_ID,
    run_timestamp: str = config.TIMESTAMP,
    data_dir=config.DATA_DIR,
    out_dir=config.OUT_DIR,
    plot_dir=None,
) -> dict:
    """
    Process datasets sequentially with a tqdm progress bar.

    The progress bar shows:
      completed / total  |  elapsed  |  ETA  |  current dataset ID

    Returns
    -------
    dict with keys:
        'results', 'auc_records', 'scate_records',
        'pate_vs_scate_records', 'icate_dfs', 'timing_records'
    """
    ids = list(data_ids)
    results = {}
    auc_records = []
    scate_records = []
    pate_vs_scate = []
    icate_dfs = {}
    timing_records = []
    batch_start = time.perf_counter()

    bar = tqdm(
        ids,
        desc="Datasets",
        unit="ds",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for did in bar:
        bar.set_postfix({"current": did}, refresh=True)
        try:
            r = process_dataset(did, team_id, subm_id, run_timestamp, data_dir, out_dir, plot_dir=plot_dir, inner_n_jobs=-1)
            results[did] = r
            icate_dfs[did] = r["icate"]
            auc_records.append(_make_auc_record(did, r["aucs"]))
            scate_records.extend(_make_scate_records(did, r["scate"]))
            pate_vs_scate.extend(_make_pate_vs_scate_records(did, r["scate"], r["pate"]))
            timing_records.append({"data_id": did, **r["timing"]})
            bar.set_postfix(
                {"last": did, "last_time": _fmt(r["timing"]["total"])},
                refresh=True,
            )
        except Exception:
            traceback.print_exc()
            tqdm.write(f"[ERROR] Dataset {did} failed — skipping.")

    batch_elapsed = time.perf_counter() - batch_start
    n_ok = len(results)
    n_total = len(ids)
    tqdm.write(
        f"\n{'═'*60}\n"
        f"  Batch complete: {n_ok}/{n_total} datasets succeeded\n"
        f"  Total wall time : {_fmt(batch_elapsed)}\n"
        f"  Avg per dataset : {_fmt(batch_elapsed / max(n_ok, 1))}\n"
        f"{'═'*60}"
    )

    return {
        "results": results,
        "auc_records": auc_records,
        "scate_records": scate_records,
        "pate_vs_scate_records": pate_vs_scate,
        "icate_dfs": icate_dfs,
        "timing_records": timing_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parallel batch  (ProcessPoolExecutor + tqdm)
# ─────────────────────────────────────────────────────────────────────────────


def _worker(args: tuple) -> tuple:
    """
    Picklable top-level worker.

    Receives cfg_overrides as the last element of args and applies them to
    the config module before calling process_dataset.  This is necessary
    because ProcessPoolExecutor spawns fresh worker processes that import
    acic2026.config from disk (getting all defaults), so CLI flags like
    --no-bootstrap that mutated config in the main process are invisible
    to workers unless explicitly forwarded.
    """
    data_id, team_id, subm_id, run_timestamp, data_dir, out_dir, plot_dir, inner_n_jobs, cfg_overrides = args
    try:
        # Apply config overrides in this worker process
        import config as _cfg

        for k, v in cfg_overrides.items():
            setattr(_cfg, k, v)

        r = process_dataset(data_id, team_id, subm_id, run_timestamp, data_dir, out_dir, plot_dir=plot_dir, inner_n_jobs=inner_n_jobs)
        return data_id, r, None
    except Exception:
        return data_id, None, traceback.format_exc()


def run_batch_parallel(
    data_ids,
    team_id: str = config.TEAM_ID,
    subm_id: str = config.SUBM_ID,
    run_timestamp: str = config.TIMESTAMP,
    data_dir=config.DATA_DIR,
    out_dir=config.OUT_DIR,
    plot_dir=None,
    max_workers: int = config.N_PARALLEL_DATASETS,
    cfg_overrides: dict = None,
) -> dict:
    """
    Process datasets concurrently (ProcessPoolExecutor) with a tqdm bar.

    cfg_overrides : dict of config attribute names → values to apply in each
    worker process.  Required because ProcessPoolExecutor workers import
    acic2026.config fresh (getting defaults), so any runtime flag changes
    made in __main__ (e.g. USE_BOOTSTRAP=False from --no-bootstrap) must be
    forwarded explicitly.  Populated automatically when called from __main__.
    """
    ids = list(data_ids)
    total = len(ids)
    cpu_count = os.cpu_count() or 4
    inner_jobs = max(1, cpu_count // max(max_workers, 1))
    overrides = cfg_overrides or {}

    print(
        f"\n{'═'*62}\n"
        f"  Parallel batch: {total} datasets\n"
        f"  Workers        : {max_workers}\n"
        f"  Inner n_jobs   : {inner_jobs}  (machine: {cpu_count} CPUs)\n"
        f"  Total cores in use ≈ {max_workers * inner_jobs}\n"
        f"{'═'*62}"
    )

    job_args = [
        (did, team_id, subm_id, run_timestamp, str(data_dir), str(out_dir), str(plot_dir) if plot_dir else None, inner_jobs, overrides)
        for did in ids
    ]

    auc_records = []
    scate_records = []
    pate_vs_scate = []
    icate_dfs = {}
    all_results = {}
    timing_records = []
    batch_start = time.perf_counter()

    bar = tqdm(
        total=total,
        desc="Datasets",
        unit="ds",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, a): a[0] for a in job_args}
        for fut in as_completed(futures):
            did = futures[fut]
            try:
                did_ret, r, err = fut.result()
            except Exception as exc:
                r, err = None, str(exc)

            if r is not None and err is None:
                all_results[did] = r
                icate_dfs[did] = r["icate"]
                auc_records.append(_make_auc_record(did, r["aucs"]))
                scate_records.extend(_make_scate_records(did, r["scate"]))
                pate_vs_scate.extend(_make_pate_vs_scate_records(did, r["scate"], r["pate"]))
                timing_records.append({"data_id": did, **r["timing"]})
                bar.set_postfix(
                    {"last": did, "time": _fmt(r["timing"]["total"])},
                    refresh=True,
                )
            else:
                first_line = (err or "unknown error").splitlines()[-1]
                tqdm.write(f"  [ERROR] {did}: {first_line}")

            bar.update(1)

    bar.close()

    batch_elapsed = time.perf_counter() - batch_start
    n_ok = len(all_results)
    n_fail = total - n_ok

    print(
        f"\n{'═'*62}\n"
        f"  Parallel batch complete\n"
        f"  Succeeded : {n_ok}/{total}\n"
        f"  Failed    : {n_fail}\n"
        f"  Wall time : {_fmt(batch_elapsed)}\n"
        f"  Avg/ds    : {_fmt(batch_elapsed / max(n_ok, 1))}\n"
        f"{'═'*62}"
    )

    return {
        "results": all_results,
        "auc_records": auc_records,
        "scate_records": scate_records,
        "pate_vs_scate_records": pate_vs_scate,
        "icate_dfs": icate_dfs,
        "timing_records": timing_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

# Import TREATMENTS here so the timing summary in process_dataset can use it
from config import TREATMENTS


def _build_parser():
    """Build the argparse parser for the ACIC 2026 pipeline CLI."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m acic2026.main",
        description="ACIC 2026 causal inference pipeline — single dataset or batch mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Single dataset (default settings):
  python -m acic2026.main --data-id 1

  # Single dataset with plots:
  python -m acic2026.main --data-id 1 --plot-dir plots/

  # Curated 18-dataset track, sequential:
  python -m acic2026.main --batch curated

  # Curated track, parallel across ALL logical cores:
  python -m acic2026.main --batch curated --parallel

  # Parallel with explicit worker count (e.g. 8 datasets at once):
  python -m acic2026.main --batch curated --parallel --workers 8

  # Full 9000-dataset track, parallel, max cores:
  python -m acic2026.main --batch all --parallel

  # Specific dataset IDs, parallel:
  python -m acic2026.main --batch ids --ids 1 42 100 --parallel

  # Speed run: fewer bootstrap replicates, no LinearDRL, no ForestDRL:
  python -m acic2026.main --batch curated --parallel --n-boot 20 --no-if-ci --no-forest-drl

  # Override all paths:
  python -m acic2026.main --batch curated --parallel \\
      --data-dir /data/acic2026/ --out-dir /results/ --plot-dir /results/plots/

Config overrides take effect for that run only — config.py is not modified.
""",
    )

    # ── Identity ──────────────────────────────────────────────────────────────
    ident = p.add_argument_group("identity")
    ident.add_argument(
        "--team-id", default=config.TEAM_ID, help=f"Team ID for output filenames (default: {config.TEAM_ID})"
    )
    ident.add_argument("--subm-id", default=config.SUBM_ID, help=f"Submission ID (default: {config.SUBM_ID})")

    # ── Dataset selection ─────────────────────────────────────────────────────
    ds = p.add_argument_group("dataset selection  (mutually exclusive: --data-id vs --batch)")
    grp = ds.add_mutually_exclusive_group()
    grp.add_argument("--data-id", metavar="ID", help="Dataset ID, e.g. 1 or 42")
    grp.add_argument(
        "--batch",
        choices=["curated", "all", "ids"],
        help=("curated = 18 representative datasets | " "all = all 9000 datasets | " "ids = explicit list via --ids"),
    )
    ds.add_argument("--ids", nargs="+", metavar="ID", help="Dataset IDs to process when --batch ids is used")

    # ── Paths ─────────────────────────────────────────────────────────────────
    paths = p.add_argument_group("paths")
    paths.add_argument(
        "--data-dir",
        default=str(config.DATA_DIR),
        help=f"Directory containing data_*.csv files (default: {config.DATA_DIR})",
    )
    paths.add_argument(
        "--out-dir",
        default=str(config.OUT_DIR),
        help=f"Output directory for submission CSVs (default: {config.OUT_DIR})",
    )
    paths.add_argument("--plot-dir", default=None, help="Output directory for PNG plots (default: skip plots)")

    # ── Parallelism ───────────────────────────────────────────────────────────
    par = p.add_argument_group("parallelism")
    par.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Run datasets concurrently via ProcessPoolExecutor. "
            "Default workers = os.cpu_count() (all logical cores). "
            "Each worker uses cpu_count // workers inner threads."
        ),
    )
    par.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of parallel dataset workers. "
            "Defaults to os.cpu_count() when --parallel is set, "
            f"or {config.N_PARALLEL_DATASETS} from config otherwise."
        ),
    )

    # ── Model knobs ───────────────────────────────────────────────────────────
    mdl = p.add_argument_group("model knobs  (override config.py for this run)")
    mdl.add_argument(
        "--n-boot",
        type=int,
        default=config.N_BOOT,
        metavar="N",
        help=f"Bootstrap replicates for DRLearner CIs (default: {config.N_BOOT})",
    )
    mdl.add_argument(
        "--n-folds",
        type=int,
        default=config.N_CV_FOLDS,
        metavar="K",
        help=f"Cross-fitting folds (default: {config.N_CV_FOLDS})",
    )
    mdl.add_argument(
        "--no-bootstrap",
        action="store_true",
        help=(
            "Skip BootstrapInference on the NonlinearDRLearner. "
            "The DRLearner still contributes its point estimate; "
            "CIs come from LinearDRLearner (HC1 sandwich) + "
            "CausalForest and ForestDRLearner (GRF variance). "
            "Significantly faster — recommended for large batches."
        ),
    )
    mdl.add_argument("--no-if-ci", action="store_true", help="Skip LinearDRLearner (influence-function CIs) — faster")
    mdl.add_argument("--no-forest-drl", action="store_true", help="Skip ForestDRLearner (4th ensemble member) — faster")
    mdl.add_argument("--automl", action="store_true", help="Use FLAML AutoML nuisance models instead of LightGBM stack")

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Apply config overrides in the main process ────────────────────────────
    config.N_BOOT = args.n_boot
    config.N_CV_FOLDS = args.n_folds
    config.USE_BOOTSTRAP = not args.no_bootstrap
    config.USE_IF_CI = not args.no_if_ci
    config.USE_FOREST_DRL = not args.no_forest_drl
    config.USE_AUTOML = args.automl

    # Build a plain dict of all runtime overrides so that _worker can apply
    # the same changes inside each worker process.  ProcessPoolExecutor workers
    # import acic2026.config fresh from disk and would otherwise use the
    # on-disk defaults (e.g. USE_BOOTSTRAP=True) regardless of CLI flags.
    cfg_overrides = {
        "N_BOOT": config.N_BOOT,
        "N_CV_FOLDS": config.N_CV_FOLDS,
        "USE_BOOTSTRAP": config.USE_BOOTSTRAP,
        "USE_IF_CI": config.USE_IF_CI,
        "USE_FOREST_DRL": config.USE_FOREST_DRL,
        "USE_AUTOML": config.USE_AUTOML,
    }

    # Resolve worker count
    cpu_count = os.cpu_count() or 4
    if args.parallel:
        workers = args.workers if args.workers else cpu_count
    else:
        workers = args.workers or config.N_PARALLEL_DATASETS
    config.N_PARALLEL_DATASETS = workers

    # ── Single dataset ────────────────────────────────────────────────────────
    if args.data_id:
        process_dataset(
            data_id=args.data_id,
            team_id=args.team_id,
            subm_id=args.subm_id,
            run_timestamp=run_timestamp,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            plot_dir=args.plot_dir,
            inner_n_jobs=-1,
        )

    # ── Batch ─────────────────────────────────────────────────────────────────
    elif args.batch:
        if args.batch == "curated":
            ids = CURATED_IDS
        elif args.batch == "all":
            ids = [str(i) for i in range(1, 9001)]
        elif args.batch == "ids":
            if not args.ids:
                parser.error("--batch ids requires --ids ID [ID ...]")
            ids = args.ids
        else:
            parser.error(f"Unknown --batch value: {args.batch}")

        batch_fn = run_batch_parallel if args.parallel else run_batch
        results = batch_fn(
            data_ids=ids,
            team_id=args.team_id,
            subm_id=args.subm_id,
            run_timestamp=run_timestamp,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            plot_dir=args.plot_dir,
            **({"max_workers": workers, "cfg_overrides": cfg_overrides} if args.parallel else {}),
        )

        # Save timing summary CSV
        if results.get("timing_records"):
            # Build output directory
            timing_dir = Path("timing_summaries")
            timing_dir.mkdir(parents=True, exist_ok=True)

            # Build filename
            filename = f"timing_summary_{args.subm_id}_{run_timestamp}.csv"
            timing_path = timing_dir / filename
            
            # Save file
            pd.DataFrame(results["timing_records"]).to_csv(timing_path, index=False)

            print(f"\n  Timing summary saved → {timing_path}")

    else:
        parser.print_help()
