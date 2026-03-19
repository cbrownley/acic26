"""
main.py
Entry point.  Orchestrates the full pipeline for one or many datasets.

v3 additions
------------
- evaluate.compute_gain_curves / evaluate_aucs called after iCATE estimation
- visualizations.plot_all_single_dataset called to produce PNGs
- AUC results stored in a per-dataset CSV (aucs_dataID_teamID_submID.csv)
- run_batch and run_batch_parallel accumulate AUC records so callers can
  pass them to the multi-dataset visualizations after the batch completes

Usage
-----
    # Single dataset (CLI):
    python -m acic2026.main 0001 XXXX 1

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
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import config as config
from data       import load_and_split, build_preprocessor
from evaluate   import cv_evaluate, compute_gain_curves, evaluate_aucs
from estimators import (
    fit_drlearner, fit_linear_drlearner, fit_causal_forest,
)
from inference  import (
    get_icates, compute_scate, compute_subcate, compute_pate,
)
from outputs    import (
    build_icate_df, best_icate, best_scalar, best_subcate, save,
)
import visualizations as viz

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Curated 18-dataset IDs
# ─────────────────────────────────────────────────────────────────────────────
CURATED_IDS = [f"{i:04d}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]


# ─────────────────────────────────────────────────────────────────────────────
# Single-dataset pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(
    data_id:      str,
    team_id:      str  = config.TEAM_ID,
    subm_id:      str  = config.SUBM_ID,
    data_dir             = config.DATA_DIR,
    out_dir               = config.OUT_DIR,
    plot_dir              = None,            # None = skip plots
    inner_n_jobs: int    = config.INNER_N_JOBS,
) -> dict:
    """
    Run the complete pipeline for one dataset.

    Produces
    --------
    8 competition CSV files (iCATE, sCATE, subCATE, PATE, BEST_*)
    1 AUC CSV file           (aucs_dataID_teamID_submID.csv)
    4 PNG plots              (if plot_dir is set)

    Returns
    -------
    dict with keys:
        'icate', 'scate', 'subcate', 'pate'  — estimand DataFrames
        'aucs'                                — {z: float} AURC per arm
        'curves'                              — {z: fklearn DataFrame}
    """
    config.INNER_N_JOBS = inner_n_jobs

    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Dataset {data_id}  |  team {team_id}  |  submission {subm_id}")
    print(f"  AutoML={config.USE_AUTOML}  IF_CI={config.USE_IF_CI}"
          f"  inner_n_jobs={inner_n_jobs}")
    print(sep)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    path = data_dir / f"data_{data_id}.csv"
    ID, Y, T, X_raw, x12 = load_and_split(path)
    n = len(Y)
    print(f"  n={n}  p={X_raw.shape[1]}  arms={sorted(set(T))}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    preprocessor = build_preprocessor(X_raw)
    X_feat       = preprocessor.transform(X_raw)

    # ── 3. CV quality check ───────────────────────────────────────────────────
    cv_evaluate(X_feat, Y, T)

    # ── 4. Fit CATE estimators ────────────────────────────────────────────────
    est_drl, drl_has_ci = fit_drlearner(X_feat, Y, T)
    est_lin = fit_linear_drlearner(X_feat, Y, T) if config.USE_IF_CI else None
    forests = fit_causal_forest(X_feat, Y, T)

    # ── 5. Variance-weighted ensemble iCATE ───────────────────────────────────
    icates, lowers, uppers = get_icates(
        est_drl, drl_has_ci, est_lin, forests, X_feat, n)

    # ── 6. Aggregate estimands ────────────────────────────────────────────────
    icate_df   = build_icate_df(icates, lowers, uppers, ID)
    scate_df   = compute_scate(icates, lowers, uppers, n)
    subcate_df = compute_subcate(icates, lowers, uppers, x12, n)
    pate_df    = compute_pate(est_drl, drl_has_ci, est_lin, X_feat)

    # ── 7. Gain curves and AUCs ───────────────────────────────────────────────
    print("\n  [AUC] Computing relative cumulative gain curves …")
    curves = {}
    aucs   = {}
    try:
        curves = compute_gain_curves(icates, T, Y)
        aucs   = evaluate_aucs(icates, T, Y)
    except ImportError as e:
        print(f"  [AUC] fklearn not installed — skipping: {e}")

    # Save AUC CSV
    if aucs:
        aucs_df = pd.DataFrame([
            {"z": z, "AURC": v} for z, v in aucs.items()
        ])
        save(aucs_df, f"aucs_{data_id}_{team_id}_{subm_id}.csv", out_dir)

    # ── 8. Best-treatment files ───────────────────────────────────────────────
    best_icate_df   = best_icate(icates, ID)
    best_scate_df   = best_scalar(scate_df)
    best_subcate_df = best_subcate(subcate_df)
    best_pate_df    = best_scalar(pate_df)

    # ── 9. Save competition CSVs ──────────────────────────────────────────────
    d, t, s = data_id, team_id, subm_id
    print()
    save(icate_df,        f"iCATE_{d}_{t}_{s}.csv",        out_dir)
    save(scate_df,        f"sCATE_{d}_{t}_{s}.csv",        out_dir)
    save(subcate_df,      f"subCATE_{d}_{t}_{s}.csv",      out_dir)
    save(pate_df,         f"PATE_{d}_{t}_{s}.csv",         out_dir)
    save(best_icate_df,   f"BEST_iCATE_{d}_{t}_{s}.csv",   out_dir)
    save(best_scate_df,   f"BEST_sCATE_{d}_{t}_{s}.csv",   out_dir)
    save(best_subcate_df, f"BEST_subCATE_{d}_{t}_{s}.csv", out_dir)
    save(best_pate_df,    f"BEST_PATE_{d}_{t}_{s}.csv",    out_dir)

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    if plot_dir is not None:
        print(f"\n  [VIZ] Generating plots → {plot_dir}")
        viz.plot_all_single_dataset(
            data_id    = data_id,
            icate_df   = icate_df,
            scate_df   = scate_df,
            subcate_df = subcate_df,
            pate_df    = pate_df,
            curves_dict= curves,
            aucs_dict  = aucs,
            save_dir   = plot_dir,
        )

    # ── 11. Console summary ───────────────────────────────────────────────────
    print("\n  sCATE summary:")
    print(scate_df.to_string(index=False))
    print(f"\n  Best sCATE : {best_scate_df['best_z'].iloc[0]}")
    print(f"  Best PATE  : {best_pate_df['best_z'].iloc[0]}")
    if aucs:
        print("  AURC by arm: " +
              "  ".join(f"{z}={v:.4f}" for z, v in aucs.items()))
    print(f"\nDataset {data_id} complete.\n")

    return {
        "icate":   icate_df,
        "scate":   scate_df,
        "subcate": subcate_df,
        "pate":    pate_df,
        "aucs":    aucs,
        "curves":  curves,
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
        rows.append({
            "data_id":    data_id,
            "z":          z,
            "sCATE":      s["Estimate"],
            "sCATE_L95":  s["L95"],
            "sCATE_U95":  s["U95"],
            "PATE":       p["Estimate"],
            "PATE_L95":   p["L95"],
            "PATE_U95":   p["U95"],
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sequential batch
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    data_ids,
    team_id:  str  = config.TEAM_ID,
    subm_id:  str  = config.SUBM_ID,
    data_dir         = config.DATA_DIR,
    out_dir           = config.OUT_DIR,
    plot_dir          = None,
) -> dict:
    """
    Process datasets sequentially.

    Returns
    -------
    dict with keys:
        'results'             — {data_id: result_dict}
        'auc_records'         — list of dicts for plot_auc_heatmap
        'scate_records'       — list of dicts for plot_scate_across_datasets
        'pate_vs_scate_records' — list for plot_pate_vs_scate
        'icate_dfs'           — {data_id: icate_df} for violin grid
    """
    ids              = list(data_ids)
    results          = {}
    auc_records      = []
    scate_records    = []
    pate_vs_scate    = []
    icate_dfs        = {}

    for idx, did in enumerate(ids, 1):
        print(f"\n[{idx}/{len(ids)}] dataset {did}")
        try:
            r = process_dataset(did, team_id, subm_id, data_dir, out_dir,
                                plot_dir=plot_dir, inner_n_jobs=-1)
            results[did]   = r
            icate_dfs[did] = r["icate"]
            auc_records.append(_make_auc_record(did, r["aucs"]))
            scate_records.extend(_make_scate_records(did, r["scate"]))
            pate_vs_scate.extend(
                _make_pate_vs_scate_records(did, r["scate"], r["pate"]))
        except Exception:
            traceback.print_exc()
            print(f"[ERROR] Dataset {did} failed — skipping.")

    return {
        "results":               results,
        "auc_records":           auc_records,
        "scate_records":         scate_records,
        "pate_vs_scate_records": pate_vs_scate,
        "icate_dfs":             icate_dfs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parallel batch
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> tuple:
    """Picklable top-level worker — returns (data_id, result_dict | None, err)."""
    data_id, team_id, subm_id, data_dir, out_dir, plot_dir, inner_n_jobs = args
    try:
        r = process_dataset(data_id, team_id, subm_id, data_dir, out_dir,
                            plot_dir=plot_dir, inner_n_jobs=inner_n_jobs)
        return data_id, r, None
    except Exception:
        return data_id, None, traceback.format_exc()


def run_batch_parallel(
    data_ids,
    team_id:     str  = config.TEAM_ID,
    subm_id:     str  = config.SUBM_ID,
    data_dir           = config.DATA_DIR,
    out_dir             = config.OUT_DIR,
    plot_dir            = None,
    max_workers: int   = config.N_PARALLEL_DATASETS,
) -> dict:
    """
    Process datasets concurrently (ProcessPoolExecutor).

    Returns the same dict structure as run_batch.
    """
    ids        = list(data_ids)
    total      = len(ids)
    cpu_count  = os.cpu_count() or 4
    inner_jobs = max(1, cpu_count // max(max_workers, 1))
    print(f"\n[PARALLEL] {total} datasets · {max_workers} workers · "
          f"{inner_jobs} inner_n_jobs  (machine: {cpu_count} CPUs)")

    job_args = [
        (did, team_id, subm_id,
         str(data_dir), str(out_dir),
         str(plot_dir) if plot_dir else None,
         inner_jobs)
        for did in ids
    ]

    auc_records   = []
    scate_records = []
    pate_vs_scate = []
    icate_dfs     = {}
    all_results   = {}
    completed     = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, a): a[0] for a in job_args}
        for fut in as_completed(futures):
            did = futures[fut]
            completed += 1
            try:
                did_ret, r, err = fut.result()
            except Exception as exc:
                err = str(exc)
                r   = None

            if r is not None and err is None:
                all_results[did] = r
                icate_dfs[did]   = r["icate"]
                auc_records.append(_make_auc_record(did, r["aucs"]))
                scate_records.extend(_make_scate_records(did, r["scate"]))
                pate_vs_scate.extend(
                    _make_pate_vs_scate_records(did, r["scate"], r["pate"]))
                status = "OK"
            else:
                status = "FAILED"
                if err:
                    print(f"    {err.splitlines()[-1]}")

            print(f"  [{completed}/{total}] dataset {did} → {status}")

    n_ok = len(all_results)
    print(f"\n[PARALLEL] Done. {n_ok} succeeded, {total-n_ok} failed.")
    return {
        "results":               all_results,
        "auc_records":           auc_records,
        "scate_records":         scate_records,
        "pate_vs_scate_records": pate_vs_scate,
        "icate_dfs":             icate_dfs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args     = sys.argv[1:]
    data_id  = args[0] if len(args) > 0 else "1"
    team_id  = args[1] if len(args) > 1 else config.TEAM_ID
    subm_id  = args[2] if len(args) > 2 else config.SUBM_ID
    plot_dir = args[3] if len(args) > 3 else "plots"
    process_dataset(data_id, team_id, subm_id, plot_dir=plot_dir)
