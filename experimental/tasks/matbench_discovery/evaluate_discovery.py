from __future__ import annotations
from pathlib import Path
import typer
from typing import Annotated
import pandas as pd
import os

from matbench_discovery import STABILITY_THRESHOLD
from matbench_discovery.data import df_wbm
from matbench_discovery.enums import MbdKey, DataFiles
from pymatgen.core import Structure
from pymatviz.enums import Key
from matbench_discovery.metrics.discovery import stable_metrics
from matbench_discovery.structure import symmetry
from matbench_discovery.metrics import geo_opt


PREDS_PATH = Path("FINAL_RESULTS/nequix.csv.gz")
GEO_OPT_PATH = Path("FINAL_RESULTS/nequix.jsonl.gz")
ID_COL = "material_id"
PRED_COLS = ("e_form_per_atom_fairchem",)

"""
    CPS config and constants (normalized performance score)
    The following constants except `KAPPA_SRME` are provided to compute CPS
"""
RMSD_BASELINE = 0.15    # Å
KAPPA_SRME = 0.446      # provided by user for this model
WEIGHT_F1 = 0.5
WEIGHT_KAPPA = 0.4
WEIGHT_RMSD = 0.1
SYMPREC = 1e-5


def _model_name_from_path(path: Path) -> str:
    name = path.name
    for suf in (".csv.gz", ".json.gz", ".csv", ".json"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name


def _format_table(full: dict[str, float], k10: dict[str, float], uniq: dict[str, float]) -> str:
    metric_order = [
        "F1",
        "DAF",
        "Precision",
        "Recall",
        "Accuracy",
        "TPR",
        "FPR",
        "TNR",
        "FNR",
        "TP",
        "FP",
        "TN",
        "FN",
        "MAE",
        "RMSE",
        "R2",
    ]

    rows = []
    for key in metric_order:
        v_full = full.get(key, float("nan"))
        v_10k = k10.get(key, float("nan"))
        v_uniq = uniq.get(key, float("nan"))
        rows.append((key, v_full, v_10k, v_uniq))

    # Determine column widths
    col1_width = max(len(r[0]) for r in rows)
    headers = ("full", "10k", "unique")
    colw = [
        max(len(headers[0]), max(len(f"{r[1]:.6f}") for r in rows)),
        max(len(headers[1]), max(len(f"{r[2]:.6f}") for r in rows)),
        max(len(headers[2]), max(len(f"{r[3]:.6f}") for r in rows)),
    ]

    # Build table string
    lines = []
    # Header aligned over numeric columns
    lines.append(
        f"{'':<{col1_width+1}}{headers[0]:>{colw[0]}}  {headers[1]:>{colw[1]}}  {headers[2]:>{colw[2]}}"
    )
    for metric, a, b, c in rows:
        lines.append(
            f"{metric:<{col1_width}}  {a:>{colw[0]}.6f}  {b:>{colw[1]}.6f}  {c:>{colw[2]}.6f}"
        )
    return "\n".join(lines)


def _normalize_f1(value: float | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_rmsd(value: float | None) -> float:
    if value is None:
        return 0.0
    try:
        rmsd = float(value)
    except Exception:
        return 0.0
    if rmsd <= 0:
        return 1.0
    if rmsd >= RMSD_BASELINE:
        return 0.0
    return (RMSD_BASELINE - rmsd) / RMSD_BASELINE


def _normalize_kappa_srme(value: float | None) -> float:
    if value is None:
        return 0.0
    try:
        kappa = float(value)
    except Exception:
        return 0.0
    return max(0.0, 1.0 - kappa / 2.0)


def _calculate_cps(f1: float | None, rmsd: float | None, kappa: float | None) -> float | None:
    # If any metric with non-zero weight is missing, return None
    if WEIGHT_F1 > 0 and (f1 is None or pd.isna(f1)):
        return None
    if WEIGHT_RMSD > 0 and (rmsd is None or pd.isna(rmsd)):
        return None
    if WEIGHT_KAPPA > 0 and (kappa is None or pd.isna(kappa)):
        return None

    total_weight = WEIGHT_F1 + WEIGHT_RMSD + WEIGHT_KAPPA
    if total_weight == 0:
        return 0.0

    weighted_sum = 0.0
    if WEIGHT_F1 > 0:
        weighted_sum += _normalize_f1(f1) * WEIGHT_F1
    if WEIGHT_RMSD > 0:
        weighted_sum += _normalize_rmsd(rmsd) * WEIGHT_RMSD
    if WEIGHT_KAPPA > 0:
        weighted_sum += _normalize_kappa_srme(kappa) * WEIGHT_KAPPA

    return weighted_sum / total_weight


def main(
    input_dir: Annotated[
        str, typer.Option(help="Input directory to `results.csv.gz` and `results.json.gz` files")
    ]
) -> None:
    input_csv_path = os.path.join(input_dir, 'results.csv.gz')
    input_json_path = os.path.join(input_dir, 'results.json.gz')
    
    print('Reading discovery csv file')
    df_preds = pd.read_csv(input_csv_path).set_index(ID_COL)
    
    # Infer prediction column and whether it's EACH or formation energy
    #pred_col = next((c for c in PRED_COLS if c in df_preds.columns), None)
    #if pred_col is None:
    #    raise SystemExit(f"None of {PRED_COLS} found in predictions file: {list(df_preds.columns)}")
    pred_col = None
    for column in list(df_preds.columns):
        if "e_form_per_atom" in column:
            pred_col = column
            break
    print('[discovery] Prediction column has: {}'.format(list(df_preds.columns)))
    print('[discovery] The column of prediction: {}'.format(pred_col))
    assert pred_col is not None

    col_type = "each" if pred_col == "each_pred" else "formation_energy"

    series_pred = df_preds[pred_col].astype(float).reindex(df_wbm.index)
    
    # Apply centralized model prediction cleaning criterion
    max_error_threshold = 5.0
    bad_mask = (abs(series_pred - df_wbm[MbdKey.e_form_dft])) > max_error_threshold
    series_pred.loc[bad_mask] = pd.NA
    print('[discovery] Number of unrealistic predictions: {}'.format(sum(bad_mask)))

    series_pred = series_pred.round(3)
    each_true = df_wbm[MbdKey.each_true].round(3)

    if col_type == "formation_energy":
        e_form_dft = df_wbm[MbdKey.e_form_dft].round(3)
        each_pred = each_true + series_pred - e_form_dft
    else:
        each_pred = series_pred

    # Discovery metrics
    uniq_proto_prevalence = (
        df_wbm.query(MbdKey.uniq_proto)[MbdKey.each_true] <= STABILITY_THRESHOLD
    ).mean()
    #   Full
    metrics_full = stable_metrics(each_true, each_pred, stability_threshold=STABILITY_THRESHOLD, fillna=True)
    #   Unique
    uniq_idx = df_wbm.query(MbdKey.uniq_proto).index
    metrics_uniq = stable_metrics(
        each_true.loc[uniq_idx],
        each_pred.loc[uniq_idx],
        stability_threshold=STABILITY_THRESHOLD,
        fillna=True,
    )
    metrics_uniq['DAF'] = metrics_uniq['Precision'] / uniq_proto_prevalence
    #   10K
    top10k_idx = each_pred.loc[uniq_idx].nsmallest(10_000).index
    metrics_10k = stable_metrics(
        each_true.loc[top10k_idx],
        each_pred.loc[top10k_idx],
        stability_threshold=STABILITY_THRESHOLD,
        fillna=True,
    )
    print(metrics_10k)
    metrics_10k['DAF'] = metrics_10k['Precision'] / uniq_proto_prevalence
    table_str = _format_table(metrics_full, metrics_10k, metrics_uniq)
    print(table_str)

    # --- Batch RMSD vs DFT reference structures ---
    print('Reading geometry optimization json file')
    df_geo_opt = pd.read_json(input_json_path, lines=True).set_index(ID_COL)
    # Detect structure column in model geo-opt JSONL
    struct_cols = [c for c in df_geo_opt.columns if "structure" in c]
    if not struct_cols:
        raise SystemExit(
            f"No structure-like column found in {input_json_path}. Columns: {list(df_geo_opt.columns)}"
        )
    print('[geo_opt] Structure column `struct_cols` has: {}'.format(struct_cols))
    struct_col = struct_cols[0]
    print('[geo_opt] The column of prediction: {}'.format(struct_col))

    # Convert predicted structures to pymatgen.Structure
    def _to_structure(obj: dict | Structure) -> Structure:
        if isinstance(obj, Structure):
            return obj
        # Some rows may store a ComputedStructureEntry-like dict with 'structure'
        if isinstance(obj, dict) and Key.structure in obj:
            obj = obj[Key.structure]
        return Structure.from_dict(obj)  # type: ignore[arg-type]

    print('[geo_opt] Getting predicted structures')
    pred_structs: dict[str, Structure] = (
        df_geo_opt[struct_col].map(_to_structure).to_dict()
    )

    # Load DFT reference structures once
    print('[geo_opt] Reading DFT reference structures')
    df_wbm_structs = pd.read_json(
        DataFiles.wbm_computed_structure_entries.path, lines=True, orient="records"
    ).set_index(Key.mat_id)
    ref_structs: dict[str, Structure] = {
        mid: Structure.from_dict(cse[Key.structure])
        for mid, cse in df_wbm_structs[Key.computed_structure_entry].items()
    }

    # Compute symmetry info and RMSD using project utilities
    print('[geo_opt] Computing the symmetry info of predicted structures')
    df_sym_pred = symmetry.get_sym_info_from_structs(
        pred_structs, pbar=True, symprec=SYMPREC
    )

    # Try to load precomputed DFT symmetry analysis; fallback to compute
    dft_sym_path = Path(DataFiles.wbm_dft_geo_opt_symprec_1e_5.path)
    if dft_sym_path.is_file():
        df_sym_ref = pd.read_csv(dft_sym_path, index_col=0)
    else:
        df_sym_ref = symmetry.get_sym_info_from_structs(
            ref_structs, pbar=True, symprec=SYMPREC
        )

    print('[geo_opt] Comparing predicted and reference structures')
    df_compare = symmetry.pred_vs_ref_struct_symmetry(
        df_sym_pred, df_sym_ref, pred_structs, ref_structs, pbar=True
    )
    """
    series_rmsd = pd.to_numeric(
        df_compare[MbdKey.structure_rmsd_vs_dft], errors="coerce"
    ).dropna()
    print(
        f"RMSD computed for {len(series_rmsd):,}/{len(pred_structs):,} structures. "
        f"median={series_rmsd.median():.4f} Å, mean={series_rmsd.mean():.4f} Å"
    )
    """
    # Calculate geometry optimization metrics (matches analyze_model_symprec)
    metrics_all = geo_opt.calc_geo_opt_metrics(df_compare)
    print(
        "Geo-opt metrics (all): "
        f"RMSD_mean={metrics_all[str(MbdKey.structure_rmsd_vs_dft)]:.4f}, "
        f"n_sym_ops_mae={metrics_all[str(Key.n_sym_ops_mae)]:.4f}, "
        f"sym_match={metrics_all[str(Key.symmetry_match)]:.3f}"
    )

    """
    # Compute normalized performance score (CPS) using UNIQUE prototype metrics
    f1_uniq = metrics_uniq.get("F1")
    # Use actual structural RMSD over unique prototypes (median for robustness)
    rmsd_uniq_series = series_rmsd.reindex(uniq_idx).dropna()
    rmsd_uniq_value = (
        float(rmsd_uniq_series.mean()) if not rmsd_uniq_series.empty else None
    )
    cps_uniq = _calculate_cps(f1_uniq, rmsd_uniq_value, KAPPA_SRME)
    if cps_uniq is not None:
        if rmsd_uniq_value is not None:
            print(
                f"\nCPS (unique prototypes): {cps_uniq:.6f}  "
                f"[RMSD_unique median={rmsd_uniq_value:.4f} Å]"
            )
        else:
            print(f"\nCPS (unique prototypes): {cps_uniq:.6f}  [RMSD_unique: n/a]")
    else:
        print("\nCPS (unique prototypes): unavailable (missing metrics)")

    # print(_calculate_cps(0.786, 0.084, 0.408)) # eqnorm test
    """


if __name__ == "__main__":
    typer.run(main)