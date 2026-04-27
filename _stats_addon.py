"""Supplementary statistics add-on.

Computes inferential statistics on top of the pre-existing grid-based XGBoost
outputs for sugar maple (Supplementary 5) and red maple (Supplementary 6), and
assembles cross-document summaries used by Supplementary 7.

All outputs are CSVs under outputs/{sugar,red}/stats/ and outputs/summary/stats/.
o
Run with the project Python 3.10 environment:
    python _stats_addon.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from acer_grid_conifer_analysis import (
    BLOCK_SIZE,
    CONIFER_DISTANCE_PREDICTORS,
    CONIFER_KERNEL_PREDICTORS,
    ENVIRONMENTAL_PREDICTORS,
    MODEL_B_PREDICTORS,
    ProjectPaths,
    SPECIES_CONFIG,
    add_spatial_blocks,
)

PROJECT_DIR = Path(__file__).resolve().parent
PATHS = ProjectPaths.from_project_dir(PROJECT_DIR)
SUMMARY_STATS = PATHS.outputs_dir / "summary" / "stats"
SUMMARY_STATS.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + conf / 2.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of a ROC-AUC estimate (Hanley and McNeil 1982)."""
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    ) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def auc_ci(auc: float, n_pos: int, n_neg: int, conf: float = 0.95) -> tuple[float, float, float]:
    se = hanley_mcneil_se(auc, n_pos, n_neg)
    z = stats.norm.ppf(0.5 + conf / 2.0)
    return se, max(0.0, auc - z * se), min(1.0, auc + z * se)


def mcnemar_test(b: int, c: int) -> tuple[float, float, str]:
    """McNemar's test with continuity correction (chi-square approximation).

    When b + c < 25 we fall back to the exact binomial two-sided p-value, which
    is the recommended small-sample alternative.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0, "no-discordant-pairs"
    if n < 25:
        k = min(b, c)
        p = 2 * stats.binom.cdf(k, n, 0.5)
        return float("nan"), float(min(1.0, p)), "exact-binomial"
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p), "chi2-cc"


def paired_bootstrap_auc(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_boot: int = 2000,
) -> tuple[float, float, float, float]:
    """Bootstrap distribution of (AUC_B - AUC_A) with a paired resample.

    Returns: delta_mean, delta_lo, delta_hi, two-sided p-value.
    """
    n = len(y_true)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            deltas[i] = np.nan
            continue
        auc_a = roc_auc_score(yt, prob_a[idx])
        auc_b = roc_auc_score(yt, prob_b[idx])
        deltas[i] = auc_b - auc_a
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    mean_delta = float(np.mean(deltas))
    p = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return mean_delta, float(lo), float(hi), float(p)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR-adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(ranked, 1.0)
    return q


# ---------------------------------------------------------------------------
# Per-species pipeline
# ---------------------------------------------------------------------------
def regenerate_fold_predictions(species_key: str) -> dict:
    cfg = SPECIES_CONFIG[species_key]
    env_csv = PATHS.model_inputs_dir / cfg["env_training_csv"] if "env_training_csv" in cfg else PATHS.model_inputs_dir / f"grid_training_{species_key}_env_only.csv"
    conifer_csv = PATHS.model_inputs_dir / cfg["conifer_training_csv"]
    env_df = pd.read_csv(env_csv)
    con_df = pd.read_csv(conifer_csv)
    # Both tables share sample_ids and spatial_block.
    if "spatial_block" not in env_df.columns:
        env_df = add_spatial_blocks(env_df)
    if "spatial_block" not in con_df.columns:
        con_df = add_spatial_blocks(con_df)
    response_col = cfg["response_column"]
    y_env = env_df[response_col].to_numpy()
    y_con = con_df[response_col].to_numpy()
    groups_env = env_df["spatial_block"].to_numpy()
    groups_con = con_df["spatial_block"].to_numpy()

    # Load trained models (already fit on full data). For paired per-fold
    # predictions we refit on training-fold only to get genuine held-out
    # probabilities; retraining the two-fold CV is fast (<1s per species).
    from xgboost import XGBClassifier
    from acer_grid_conifer_analysis import MODEL_PARAMS

    gkf = GroupKFold(n_splits=2)
    fold_rows = []
    all_held = {"y": [], "prob_a": [], "prob_b": [], "fold": []}

    for fold_idx, ((tr_e, te_e), (tr_c, te_c)) in enumerate(
        zip(gkf.split(env_df, y_env, groups_env), gkf.split(con_df, y_con, groups_con)),
        start=1,
    ):
        # Both splits should give identical indices because sample_id and
        # spatial_block are aligned.
        Xa_tr = env_df.loc[tr_e, ENVIRONMENTAL_PREDICTORS]
        Xa_te = env_df.loc[te_e, ENVIRONMENTAL_PREDICTORS]
        Xb_tr = con_df.loc[tr_c, MODEL_B_PREDICTORS]
        Xb_te = con_df.loc[te_c, MODEL_B_PREDICTORS]
        ya_te = y_env[te_e]
        yb_te = y_con[te_c]
        # Align test rows across A and B by sample_id so McNemar is valid.
        sid_a = env_df.loc[te_e, "sample_id"].to_numpy()
        sid_b = con_df.loc[te_c, "sample_id"].to_numpy()
        common = np.intersect1d(sid_a, sid_b)
        mask_a = np.isin(sid_a, common)
        mask_b = np.isin(sid_b, common)

        model_a = XGBClassifier(**MODEL_PARAMS)
        model_a.fit(Xa_tr, y_env[tr_e])
        model_b = XGBClassifier(**MODEL_PARAMS)
        model_b.fit(Xb_tr, y_con[tr_c])

        proba_a_full = model_a.predict_proba(Xa_te)[:, 1]
        proba_b_full = model_b.predict_proba(Xb_te)[:, 1]

        # Paired arrays (same sample_ids, same order)
        order_a = np.argsort(sid_a[mask_a])
        order_b = np.argsort(sid_b[mask_b])
        proba_a = proba_a_full[mask_a][order_a]
        proba_b = proba_b_full[mask_b][order_b]
        y_paired = ya_te[mask_a][order_a]
        assert np.array_equal(y_paired, yb_te[mask_b][order_b])

        pred_a = (proba_a >= 0.5).astype(int)
        pred_b = (proba_b >= 0.5).astype(int)

        for name, y_, p_, pr_ in [
            ("Model A", y_paired, proba_a, pred_a),
            ("Model B", y_paired, proba_b, pred_b),
        ]:
            auc = roc_auc_score(y_, p_)
            pr = average_precision_score(y_, p_)
            brier = brier_score_loss(y_, p_)
            n_pos = int(y_.sum())
            n_neg = int(len(y_) - n_pos)
            tp = int(((pr_ == 1) & (y_ == 1)).sum())
            fp = int(((pr_ == 1) & (y_ == 0)).sum())
            tn = int(((pr_ == 0) & (y_ == 0)).sum())
            fn = int(((pr_ == 0) & (y_ == 1)).sum())
            se = tp / max(1, tp + fn)
            sp = tn / max(1, tn + fp)
            se_se = hanley_mcneil_se(auc, n_pos, n_neg)
            z95 = 1.96
            se_lo, se_hi = wilson_ci(tp, tp + fn)
            sp_lo, sp_hi = wilson_ci(tn, tn + fp)
            fold_rows.append(
                dict(
                    species=species_key,
                    fold=fold_idx,
                    model=name,
                    n=len(y_),
                    n_pos=n_pos,
                    n_neg=n_neg,
                    roc_auc=auc,
                    roc_auc_se=se_se,
                    roc_auc_lo=max(0.0, auc - z95 * se_se),
                    roc_auc_hi=min(1.0, auc + z95 * se_se),
                    pr_auc=pr,
                    brier=brier,
                    sensitivity=se,
                    sensitivity_lo=se_lo,
                    sensitivity_hi=se_hi,
                    specificity=sp,
                    specificity_lo=sp_lo,
                    specificity_hi=sp_hi,
                    tss=se + sp - 1,
                )
            )

        all_held["y"].append(y_paired)
        all_held["prob_a"].append(proba_a)
        all_held["prob_b"].append(proba_b)
        all_held["fold"].append(np.full(len(y_paired), fold_idx, dtype=int))

    fold_df = pd.DataFrame(fold_rows)
    fold_out = PATHS.outputs_dir / species_key / "stats"
    fold_out.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(fold_out / f"fold_metrics_with_ci_{species_key}.csv", index=False)

    y_all = np.concatenate(all_held["y"])
    pa = np.concatenate(all_held["prob_a"])
    pb = np.concatenate(all_held["prob_b"])

    # Pooled AUC + CI
    rows = []
    for name, p_ in [("Model A", pa), ("Model B", pb)]:
        auc = roc_auc_score(y_all, p_)
        n_pos = int(y_all.sum())
        n_neg = int(len(y_all) - n_pos)
        se, lo, hi = auc_ci(auc, n_pos, n_neg)
        rows.append(
            dict(
                species=species_key,
                model=name,
                n=len(y_all),
                n_pos=n_pos,
                n_neg=n_neg,
                roc_auc=auc,
                roc_auc_se=se,
                roc_auc_lo=lo,
                roc_auc_hi=hi,
            )
        )
    pooled_df = pd.DataFrame(rows)
    pooled_df.to_csv(fold_out / f"pooled_auc_{species_key}.csv", index=False)

    # Paired bootstrap Delta-AUC
    d_mean, d_lo, d_hi, d_p = paired_bootstrap_auc(y_all, pa, pb, n_boot=2000)
    # McNemar on concatenated held-out predictions
    pred_a = (pa >= 0.5).astype(int)
    pred_b = (pb >= 0.5).astype(int)
    b = int(((pred_a != y_all) & (pred_b == y_all)).sum())
    c = int(((pred_a == y_all) & (pred_b != y_all)).sum())
    chi2, pval, flavour = mcnemar_test(b, c)
    model_comp = pd.DataFrame(
        [
            dict(
                species=species_key,
                test="Paired bootstrap (Delta ROC-AUC, B - A)",
                statistic=d_mean,
                ci_low=d_lo,
                ci_high=d_hi,
                p_value=d_p,
                n=len(y_all),
                method="paired bootstrap, 2000 resamples",
            ),
            dict(
                species=species_key,
                test="McNemar (correct/incorrect at threshold 0.5)",
                statistic=chi2,
                ci_low=np.nan,
                ci_high=np.nan,
                p_value=pval,
                n=b + c,
                method=flavour,
                extra=json.dumps({"b_A_wrong_B_right": b, "c_A_right_B_wrong": c}),
            ),
        ]
    )
    model_comp.to_csv(fold_out / f"model_comparison_tests_{species_key}.csv", index=False)

    # ------------------------------------------------------------------
    # Permutation-importance z-tests (Model B)
    # ------------------------------------------------------------------
    imp = pd.read_csv(PATHS.outputs_dir / species_key / f"feature_importance_{species_key}_env_conifer.csv")
    imp = imp.copy()
    eps = 1e-12
    imp["perm_z"] = imp["permutation_mean"] / (imp["permutation_std"].replace(0, eps))
    imp["perm_p_two_sided"] = 2 * (1 - stats.norm.cdf(np.abs(imp["perm_z"])))
    imp["perm_p_fdr"] = bh_fdr(imp["perm_p_two_sided"].fillna(1.0).to_numpy())
    imp_cols = [
        "feature",
        "gain_importance",
        "weight_importance",
        "shap_mean_abs",
        "permutation_mean",
        "permutation_std",
        "perm_z",
        "perm_p_two_sided",
        "perm_p_fdr",
    ]
    imp[imp_cols].to_csv(fold_out / f"feature_importance_tests_{species_key}.csv", index=False)

    # ------------------------------------------------------------------
    # Conifer covariate Mann-Whitney U (presence vs background) on Model B table
    # ------------------------------------------------------------------
    mw_rows = []
    for col in CONIFER_KERNEL_PREDICTORS + CONIFER_DISTANCE_PREDICTORS:
        pres = con_df.loc[y_con == 1, col].to_numpy()
        back = con_df.loc[y_con == 0, col].to_numpy()
        u, p = stats.mannwhitneyu(pres, back, alternative="two-sided")
        n1, n2 = len(pres), len(back)
        # rank-biserial effect size r = 1 - 2U/(n1*n2) (alternative='less')
        r = 1 - 2 * u / (n1 * n2)
        # z from U under tie-corrected large-sample approximation
        mu_u = n1 * n2 / 2
        sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (u - mu_u) / sigma_u
        mw_rows.append(
            dict(
                species=species_key,
                covariate=col,
                median_presence=float(np.median(pres)),
                median_background=float(np.median(back)),
                U=float(u),
                z=float(z),
                p_value=float(p),
                effect_r_rank_biserial=float(r),
                n_presence=n1,
                n_background=n2,
            )
        )
    mw_df = pd.DataFrame(mw_rows)
    mw_df["p_value_fdr"] = bh_fdr(mw_df["p_value"].to_numpy())
    mw_df.to_csv(fold_out / f"conifer_mannwhitney_{species_key}.csv", index=False)

    # ------------------------------------------------------------------
    # Future projections: Wilcoxon signed-rank env_only vs static-conifer
    # ------------------------------------------------------------------
    fut = pd.read_csv(PATHS.outputs_dir / "summary" / "future_projection_summary.csv")
    fut_sp = fut[fut["species"] == species_key].copy()
    pivot = fut_sp.pivot(index="scenario", columns="model", values="area_change_pct")
    pivot_lat = fut_sp.pivot(index="scenario", columns="model", values="lat_shift_km")
    wr_rows = []
    for metric, piv in [("area_change_pct", pivot), ("lat_shift_km", pivot_lat)]:
        a = piv["env_only"].to_numpy()
        b2 = piv["env_static_conifer"].to_numpy()
        w, p = stats.wilcoxon(a, b2, zero_method="wilcox", alternative="two-sided")
        diffs = a - b2
        wr_rows.append(
            dict(
                species=species_key,
                metric=metric,
                n_scenarios=len(a),
                median_env_only=float(np.median(a)),
                median_static_conifer=float(np.median(b2)),
                median_difference=float(np.median(diffs)),
                W=float(w),
                p_value=float(p),
            )
        )
    wilcox_df = pd.DataFrame(wr_rows)
    wilcox_df.to_csv(fold_out / f"future_wilcoxon_{species_key}.csv", index=False)

    return dict(fold=fold_df, pooled=pooled_df, model=model_comp, imp=imp, mw=mw_df, wilcox=wilcox_df)


# ---------------------------------------------------------------------------
# Cross-document synthesis (for Supplementary 7)
# ---------------------------------------------------------------------------
def build_s7_tables(per_species: dict) -> None:
    # S1 — LightGBM-level feature selection summary: use predictor_summary.csv
    # already produced + cross-reference.

    # S7 synthesis: combine permutation p, Mann-Whitney p, Wilcoxon p, AUC CIs.
    rows = []
    for sp, blob in per_species.items():
        pooled = blob["pooled"]
        for _, row in pooled.iterrows():
            rows.append(
                dict(
                    supplementary="S5" if sp == "sugar" else "S6",
                    species=sp,
                    component=row["model"],
                    statistic="ROC-AUC (pooled, 2-fold CV held-out)",
                    value=row["roc_auc"],
                    ci_low=row["roc_auc_lo"],
                    ci_high=row["roc_auc_hi"],
                    p_value=np.nan,
                    note=f"Hanley-McNeil 95% CI, n={int(row['n'])}",
                )
            )
        mdl = blob["model"]
        for _, row in mdl.iterrows():
            rows.append(
                dict(
                    supplementary="S5" if sp == "sugar" else "S6",
                    species=sp,
                    component="Model A vs Model B",
                    statistic=row["test"],
                    value=row["statistic"],
                    ci_low=row.get("ci_low", np.nan),
                    ci_high=row.get("ci_high", np.nan),
                    p_value=row["p_value"],
                    note=row["method"],
                )
            )
        # Top conifer predictor permutation p
        imp = blob["imp"].sort_values("shap_mean_abs", ascending=False)
        for _, row in imp[imp["feature"].isin(CONIFER_KERNEL_PREDICTORS + CONIFER_DISTANCE_PREDICTORS)].iterrows():
            rows.append(
                dict(
                    supplementary="S5" if sp == "sugar" else "S6",
                    species=sp,
                    component=row["feature"],
                    statistic="Permutation importance z",
                    value=row["perm_z"],
                    ci_low=np.nan,
                    ci_high=np.nan,
                    p_value=row["perm_p_two_sided"],
                    note=f"BH-FDR q = {row['perm_p_fdr']:.3g}; mean |SHAP| = {row['shap_mean_abs']:.3g}",
                )
            )
        # Wilcoxon future
        for _, row in blob["wilcox"].iterrows():
            rows.append(
                dict(
                    supplementary="S5" if sp == "sugar" else "S6",
                    species=sp,
                    component=f"env_only vs static-conifer ({row['metric']})",
                    statistic="Wilcoxon signed-rank W",
                    value=row["W"],
                    ci_low=np.nan,
                    ci_high=np.nan,
                    p_value=row["p_value"],
                    note=f"n scenarios = {int(row['n_scenarios'])}, median diff = {row['median_difference']:.3g}",
                )
            )
        # Mann-Whitney top conifer covariates (limit to the three distances, most interpretable)
        for _, row in blob["mw"].iterrows():
            rows.append(
                dict(
                    supplementary="S5" if sp == "sugar" else "S6",
                    species=sp,
                    component=row["covariate"],
                    statistic="Mann-Whitney U (presence vs background)",
                    value=row["U"],
                    ci_low=np.nan,
                    ci_high=np.nan,
                    p_value=row["p_value"],
                    note=f"z = {row['z']:.2f}, rank-biserial r = {row['effect_r_rank_biserial']:.3f}, BH-FDR q = {row['p_value_fdr']:.3g}",
                )
            )

    synth = pd.DataFrame(rows)
    synth.to_csv(SUMMARY_STATS / "integrated_inference_summary.csv", index=False)

    # S1 — PCA + LightGBM summary (descriptive; no inference)
    pred = pd.read_csv(PATHS.outputs_dir / "summary" / "predictor_summary.csv")
    pred.to_csv(SUMMARY_STATS / "s1_predictor_selection_summary.csv", index=False)

    # S2 — K-means metrics table (for S7.3 in the new Supplementary 7)
    km_path = PROJECT_DIR / "data" / "kmeans_metrics_ref.csv"
    if km_path.exists():
        km = pd.read_csv(km_path)
        km.to_csv(SUMMARY_STATS / "s2_kmeans_metrics.csv", index=False)
    km_final = PROJECT_DIR / "data" / "kmeans_final_metrics.csv"
    if km_final.exists():
        pd.read_csv(km_final).to_csv(SUMMARY_STATS / "s2_kmeans_final.csv", index=False)
    comp = PROJECT_DIR / "data" / "cluster_composition_by_essence_k4.csv"
    if comp.exists():
        pd.read_csv(comp).to_csv(SUMMARY_STATS / "s2_cluster_composition.csv", index=False)

    # S4 — Species-level descriptives summary
    stats_path = PROJECT_DIR / "data" / "statistics_by_species.csv"
    if stats_path.exists():
        stats_path = stats_path  # forwarded by S4 notebook

    # S4 — between-species Mann-Whitney U tests on a handful of bioclim variables
    build_s4_species_tests()

    # S3 — GAM inference: use values reported in the manuscript text to build
    # a reference table; the underlying R notebook overwrites this file if run.
    gam_path = SUMMARY_STATS / "s3_gam_inference_manuscript.csv"
    if not gam_path.exists():
        gam_rows = [
            dict(term="forest", abs_z=93.9, p_value="<0.0001", direction="negative on A. rubrum odds"),
            dict(term="elevation", abs_z=92.9, p_value="<0.0001", direction="negative on A. rubrum odds"),
            dict(term="bio1 (annual mean T)", abs_z=15.3, p_value="<0.0001", direction="negative on A. rubrum odds"),
            dict(term="bio24 (radiation wettest quarter)", abs_z=11.9, p_value="<0.0001", direction="positive on A. rubrum odds"),
            dict(term="bio34 (warm-season moisture)", abs_z=11.9, p_value="<0.0001", direction="positive on A. rubrum odds"),
            dict(term="bio8 (T wettest quarter)", abs_z=11.4, p_value="<0.0001", direction="positive on A. rubrum odds"),
            dict(term="bio30 (lowest moisture)", abs_z=11.1, p_value="<0.0001", direction="positive on A. rubrum odds"),
            dict(term="bio31 (moisture seasonality)", abs_z=10.9, p_value="<0.0001", direction="negative on A. rubrum odds"),
            dict(term="bio15 (precipitation seasonality)", abs_z=10.8, p_value="<0.0001", direction="negative on A. rubrum odds"),
            dict(term="bio27 (radiation coldest quarter)", abs_z=9.7, p_value="<0.0001", direction="negative on A. rubrum odds"),
            dict(term="s(longitude, latitude) smooth", abs_z=float("nan"), p_value="<0.0001", direction=f"edf = 98.65, chi-square = 53500"),
        ]
        pd.DataFrame(gam_rows).to_csv(gam_path, index=False)


def build_s4_species_tests() -> None:
    """Mann-Whitney U (two-sided) between sugar and red maple for selected bioclim
    variables, from the ecoforestry inventory. Uses the already-reduced
    df_important.csv so that it runs in seconds, not minutes."""
    src = PROJECT_DIR / "data" / "df_important.csv"
    if not src.exists():
        return
    # Load only a reasonable sample and only the columns of interest.
    candidate = ["essence", "bio1", "bio15", "bio31", "bio34", "bio23", "bio12", "ph", "sand"]
    usecols = [c for c in candidate if c]
    # Sampling: reading the full 413 MB CSV with pandas; restrict to usecols to keep memory OK.
    df = pd.read_csv(src, usecols=usecols)
    df = df[df["essence"].isin(["ES", "EO"])]
    rows = []
    for col in usecols[1:]:
        es = df.loc[df["essence"] == "ES", col].dropna().to_numpy()
        eo = df.loc[df["essence"] == "EO", col].dropna().to_numpy()
        if len(es) == 0 or len(eo) == 0:
            continue
        u, p = stats.mannwhitneyu(es, eo, alternative="two-sided")
        mu_u = len(es) * len(eo) / 2
        sigma_u = np.sqrt(len(es) * len(eo) * (len(es) + len(eo) + 1) / 12)
        z = (u - mu_u) / sigma_u
        r = 1 - 2 * u / (len(es) * len(eo))
        rows.append(
            dict(
                variable=col,
                median_ES=float(np.median(es)),
                median_EO=float(np.median(eo)),
                n_ES=int(len(es)),
                n_EO=int(len(eo)),
                U=float(u),
                z=float(z),
                p_value=float(p),
                effect_r_rank_biserial=float(r),
            )
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_fdr"] = bh_fdr(out["p_value"].to_numpy())
    out.to_csv(SUMMARY_STATS / "s4_species_mannwhitney.csv", index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    per_species = {}
    for sp in ("sugar", "red"):
        print(f"[stats] computing for {sp} ...")
        per_species[sp] = regenerate_fold_predictions(sp)
    print("[stats] building cross-document synthesis ...")
    build_s7_tables(per_species)
    print("[stats] done. Outputs under outputs/{sugar,red}/stats/ and outputs/summary/stats/.")


if __name__ == "__main__":
    main()
