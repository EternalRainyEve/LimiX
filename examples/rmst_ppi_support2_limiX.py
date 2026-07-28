"""
SUPPORT2 real-data RMST experiment with LimiX imputation.

Main analysis:
    X2 is the naturally observed costly renal/metabolic panel
    log1p(bun), log1p(glucose), log1p(urine).

    R = 1 if the full X2 panel is observed and R = 0 otherwise.

The total SUPPORT2 cohort is first split into:
    context set: complete-panel patients used only as the LimiX context set
                 for learning/applying f: X1 -> X2;
    inference set: all remaining patients used for Classical, Naive, and PPI.

The estimators mirror the MCAR-PPI simulation code:
    Classical: labeled complete-panel patients only.
    Naive: all patients with LimiX-imputed X2.
    PPI: unlabeled imputed score plus labeled correction.

No Oracle estimator is reported because the true X2 values are unavailable for
inference-set patients with natural panel missingness.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from sklearn.model_selection import StratifiedKFold


# LimiX package root (this file: LimiX/examples/...)
_LIMIX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LIMIX_ROOT not in sys.path:
    sys.path.insert(0, _LIMIX_ROOT)
from inference.predictor import LimiXPredictor  # noqa: E402


X2_PANEL_OPTIONS = {
    "liver": ["alb", "bili"],
    "renal_metabolic": ["bun", "glucose", "urine"],
}

X2_PANEL_LABELS = {
    "liver": {
        "name": "liver/albumin laboratory panel",
        "formula": r"\{\log(1+alb), \log(1+bili)\}",
        "description": "serum albumin and bilirubin, two baseline laboratory measurements commonly used in SUPPORT prognostic modeling",
    },
    "renal_metabolic": {
        "name": "renal/metabolic measurement panel",
        "formula": r"\{\log(1+bun), \log(1+glucose), \log(1+urine)\}",
        "description": "blood urea nitrogen, glucose, and urine output, which summarize renal and metabolic status",
    },
}

DEFAULT_NUMERIC_X1 = [
    "age",
    "num.co",
    "edu",
    "scoma",
    "hday",
    "diabetes",
    "dementia",
    "meanbp",
    "wblc",
    "hrt",
    "resp",
    "temp",
    "pafi",
    "crea",
    "sod",
    "ph",
    "glucose",
    "bun",
    "urine",
    "adlsc",
]
DEFAULT_CATEGORICAL_X1 = ["sex", "race", "income", "dzgroup", "dzclass", "ca"]


@dataclass
class FittedCox:
    cph: CoxPHFitter
    x_cols: list[str]


def sanitize_col(name: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name))
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "col"
    if name[0].isdigit():
        name = f"x_{name}"
    return name


def make_unique(cols: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for col in cols:
        base = sanitize_col(col)
        k = seen.get(base, 0)
        seen[base] = k + 1
        out.append(base if k == 0 else f"{base}_{k}")
    return out


def load_support2() -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise SystemExit("Install ucimlrepo in the active environment first.") from e

    support2 = fetch_ucirepo(id=880)
    if support2.data.original is None:
        raise RuntimeError("support2.data.original is required because d.time is not a target column.")
    return support2.data.original.copy()


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_log1p(s: pd.Series, name: str) -> pd.Series:
    x = safe_numeric(s).astype(float)
    bad = x < -0.999
    if bool(bad.any()):
        raise ValueError(f"{name} contains values below -0.999; log1p is not defined.")
    return np.log1p(x)


def build_support2_analysis_frame(
    raw: pd.DataFrame,
    tau: float,
    x2_panel: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if x2_panel not in X2_PANEL_OPTIONS:
        raise ValueError(f"Unknown X2 panel {x2_panel!r}; choose from {sorted(X2_PANEL_OPTIONS)}.")

    x2_raw_cols = X2_PANEL_OPTIONS[x2_panel]
    numeric_x1_raw = [c for c in DEFAULT_NUMERIC_X1 if c not in x2_raw_cols]
    categorical_x1_raw = [c for c in DEFAULT_CATEGORICAL_X1 if c not in x2_raw_cols]

    required = ["death", "d.time", *x2_raw_cols, *numeric_x1_raw, *categorical_x1_raw]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"SUPPORT2 data are missing required columns: {missing}")

    death = safe_numeric(raw["death"])
    dtime = safe_numeric(raw["d.time"])

    base = pd.DataFrame(index=raw.index)
    base["death"] = death
    base["dtime"] = dtime
    base["Y"] = np.minimum(dtime, tau)
    base["Delta"] = ((dtime >= tau) | ((death == 1) & (dtime <= tau))).astype(float)
    base["event_T"] = ((death == 1) & (dtime <= tau)).astype(int)
    base["event_C"] = ((death == 0) & (dtime < tau)).astype(int)

    r_complete = pd.Series(True, index=raw.index)
    for col in x2_raw_cols:
        r_complete &= raw[col].notna()
    base["R"] = r_complete.astype(float)

    x1_num = pd.DataFrame(index=raw.index)
    for col in numeric_x1_raw:
        v = safe_numeric(raw[col])
        med = float(v.median(skipna=True))
        x1_num[f"x1_{sanitize_col(col)}"] = v.fillna(med).astype(float)

    x1_cat_raw = raw[categorical_x1_raw].copy()
    for col in categorical_x1_raw:
        x1_cat_raw[col] = x1_cat_raw[col].astype("string").fillna("missing")
    x1_cat = pd.get_dummies(x1_cat_raw, prefix=[f"x1_{sanitize_col(c)}" for c in categorical_x1_raw], dtype=float)
    x1_cat.columns = make_unique(x1_cat.columns)

    x2 = pd.DataFrame(index=raw.index)
    for col in x2_raw_cols:
        x2[f"x2_log1p_{sanitize_col(col)}"] = safe_log1p(raw[col], col)

    x1_cols = list(x1_num.columns) + list(x1_cat.columns)
    x2_cols = list(x2.columns)
    design = pd.concat([base, x1_num, x1_cat, x2], axis=1)

    keep = design[["death", "dtime", "Y", "Delta", "event_T", "event_C"]].notna().all(axis=1)
    design = design.loc[keep].reset_index(drop=True)

    return design, x1_cols, x2_cols, x2_raw_cols


def split_context_inference(
    data: pd.DataFrame,
    n_context: int,
    context_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_context <= 0:
        raise ValueError("--n-context must be positive.")

    complete_idx = data.index[data["R"] == 1].to_numpy()
    if n_context >= len(complete_idx):
        raise ValueError(
            f"--n-context={n_context} leaves no complete-panel inference samples; "
            f"there are only {len(complete_idx)} complete-panel rows."
        )

    rng = np.random.default_rng(context_seed)
    context_idx = rng.choice(complete_idx, size=n_context, replace=False)
    context_mask = data.index.isin(context_idx)

    context = data.loc[context_mask].copy().reset_index(drop=True)
    inference = data.loc[~context_mask].copy().reset_index(drop=True)
    return context, inference


def summarize_sets(
    full_data: pd.DataFrame,
    context: pd.DataFrame,
    inference: pd.DataFrame,
    tau: float,
    x1_cols: list[str],
    x2_cols: list[str],
    x2_raw_cols: list[str],
) -> dict:
    r1 = inference["R"] == 1
    r0 = inference["R"] == 0
    return {
        "n_total": int(len(full_data)),
        "n_context": int(len(context)),
        "n_context_R1": int(context["R"].sum()),
        "n_inference": int(len(inference)),
        "n": int(len(inference)),
        "n_R1": int(r1.sum()),
        "n_R0": int(r0.sum()),
        "pi_hat": float(inference["R"].mean()),
        "missing_fraction": float(1.0 - inference["R"].mean()),
        "death_rate": float(inference["death"].mean()),
        "death_rate_R1": float(inference.loc[r1, "death"].mean()),
        "death_rate_R0": float(inference.loc[r0, "death"].mean()),
        "dtime_median": float(inference["dtime"].median()),
        "dtime_median_R1": float(inference.loc[r1, "dtime"].median()),
        "dtime_median_R0": float(inference.loc[r0, "dtime"].median()),
        "event_before_tau": float(inference["event_T"].mean()),
        "censoring_fraction": float(inference["event_C"].mean()),
        "observed_truncated_fraction": float(inference["Delta"].mean()),
        "tau": float(tau),
        "x2_raw_cols": ",".join(x2_raw_cols),
        "x2_cols": ",".join(x2_cols),
        "n_x1_cols": int(len(x1_cols)),
    }


def _cumulative_baseline_on_grid(cph: CoxPHFitter, grid_t: np.ndarray) -> np.ndarray:
    base = cph.baseline_cumulative_hazard_.iloc[:, 0]
    t_bh = base.index.to_numpy(dtype=np.float64)
    h_bh = base.to_numpy(dtype=np.float64)
    if len(t_bh) == 0:
        return np.zeros_like(grid_t, dtype=np.float64)
    j = np.searchsorted(t_bh, grid_t, side="right") - 1
    h0 = np.empty_like(grid_t, dtype=np.float64)
    neg = j < 0
    h0[neg] = 0.0
    h0[~neg] = h_bh[j[~neg]]
    return h0


def _match_r_find_interval(y: np.ndarray, grid_t: np.ndarray, k: int) -> np.ndarray:
    g = np.sort(np.asarray(grid_t, dtype=np.float64))
    j = np.searchsorted(g, y, side="right")
    j = np.clip(j, 1, k)
    return (j - 1).astype(np.int64)


def fit_cox_model(
    frame: pd.DataFrame,
    x_cols: list[str],
    event_col: str,
    penalizer: float,
) -> FittedCox:
    if frame[event_col].nunique() < 2:
        raise ValueError(f"Cox model for {event_col} requires both event classes.")

    variances = frame[x_cols].astype(float).var(axis=0, ddof=0)
    use_cols = [c for c in x_cols if np.isfinite(variances[c]) and variances[c] > 1e-12]
    if not use_cols:
        raise ValueError(f"No non-constant covariates available for {event_col}.")

    d = frame[["Y", *use_cols]].copy()
    d[event_col] = frame[event_col].astype(int).values
    cph = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d, duration_col="Y", event_col=event_col, show_progress=False)
    return FittedCox(cph=cph, x_cols=use_cols)


def fit_cox_t(frame: pd.DataFrame, x_cols: list[str], penalizer: float) -> FittedCox:
    return fit_cox_model(frame, x_cols, "event_T", penalizer)


def fit_cox_c(frame: pd.DataFrame, x_cols: list[str], penalizer: float) -> FittedCox:
    return fit_cox_model(frame, x_cols, "event_C", penalizer)


def compute_scores(
    newdata: pd.DataFrame,
    cox_t: FittedCox,
    cox_c: FittedCox,
    tau: float,
) -> dict[str, np.ndarray]:
    n_obs = len(newdata)
    y = newdata["Y"].to_numpy(dtype=np.float64)
    delta = newdata["Delta"].to_numpy(dtype=np.float64)

    bht = cox_t.cph.baseline_cumulative_hazard_.index.to_numpy(dtype=np.float64)
    bhc = cox_c.cph.baseline_cumulative_hazard_.index.to_numpy(dtype=np.float64)
    all_times = np.sort(np.unique(np.concatenate([np.array([0.0]), bht, bhc, [tau]])))
    grid_t = all_times[(all_times >= 0.0) & (all_times <= tau)]
    if len(grid_t) < 2:
        grid_t = np.array([0.0, tau], dtype=np.float64)
    k = len(grid_t)
    dt = np.append(np.diff(grid_t), 0.0)

    h0_t = _cumulative_baseline_on_grid(cox_t.cph, grid_t)
    h0_c = _cumulative_baseline_on_grid(cox_c.cph, grid_t)
    dh0_c = np.empty(k, dtype=np.float64)
    dh0_c[0] = 0.0
    dh0_c[1:] = h0_c[1:] - h0_c[:-1]

    risk_t = cox_t.cph.predict_partial_hazard(newdata[cox_t.x_cols].astype(np.float64)).to_numpy().ravel()
    risk_c = cox_c.cph.predict_partial_hazard(newdata[cox_c.x_cols].astype(np.float64)).to_numpy().ravel()

    s_t_mat = np.exp(-np.outer(risk_t, h0_t))
    s_c_mat = np.exp(-np.outer(risk_c, h0_c))
    s_t_mat = np.maximum(s_t_mat, 1e-3)
    s_c_mat = np.maximum(s_c_mat, 1e-3)

    area_s_t = s_t_mat * dt[None, :]
    int_s_t = np.cumsum(area_s_t[:, ::-1], axis=1)[:, ::-1]
    q_mat = (int_s_t / s_t_mat) + grid_t[None, :]

    integrand_mat = (q_mat / s_c_mat) * risk_c[:, None] * dh0_c[None, :]
    integral_mat = np.cumsum(integrand_mat, axis=1)

    idx = _match_r_find_interval(y, grid_t, k)
    row = np.arange(n_obs)
    sc_y = s_c_mat[row, idx]
    q_y = q_mat[row, idx]
    int_y = integral_mat[row, idx]

    term1 = (y * delta) / sc_y
    term2 = (q_y * (1.0 - delta)) / sc_y
    term3 = int_y
    return {"ipcw": term1, "dr": term1 + term2 - term3}


def load_or_fetch_ckpt(ckpt: Optional[str], root: str) -> str:
    if ckpt and os.path.isfile(ckpt):
        return os.path.abspath(ckpt)
    def_dir = os.path.join(root, "cache", "LimiX-16M.ckpt")
    if os.path.isfile(def_dir):
        return def_dir
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("Install huggingface_hub and download the checkpoint, or pass --model_path.") from e
    return hf_hub_download(
        repo_id="stableai-org/LimiX-16M",
        filename="LimiX-16M.ckpt",
        local_dir=os.path.join(root, "cache"),
    )


def resolve_limiX_device(device_arg: str) -> torch.device:
    s = (device_arg or "auto").strip().lower()
    if s in ("", "auto", "default"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def limix_impute_panel(
    context: pd.DataFrame,
    target: pd.DataFrame,
    covariate_cols: list[str],
    imputed_cols: list[str],
    predictor: LimiXPredictor,
    n_anchor: int,
) -> np.ndarray:
    if len(context) == 0:
        raise ValueError("LimiX imputation requires at least one context row.")
    x_train = context[covariate_cols].to_numpy(np.float64)
    anchor_y = np.zeros(len(x_train), dtype=np.float64)

    n_anchor = max(1, min(n_anchor, len(context)))
    anchor_x = context[covariate_cols].iloc[:n_anchor].to_numpy(np.float64)

    x_test_main = target[covariate_cols].to_numpy(np.float64)
    imputed_idx = [covariate_cols.index(col) for col in imputed_cols]
    x_test_main[:, imputed_idx] = np.nan

    x_test = np.vstack([anchor_x, x_test_main])
    _, reconstructed_x = predictor.predict(
        x_train,
        anchor_y,
        x_test,
        task_type="Regression",
    )
    if reconstructed_x is None:
        raise RuntimeError("LimiX predictor did not return reconstructed_X; set mask_prediction=True.")

    reconstructed = as_numpy(reconstructed_x)
    reconstructed_test = reconstructed[-len(x_test) :]
    imputed_target = reconstructed_test[n_anchor:]
    return imputed_target[:, imputed_idx].astype(np.float64, copy=False)


def median_impute_panel(
    context: pd.DataFrame,
    target: pd.DataFrame,
    imputed_cols: list[str],
) -> np.ndarray:
    med = context[imputed_cols].median(axis=0, skipna=True).to_numpy(dtype=np.float64)
    if np.isnan(med).any():
        raise ValueError("Median imputation failed because a costly covariate has no labeled values.")
    return np.tile(med[None, :], (len(target), 1))


def make_proxy_frames(
    context: pd.DataFrame,
    dt_train: pd.DataFrame,
    dt_target: pd.DataFrame,
    covariate_cols: list[str],
    imputed_cols: list[str],
    predictor: Optional[LimiXPredictor],
    n_anchor: int,
    debug_median_impute: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if debug_median_impute:
        train_hat = median_impute_panel(context, dt_train, imputed_cols)
        target_hat = median_impute_panel(context, dt_target, imputed_cols)
    else:
        if predictor is None:
            raise ValueError("predictor must be provided unless --debug_median_impute is used.")
        train_hat = limix_impute_panel(context, dt_train, covariate_cols, imputed_cols, predictor, n_anchor)
        target_hat = limix_impute_panel(context, dt_target, covariate_cols, imputed_cols, predictor, n_anchor)

    train_proxy = dt_train.copy()
    target_proxy = dt_target.copy()
    train_proxy.loc[:, imputed_cols] = train_hat
    target_proxy.loc[:, imputed_cols] = target_hat
    return train_proxy, target_proxy


def mean_score_stats(scores: np.ndarray, name: str) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size <= 1:
        raise ValueError(f"Not enough observations to compute variance for {name}: n={scores.size}")
    est = float(np.mean(scores))
    var = float(np.var(scores, ddof=1) / scores.size)
    return est, var


def ppi_score_stats(score_r0: np.ndarray, correction_r1: np.ndarray, name: str) -> tuple[float, float]:
    score_r0 = np.asarray(score_r0, dtype=np.float64)
    correction_r1 = np.asarray(correction_r1, dtype=np.float64)
    n0 = score_r0.size
    n1 = correction_r1.size
    if n0 <= 1 or n1 <= 1:
        raise ValueError(f"Not enough observations for PPI variance {name}: n0={n0}, n1={n1}")
    est = float(np.mean(score_r0) + np.mean(correction_r1))
    var = float(np.var(score_r0, ddof=1) / n0 + np.var(correction_r1, ddof=1) / n1)
    return est, var


def compute_fold_theta(
    context: pd.DataFrame,
    dt_train: pd.DataFrame,
    dt_target: pd.DataFrame,
    covariate_cols: list[str],
    imputed_cols: list[str],
    tau: float,
    penalizer: float,
    predictor: Optional[LimiXPredictor],
    n_anchor: int,
    debug_median_impute: bool,
) -> dict:
    train_proxy, target_proxy = make_proxy_frames(
        context,
        dt_train,
        dt_target,
        covariate_cols,
        imputed_cols,
        predictor,
        n_anchor,
        debug_median_impute,
    )

    mask_r1_train = dt_train["R"] == 1.0
    mask_r0_train = dt_train["R"] == 0.0
    mask_r1_target = dt_target["R"] == 1.0
    mask_r0_target = dt_target["R"] == 0.0

    x_cols = covariate_cols
    out: dict[str, float] = {
        "n_target": float(len(dt_target)),
        "n_r1_target": float(mask_r1_target.sum()),
        "n_r0_target": float(mask_r0_target.sum()),
    }

    # Classical: true X2 among labeled patients.
    cox_t_cla = fit_cox_t(dt_train.loc[mask_r1_train], x_cols, penalizer)
    cox_c_cla = fit_cox_c(dt_train.loc[mask_r1_train], x_cols, penalizer)
    res_cla = compute_scores(dt_target.loc[mask_r1_target], cox_t_cla, cox_c_cla, tau)

    # Naive: imputed X2 for every patient.
    cox_t_nai = fit_cox_t(train_proxy, x_cols, penalizer)
    cox_c_nai = fit_cox_c(train_proxy, x_cols, penalizer)
    res_nai = compute_scores(target_proxy, cox_t_nai, cox_c_nai, tau)

    # PPI components: R=0 imputed score and R=1 correction.
    cox_t_r0 = fit_cox_t(train_proxy.loc[mask_r0_train], x_cols, penalizer)
    cox_c_r0 = fit_cox_c(train_proxy.loc[mask_r0_train], x_cols, penalizer)
    res_r0 = compute_scores(target_proxy.loc[mask_r0_target], cox_t_r0, cox_c_r0, tau)

    cox_t_r1_hat = fit_cox_t(train_proxy.loc[mask_r1_train], x_cols, penalizer)
    cox_c_r1_hat = fit_cox_c(train_proxy.loc[mask_r1_train], x_cols, penalizer)
    res_r1_hat = compute_scores(target_proxy.loc[mask_r1_target], cox_t_r1_hat, cox_c_r1_hat, tau)

    for metric in ["ipcw", "dr"]:
        out[f"classical_{metric}_est"], out[f"classical_{metric}_var"] = mean_score_stats(
            res_cla[metric], f"classical_{metric}"
        )
        out[f"naive_{metric}_est"], out[f"naive_{metric}_var"] = mean_score_stats(
            res_nai[metric], f"naive_{metric}"
        )
        correction = res_cla[metric] - res_r1_hat[metric]
        out[f"ppi_{metric}_est"], out[f"ppi_{metric}_var"] = ppi_score_stats(
            res_r0[metric], correction, f"ppi_{metric}"
        )
    return out


def combine_folds(fold_results: list[dict], total_n: int, total_r1: int, tau: float) -> pd.DataFrame:
    rows = []
    for method_key, method_label in [
        ("classical", "Classical"),
        ("naive", "Naive"),
        ("ppi", "PPI"),
    ]:
        for metric_key, metric_label in [("ipcw", "IPCW"), ("dr", "DR")]:
            est = 0.0
            var = 0.0
            for fr in fold_results:
                if method_key == "classical":
                    weight = fr["n_r1_target"] / total_r1
                else:
                    weight = fr["n_target"] / total_n
                est += weight * fr[f"{method_key}_{metric_key}_est"]
                var += (weight**2) * fr[f"{method_key}_{metric_key}_var"]
            se = float(np.sqrt(var))
            ci_l = float(est - 1.96 * se)
            ci_u = float(est + 1.96 * se)
            rows.append(
                {
                    "method": method_label,
                    "score": metric_label,
                    "estimate": float(est),
                    "se": se,
                    "ci_lower": ci_l,
                    "ci_upper": ci_u,
                    "ci_width": float(ci_u - ci_l),
                    "n": int(total_n if method_key != "classical" else total_r1),
                    "n_R1": int(total_r1),
                    "pi_hat": float(total_r1 / total_n),
                    "tau": float(tau),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SUPPORT2 RMST PPI real-data experiment with LimiX MVI")
    p.add_argument("--tau", type=float, default=1000.0)
    p.add_argument("--x2-panel", type=str, default="renal_metabolic", choices=sorted(X2_PANEL_OPTIONS))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--context-seed", type=int, default=42)
    p.add_argument("--n-context", type=int, default=1000)
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--inference_config", type=str, default="")
    p.add_argument("--out_dir", type=str, default=os.path.join("results", "support2_realdata"))
    p.add_argument("--n-anchor", type=int, default=16)
    p.add_argument("--cox-penalizer", type=float, default=0.01)
    p.add_argument("--max_rows", type=int, default=0, help="Optional subsample for smoke tests.")
    p.add_argument(
        "--debug_median_impute",
        action="store_true",
        help="Use fold-specific median imputation instead of loading LimiX; intended for smoke tests.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _LIMIX_ROOT
    os.chdir(root)

    raw = load_support2()
    full_data, x1_cols, x2_cols, x2_raw_cols = build_support2_analysis_frame(raw, args.tau, args.x2_panel)
    covariate_cols = x1_cols + x2_cols
    context, data = split_context_inference(full_data, args.n_context, args.context_seed)

    if args.max_rows and args.max_rows > 0 and args.max_rows < len(data):
        data = data.sample(n=args.max_rows, random_state=args.fold_seed).reset_index(drop=True)

    diagnostics = summarize_sets(full_data, context, data, args.tau, x1_cols, x2_cols, x2_raw_cols)
    panel_info = X2_PANEL_LABELS[args.x2_panel]
    diagnostics["x2_panel"] = args.x2_panel
    diagnostics["x2_panel_name"] = panel_info["name"]
    diagnostics["x2_formula"] = panel_info["formula"]
    diagnostics["x2_description"] = panel_info["description"]
    diagnostics["x1_raw_cols"] = ",".join(
        [c for c in [*DEFAULT_NUMERIC_X1, *DEFAULT_CATEGORICAL_X1] if c not in x2_raw_cols]
    )
    diagnostics["fold_seed"] = int(args.fold_seed)
    diagnostics["context_seed"] = int(args.context_seed)
    diagnostics["predictor_seed"] = int(args.fold_seed)

    print("SUPPORT2 total rows:", len(full_data), flush=True)
    print("LimiX context rows:", len(context), flush=True)
    print("inference rows:", len(data), flush=True)
    print("inference R=1 complete-panel rows:", int(data["R"].sum()), flush=True)
    print("inference label fraction:", f"{data['R'].mean():.4f}", flush=True)
    print("inference censoring before tau:", f"{data['event_C'].mean():.4f}", flush=True)

    predictor = None
    if not args.debug_median_impute:
        ckpt = load_or_fetch_ckpt((args.model_path or None) if args.model_path else None, root)
        icfg = args.inference_config or os.path.join(root, "config", "reg_default_noretrieval_MVI.json")
        predictor = LimiXPredictor(
            device=resolve_limiX_device(args.device),
            model_path=ckpt,
            inference_config=icfg,
            mask_prediction=True,
            seed=args.fold_seed,
        )

    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.fold_seed)
    fold_results: list[dict] = []
    for fold_id, (train_idx, target_idx) in enumerate(splitter.split(data, data["R"].astype(int)), start=1):
        print(f"cross-fit fold {fold_id}/2 ...", flush=True)
        dt_train = data.iloc[train_idx].copy()
        dt_target = data.iloc[target_idx].copy()
        fold_results.append(
            compute_fold_theta(
                context=context,
                dt_train=dt_train,
                dt_target=dt_target,
                covariate_cols=covariate_cols,
                imputed_cols=x2_cols,
                tau=args.tau,
                penalizer=args.cox_penalizer,
                predictor=predictor,
                n_anchor=args.n_anchor,
                debug_median_impute=args.debug_median_impute,
            )
        )

    results = combine_folds(
        fold_results=fold_results,
        total_n=len(data),
        total_r1=int(data["R"].sum()),
        tau=args.tau,
    )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(root) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "support2_realdata_results.csv"
    diag_path = out_dir / "support2_realdata_diagnostics.csv"

    pd.DataFrame([diagnostics]).to_csv(diag_path, index=False, encoding="utf-8-sig")
    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    print("\n--- SUPPORT2 Real-Data Results ---")
    print(results.to_string(index=False))
    print("\nSaved:")
    print(results_path)
    print(diag_path)


if __name__ == "__main__":
    main()
