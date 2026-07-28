"""
Five-covariate survival simulation with LimiX missing value imputation:
IPCW, doubly robust, and PPI-Old (Ablation Test)
2-Fold Cross-Fitting + SEPARATE NUISANCE MODELS FOR EVERY TERM
Includes: Coverage Rate (CR) and 95% Confidence Interval width calculations.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import List, Optional
from lifelines import CoxPHFitter

import numpy as np
import pandas as pd
import torch


# LimiX package root (this file: LimiX/examples/...)
_LIMIX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LIMIX_ROOT not in sys.path:
    sys.path.insert(0, _LIMIX_ROOT)
from inference.predictor import LimiXPredictor  # noqa: E402


COVARIATE_COLS = ["X1", "X2", "X3", "X4", "X5"]
IMPUTED_COLS = ["X4", "X5"]


# ==============================================================================
# 1. DGP: S^C misspecified, S^T correctly specified
# ==============================================================================
def generate_data(
    n: int,
    rng: np.random.Generator,
    tau: float,
    rho: float = 0.7,
    c_base_hazard: float = 0.51,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Failure-time model S^T is correctly specified.
    Censoring model S^C is misspecified.

    The true censoring model contains nonlinear and interaction terms,
    while the working Cox model contains only linear main effects.
    """

    cov = np.full((5, 5), rho, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)

    x1, x2, x3, x4, x5 = rng.multivariate_normal(
        np.zeros(5),
        cov,
        size=n,
    ).T

    # --------------------------------------------------------------------------
    # Failure-time model: correctly specified
    # --------------------------------------------------------------------------
    beta_t = np.array([-0.5, 0.8, 0.4, 2.1, -1.3])

    linear_predictor_t = (
        beta_t[0] * x1
        + beta_t[1] * x2
        + beta_t[2] * x3
        + beta_t[3] * x4
        + beta_t[4] * x5
    )

    risk_score_t = np.exp(linear_predictor_t)

    nu_t = 2.0
    lambda_t = 1.0
    u_t = rng.random(n)

    t_true = (
        -np.log(u_t) / (lambda_t * risk_score_t)
    ) ** (1.0 / nu_t)

    t_trunc = np.minimum(t_true, tau)

    if verbose:
        q85 = np.quantile(t_true, 0.85)
        print(f"85th percentile of t_true: {q85:.4f}")

    # --------------------------------------------------------------------------
    # Censoring model: misspecified
    # --------------------------------------------------------------------------
    beta_c = np.array([0.1, 0.2, -0.1, 0.6, -0.5])

    linear_predictor_c = (
        beta_c[0] * x1
        + beta_c[1] * x2
        + beta_c[2] * x3
        + beta_c[3] * x4
        + beta_c[4] * x5
    )

    # Centered nonlinear term:
    # E(X4^2 - 1) = 0
    # E(X1 X5 - rho) = 0
    # E(X2 X3 - rho) = 0
    q_c = (
        0.6 * (x4**2 - 1.0)
        - 0.5 * (x1 * x5 - rho)
        + 0.5 * (x2 * x3 - rho)
    )

    misspec_strength = 0.5
    linear_predictor_c = (
        linear_predictor_c
        + misspec_strength * q_c
    )

    risk_score_c = np.exp(linear_predictor_c)

    nu_c = 1.0
    lambda_c = c_base_hazard
    u_c = rng.random(n)

    c = (
        -np.log(u_c) / (lambda_c * risk_score_c)
    ) ** (1.0 / nu_c)

    if verbose:
        q85_c = np.quantile(c, 0.85)
        print(f"85th percentile of c: {q85_c:.4f}")

    y = np.minimum(t_trunc, c)
    delta = (t_trunc <= c).astype(np.float64)

    return pd.DataFrame(
        {
            "X1": x1,
            "X2": x2,
            "X3": x3,
            "X4": x4,
            "X5": x5,
            "T_true": t_trunc,
            "Y": y,
            "Delta": delta,
        }
    )


def truth_theta0_rmst(
    rng: np.random.Generator,
    n_big: int,
    tau: float,
) -> tuple[float, float]:
    d = generate_data(
        n=n_big,
        rng=rng,
        tau=tau,
        verbose=False,
    )

    censoring_rate = 1.0 - float(d["Delta"].mean())

    return float(d["T_true"].mean()), censoring_rate

# ==============================================================================
# 1. DGP: S^T misspecified, S^C correctly specified
# ==============================================================================
def generate_data(
    n: int,
    rng: np.random.Generator,
    tau: float,
    rho: float = 0.7,
    c_base_hazard: float = 0.23,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Failure-time model S^T is misspecified.
    Censoring model S^C is correctly specified.

    The true failure-time model contains nonlinear and interaction terms,
    while the working Cox model contains only linear main effects.
    """

    cov = np.full((5, 5), rho, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)

    x1, x2, x3, x4, x5 = rng.multivariate_normal(
        np.zeros(5),
        cov,
        size=n,
    ).T

    # --------------------------------------------------------------------------
    # Failure-time model: misspecified
    # --------------------------------------------------------------------------
    beta_t = np.array([-0.5, 0.8, 0.4, 2.1, -1.3])

    linear_predictor_t = (
        beta_t[0] * x1
        + beta_t[1] * x2
        + beta_t[2] * x3
        + beta_t[3] * x4
        + beta_t[4] * x5
    )

    # Centered nonlinear term:
    # E(X1^2 - 1) = 0
    # E(X2 X4 - rho) = 0
    # E(X3 X5 - rho) = 0
    q_t = (
        0.6 * (x1**2 - 1.0)
        + 0.5 * (x2 * x4 - rho)
        - 0.5 * (x3 * x5 - rho)
    )

    misspec_strength = 0.5
    linear_predictor_t = (
        linear_predictor_t
        + misspec_strength * q_t
    )

    risk_score_t = np.exp(linear_predictor_t)

    nu_t = 2.0
    lambda_t = 1.0
    u_t = rng.random(n)

    t_true = (
        -np.log(u_t) / (lambda_t * risk_score_t)
    ) ** (1.0 / nu_t)

    t_trunc = np.minimum(t_true, tau)

    if verbose:
        q85 = np.quantile(t_true, 0.85)
        print(f"85th percentile of t_true: {q85:.4f}")

    # --------------------------------------------------------------------------
    # Censoring model: correctly specified
    # --------------------------------------------------------------------------
    beta_c = np.array([0.1, 0.2, -0.1, 0.6, -0.5])

    linear_predictor_c = (
        beta_c[0] * x1
        + beta_c[1] * x2
        + beta_c[2] * x3
        + beta_c[3] * x4
        + beta_c[4] * x5
    )

    risk_score_c = np.exp(linear_predictor_c)

    nu_c = 1.0
    lambda_c = c_base_hazard
    u_c = rng.random(n)

    c = (
        -np.log(u_c) / (lambda_c * risk_score_c)
    ) ** (1.0 / nu_c)

    if verbose:
        q85_c = np.quantile(c, 0.85)
        print(f"85th percentile of c: {q85_c:.4f}")

    y = np.minimum(t_trunc, c)
    delta = (t_trunc <= c).astype(np.float64)

    return pd.DataFrame(
        {
            "X1": x1,
            "X2": x2,
            "X3": x3,
            "X4": x4,
            "X5": x5,
            "T_true": t_trunc,
            "Y": y,
            "Delta": delta,
        }
    )


def truth_theta0_rmst(
    rng: np.random.Generator,
    n_big: int,
    tau: float,
) -> tuple[float, float]:
    d = generate_data(
        n=n_big,
        rng=rng,
        tau=tau,
        verbose=False,
    )

    censoring_rate = 1.0 - float(d["Delta"].mean())

    return float(d["T_true"].mean()), censoring_rate


# ==============================================================================
# 1. DGP: both S^T and S^C misspecified
# ==============================================================================
def generate_data(
    n: int,
    rng: np.random.Generator,
    tau: float,
    rho: float = 0.7,
    c_base_hazard: float = 0.23,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Both the failure-time model S^T and censoring model S^C are misspecified.

    Both true models contain nonlinear and interaction terms, while the
    working Cox models contain only linear main effects.
    """

    cov = np.full((5, 5), rho, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)

    x1, x2, x3, x4, x5 = rng.multivariate_normal(
        np.zeros(5),
        cov,
        size=n,
    ).T

    # --------------------------------------------------------------------------
    # Failure-time model: misspecified
    # --------------------------------------------------------------------------
    beta_t = np.array([-0.5, 0.8, 0.4, 2.1, -1.3])

    linear_predictor_t = (
        beta_t[0] * x1
        + beta_t[1] * x2
        + beta_t[2] * x3
        + beta_t[3] * x4
        + beta_t[4] * x5
    )

    q_t = (
        0.6 * (x1**2 - 1.0)
        + 0.5 * (x2 * x4 - rho)
        - 0.5 * (x3 * x5 - rho)
    )

    misspec_strength = 0.5
    linear_predictor_t = (
        linear_predictor_t
        + misspec_strength * q_t
    )

    risk_score_t = np.exp(linear_predictor_t)

    nu_t = 2.0
    lambda_t = 1.0
    u_t = rng.random(n)

    t_true = (
        -np.log(u_t) / (lambda_t * risk_score_t)
    ) ** (1.0 / nu_t)

    t_trunc = np.minimum(t_true, tau)

    if verbose:
        q85 = np.quantile(t_true, 0.85)
        print(f"85th percentile of t_true: {q85:.4f}")

    # --------------------------------------------------------------------------
    # Censoring model: misspecified
    # --------------------------------------------------------------------------
    beta_c = np.array([0.1, 0.2, -0.1, 0.6, -0.5])

    linear_predictor_c = (
        beta_c[0] * x1
        + beta_c[1] * x2
        + beta_c[2] * x3
        + beta_c[3] * x4
        + beta_c[4] * x5
    )

    q_c = (
        0.6 * (x4**2 - 1.0)
        - 0.5 * (x1 * x5 - rho)
        + 0.5 * (x2 * x3 - rho)
    )

    linear_predictor_c = (
        linear_predictor_c
        + misspec_strength * q_c
    )

    risk_score_c = np.exp(linear_predictor_c)

    nu_c = 1.0
    lambda_c = c_base_hazard
    u_c = rng.random(n)

    c = (
        -np.log(u_c) / (lambda_c * risk_score_c)
    ) ** (1.0 / nu_c)

    if verbose:
        q85_c = np.quantile(c, 0.85)
        print(f"85th percentile of c: {q85_c:.4f}")

    y = np.minimum(t_trunc, c)
    delta = (t_trunc <= c).astype(np.float64)

    return pd.DataFrame(
        {
            "X1": x1,
            "X2": x2,
            "X3": x3,
            "X4": x4,
            "X5": x5,
            "T_true": t_trunc,
            "Y": y,
            "Delta": delta,
        }
    )


def truth_theta0_rmst(
    rng: np.random.Generator,
    n_big: int,
    tau: float,
) -> tuple[float, float]:
    d = generate_data(
        n=n_big,
        rng=rng,
        tau=tau,
        verbose=False,
    )

    censoring_rate = 1.0 - float(d["Delta"].mean())

    return float(d["T_true"].mean()), censoring_rate


# ==============================================================================
# 2. core functions: fast computation of IPCW and DR-CUT scores
# ==============================================================================
def _cumulative_baseline_on_grid(
    cph: CoxPHFitter, grid_t: np.ndarray
) -> np.ndarray:
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

def _match_r_find_interval(
    y: np.ndarray, grid_t: np.ndarray, k: int
) -> np.ndarray:
    g = np.sort(np.asarray(grid_t, dtype=np.float64))
    j = np.searchsorted(g, y, side="right")
    j = np.clip(j, 1, k)
    return (j - 1).astype(np.int64)

def compute_scores(
    newdata: pd.DataFrame,
    cph_t: CoxPHFitter,
    cph_c: CoxPHFitter,
    tau: float,
    x_cols: List[str],
) -> dict:
    n_obs = len(newdata)
    y = newdata["Y"].to_numpy(dtype=np.float64)
    delta = newdata["Delta"].to_numpy(dtype=np.float64)

    bht = cph_t.baseline_cumulative_hazard_.index.to_numpy(dtype=np.float64)
    bhc = cph_c.baseline_cumulative_hazard_.index.to_numpy(dtype=np.float64)
    all_times = np.sort(np.unique(np.concatenate([np.array([0.0]), bht, bhc, [tau]])))
    grid_t = all_times[all_times <= tau]
    k = len(grid_t)
    
    dt = np.append(np.diff(grid_t), 0.0)

    h0_t = _cumulative_baseline_on_grid(cph_t, grid_t)
    h0_c = _cumulative_baseline_on_grid(cph_c, grid_t)
    dh0_c = np.empty(k, dtype=np.float64)
    dh0_c[0] = 0.0
    dh0_c[1:] = h0_c[1:] - h0_c[:-1]

    r_t = cph_t.predict_partial_hazard(newdata[x_cols].astype(np.float64))
    r_c = cph_c.predict_partial_hazard(newdata[x_cols].astype(np.float64))
    risk_t = r_t.to_numpy().ravel()
    risk_c = r_c.to_numpy().ravel()

    s_t_mat = np.exp(-np.outer(risk_t, h0_t))
    s_c_mat = np.exp(-np.outer(risk_c, h0_c))
    s_c_mat = np.maximum(s_c_mat, 1e-3)
    s_t_mat = np.maximum(s_t_mat, 1e-3)

    area_s_t = s_t_mat * dt[None, :]
    int_s_t = np.cumsum(area_s_t[:, ::-1], axis=1)[:, ::-1]
    q_mat = (int_s_t / s_t_mat) + grid_t[None, :]

    integrand_mat = (q_mat / s_c_mat) * (risk_c[:, None]) * dh0_c[None, :]
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

def fit_cox_t(frame: pd.DataFrame, x_cols: List[str]) -> CoxPHFitter:
    d = frame[["Y", *x_cols]].copy()
    d["Delta"] = frame["Delta"].values.astype(int)
    cph = CoxPHFitter(penalizer=0.0) 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d, duration_col="Y", event_col="Delta", show_progress=False)
    return cph

def fit_cox_c(frame: pd.DataFrame, x_cols: List[str]) -> CoxPHFitter:
    d = frame[["Y", *x_cols]].copy()
    d["eventC"] = (1.0 - frame["Delta"].to_numpy() > 0.5)
    cph = CoxPHFitter(penalizer=0.0) 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d, duration_col="Y", event_col="eventC", show_progress=False)
    return cph

def limix_impute_x4x5(
    dt_pre: pd.DataFrame,
    dt_inf: pd.DataFrame,
    predictor: LimiXPredictor,
    n_anchor: int = 16,
) -> np.ndarray:
    x_train = dt_pre[COVARIATE_COLS].to_numpy(np.float64)
    anchor_y = np.zeros(len(x_train), dtype=np.float64)

    n_anchor = max(1, min(n_anchor, len(dt_pre)))
    anchor_x = dt_pre[COVARIATE_COLS].iloc[:n_anchor].to_numpy(np.float64)
    inf_x = dt_inf[COVARIATE_COLS].to_numpy(np.float64)
    inf_x[:, [COVARIATE_COLS.index(col) for col in IMPUTED_COLS]] = np.nan

    x_test = np.vstack([anchor_x, inf_x])
    _, reconstructed_x = predictor.predict(
        x_train,
        anchor_y,
        x_test,
        task_type="Regression",
    )
    if reconstructed_x is None:
        raise RuntimeError("LimiX predictor did not return reconstructed_X; set mask_prediction=True.")

    reconstructed_test = reconstructed_x[-len(x_test):]
    imputed_inf = reconstructed_test[n_anchor:]
    imputed_indices = [COVARIATE_COLS.index(col) for col in IMPUTED_COLS]
    return imputed_inf[:, imputed_indices].astype(np.float64, copy=False)


# ==============================================================================
# 3. core function: single-directional inference (calculate the point estimate and variance inside one fold)
# ==============================================================================
def compute_fold_theta_separate(
    dt_train: pd.DataFrame, dt_target: pd.DataFrame, tau: float
) -> dict:
    
    x_cols = COVARIATE_COLS
    
    mask_r1_train = dt_train["R"] == 1.0
    mask_r0_train = dt_train["R"] == 0.0
    mask_r1_target = dt_target["R"] == 1.0
    mask_r0_target = dt_target["R"] == 0.0

    # --- Baseline 1: Oracle ---
    cox_t_ora = fit_cox_t(dt_train, x_cols)
    cox_c_ora = fit_cox_c(dt_train, x_cols)
    res_ora = compute_scores(dt_target, cox_t_ora, cox_c_ora, tau, x_cols)
    
    dt_train_proxy = dt_train.copy()
    dt_train_proxy["X4"] = dt_train["Xhat4"]
    dt_train_proxy["X5"] = dt_train["Xhat5"]
    dt_target_proxy = dt_target.copy()
    dt_target_proxy["X4"] = dt_target["Xhat4"]
    dt_target_proxy["X5"] = dt_target["Xhat5"]
    
    # --- Component 1: R=0 samples ---
    if mask_r0_train.sum() > 0 and mask_r0_target.sum() > 0:
        cox_t_r0 = fit_cox_t(dt_train_proxy[mask_r0_train], x_cols)
        cox_c_r0 = fit_cox_c(dt_train_proxy[mask_r0_train], x_cols)
        res_t1 = compute_scores(dt_target_proxy[mask_r0_target], cox_t_r0, cox_c_r0, tau, x_cols)
    else:
        res_t1 = {"ipcw": np.array([]), "dr": np.array([])}

    # --- Component 2: R=1 (Hat and True) ---
    if mask_r1_train.sum() > 0 and mask_r1_target.sum() > 0:
        cox_t_r1_hat = fit_cox_t(dt_train_proxy[mask_r1_train], x_cols)
        cox_c_r1_hat = fit_cox_c(dt_train_proxy[mask_r1_train], x_cols)
        res_t2 = compute_scores(dt_target_proxy[mask_r1_target], cox_t_r1_hat, cox_c_r1_hat, tau, x_cols)
        
        cox_t_cla = fit_cox_t(dt_train[mask_r1_train], x_cols)
        cox_c_cla = fit_cox_c(dt_train[mask_r1_train], x_cols)
        res_cla = compute_scores(dt_target[mask_r1_target], cox_t_cla, cox_c_cla, tau, x_cols)
    else:
        res_t2 = {"ipcw": np.array([]), "dr": np.array([])}
        res_cla = {"ipcw": np.array([]), "dr": np.array([])}

    # ------------------------------------------------------------------
    # Fold-level point estimates and theorem-based variances
    # ------------------------------------------------------------------
    def mean_score_stats(scores: np.ndarray, name: str) -> tuple[float, float]:
        """
        For Oracle / Classical / Naive estimators in one target fold.

        Estimator:
            theta_hat = mean(scores)

        Variance estimator, conditional on the fitted nuisances:
            Var(theta_hat) = sample_var(scores) / len(scores)
        """
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size <= 1:
            raise ValueError(f"Not enough target observations to compute variance for {name}: n={scores.size}")

        est = float(np.mean(scores))
        var = float(np.var(scores, ddof=1) / scores.size)
        return est, var

    def ppi_score_stats(
        score_r0: np.ndarray,
        correction_r1: np.ndarray,
        name: str,
    ) -> tuple[float, float]:
        """
        For PPI estimators in one target fold.

        score_r0:
            A_i = H(Xhat_i; S_1), evaluated on R=0 target samples.

        correction_r1:
            B_i = H(X_i; S_3) - H(Xhat_i; S_2), evaluated on R=1 target samples.

        Fold estimator:
            theta_hat_k = mean(A_i | R=0) + mean(B_i | R=1)

        Theorem-based asymptotic variance of theta_hat_k:
            Var(theta_hat_k)
            = [ Var(A_i)/(1 - pi_hat_k) + Var(B_i)/pi_hat_k ] / n_k

        This is algebraically equal to:
            Var(A_i)/n0_k + Var(B_i)/n1_k
        where n0_k is the number of R=0 target samples and n1_k is the number of R=1 target samples.
        """
        score_r0 = np.asarray(score_r0, dtype=np.float64)
        correction_r1 = np.asarray(correction_r1, dtype=np.float64)

        n0 = score_r0.size
        n1 = correction_r1.size
        n_fold = n0 + n1
        if n0 <= 1 or n1 <= 1:
            raise ValueError(
                f"Not enough target observations to compute PPI variance for {name}: "
                f"n0={n0}, n1={n1}"
            )

        est = float(np.mean(score_r0) + np.mean(correction_r1))

        pi_hat = n1 / n_fold
        s2_r0 = float(np.var(score_r0, ddof=1))
        s2_corr = float(np.var(correction_r1, ddof=1))

        sigma_hat = s2_r0 / (1.0 - pi_hat) + s2_corr / pi_hat
        var = sigma_hat / n_fold

        # Equivalent expression:
        # var = s2_r0 / n0 + s2_corr / n1
        return est, float(var)

    res_dict = {}
    for metric in ["ipcw", "dr"]:
        # Oracle / Classical / Naive: ordinary mean-score variance in the target fold.
        res_dict[f"ora_{metric}_est"], res_dict[f"ora_{metric}_var"] = mean_score_stats(
            res_ora[metric], f"ora_{metric}"
        )
        res_dict[f"cla_{metric}_est"], res_dict[f"cla_{metric}_var"] = mean_score_stats(
            res_cla[metric], f"cla_{metric}"
        )
        res_dict[f"nai_{metric}_est"], res_dict[f"nai_{metric}_var"] = mean_score_stats(
            res_t1[metric], f"nai_{metric}"
        )

        # PPI: R=0 imputed score plus R=1 correction score.
        # correction = H(true X; S_3) - H(imputed X; S_2)
        correction = res_cla[metric] - res_t2[metric]
        res_dict[f"ppi_{metric}_est"], res_dict[f"ppi_{metric}_var"] = ppi_score_stats(
            res_t1[metric], correction, f"ppi_{metric}"
        )

    return res_dict


# ==============================================================================
# 4. single simulation function 
# ==============================================================================
def run_single_sim(
    seed: int,
    n_pre: int,
    n_inf: int,
    p_label: float,
    tau: float,
    predictor: LimiXPredictor,
    theta0: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    
    dt_pre = generate_data(n_pre, rng, tau, verbose=False)
    dt_inf = generate_data(n_inf, rng, tau, verbose=False)
    r_bern = rng.binomial(1, p_label, size=n_inf).astype(np.float64)
    
    x45_hat = limix_impute_x4x5(dt_pre, dt_inf, predictor)

    dt_inf["Xhat4"] = x45_hat[:, 0]
    dt_inf["Xhat5"] = x45_hat[:, 1]
    dt_inf["R"] = r_bern

    # --- 2-Fold Split ---
    idx_perm = rng.permutation(n_inf)
    split_point = n_inf // 2
    idx_A = idx_perm[:split_point]
    idx_B = idx_perm[split_point:]

    dt_A = dt_inf.iloc[idx_A].copy()
    dt_B = dt_inf.iloc[idx_B].copy()

    # --- Cross-fitting ---
    res_B = compute_fold_theta_separate(dt_A, dt_B, tau) # Train on A, Target is B
    res_A = compute_fold_theta_separate(dt_B, dt_A, tau) # Train on B, Target is A


    n_A = len(dt_A)
    n_B = len(dt_B)
    n_total = n_A + n_B
    w_A = n_A / n_total 
    w_B = n_B / n_total

    # aggregate helper function: apply DML1 rule to merge point estimates and variances, output the required 6 statistics
    def combine_dml1(prefix):
        # 1. aggregate point estimates: w_A * est_A + w_B * est_B
        est = w_A * res_A[f"{prefix}_est"] + w_B * res_B[f"{prefix}_est"]
        
        # 2. aggregate variances: (w_A^2) * var_A + (w_B^2) * var_B
        var = (w_A**2) * res_A[f"{prefix}_var"] + (w_B**2) * res_B[f"{prefix}_var"]
        
        # 3. calculate standard error and confidence interval
        se = np.sqrt(var)
        ci_l = est - 1.96 * se
        ci_u = est + 1.96 * se
        width = ci_u - ci_l
        
        # 4. calculate coverage
        cover = 1.0 if (ci_l <= theta0 <= ci_u) else 0.0
        
        return [est, se, ci_l, ci_u, width, cover]

    results = [
        combine_dml1("ora_ipcw"),
        combine_dml1("cla_ipcw"),
        combine_dml1("nai_ipcw"),
        combine_dml1("ppi_ipcw"),
        combine_dml1("ora_dr"),
        combine_dml1("cla_dr"),
        combine_dml1("nai_dr"),
        combine_dml1("ppi_dr"),
    ]
    return np.array(results, dtype=np.float64)


# ==============================================================================
# environment and model loading
# ==============================================================================
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
    p = hf_hub_download(
        repo_id="stableai-org/LimiX-16M",
        filename="LimiX-16M.ckpt",
        local_dir=os.path.join(root, "cache"),
    )
    return p

def resolve_limiX_device(device_arg: str) -> torch.device:
    s = (device_arg or "auto").strip().lower()
    if s in ("", "auto", "default"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Five-covariate survival 2-Fold CF + SEPARATE Models")
    p.add_argument("--m", type=int, default=30, help="Monte Carlo replicates")
    p.add_argument("--seed-gold", type=int, default=999)
    p.add_argument("--theta-n", type=int, default=10_000_000)
    p.add_argument("--n-pre", type=int, default=1000)
    p.add_argument("--n-inf", type=int, default=10000)
    p.add_argument("--p-label", type=float, default=0.10)
    p.add_argument("--tau", type=float, default=1.4)
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--inference_config", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--predictor-seed", type=int, default=0)
    p.add_argument("--base-seed", type=int, default=1)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    root = _LIMIX_ROOT
    try:
        os.chdir(root)
    except OSError:
        pass
    
    rng_g = np.random.default_rng(args.seed_gold)
    theta0, censoring_rate = truth_theta0_rmst(rng_g, args.theta_n, args.tau)
    print("theta0 (E[T_true]):", theta0, "Censoring rate:", censoring_rate, flush=True)

    ck = load_or_fetch_ckpt((args.model_path or None) if args.model_path else None, root)
    icfg = args.inference_config or os.path.join(root, "config", "reg_default_noretrieval_MVI.json")
    
    dev = resolve_limiX_device(args.device)
    pred = LimiXPredictor(
        device=dev,
        model_path=ck,
        inference_config=icfg,
        mask_prediction=True,
        seed=args.predictor_seed,
    )

    M = max(1, int(args.m))
    names = [
        "IPCW_Oracle",
        "IPCW_Classical",
        "IPCW_Naive",
        "IPCW_PPI",
        "DR_Oracle",
        "DR_Classical",
        "DR_Naive",
        "DR_PPI",
    ]
    
    # result array shape: (M replicates, 8 methods, 6 metrics [Est, SE, CI_L, CI_U, Width, Coverage])
    res = np.zeros((M, len(names), 6), dtype=np.float64)
    
    for m2 in range(1, M + 1):
        print(f"replicate {m2}/{M} ...", flush=True)
        res[m2 - 1, :, :] = run_single_sim(
            int(args.base_seed) + m2,
            n_pre=args.n_pre,
            n_inf=args.n_inf,
            p_label=args.p_label,
            tau=args.tau,
            predictor=pred,
            theta0=theta0,
        )

    # extract the Monte Carlo results and calculate the final indicators
    res_mean = res.mean(axis=0)  # Shape (8, 6)
    
    est_mean = res_mean[:, 0]
    est_se = res_mean[:, 1]      # Method's own estimated average SE
    width_mean = res_mean[:, 4]  # average CI width
    cr_mean = res_mean[:, 5]     # coverage rate (Coverage Rate)

    bias = est_mean - theta0
    sd = res.std(axis=0, ddof=0)[:, 0] # Empirical SD (empirical standard deviation of the point estimates)
    rmse = np.sqrt(bias**2 + sd**2)
    
    out = pd.DataFrame({
        "Method": [f"{i+1}. {n}" for i, n in enumerate(names)],
        "Bias": np.round(bias, 6),
        "EmpSD": np.round(sd, 6),
        "EstSE": np.round(est_se, 6),
        "RMSE": np.round(rmse, 6),
        "CR": np.round(cr_mean, 6),
        "Width": np.round(width_mean, 6),
    })
    
    print(f"\n--- Final Results (2-Fold LimiX MVI + Separate Models) (M = {M}) ---\n")
    print(f"Missing rate: {1 - args.p_label:.2f}")
    print(f"Censoring rate: {censoring_rate:.4f}")
    print("----------------------------------------------------------------------------------")
    print("Note: EmpSD = Empirical Standard Deviation of Estimates across simulations")
    print("      EstSE = Average of the method's own estimated Standard Error")
    print("      CR = Empirical Coverage Rate of the 95% Confidence Interval")
    print("      Width = Average width of the 95% Confidence Interval")
    print("----------------------------------------------------------------------------------")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()