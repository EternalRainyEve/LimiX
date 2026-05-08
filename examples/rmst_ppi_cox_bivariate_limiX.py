"""
Five-covariate survival simulation with LimiX missing value imputation:
Strict Original PPI Formula + 2-Fold Cross-Fitting + Separate Nuisance Models
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


# LimiX package root
_LIMIX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LIMIX_ROOT not in sys.path:
    sys.path.insert(0, _LIMIX_ROOT)
from inference.predictor import LimiXPredictor  # noqa: E402


COVARIATE_COLS = ["X1", "X2", "X3", "X4", "X5"]
IMPUTED_COLS = ["X4", "X5"]


# ==============================================================================
# 1. 数据生成函数 (DGP) - 保持不变
# ==============================================================================
def generate_data(
    n: int,
    rng: np.random.Generator,
    rho: float = 0.7,
    tau: float = 1.4,
    c_base_hazard: float = 1.05
) -> pd.DataFrame:
    cov = np.full((5, 5), rho, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)
    x1, x2, x3, x4, x5 = rng.multivariate_normal(np.zeros(5), cov, size=n).T

    beta_t = np.array([-0.5, 0.8, 0.4, 2.1, -1.3])
    risk_score_t = np.exp(
        beta_t[0] * x1 + beta_t[1] * x2 + beta_t[2] * x3 + beta_t[3] * x4 + beta_t[4] * x5
    )
    
    nu_t = 2.0     
    lambda_t = 1.0  
    u_t = rng.random(n) 
    
    t_true = (-np.log(u_t) / (lambda_t * risk_score_t)) ** (1.0 / nu_t)
    t_trunc = np.minimum(t_true, tau)

    beta_c = np.array([0.1, 0.2, -0.1, 0.6, -0.5])
    risk_score_c = np.exp(
        beta_c[0] * x1 + beta_c[1] * x2 + beta_c[2] * x3 + beta_c[3] * x4 + beta_c[4] * x5
    )
    
    nu_c = 1.0
    lambda_c = c_base_hazard  
    u_c = rng.random(n)
    
    c = (-np.log(u_c) / (lambda_c * risk_score_c)) ** (1.0 / nu_c)

    y = np.minimum(t_trunc, c)
    delta = (t_trunc <= c).astype(np.float64)

    return pd.DataFrame({
        "X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5,
        "T_true": t_trunc, "Y": y, "Delta": delta,
    })

def truth_theta0_rmst(
    rng: np.random.Generator, n_big: int, tau: float
) -> tuple[float, float]:
    d = generate_data(n_big, rng, tau=tau)
    censor_rate = 1 - float(d["Delta"].mean())
    return float(d["T_true"].mean()), censor_rate


# ==============================================================================
# 2. 核心辅助函数 - 保持不变
# ==============================================================================
def _cumulative_baseline_on_grid(cph: CoxPHFitter, grid_t: np.ndarray) -> np.ndarray:
    base = cph.baseline_cumulative_hazard_.iloc[:, 0]
    t_bh = base.index.to_numpy(dtype=np.float64)
    h_bh = base.to_numpy(dtype=np.float64)
    if len(t_bh) == 0: return np.zeros_like(grid_t, dtype=np.float64)
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

def compute_scores(newdata: pd.DataFrame, cph_t: CoxPHFitter, cph_c: CoxPHFitter, tau: float, x_cols: List[str]) -> dict:
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
    dh0_c[0] = 0.0; dh0_c[1:] = h0_c[1:] - h0_c[:-1]

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

def limix_impute_x4x5(pre: pd.DataFrame, inf: pd.DataFrame, predictor: LimiXPredictor, n_anchor: int = 16) -> np.ndarray:
    x_train = pre[COVARIATE_COLS].to_numpy(np.float64)
    anchor_y = np.zeros(len(x_train), dtype=np.float64)

    n_anchor = max(1, min(n_anchor, len(pre)))
    anchor_x = pre[COVARIATE_COLS].iloc[:n_anchor].to_numpy(np.float64)
    inf_x = inf[COVARIATE_COLS].to_numpy(np.float64)
    inf_x[:, [COVARIATE_COLS.index(col) for col in IMPUTED_COLS]] = np.nan

    x_test = np.vstack([anchor_x, inf_x])
    _, reconstructed_x = predictor.predict(x_train, anchor_y, x_test, task_type="Regression")
    if reconstructed_x is None:
        raise RuntimeError("LimiX predictor failed.")

    reconstructed_test = reconstructed_x[-len(x_test):]
    imputed_inf = reconstructed_test[n_anchor:]
    imputed_indices = [COVARIATE_COLS.index(col) for col in IMPUTED_COLS]
    return imputed_inf[:, imputed_indices].astype(np.float64, copy=False)


# ==============================================================================
# 3. 提取各个独立子集的得分向量 (严格遵循原初 PPI 设定)
# ==============================================================================
def compute_fold_theta_separate(
    dt_train: pd.DataFrame, dt_target: pd.DataFrame, tau: float
) -> dict:
    
    x_cols = COVARIATE_COLS
    
    mask_r1_train = dt_train["R"] == 1.0
    mask_r0_train = dt_train["R"] == 0.0
    
    mask_r1_target = dt_target["R"] == 1.0
    mask_r0_target = dt_target["R"] == 0.0
    
    n0_target = mask_r0_target.sum()
    n1_target = mask_r1_target.sum()
    
    # --- Oracle (All Target Data, True X) ---
    cox_t_ora = fit_cox_t(dt_train, x_cols)
    cox_c_ora = fit_cox_c(dt_train, x_cols)
    res_ora = compute_scores(dt_target, cox_t_ora, cox_c_ora, tau, x_cols)
    
    dt_train_proxy = dt_train.copy()
    dt_train_proxy["X4"], dt_train_proxy["X5"] = dt_train["Xhat4"], dt_train["Xhat5"]
    dt_target_proxy = dt_target.copy()
    dt_target_proxy["X4"], dt_target_proxy["X5"] = dt_target["Xhat4"], dt_target["Xhat5"]
    
    # --- Naive (Only R=0 Target Data, ML Imputed X) ---
    if mask_r0_train.sum() > 0 and n0_target > 0:
        cox_t_r0 = fit_cox_t(dt_train_proxy[mask_r0_train], x_cols)
        cox_c_r0 = fit_cox_c(dt_train_proxy[mask_r0_train], x_cols)
        res_naive = compute_scores(dt_target_proxy[mask_r0_target], cox_t_r0, cox_c_r0, tau, x_cols)
    else:
        res_naive = {"ipcw": np.array([]), "dr": np.array([])}

    # --- Classical & Rectifier Term (Only R=1 Target Data) ---
    if mask_r1_train.sum() > 0 and n1_target > 0:
        # Classical (True X on R=1)
        cox_t_cla = fit_cox_t(dt_train[mask_r1_train], x_cols)
        cox_c_cla = fit_cox_c(dt_train[mask_r1_train], x_cols)
        res_class = compute_scores(dt_target[mask_r1_target], cox_t_cla, cox_c_cla, tau, x_cols)
        
        # ML Imputed X on R=1
        cox_t_ml1 = fit_cox_t(dt_train_proxy[mask_r1_train], x_cols)
        cox_c_ml1 = fit_cox_c(dt_train_proxy[mask_r1_train], x_cols)
        res_ml1 = compute_scores(dt_target_proxy[mask_r1_target], cox_t_ml1, cox_c_ml1, tau, x_cols)
    else:
        res_class = {"ipcw": np.array([]), "dr": np.array([])}
        res_ml1 = {"ipcw": np.array([]), "dr": np.array([])}
    
    return {
        "ora_ipcw": res_ora["ipcw"], "ora_dr": res_ora["dr"],
        "naive_ipcw": res_naive["ipcw"], "naive_dr": res_naive["dr"],
        "class_ipcw": res_class["ipcw"], "class_dr": res_class["dr"],
        "ml1_ipcw": res_ml1["ipcw"], "ml1_dr": res_ml1["dr"]
    }


# ==============================================================================
# 评估工具函数：包含 Coverage 和 CI Width 的严格方差计算
# ==============================================================================
def calc_metrics_standard(scores: np.ndarray, truth: float) -> tuple[float, float, float]:
    """For Oracle, Classical, and Naive estimators"""
    if len(scores) == 0: return 0.0, 0.0, 0.0
    est = np.mean(scores)
    var = np.var(scores, ddof=1) / len(scores)
    se = np.sqrt(var)
    cov = 1.0 if (est - 1.96 * se) <= truth <= (est + 1.96 * se) else 0.0
    return est, cov, 2 * 1.96 * se

def calc_metrics_ppi(scores_r0_ml: np.ndarray, scores_r1_true: np.ndarray, scores_r1_ml: np.ndarray, truth: float) -> tuple[float, float, float]:
    """For Strict Original PPI Estimator"""
    if len(scores_r0_ml) == 0 or len(scores_r1_true) == 0: return 0.0, 0.0, 0.0
    
    n0 = len(scores_r0_ml)
    n1 = len(scores_r1_true)
    
    # 完美对齐用户公式：PPI = Mean(ML on R=0) - Mean(ML on R=1) + Mean(True on R=1)
    est_naive = np.mean(scores_r0_ml)
    est_diff = np.mean(scores_r1_true - scores_r1_ml)
    est_ppi = est_naive + est_diff
    
    # 完美对齐 PPI 论文方差公式：Var(Naive)/n0 + Var(True - ML)/n1
    var_naive = np.var(scores_r0_ml, ddof=1) / n0
    var_diff = np.var(scores_r1_true - scores_r1_ml, ddof=1) / n1
    se_ppi = np.sqrt(var_naive + var_diff)
    
    cov = 1.0 if (est_ppi - 1.96 * se_ppi) <= truth <= (est_ppi + 1.96 * se_ppi) else 0.0
    return est_ppi, cov, 2 * 1.96 * se_ppi


# ==============================================================================
# 4. 单次模拟函数 (无缝拼接两折得分)
# ==============================================================================
def run_single_sim(
    seed: int,
    n_pre: int,
    n_inf: int,
    p_label: float,
    tau: float,
    predictor: LimiXPredictor,
    theta0: float
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    
    pre = generate_data(n_pre, rng, tau=tau)
    dt_inf = generate_data(n_inf, rng, tau=tau)
    r_bern = rng.binomial(1, p_label, size=n_inf).astype(np.float64)
    
    x45_hat = limix_impute_x4x5(pre, dt_inf, predictor)
    dt_inf["Xhat4"] = x45_hat[:, 0]
    dt_inf["Xhat5"] = x45_hat[:, 1]
    dt_inf["R"] = r_bern

    idx_perm = rng.permutation(n_inf)
    split_point = n_inf // 2
    idx_A = idx_perm[:split_point]
    idx_B = idx_perm[split_point:]

    dt_A = dt_inf.iloc[idx_A].copy()
    dt_B = dt_inf.iloc[idx_B].copy()

    # 交叉提取各折 OOF (Out-Of-Fold) 得分
    res_B = compute_fold_theta_separate(dt_A, dt_B, tau)
    res_A = compute_fold_theta_separate(dt_B, dt_A, tau)

    results = []
    
    for metric_type in ["ipcw", "dr"]:
        # 拼接全量 OOF 得分
        ora_all = np.concatenate([res_B[f"ora_{metric_type}"], res_A[f"ora_{metric_type}"]])
        naive_all = np.concatenate([res_B[f"naive_{metric_type}"], res_A[f"naive_{metric_type}"]])
        class_all = np.concatenate([res_B[f"class_{metric_type}"], res_A[f"class_{metric_type}"]])
        ml1_all = np.concatenate([res_B[f"ml1_{metric_type}"], res_A[f"ml1_{metric_type}"]])
        
        # 依次计算 Oracle, Classical, Naive, PPI
        results.append(list(calc_metrics_standard(ora_all, theta0)))
        results.append(list(calc_metrics_standard(class_all, theta0)))
        results.append(list(calc_metrics_standard(naive_all, theta0)))
        # 调用专属于原初 PPI 的方差计算函数！
        results.append(list(calc_metrics_ppi(naive_all, class_all, ml1_all, theta0)))

    return np.array(results, dtype=np.float64)


# ==============================================================================
# 环境及主程序 (保持不变)
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
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=50)
    p.add_argument("--seed-gold", type=int, default=999)
    p.add_argument("--theta-n", type=int, default=10_000_000)
    p.add_argument("--n-pre", type=int, default=1000)
    p.add_argument("--n-inf", type=int, default=20000)
    p.add_argument("--p-label", type=float, default=0.10)
    p.add_argument("--tau", type=float, default=1.4)
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--inference_config", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--predictor-seed", type=int, default=0)
    p.add_argument("--base-seed", type=int, default=2024)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    root = _LIMIX_ROOT
    try: os.chdir(root)
    except OSError: pass
    
    rng_g = np.random.default_rng(args.seed_gold)
    theta0, censor_rate = truth_theta0_rmst(rng_g, args.theta_n, args.tau)
    print("theta0 (E[T_true]):", theta0, "Censor rate:", censor_rate, flush=True)

    ck = load_or_fetch_ckpt((args.model_path or None) if args.model_path else None, root)
    icfg = args.inference_config or os.path.join(root, "config", "reg_default_noretrieval_MVI.json")
    
    dev = resolve_limiX_device(args.device)
    pred = LimiXPredictor(
        device=dev, model_path=ck, inference_config=icfg, mask_prediction=True, seed=args.predictor_seed,
    )

    M = max(1, int(args.m))
    names = [
        "IPCW_Oracle", "IPCW_Classical", "IPCW_Naive", "IPCW_PPI",
        "DR_Oracle", "DR_Classical", "DR_Naive", "DR_PPI",
    ]
    
    res = np.zeros((M, len(names), 3), dtype=np.float64)
    
    for m2 in range(1, M + 1):
        print(f"replicate {m2}/{M} ...", flush=True)
        res[m2 - 1, :, :] = run_single_sim(
            int(args.base_seed) + m2, args.n_pre, args.n_inf, args.p_label, args.tau, pred, theta0
        )

    point_ests = res[:, :, 0]
    bias = point_ests.mean(axis=0) - theta0
    sd = point_ests.std(axis=0, ddof=1)
    rmse = np.sqrt(bias**2 + sd**2)
    cov_rate = res[:, :, 1].mean(axis=0)
    avg_width = res[:, :, 2].mean(axis=0)
    
    out = pd.DataFrame({
        "Method": [f"{i+1}. {n}" for i, n in enumerate(names)],
        "Bias": np.round(bias, 6),
        "SD": np.round(sd, 6),
        "RMSE": np.round(rmse, 6),
        "Cov Rate": np.round(cov_rate, 4),
        "Avg Width": np.round(avg_width, 6),
    })
    
    print(f"\n--- Final Results (Strict Original PPI + 2-Fold CF) (M = {M}) ---\n")
    print(f"missing rate (Expected): {1 - args.p_label}")
    print(f"censor rate: {censor_rate}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()