import os
import glob
import gc
import tempfile
import traceback
import faulthandler
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

# Better defaults for multi-process execution on servers:
# prevent MKL / OpenBLAS / OMP from oversubscribing CPU cores
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Make matplotlib / numba cache writable on servers
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

try:
    faulthandler.enable()
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from numba import njit

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


PILOT_NAMES = ["LLCV", "LSCV", "Campbell"]
PRED_BW_METHODS = ["Fixed"]
ESTIMATOR_MODELS = {
    "Uncorrected": False,
}
COLORS = {
    "Fixed": "black",
}
STYLES = {
    "Fixed": "-",
}

KERNEL_IDS = {
    "Gaussian": 0,
    "Epanechnikov": 1,
    "Triangular": 2,
    "Biweight": 3,
    "Uniform": 4,
}


# ============================================================
# 1) Kernel library
# ============================================================
class Kernels:
    @staticmethod
    def Gaussian(t, h):
        x = t / h
        return (1 / h) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

    @staticmethod
    def Triangular(t, h):
        x = t / h
        return np.where(np.abs(x) <= 1, (1 - np.abs(x)) / h, 0)

    @staticmethod
    def Biweight(t, h):
        x = t / h
        return np.where(np.abs(x) <= 1, (15 / 16) * ((1 - x**2) ** 2) / h, 0)

    @staticmethod
    def Epanechnikov(t, h):
        x = t / h
        return np.where(np.abs(x) <= 1, 0.75 * (1 - x**2) / h, 0)

    @staticmethod
    def Uniform(t, h):
        x = t / h
        return np.where(np.abs(x) <= 1, 0.5 / h, 0)


def get_kernel_map():
    return {
        "Gaussian": Kernels.Gaussian,
        "Epanechnikov": Kernels.Epanechnikov,
        "Triangular": Kernels.Triangular,
        "Biweight": Kernels.Biweight,
        "Uniform": Kernels.Uniform,
    }


def _kernel_id_from_func(kernel_func):
    name = getattr(kernel_func, "__name__", None)
    if name not in KERNEL_IDS:
        raise ValueError(f"Unsupported kernel function: {name}")
    return KERNEL_IDS[name]


# ============================================================
# 1.5) NUMBA cores
# ============================================================
@njit(cache=True)
def _kernel_scalar_numba(t, h, kernel_id):
    x = t / h
    ax = abs(x)

    if kernel_id == 0:  # Gaussian
        return (1.0 / h) * (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)
    elif kernel_id == 1:  # Epanechnikov
        if ax <= 1.0:
            return 0.75 * (1.0 - x * x) / h
        return 0.0
    elif kernel_id == 2:  # Triangular
        if ax <= 1.0:
            return (1.0 - ax) / h
        return 0.0
    elif kernel_id == 3:  # Biweight
        if ax <= 1.0:
            return (15.0 / 16.0) * ((1.0 - x * x) ** 2) / h
        return 0.0
    else:  # Uniform
        if ax <= 1.0:
            return 0.5 / h
        return 0.0


@njit(cache=True)
def _trapz_numba(y, x):
    n = len(x)
    if n <= 1:
        return 0.0
    s = 0.0
    for i in range(n - 1):
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return s


@njit(cache=True)
def _cumulative_trapz_numba(y, x):
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return out
    s = 0.0
    for i in range(n - 1):
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
        out[i + 1] = s
    return out


@njit(cache=True)
def _get_intensity_scalar_numba(x_grid, X_events, h, kernel_id):
    n_grid = len(x_grid)
    n_evt = len(X_events)
    out = np.zeros(n_grid, dtype=np.float64)

    for i in range(n_grid):
        xi = x_grid[i]
        s = 0.0
        for j in range(n_evt):
            s += _kernel_scalar_numba(xi - X_events[j], h, kernel_id)
        out[i] = s
    return out


@njit(cache=True)
def _get_intensity_vector_numba(x_grid, X_events, h_vec, kernel_id):
    n_grid = len(x_grid)
    n_evt = len(X_events)
    out = np.zeros(n_grid, dtype=np.float64)

    for i in range(n_grid):
        xi = x_grid[i]
        s = 0.0
        for j in range(n_evt):
            s += _kernel_scalar_numba(xi - X_events[j], h_vec[j], kernel_id)
        out[i] = s
    return out


@njit(cache=True)
def _boundary_corr_scalar_numba(u_grid, h, kernel_id, window_left, window_right, v_grid_size):
    n_u = len(u_grid)
    p_u = np.zeros(n_u, dtype=np.float64)

    if v_grid_size <= 1:
        for i in range(n_u):
            p_u[i] = 1.0
        return p_u

    v_grid = np.linspace(window_left, window_right, v_grid_size)

    for i in range(n_u):
        ui = u_grid[i]
        vals = np.empty(v_grid_size, dtype=np.float64)
        for k in range(v_grid_size):
            vals[k] = _kernel_scalar_numba(ui - v_grid[k], h, kernel_id)
        p = _trapz_numba(vals, v_grid)
        if p < 1e-12:
            p = 1e-12
        elif p > 1.0:
            p = 1.0
        p_u[i] = p

    return p_u


@njit(cache=True)
def _get_intensity_corrected_scalar_numba(
    u_grid,
    U_events,
    h,
    kernel_id,
    window_left,
    window_right,
    v_grid_size,
):
    n_u = len(u_grid)
    n_evt = len(U_events)
    out = np.zeros(n_u, dtype=np.float64)

    if n_evt == 0:
        return out

    p_u = _boundary_corr_scalar_numba(
        u_grid, h, kernel_id, window_left, window_right, v_grid_size
    )

    for i in range(n_u):
        ui = u_grid[i]
        s = 0.0
        for j in range(n_evt):
            s += _kernel_scalar_numba(ui - U_events[j], h, kernel_id)
        out[i] = s / p_u[i]

    return out


@njit(cache=True)
def _get_intensity_corrected_vector_numba(
    u_grid,
    U_events,
    h_vec,
    kernel_id,
    window_left,
    window_right,
    v_grid_size,
):
    n_u = len(u_grid)
    n_evt = len(U_events)
    out = np.zeros(n_u, dtype=np.float64)

    if n_evt == 0:
        return out

    if v_grid_size <= 1:
        for i in range(n_u):
            ui = u_grid[i]
            s = 0.0
            for j in range(n_evt):
                s += _kernel_scalar_numba(ui - U_events[j], h_vec[j], kernel_id)
            out[i] = s
        return out

    v_grid = np.linspace(window_left, window_right, v_grid_size)

    for i in range(n_u):
        ui = u_grid[i]
        s = 0.0
        for j in range(n_evt):
            hj = h_vec[j]
            num = _kernel_scalar_numba(ui - U_events[j], hj, kernel_id)

            vals = np.empty(v_grid_size, dtype=np.float64)
            for k in range(v_grid_size):
                vals[k] = _kernel_scalar_numba(ui - v_grid[k], hj, kernel_id)

            p = _trapz_numba(vals, v_grid)
            if p < 1e-12:
                p = 1e-12
            elif p > 1.0:
                p = 1.0

            s += num / p
        out[i] = s

    return out


@njit(cache=True)
def _loo_row_sums_scalar_numba(X_obs, h, kernel_id):
    n = len(X_obs)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        xi = X_obs[i]
        s = 0.0
        for j in range(n):
            if i != j:
                s += _kernel_scalar_numba(xi - X_obs[j], h, kernel_id)
        out[i] = s
    return out


@njit(cache=True)
def _weighted_pred_intensity_from_khat_scalar_numba(x_grid, X_events, h, k_hat_at_events, kernel_id):
    n_grid = len(x_grid)
    n_evt = len(X_events)
    out = np.zeros(n_grid, dtype=np.float64)

    if n_evt == 0:
        return out

    for i in range(n_grid):
        xi = x_grid[i]

        # Gaussian: numerically stable normalized local weights.
        # The common Gaussian factor cancels in the ratio, preserving
        # the original formula while avoiding far-tail underflow.
        if kernel_id == 0:
            max_logw = -1.0e308

            for j in range(n_evt):
                z = (xi - X_events[j]) / h
                logw = -0.5 * z * z
                if logw > max_logw:
                    max_logw = logw

            numer = 0.0
            denom = 0.0
            for j in range(n_evt):
                z = (xi - X_events[j]) / h
                logw = -0.5 * z * z
                w = np.exp(logw - max_logw)
                denom += w
                numer += w * k_hat_at_events[j]

            if denom <= 0.0:
                out[i] = 0.0
            else:
                val = numer / denom
                out[i] = val if val > 0.0 else 0.0

        # Compact-support kernels: zero denominator outside support is genuine.
        else:
            numer = 0.0
            denom = 0.0
            for j in range(n_evt):
                kval = _kernel_scalar_numba(xi - X_events[j], h, kernel_id)
                denom += kval
                numer += kval * k_hat_at_events[j]

            if denom < 1e-12:
                out[i] = 0.0
            else:
                val = numer / denom
                out[i] = val if val > 0.0 else 0.0

    return out


@njit(cache=True)
def _weighted_pred_intensity_from_khat_vector_numba(x_grid, X_events, h_vec, k_hat_at_events, kernel_id):
    n_grid = len(x_grid)
    n_evt = len(X_events)
    out = np.zeros(n_grid, dtype=np.float64)

    for i in range(n_grid):
        xi = x_grid[i]
        numer = 0.0
        denom = 0.0
        for j in range(n_evt):
            kval = _kernel_scalar_numba(xi - X_events[j], h_vec[j], kernel_id)
            denom += kval
            numer += kval * k_hat_at_events[j]

        if denom < 1e-12:
            out[i] = 0.0
        else:
            val = numer / denom
            out[i] = val if val > 0.0 else 0.0

    return out


# ============================================================
# 2) Intensity estimation and bandwidth scoring
# ============================================================
def get_intensity(x_grid, X_events, h, kernel_func):
    x_grid = np.asarray(x_grid, dtype=np.float64)
    X_events = np.asarray(X_events, dtype=np.float64)
    kernel_id = _kernel_id_from_func(kernel_func)

    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 0:
        return _get_intensity_scalar_numba(x_grid, X_events, float(h_arr), kernel_id)

    if h_arr.ndim != 1 or h_arr.size != len(X_events):
        raise ValueError("When h is a vector, its length must equal len(X_events).")

    return _get_intensity_vector_numba(x_grid, X_events, h_arr, kernel_id)


def get_intensity_corrected(
    u_grid,
    U_events,
    h,
    kernel_func,
    window_left=0.0,
    window_right=1.0,
    v_grid_size=200,
):
    u_grid = np.asarray(u_grid, dtype=np.float64)
    U_events = np.asarray(U_events, dtype=np.float64)

    if len(U_events) == 0:
        return np.zeros_like(u_grid, dtype=np.float64)

    window_left = float(window_left)
    window_right = float(window_right)
    if window_right <= window_left:
        raise ValueError("window_right must be greater than window_left")

    kernel_id = _kernel_id_from_func(kernel_func)

    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 0:
        return _get_intensity_corrected_scalar_numba(
            u_grid,
            U_events,
            float(h_arr),
            kernel_id,
            window_left,
            window_right,
            int(v_grid_size),
        )

    if h_arr.ndim == 1 and h_arr.size > 0 and np.allclose(h_arr, h_arr[0]):
        return _get_intensity_corrected_scalar_numba(
            u_grid,
            U_events,
            float(h_arr[0]),
            kernel_id,
            window_left,
            window_right,
            int(v_grid_size),
        )

    if h_arr.ndim != 1 or h_arr.size != len(U_events):
        raise ValueError("When h is a vector, its length must equal len(U_events).")

    return _get_intensity_corrected_vector_numba(
        u_grid,
        U_events,
        h_arr,
        kernel_id,
        window_left,
        window_right,
        int(v_grid_size),
    )


def evaluate_intensity_at_events(
    U_events_eval,
    U_events_train,
    h,
    kernel_func,
    corrected=False,
    window_left=0.0,
    window_right=1.0,
    v_grid_size=200,
):
    U_events_eval = np.asarray(U_events_eval, dtype=np.float64)
    U_events_train = np.asarray(U_events_train, dtype=np.float64)

    if corrected:
        return get_intensity_corrected(
            U_events_eval,
            U_events_train,
            h,
            kernel_func,
            window_left=window_left,
            window_right=window_right,
            v_grid_size=v_grid_size,
        )

    return get_intensity(U_events_eval, U_events_train, h, kernel_func)


def compute_campbell_score(
    U_events,
    h,
    kernel_func,
    corrected=False,
    window_left=0.0,
    window_right=1.0,
    v_grid_size=200,
):
    t_total = window_right - window_left

    lam_at_events = evaluate_intensity_at_events(
        U_events_eval=U_events,
        U_events_train=U_events,
        h=h,
        kernel_func=kernel_func,
        corrected=corrected,
        window_left=window_left,
        window_right=window_right,
        v_grid_size=v_grid_size,
    )

    lam_at_events = np.maximum(lam_at_events, 1e-12)
    t_kappa = np.sum(1.0 / lam_at_events)
    return (t_kappa - t_total) ** 2


def evaluate_score_values(
    X_obs,
    h,
    x_grid_obs,
    kernel_func,
    corrected=False,
    v_grid_size=200,
):
    obs_left, obs_right = 0.0, 1.0
    h = float(h)
    kernel_id = _kernel_id_from_func(kernel_func)

    if corrected:
        lam = get_intensity_corrected(
            x_grid_obs,
            X_obs,
            h,
            kernel_func,
            window_left=obs_left,
            window_right=obs_right,
            v_grid_size=v_grid_size,
        )
        lam_at_events = get_intensity_corrected(
            X_obs,
            X_obs,
            h,
            kernel_func,
            window_left=obs_left,
            window_right=obs_right,
            v_grid_size=v_grid_size,
        )

        row_sums = _loo_row_sums_scalar_numba(
            np.asarray(X_obs, dtype=np.float64),
            h,
            kernel_id,
        )
        p_u = _boundary_corr_scalar_numba(
            np.asarray(X_obs, dtype=np.float64),
            h,
            kernel_id,
            obs_left,
            obs_right,
            int(v_grid_size),
        )
        lam_loo = row_sums / np.maximum(p_u, 1e-12)
    else:
        lam = get_intensity(x_grid_obs, X_obs, h, kernel_func)
        lam_at_events = np.interp(X_obs, x_grid_obs, lam)
        lam_loo = _loo_row_sums_scalar_numba(
            np.asarray(X_obs, dtype=np.float64),
            h,
            kernel_id,
        )

    integral = _trapz_numba(
        np.asarray(lam, dtype=np.float64),
        np.asarray(x_grid_obs, dtype=np.float64),
    )
    s_ml = np.sum(np.log(np.maximum(lam_at_events, 1e-12))) - integral
    s_llcv = np.sum(np.log(np.maximum(lam_loo, 1e-12))) - integral

    term1 = _trapz_numba(
        np.asarray(lam ** 2, dtype=np.float64),
        np.asarray(x_grid_obs, dtype=np.float64),
    )
    term2 = 2 * np.sum(lam_loo)
    s_lscv = term1 - term2

    s_campbell = compute_campbell_score(
        X_obs,
        h,
        kernel_func,
        corrected=corrected,
        window_left=obs_left,
        window_right=obs_right,
        v_grid_size=v_grid_size,
    )
    return s_ml, s_llcv, s_lscv, s_campbell


def compute_scores(
    X_obs,
    h_cands,
    x_grid_obs,
    kernel_func,
    corrected=False,
    v_grid_size=200,
):
    s_ml, s_llcv, s_lscv, s_campbell = [], [], [], []

    for h in h_cands:
        ml, llcv, lscv, campbell = evaluate_score_values(
            X_obs,
            h,
            x_grid_obs,
            kernel_func,
            corrected=corrected,
            v_grid_size=v_grid_size,
        )
        s_ml.append(ml)
        s_llcv.append(llcv)
        s_lscv.append(lscv)
        s_campbell.append(campbell)

    return np.array(s_ml), np.array(s_llcv), np.array(s_lscv), np.array(s_campbell)


def optimize_pilot_bandwidths(
    X_obs,
    h_bounds,
    h_coarse,
    x_grid_obs,
    kernel_func,
    corrected=False,
    v_grid_size=200,
    fine_points=25,
):
    h_min, h_max = float(h_bounds[0]), float(h_bounds[1])
    h_coarse = np.asarray(h_coarse, dtype=float)

    _, llcv_grid, lscv_grid, campbell_grid = compute_scores(
        X_obs,
        h_coarse,
        x_grid_obs,
        kernel_func,
        corrected=corrected,
        v_grid_size=v_grid_size,
    )

    coarse_best_idx = {
        "LLCV": int(np.argmax(llcv_grid)),
        "LSCV": int(np.argmin(lscv_grid)),
        "Campbell": int(np.argmin(campbell_grid)),
    }

    def make_objective(name):
        def obj(h):
            _, llcv_val, lscv_val, campbell_val = evaluate_score_values(
                X_obs,
                h,
                x_grid_obs,
                kernel_func,
                corrected=corrected,
                v_grid_size=v_grid_size,
            )
            if name == "LLCV":
                return -llcv_val
            if name == "LSCV":
                return lscv_val
            if name == "Campbell":
                return campbell_val
            raise ValueError(f"Unknown pilot criterion: {name}")

        return obj

    def best_index(values, name):
        if name == "LLCV":
            return int(np.argmax(values))
        return int(np.argmin(values))

    pilot_opts = {}
    optimize_info = {}
    n_coarse = len(h_coarse)

    for name, idx_coarse in coarse_best_idx.items():
        coarse_left = h_coarse[max(0, idx_coarse - 1)]
        coarse_right = h_coarse[min(n_coarse - 1, idx_coarse + 1)]

        local_left = max(h_min, min(coarse_left, coarse_right))
        local_right = min(h_max, max(coarse_left, coarse_right))

        if local_right <= local_left:
            local_left, local_right = h_min, h_max

        h_fine = np.logspace(
            np.log10(local_left),
            np.log10(local_right),
            fine_points,
        )

        _, llcv_fine, lscv_fine, campbell_fine = compute_scores(
            X_obs,
            h_fine,
            x_grid_obs,
            kernel_func,
            corrected=corrected,
            v_grid_size=v_grid_size,
        )

        if name == "LLCV":
            fine_curve = llcv_fine
        elif name == "LSCV":
            fine_curve = lscv_fine
        else:
            fine_curve = campbell_fine

        idx_fine = best_index(fine_curve, name)
        fine_h = float(h_fine[idx_fine])

        n_fine = len(h_fine)
        fine_left = h_fine[max(0, idx_fine - 1)]
        fine_right = h_fine[min(n_fine - 1, idx_fine + 1)]

        micro_left = max(h_min, min(fine_left, fine_right))
        micro_right = min(h_max, max(fine_left, fine_right))

        if micro_right <= micro_left:
            micro_left, micro_right = local_left, local_right

        res = minimize_scalar(
            make_objective(name),
            bounds=(micro_left, micro_right),
            method="bounded",
            options={"xatol": 1e-6, "maxiter": 200},
        )

        if res.success and np.isfinite(res.x):
            h_star = float(np.clip(res.x, h_min, h_max))
        else:
            h_star = fine_h

        pilot_opts[name] = h_star
        optimize_info[name] = {
            "coarse_h": float(h_coarse[idx_coarse]),
            "coarse_bounds": (float(local_left), float(local_right)),
            "fine_h": float(fine_h),
            "fine_bounds": (float(micro_left), float(micro_right)),
            "success": bool(res.success),
        }

    return pilot_opts, (llcv_grid, lscv_grid, campbell_grid), optimize_info


# ============================================================
# 3) Prediction intensity
# ============================================================
def get_fixed_prediction_bandwidth(h_pilot):
    return float(h_pilot)


def weighted_pred_intensity(
    x_grid,
    X_events,
    h,
    kernel_func,
    corrected=False,
    history_right=None,
    v_grid_size=200,
):
    x_grid = np.asarray(x_grid, dtype=np.float64)
    X_events = np.asarray(X_events, dtype=np.float64)
    kernel_id = _kernel_id_from_func(kernel_func)

    if corrected:
        if history_right is None:
            history_right = float(np.max(X_events))
        k_hat_at_events = get_intensity_corrected(
            X_events,
            X_events,
            h,
            kernel_func,
            window_left=0.0,
            window_right=float(history_right),
            v_grid_size=v_grid_size,
        )
    else:
        k_hat_at_events = get_intensity(X_events, X_events, h, kernel_func)

    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 0:
        return _weighted_pred_intensity_from_khat_scalar_numba(
            x_grid,
            X_events,
            float(h_arr),
            np.asarray(k_hat_at_events, dtype=np.float64),
            kernel_id,
        )

    if h_arr.ndim != 1 or h_arr.size != len(X_events):
        raise ValueError("When h is a vector, its length must equal len(X_events).")

    return _weighted_pred_intensity_from_khat_vector_numba(
        x_grid,
        X_events,
        h_arr,
        np.asarray(k_hat_at_events, dtype=np.float64),
        kernel_id,
    )


def cumulative_trapz(y, x):
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return _cumulative_trapz_numba(y, x)


def find_next_event_time(
    x_cur,
    X_events,
    h,
    kernel_func,
    x_end,
    base_ngrid=600,
    max_expand=60,
    verbose=False,
    debug_expand=False,
    corrected=False,
    history_right=None,
    v_grid_size=200,
):
    if x_cur >= x_end:
        return x_end, False

    if history_right is None:
        history_right = float(np.max(X_events))

    h_eff = float(np.mean(h)) if np.ndim(h) > 0 else float(h)

    initial_span = max(2.0 * h_eff, 1e-3)
    total_span = max(x_end - x_cur, 0.0)
    upper = min(x_end, x_cur + initial_span)

    # Ensure that the search can always reach x_end.
    # max_expand is only a minimum safety budget.
    if total_span > initial_span and initial_span > 0.0:
        needed_expand = int(np.ceil(np.log2(total_span / initial_span))) + 2
    else:
        needed_expand = 1
    effective_max_expand = max(int(max_expand), needed_expand)

    for expand_i in range(effective_max_expand):
        if upper <= x_cur + 1e-12:
            upper = min(x_end, x_cur + 1e-3)

        # Increase integration resolution as the searched interval expands.
        span_ratio = max((upper - x_cur) / initial_span, 1.0)
        n_grid = int(min(5000, max(base_ngrid, np.ceil(base_ngrid * span_ratio))))

        x_grid = np.linspace(x_cur, upper, n_grid)
        kp = weighted_pred_intensity(
            x_grid,
            X_events,
            h,
            kernel_func,
            corrected=corrected,
            history_right=history_right,
            v_grid_size=v_grid_size,
        )
        cum = cumulative_trapz(kp, x_grid)
        integral_val = cum[-1]

        if verbose and debug_expand:
            print(
                f"        [expand {expand_i+1:02d}] "
                f"x_cur={x_cur:.10f} upper={upper:.10f} integral={integral_val:.6f}"
            )

        if integral_val >= 1.0:
            x_next = np.interp(1.0, cum, x_grid)
            if verbose and debug_expand:
                print(
                    f"        [hit] integral>=1, "
                    f"x_next={x_next:.10f} (upper={upper:.10f})"
                )
            return float(x_next), True

        if np.isclose(upper, x_end) or upper >= x_end:
            if verbose and debug_expand:
                print(
                    f"        [end] reached x_end but integral<1: "
                    f"return x_end={x_end:.10f}, integral={integral_val:.6f}"
                )
            return float(x_end), False

        new_upper = min(x_end, x_cur + 2.0 * (upper - x_cur))
        if new_upper <= upper + 1e-12:
            upper = x_end
        else:
            upper = new_upper

    # Final fallback: explicitly check the complete remaining interval.
    x_grid = np.linspace(x_cur, x_end, 5000)
    kp = weighted_pred_intensity(
        x_grid,
        X_events,
        h,
        kernel_func,
        corrected=corrected,
        history_right=history_right,
        v_grid_size=v_grid_size,
    )
    cum = cumulative_trapz(kp, x_grid)
    integral_val = cum[-1]

    if integral_val >= 1.0:
        x_next = np.interp(1.0, cum, x_grid)
        return float(x_next), True

    if verbose and debug_expand:
        print(
            f"        [final-check] reached x_end but integral<1: "
            f"return x_end={x_end:.10f}, integral={integral_val:.6f}"
        )
    return float(x_end), False


def recursive_predict(
    X_obs,
    h_pilot,
    kernel_func,
    x_end,
    max_steps=10000,
    verbose=False,
    debug_expand=False,
    pred_bw_method="Fixed",
    corrected=False,
    v_grid_size=200,
    save_step_details=False,
):
    X_events = list(np.asarray(X_obs, dtype=float))
    X_events.sort()

    preds = []
    flags = []
    step_records = []
    bandwidth_detail_rows = []
    x_cur = X_events[-1]

    for step_idx in range(1, max_steps + 1):
        if x_cur >= x_end - 1e-12:
            break

        X_events_before = np.asarray(sorted(X_events), dtype=float)
        history_count_before = int(len(X_events_before))
        history_right = float(np.max(X_events_before))

        h_used = get_fixed_prediction_bandwidth(h_pilot)

        h_used_arr = np.asarray(h_used, dtype=float)
        if h_used_arr.ndim == 0:
            h_used_mean = float(h_used_arr)
            h_used_min = float(h_used_arr)
            h_used_max = float(h_used_arr)
            h_vector_type = "scalar"
        else:
            h_used_mean = float(np.mean(h_used_arr))
            h_used_min = float(np.min(h_used_arr))
            h_used_max = float(np.max(h_used_arr))
            h_vector_type = "vector"

        x_next, ok = find_next_event_time(
            x_cur,
            X_events_before,
            h_used,
            kernel_func,
            x_end,
            verbose=verbose,
            debug_expand=debug_expand,
            corrected=corrected,
            history_right=history_right,
            v_grid_size=v_grid_size,
        )

        if save_step_details:
            step_records.append(
                {
                    "step": int(step_idx),
                    "history_count_before": history_count_before,
                    "x_cur": float(x_cur),
                    "history_right": history_right,
                    "h_pilot": float(h_pilot),
                    "pred_bw_method": pred_bw_method,
                    "h_vector_type": h_vector_type,
                    "h_used_mean": h_used_mean,
                    "h_used_min": h_used_min,
                    "h_used_max": h_used_max,
                    "x_next": float(x_next),
                    "ok": bool(ok),
                }
            )

            if h_used_arr.ndim == 0:
                bandwidth_detail_rows.append(
                    {
                        "step": int(step_idx),
                        "hist_event_index": -1,
                        "hist_event_time": np.nan,
                        "h_value": float(h_used_arr),
                        "note": "scalar_bandwidth",
                    }
                )
            else:
                for hist_idx, (event_time, h_val) in enumerate(zip(X_events_before, h_used_arr), start=1):
                    bandwidth_detail_rows.append(
                        {
                            "step": int(step_idx),
                            "hist_event_index": int(hist_idx),
                            "hist_event_time": float(event_time),
                            "h_value": float(h_val),
                            "note": "eventwise_bandwidth",
                        }
                    )

        if not ok or x_next <= x_cur + 1e-10:
            break

        preds.append(x_next)
        flags.append(True)
        X_events.append(x_next)
        X_events.sort()
        x_cur = x_next

        if np.isclose(x_cur, x_end) or x_cur >= x_end:
            break

    return (
        np.array(preds, dtype=float),
        np.array(flags, dtype=bool),
        step_records,
        bandwidth_detail_rows,
    )


# ============================================================
# 4) One parallel task
# ============================================================
def analyze_single_configuration(
    data_name,
    raw_sorted,
    frac,
    kernel_name,
    model_name,
    corrected_flag,
    h_bounds,
    h_coarse,
    x_grid_obs,
    v_grid_size,
):
    kernel_func = get_kernel_map()[kernel_name]

    raw_sorted = np.sort(np.asarray(raw_sorted, dtype=float))
    n_all = len(raw_sorted)

    m = max(3, int(np.ceil(n_all * frac)))
    raw_obs = raw_sorted[:m]
    raw_val = raw_sorted[m:]

    t_cut = raw_obs[-1]
    X_obs = np.sort(raw_obs / t_cut)
    X_all = raw_sorted / t_cut
    X_val = raw_val / t_cut
    x_end = float(X_all[-1])

    emp_x_all = np.concatenate(([0.0], X_all, [x_end]))
    emp_y_all = np.concatenate(([0.0], np.arange(1, n_all + 1), [n_all]))

    pilot_opts, (llcv, lscv, campbell), optimize_info = optimize_pilot_bandwidths(
        X_obs,
        h_bounds=h_bounds,
        h_coarse=h_coarse,
        x_grid_obs=x_grid_obs,
        kernel_func=kernel_func,
        corrected=corrected_flag,
        v_grid_size=v_grid_size,
    )

    pilot_results = {}
    all_rows = []

    for pilot_name in PILOT_NAMES:
        h_pilot = float(pilot_opts[pilot_name])
        metrics_lines = []
        bw_results = {}

        for bw_name in PRED_BW_METHODS:
            preds, ok_flags, step_records, bandwidth_detail_rows = recursive_predict(
                X_obs,
                h_pilot,
                kernel_func,
                x_end,
                verbose=False,
                debug_expand=False,
                pred_bw_method=bw_name,
                corrected=corrected_flag,
                v_grid_size=v_grid_size,
                save_step_details=True,
            )

            X_pred_all = np.concatenate([X_obs, preds])
            X_pred_all.sort()
            y_pred = np.arange(1, len(X_pred_all) + 1)
            x_knots = np.concatenate(([0.0], X_pred_all, [x_end]))
            y_knots = np.concatenate(
                ([0.0], y_pred, [y_pred[-1] if len(y_pred) else 0.0])
            )

            valid_len = min(len(preds), len(X_val))
            if valid_len > 0:
                err = preds[:valid_len] - X_val[:valid_len]
                mae = float(np.mean(np.abs(err)))
                rmse = float(np.sqrt(np.mean(err**2)))
            else:
                mae, rmse = np.nan, np.nan

            ok_rate = float(np.mean(ok_flags)) if len(ok_flags) else np.nan
            tce = int(len(preds) - len(X_val))

            if (x_end - 1.0) <= 1e-12:
                ct_strict = np.nan
                ct_time = np.nan
            else:
                if len(preds) > 0:
                    last_pred = float(min(preds[-1], x_end))
                    ct_time = float((last_pred - 1.0) / (x_end - 1.0))

                    if np.any(ok_flags):
                        t_last_ok = float(preds[np.where(ok_flags)[0][-1]])
                    else:
                        t_last_ok = 1.0
                    ct_strict = float((t_last_ok - 1.0) / (x_end - 1.0))
                else:
                    ct_time = 0.0
                    ct_strict = 0.0

            if len(X_val) > 0:
                m_obs = len(X_obs)
                y_true_at_val = m_obs + np.arange(1, len(X_val) + 1)
                a_at_val = np.interp(X_val, x_knots, y_knots)
                ae_curve = np.abs(y_true_at_val - a_at_val)
                pae_mae = float(np.mean(ae_curve))
                pae_rmse = float(np.sqrt(np.mean(ae_curve**2)))
                pae_max = float(np.max(ae_curve))
            else:
                pae_mae, pae_rmse, pae_max = np.nan, np.nan, np.nan

            metrics_line = (
                f"[Model={model_name} | {kernel_name} | Pilot={pilot_name} | "
                f"PredBW={bw_name}] h_pilot={h_pilot:.8f} | "
                f"pred_steps={len(preds)} | "
                f"Ct_strict={ct_strict:.4f} | Ct_time={ct_time:.4f} | "
                f"TCE={tce:+d} | ok_rate={ok_rate:.3f} | "
                f"MAE_time={mae:.6f} | RMSE_time={rmse:.6f} | "
                f"PAE_MAE={pae_mae:.4f} | PAE_RMSE={pae_rmse:.4f} | "
                f"PAE_MAX={pae_max:.4f}"
            )
            metrics_lines.append(metrics_line)

            row = {
                "dataset": data_name,
                "obs_pct": int(frac * 100),
                "kernel": kernel_name,
                "estimator_model": model_name,
                "pilot_criterion": pilot_name,
                "h_pilot": float(h_pilot),
                "pred_bw_method": bw_name,
                "pred_steps": int(len(preds)),
                "val_count": int(len(X_val)),
                "TCE": int(tce),
                "Ct_strict": float(ct_strict) if np.isfinite(ct_strict) else np.nan,
                "Ct_time": float(ct_time) if np.isfinite(ct_time) else np.nan,
                "ok_rate": float(ok_rate) if np.isfinite(ok_rate) else np.nan,
                "MAE_time": float(mae) if np.isfinite(mae) else np.nan,
                "RMSE_time": float(rmse) if np.isfinite(rmse) else np.nan,
                "PAE_MAE": float(pae_mae) if np.isfinite(pae_mae) else np.nan,
                "PAE_RMSE": float(pae_rmse) if np.isfinite(pae_rmse) else np.nan,
                "PAE_MAX": float(pae_max) if np.isfinite(pae_max) else np.nan,
            }
            all_rows.append(row)

            predicted_events_df = pd.DataFrame(
                {
                    "pred_step": np.arange(1, len(preds) + 1, dtype=int),
                    "pred_time_normalized": preds,
                    "ok_flag": ok_flags.astype(bool) if len(ok_flags) else np.array([], dtype=bool),
                }
            )

            curve_points_df = pd.DataFrame(
                {
                    "point_index": np.arange(1, len(x_knots) + 1, dtype=int),
                    "x_knots": x_knots,
                    "y_knots": y_knots,
                }
            )

            bw_results[bw_name] = {
                "x_knots": x_knots,
                "y_knots": y_knots,
                "preds": preds,
                "ok_flags": ok_flags,
                "step_records": step_records,
                "bandwidth_detail_rows": bandwidth_detail_rows,
                "predicted_events_df": predicted_events_df,
                "curve_points_df": curve_points_df,
                "summary_row": row.copy(),
            }

        pilot_results[pilot_name] = {
            "h_pilot": h_pilot,
            "metrics_lines": metrics_lines,
            "bw_results": bw_results,
        }

    return {
        "data_name": data_name,
        "obs_pct": int(frac * 100),
        "frac": frac,
        "kernel_name": kernel_name,
        "model_name": model_name,
        "n_all": n_all,
        "t_cut": float(t_cut),
        "x_end": x_end,
        "raw_obs": raw_obs,
        "raw_val": raw_val,
        "raw_sorted": raw_sorted,
        "X_obs": X_obs,
        "X_all": X_all,
        "X_val": X_val,
        "emp_x_all": emp_x_all,
        "emp_y_all": emp_y_all,
        "llcv": llcv,
        "lscv": lscv,
        "campbell": campbell,
        "pilot_opts": pilot_opts,
        "optimize_info": optimize_info,
        "pilot_results": pilot_results,
        "all_rows": all_rows,
        "h_coarse": np.asarray(h_coarse, dtype=float),
    }


# ============================================================
# 5) Output writing
# ============================================================
def append_rows_to_csv(rows, csv_path):
    if not rows:
        return

    df = pd.DataFrame(rows)

    if os.path.exists(csv_path):
        df.to_csv(
            csv_path,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        df.to_csv(
            csv_path,
            mode="w",
            header=True,
            index=False,
            encoding="utf-8-sig",
        )


def write_configuration_outputs(result, frac_folder, h_coarse):
    data_name = result["data_name"]
    kernel_name = result["kernel_name"]
    model_name = result["model_name"]
    obs_pct = result["obs_pct"]
    x_end = result["x_end"]
    X_all = result["X_all"]
    emp_x_all = result["emp_x_all"]
    emp_y_all = result["emp_y_all"]
    n_all = result["n_all"]
    X_obs = result["X_obs"]
    llcv = result["llcv"]
    lscv = result["lscv"]
    campbell = result["campbell"]
    pilot_opts = result["pilot_opts"]
    optimize_info = result["optimize_info"]
    pilot_results = result["pilot_results"]

    fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle(
        f"{data_name} | {kernel_name} | {model_name}\n"
        f"First {obs_pct}% observed (n={len(X_obs)}), normalized endpoint = 1 | "
        f"Bandwidth Selection (LLCV / LSCV / Campbell)",
        fontsize=14,
    )

    axes[0].semilogx(h_coarse, llcv, "b")
    axes[0].set_title(f"LLCV (h*={pilot_opts['LLCV']:.6f})")

    axes[1].semilogx(h_coarse, lscv, "orange")
    axes[1].set_title(f"LSCV (h*={pilot_opts['LSCV']:.6f})")

    axes[2].semilogx(h_coarse, campbell, "green")
    axes[2].set_title(f"Campbell (h*={pilot_opts['Campbell']:.6f})")

    for ax_bw, pilot_name in zip(axes, PILOT_NAMES):
        ax_bw.axvline(optimize_info[pilot_name]["coarse_h"], color="gray", linestyle="--", alpha=0.5)
        ax_bw.axvline(optimize_info[pilot_name]["fine_h"], color="black", linestyle=":", alpha=0.5)
        ax_bw.axvline(pilot_opts[pilot_name], color="red", linestyle="-", alpha=0.5)
        ax_bw.grid(True, which="both", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        os.path.join(
            frac_folder,
            f"{data_name}_{kernel_name}_{model_name}_1_BWSelect_LLCV_LSCV_Campbell_OBS{obs_pct}.png",
        )
    )
    plt.close(fig1)

    for pilot_name in PILOT_NAMES:
        h_pilot = pilot_results[pilot_name]["h_pilot"]
        metrics_lines = pilot_results[pilot_name]["metrics_lines"]
        bw_results = pilot_results[pilot_name]["bw_results"]

        fig2, ax = plt.subplots(1, 1, figsize=(11, 6))
        ax.step(
            emp_x_all,
            emp_y_all,
            where="post",
            color="gray",
            lw=3,
            alpha=0.5,
            label="True Data (step)",
            zorder=5,
        )
        ax.axvline(1.0, linestyle="--", alpha=0.6)
        ax.scatter(X_all, np.arange(1, n_all + 1), s=10, alpha=0.25)

        for bw_name in PRED_BW_METHODS:
            x_knots = bw_results[bw_name]["x_knots"]
            y_knots = bw_results[bw_name]["y_knots"]
            pred_steps = max(len(x_knots) - len(X_obs) - 2, 0)
            ax.plot(
                x_knots,
                y_knots,
                lw=1.8,
                alpha=0.9,
                color=COLORS[bw_name],
                linestyle=STYLES[bw_name],
                label=f"{bw_name} a(x) (steps={pred_steps})",
            )

        ax.set_title(
            f"{data_name} | {kernel_name} | {model_name} | OBS={obs_pct}% | "
            f"Pilot={pilot_name} (LLCV/LSCV/Campbell, h={h_pilot:.4f})\n"
            f"Prediction Bandwidth: Fixed | "
            f"Predicted curve = piecewise-linear a(x)"
        )
        ax.set_xlabel("Normalized time x (observation endpoint = 1, validation > 1)")
        ax.set_ylabel("Cumulative failure count")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize="small", ncol=2)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                frac_folder,
                f"{data_name}_{kernel_name}_{model_name}_2_ComparePredBW_{pilot_name}_LLCV_LSCV_Campbell_OBS{obs_pct}.png",
            )
        )
        plt.close(fig2)

        metrics_path = os.path.join(
            frac_folder,
            f"{data_name}_{kernel_name}_{model_name}_Metrics_{pilot_name}_LLCV_LSCV_Campbell_OBS{obs_pct}.txt",
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(
                f"data={data_name}\n"
                f"kernel={kernel_name}\n"
                f"model={model_name}\n"
                f"OBS={obs_pct}%\n"
            )
            f.write(f"pilot_criterion={pilot_name}\n")
            f.write(f"h_pilot={h_pilot}\n")
            f.write(f"t_cut={result['t_cut']}\n")
            f.write(f"x_end={x_end}\n\n")
            f.write("\n".join(metrics_lines))

        for line in metrics_lines:
            print("          ", line, flush=True)


def _autosize_worksheet(ws, min_width=10, max_width=40):
    for col_cells in ws.columns:
        col_idx = col_cells[0].column
        max_len = 0
        for cell in col_cells:
            try:
                cell_value = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(cell_value))
            except Exception:
                pass
        ws.column_dimensions[get_column_letter(col_idx)].width = max(min_width, min(max_len + 2, max_width))


def _format_header_row(ws, row_idx=1):
    fill = PatternFill("solid", fgColor="1F4E78")
    font = Font(color="FFFFFF", bold=True)
    for cell in ws[row_idx]:
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal="center", vertical="center")


def _freeze_and_filter(ws):
    ws.freeze_panes = "A2"
    if ws.max_row >= 1 and ws.max_column >= 1:
        ws.auto_filter.ref = ws.dimensions


def write_case_excel(result, frac_folder):
    os.makedirs(frac_folder, exist_ok=True)

    data_name = result["data_name"]
    kernel_name = result["kernel_name"]
    model_name = result["model_name"]
    obs_pct = result["obs_pct"]

    excel_path = os.path.join(
        frac_folder,
        f"{data_name}_{kernel_name}_{model_name}_CaseResults_OBS{obs_pct}.xlsx",
    )

    meta_rows = [
        ("dataset", result["data_name"]),
        ("kernel", result["kernel_name"]),
        ("model", result["model_name"]),
        ("obs_pct", result["obs_pct"]),
        ("frac", result["frac"]),
        ("n_all", result["n_all"]),
        ("t_cut", result["t_cut"]),
        ("x_end", result["x_end"]),
        ("obs_count", len(result["X_obs"])),
        ("val_count", len(result["X_val"])),
    ]
    meta_df = pd.DataFrame(meta_rows, columns=["field", "value"])

    summary_df = pd.DataFrame(result["all_rows"]).copy()

    bw_curve_df = pd.DataFrame(
        {
            "h_coarse": np.asarray(result["h_coarse"], dtype=float),
            "LLCV": np.asarray(result["llcv"], dtype=float),
            "LSCV": np.asarray(result["lscv"], dtype=float),
            "Campbell": np.asarray(result["campbell"], dtype=float),
        }
    )

    optimize_rows = []
    for pilot_name, info in result["optimize_info"].items():
        optimize_rows.append(
            {
                "pilot_criterion": pilot_name,
                "h_star": float(result["pilot_opts"][pilot_name]),
                "coarse_h": info["coarse_h"],
                "coarse_left": info["coarse_bounds"][0],
                "coarse_right": info["coarse_bounds"][1],
                "fine_h": info["fine_h"],
                "fine_left": info["fine_bounds"][0],
                "fine_right": info["fine_bounds"][1],
                "success": info["success"],
            }
        )
    optimize_df = pd.DataFrame(optimize_rows)

    predicted_events_all = []
    curve_points_all = []
    step_records_all = []

    for pilot_name, pilot_pack in result["pilot_results"].items():
        h_pilot = pilot_pack["h_pilot"]
        bw_results = pilot_pack["bw_results"]

        for bw_name, bw_pack in bw_results.items():
            pred_df = bw_pack["predicted_events_df"].copy()
            pred_df.insert(0, "pred_bw_method", bw_name)
            pred_df.insert(0, "pilot_criterion", pilot_name)
            pred_df.insert(0, "h_pilot", h_pilot)
            predicted_events_all.append(pred_df)

            curve_df = bw_pack["curve_points_df"].copy()
            curve_df.insert(0, "pred_bw_method", bw_name)
            curve_df.insert(0, "pilot_criterion", pilot_name)
            curve_df.insert(0, "h_pilot", h_pilot)
            curve_points_all.append(curve_df)

            if bw_pack["step_records"]:
                step_df = pd.DataFrame(bw_pack["step_records"]).copy()
            else:
                step_df = pd.DataFrame(
                    columns=[
                        "step",
                        "history_count_before",
                        "x_cur",
                        "history_right",
                        "h_pilot",
                        "pred_bw_method",
                        "h_vector_type",
                        "h_used_mean",
                        "h_used_min",
                        "h_used_max",
                        "x_next",
                        "ok",
                    ]
                )
            step_df.insert(0, "pred_bw_method_case", bw_name)
            step_df.insert(0, "pilot_criterion_case", pilot_name)
            step_records_all.append(step_df)


    predicted_events_df = (
        pd.concat(predicted_events_all, ignore_index=True)
        if predicted_events_all
        else pd.DataFrame()
    )
    curve_points_df = (
        pd.concat(curve_points_all, ignore_index=True)
        if curve_points_all
        else pd.DataFrame()
    )
    step_records_df = (
        pd.concat(step_records_all, ignore_index=True)
        if step_records_all
        else pd.DataFrame()
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        meta_df.to_excel(writer, sheet_name="Meta", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        bw_curve_df.to_excel(writer, sheet_name="Bandwidth_Curves", index=False)
        optimize_df.to_excel(writer, sheet_name="Optimize_Info", index=False)
        predicted_events_df.to_excel(writer, sheet_name="Predicted_Events", index=False)
        curve_points_df.to_excel(writer, sheet_name="Curve_Points", index=False)
        step_records_df.to_excel(writer, sheet_name="Step_Records", index=False)

    wb = load_workbook(excel_path)

    for ws in wb.worksheets:
        if ws.max_row >= 1:
            _format_header_row(ws, 1)
        _freeze_and_filter(ws)
        _autosize_worksheet(ws)

    if "Summary" in wb.sheetnames:
        ws = wb["Summary"]
        header_map = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
        for name in ["h_pilot", "Ct_strict", "Ct_time", "ok_rate", "MAE_time", "RMSE_time", "PAE_MAE", "PAE_RMSE", "PAE_MAX"]:
            if name in header_map:
                col = header_map[name]
                for r in range(2, ws.max_row + 1):
                    ws.cell(r, col).number_format = "0.000000"

    if "Bandwidth_Curves" in wb.sheetnames:
        ws = wb["Bandwidth_Curves"]
        for c in range(1, ws.max_column + 1):
            for r in range(2, ws.max_row + 1):
                ws.cell(r, c).number_format = "0.000000"

    if "Optimize_Info" in wb.sheetnames:
        ws = wb["Optimize_Info"]
        for c in range(1, ws.max_column + 1):
            for r in range(2, ws.max_row + 1):
                if isinstance(ws.cell(r, c).value, (int, float)):
                    ws.cell(r, c).number_format = "0.000000"

    for sheet_name in ["Predicted_Events", "Curve_Points", "Step_Records"]:
        if sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for c in range(1, ws.max_column + 1):
                for r in range(2, ws.max_row + 1):
                    val = ws.cell(r, c).value
                    if isinstance(val, float):
                        ws.cell(r, c).number_format = "0.000000"

    wb.save(excel_path)
    print(f"[Excel saved] {os.path.abspath(excel_path)}", flush=True)


# ============================================================
# 5.5) Server helpers
# ============================================================
def recommend_max_workers(requested=None):
    """
    Safer default for servers:
    use at most half CPU cores, capped at 8 workers.
    """
    cpu = os.cpu_count() or 2
    safe_default = max(1, min(cpu // 2, 8))

    if requested is None:
        return safe_default

    requested = int(requested)
    return max(1, min(requested, cpu))


def init_worker():
    """
    Warm up numba in each worker to reduce first-task latency spikes.
    """
    try:
        faulthandler.enable()
    except Exception:
        pass

    try:
        x_grid = np.linspace(0.0, 1.0, 8, dtype=np.float64)
        X_events = np.array([0.1, 0.4, 0.9], dtype=np.float64)

        _ = _get_intensity_scalar_numba(x_grid, X_events, 0.2, 0)
        _ = _get_intensity_corrected_scalar_numba(
            x_grid, X_events, 0.2, 0, 0.0, 1.0, 16
        )
        _ = _cumulative_trapz_numba(x_grid, x_grid)
    except Exception as e:
        print(f"[Worker init warning] {e}", flush=True)


def write_failed_tasks_log(failed_rows, base_folder):
    if not failed_rows:
        return

    failed_csv = os.path.join(base_folder, "failed_tasks.csv")
    failed_txt = os.path.join(base_folder, "failed_tasks.txt")

    df = pd.DataFrame(failed_rows)
    df.to_csv(failed_csv, index=False, encoding="utf-8-sig")

    with open(failed_txt, "w", encoding="utf-8") as f:
        for i, row in enumerate(failed_rows, start=1):
            f.write(f"[{i}]\n")
            for k, v in row.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    print(f"[Failed task log saved] {os.path.abspath(failed_csv)}", flush=True)


# ============================================================
# 6) Batch runner
# ============================================================
def run_batch_analysis(
    dataset_dict,
    base_folder="FixedBW_Out",
    fractions=(0.2, 0.5, 0.8),
    v_grid_size=200,
    max_workers=None,
    mp_start_method="spawn",
    warmup_workers=True,
):
    os.makedirs(base_folder, exist_ok=True)

    print("Current working directory (cwd):", os.getcwd(), flush=True)
    print("Output root directory (abs):", os.path.abspath(base_folder), flush=True)

    kernel_map = get_kernel_map()

    h_bounds = (0.01, 1.0)
    h_coarse = np.logspace(np.log10(h_bounds[0]), np.log10(h_bounds[1]), 40)
    x_grid_obs = np.linspace(0, 1, 2000)

    max_workers = recommend_max_workers(max_workers)

    task_specs = []
    failed_rows = []
    detail_csv_path = os.path.join(base_folder, "detail_results_append.csv")

    for data_name, raw_data in dataset_dict.items():
        raw_sorted = np.sort(np.asarray(raw_data, dtype=float))
        n_all = len(raw_sorted)
        sub_folder = os.path.join(base_folder, data_name)
        os.makedirs(sub_folder, exist_ok=True)

        print(f"\n[Dataset] {data_name} | n_all={n_all} -> {sub_folder}", flush=True)

        for frac in fractions:
            m = max(3, int(np.ceil(n_all * frac)))
            raw_obs = raw_sorted[:m]
            raw_val = raw_sorted[m:]
            t_cut = raw_obs[-1]
            x_end = float(raw_sorted[-1] / t_cut)

            frac_folder = os.path.join(sub_folder, f"OBS_{int(frac * 100)}pct")
            os.makedirs(frac_folder, exist_ok=True)

            print(
                f"  [Observation ratio] {int(frac*100)}% | "
                f"obs={len(raw_obs)}, val={len(raw_val)} | "
                f"t_cut={t_cut:.4f} | x_end={x_end:.4f}",
                flush=True,
            )

            for kernel_name in kernel_map.keys():
                for model_name, corrected_flag in ESTIMATOR_MODELS.items():
                    task_specs.append(
                        {
                            "data_name": data_name,
                            "raw_sorted": raw_sorted,
                            "frac": frac,
                            "kernel_name": kernel_name,
                            "model_name": model_name,
                            "corrected_flag": corrected_flag,
                            "h_bounds": h_bounds,
                            "h_coarse": h_coarse,
                            "x_grid_obs": x_grid_obs,
                            "v_grid_size": v_grid_size,
                            "frac_folder": frac_folder,
                        }
                    )

    print(
        f"\n[Parallel execution] tasks={len(task_specs)} | "
        f"max_workers={max_workers} | "
        f"backend=ProcessPoolExecutor | "
        f"start_method={mp_start_method} | "
        f"numba_warmup={warmup_workers}",
        flush=True,
    )

    mp_context = mp.get_context(mp_start_method)
    initializer = init_worker if warmup_workers else None

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=initializer,
    ) as executor:
        future_map = {}

        for spec in task_specs:
            future = executor.submit(
                analyze_single_configuration,
                spec["data_name"],
                spec["raw_sorted"],
                spec["frac"],
                spec["kernel_name"],
                spec["model_name"],
                spec["corrected_flag"],
                spec["h_bounds"],
                spec["h_coarse"],
                spec["x_grid_obs"],
                spec["v_grid_size"],
            )
            future_map[future] = spec

        total_tasks = len(future_map)

        for done_idx, future in enumerate(as_completed(future_map), start=1):
            spec = future_map[future]

            try:
                result = future.result()
            except Exception as e:
                err_text = "".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                print(
                    f"[Task failed] dataset={spec['data_name']} | "
                    f"OBS={int(spec['frac']*100)}% | "
                    f"kernel={spec['kernel_name']} | "
                    f"model={spec['model_name']}\n{err_text}",
                    flush=True,
                )
                failed_rows.append(
                    {
                        "stage": "compute",
                        "dataset": spec["data_name"],
                        "obs_pct": int(spec["frac"] * 100),
                        "kernel": spec["kernel_name"],
                        "model": spec["model_name"],
                        "error": str(e),
                    }
                )
                continue

            try:
                print(
                    f"    > Finished compute: dataset={result['data_name']} | "
                    f"OBS={result['obs_pct']}% | "
                    f"kernel={result['kernel_name']} | "
                    f"model={result['model_name']}",
                    flush=True,
                )

                write_configuration_outputs(result, spec["frac_folder"], h_coarse)
                write_case_excel(result, spec["frac_folder"])
                append_rows_to_csv(result["all_rows"], detail_csv_path)

            except Exception as e:
                err_text = "".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                print(
                    f"[Output failed] dataset={spec['data_name']} | "
                    f"OBS={int(spec['frac']*100)}% | "
                    f"kernel={spec['kernel_name']} | "
                    f"model={spec['model_name']}\n{err_text}",
                    flush=True,
                )
                failed_rows.append(
                    {
                        "stage": "output",
                        "dataset": spec["data_name"],
                        "obs_pct": int(spec["frac"] * 100),
                        "kernel": spec["kernel_name"],
                        "model": spec["model_name"],
                        "error": str(e),
                    }
                )
            finally:
                try:
                    del result
                except Exception:
                    pass
                gc.collect()

            print(f"[Progress] {done_idx}/{total_tasks} tasks handled", flush=True)

    if os.path.exists(detail_csv_path):
        print("\n[Append CSV saved]", os.path.abspath(detail_csv_path), flush=True)
    else:
        print("\n[No CSV generated] No results were collected.", flush=True)

    write_failed_tasks_log(failed_rows, base_folder)


# ============================================================
# 7) Data loading
# ============================================================
def load_datasets_from_data_folder(data_folder="DATA"):
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, data_folder)

    dataset_dict = {}

    if not os.path.exists(data_dir):
        print(f"Error: folder not found: {data_dir}")
        return dataset_dict

    file_list = glob.glob(os.path.join(data_dir, "*.xlsx"))

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        dataset_name = os.path.splitext(file_name)[0]

        try:
            df = pd.read_excel(file_path, header=None)
            raw_data = df.iloc[:, 0].dropna().values

            if len(raw_data) > 0:
                dataset_dict[dataset_name] = np.sort(raw_data.astype(float))
                print(f"Successfully loaded dataset: {dataset_name} | Number of events: {len(raw_data)}")
            else:
                print(f"Skipped file {file_name}: empty data")

        except Exception as e:
            print(f"Skipped file {file_name}: failed to read ({e})")

    return dataset_dict


# ============================================================
# 8) Main entry
# ============================================================
if __name__ == "__main__":
    freeze_support()

    my_datasets = load_datasets_from_data_folder("SDATA")

    if not my_datasets:
        print("No valid dataset detected. Program exits. Please check whether the DATA folder contains .xlsx files.")
    else:
        print(f"[Loaded Excel datasets] {list(my_datasets.keys())}")

        out_dir = "FixedBW_Out_ServerStable_20260402"

        run_batch_analysis(
            my_datasets,
            base_folder=out_dir,
            fractions=(0.2, 0.5, 0.8),
            v_grid_size=1000,
            max_workers=6,          # safer default: auto choose
            mp_start_method="spawn",   # safest on servers
            warmup_workers=True,       # reduce first-task JIT spikes
        )

        print(f"\n[All tasks completed] Results saved in: {os.path.abspath(out_dir)}")