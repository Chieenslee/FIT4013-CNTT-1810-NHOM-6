"""End-to-end GTSRB training, track-disjoint GroupKFold tuning, statistical testing, and plotting pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import math
import os
import platform
import sys
import time
from typing import Any

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import skimage
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .constants import CLASS_NAMES_EN, CLASS_NAMES_VI, CLASS_SHORT_VI  # noqa: E402
from .data_loader import load_rgb_image, prepare_gtsrb_csvs  # noqa: E402
from .features import (  # noqa: E402
    crop_center_margin,
    extract_hog_color_features,
    extract_hog_features,
    extract_raw_gray,
    extract_raw_rgb,
    preprocess_image,
)

RANDOM_STATE = 42
SPEED_SIGN_ORDER = {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120}


def extract_track_ids(df: pd.DataFrame) -> np.ndarray:
    """Build physical GTSRB track identifiers (`{ClassId}_{TrackPrefix}`)."""
    tracks = []
    for _, row in df.iterrows():
        basename = os.path.basename(str(row["Path"]))
        prefix = basename.split("_")[0]
        tracks.append(f"{int(row['ClassId'])}_{prefix}")
    return np.asarray(tracks, dtype=object)


def count_unique_tracks(df_train: pd.DataFrame) -> dict[str, Any]:
    """Count physical GTSRB sequence tracks using ClassId and TrackId prefix."""
    tracks: dict[str, int] = {}
    for tid in extract_track_ids(df_train):
        key = str(tid)
        tracks[key] = tracks.get(key, 0) + 1
    size_hist: dict[int, int] = {}
    for cnt in tracks.values():
        size_hist[cnt] = size_hist.get(cnt, 0) + 1
    return {
        "unique_tracks": len(tracks),
        "frames_per_track_counts": dict(sorted(size_hist.items())),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit_time: float = 0.0,
    pred_time: float = 0.0,
) -> dict[str, float]:
    """Compute Accuracy and Macro/Weighted Precision, Recall, and F1 metrics."""
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_wgt, r_wgt, f_wgt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_mac),
        "recall_macro": float(r_mac),
        "f1_macro": float(f_mac),
        "precision_weighted": float(p_wgt),
        "recall_weighted": float(r_wgt),
        "f1_weighted": float(f_wgt),
        "fit_time_sec": float(fit_time),
        "pred_time_sec": float(pred_time),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 43,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Compute exact multinomial bootstrap 95% confidence intervals for Accuracy and Macro-F1."""
    n = int(len(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    flat = cm.ravel().astype(np.float64)
    probs = flat / float(n)
    rng = np.random.default_rng(seed)

    draws = rng.multinomial(n, probs, size=n_boot).reshape(n_boot, n_classes, n_classes)
    tp = np.diagonal(draws, axis1=1, axis2=2).astype(np.float64)
    row_sum = draws.sum(axis=2).astype(np.float64)
    col_sum = draws.sum(axis=1).astype(np.float64)

    accs = tp.sum(axis=1) / float(n)
    denom = row_sum + col_sum
    f1_cls = np.where(denom > 0.0, 2.0 * tp / denom, 0.0)
    f1s = f1_cls.mean(axis=1)

    acc_lo, acc_hi = np.percentile(accs, [2.5, 97.5])
    f1_lo, f1_hi = np.percentile(f1s, [2.5, 97.5])
    return {
        "n_boot": int(n_boot),
        "accuracy_ci95": [float(acc_lo), float(acc_hi)],
        "accuracy_halfwidth": float((acc_hi - acc_lo) / 2.0),
        "f1_macro_ci95": [float(f1_lo), float(f1_hi)],
        "f1_macro_halfwidth": float((f1_hi - f1_lo) / 2.0),
    }


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, Any]:
    """Compute continuity-corrected McNemar chi-square statistic and asymptotic p-value."""
    ok_a = y_true == pred_a
    ok_b = y_true == pred_b
    n01 = int(np.sum((~ok_a) & ok_b))
    n10 = int(np.sum(ok_a & (~ok_b)))
    disc = n01 + n10
    if disc == 0:
        return {"n01_b_better": 0, "n10_a_better": 0, "chi2": 0.0, "p_value": 1.0}
    chi2_val = ((abs(n01 - n10) - 1.0) ** 2) / float(disc)
    # 1-df chi-square survival function: erfc(sqrt(chi2 / 2))
    p_val = float(math.erfc(math.sqrt(max(0.0, chi2_val) / 2.0)))
    return {
        "n01_b_better": n01,
        "n10_a_better": n10,
        "chi2": float(chi2_val),
        "p_value": p_val,
    }


def apply_holm_bonferroni(tests: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Apply step-down Holm-Bonferroni correction across a dictionary of McNemar tests."""
    keys = list(tests.keys())
    m = len(keys)
    ordered = sorted(keys, key=lambda k: float(tests[k]["p_value"]))
    running_max = 0.0
    for rank, k in enumerate(ordered):
        adj = min(1.0, float(tests[k]["p_value"]) * float(m - rank))
        running_max = max(running_max, adj)
        tests[k]["p_value_holm"] = float(running_max)
    return tests


def _wilson_ci_pct(k_err: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    """Compute exact 95% Wilson binomial confidence interval expressed in percent."""
    if n <= 0:
        return 0.0, 0.0
    p = float(k_err) / float(n)
    denom = 1.0 + (z ** 2) / float(n)
    center = (p + (z ** 2) / (2.0 * float(n))) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * float(n))) / float(n))) / denom
    return max(0.0, (center - half) * 100.0), min(100.0, (center + half) * 100.0)


def fit_logistic_error_model(
    y_true: np.ndarray,
    is_err: np.ndarray,
    widths: np.ndarray,
    brightness: np.ndarray,
) -> dict[str, Any]:
    """Fit exact MLE binary logistic regression for error risk controlling confounders."""
    y_bin = is_err.astype(np.float64)
    log2_w = np.log2(np.maximum(8.0, widths.astype(np.float64)))
    z_logw = (log2_w - log2_w.mean()) / (log2_w.std() + 1e-8)
    bright_dev = np.abs(brightness.astype(np.float64) - 120.0)
    z_bdev = (bright_dev - bright_dev.mean()) / (bright_dev.std() + 1e-8)
    is_speed = np.isin(y_true, list(SPEED_SIGN_ORDER.keys())).astype(np.float64)

    X = np.column_stack([np.ones(len(y_bin)), z_logw, z_bdev, is_speed])
    names = ["intercept", "z_log2_width", "z_abs_bright_dev_120", "is_speed_limit_sign"]

    def _nll(beta: np.ndarray) -> float:
        logits = np.clip(X @ beta, -30.0, 30.0)
        return float(np.sum(np.log1p(np.exp(logits)) - y_bin * logits))

    def _grad(beta: np.ndarray) -> np.ndarray:
        logits = np.clip(X @ beta, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return X.T @ (probs - y_bin)

    res = minimize(_nll, np.zeros(X.shape[1]), jac=_grad, method="BFGS")
    beta = res.x
    probs = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30.0, 30.0)))
    weights_w = probs * (1.0 - probs)
    fisher = (X.T * weights_w) @ X
    cov = np.linalg.inv(fisher + 1e-8 * np.eye(X.shape[1]))
    ses = np.sqrt(np.diag(cov))

    rows = []
    for name, b, se in zip(names, beta, ses):
        z_val = float(b / max(se, 1e-12))
        p_val = float(math.erfc(abs(z_val) / math.sqrt(2.0)))
        or_val = float(math.exp(b))
        ci_lo = float(math.exp(b - 1.95996 * se))
        ci_hi = float(math.exp(b + 1.95996 * se))
        rows.append(
            {
                "term": name,
                "coef": float(b),
                "se": float(se),
                "odds_ratio": or_val,
                "or_ci95": [ci_lo, ci_hi],
                "z_stat": z_val,
                "p_value": p_val,
            }
        )
    return {"n_obs": int(len(y_bin)), "error_count": int(y_bin.sum()), "terms": rows}


def _extract_margin_chunk(args: tuple[list[dict[str, Any]], str]) -> np.ndarray:
    """Extract F4 features with automatic 10% center margin trimming (no CSV ROI)."""
    records, data_root = args
    n = len(records)
    out = np.zeros((n, 1860), dtype=np.float32)
    for idx, row in enumerate(records):
        img_rgb = load_rgb_image(os.path.join(data_root, str(row["Path"])))
        trimmed = crop_center_margin(img_rgb, 0.10)
        pre = preprocess_image(trimmed, roi=None, size=(32, 32), use_roi=False, use_clahe=True)
        out[idx] = extract_hog_color_features(pre)
    return out


def _extract_chunk(args: tuple[list[dict[str, Any]], str]) -> dict[str, np.ndarray]:
    """Extract F0-F4 feature representations and metadata for a chunk of image rows."""
    records, data_root = args
    n = len(records)
    f0 = np.zeros((n, 1024), dtype=np.float32)
    f1 = np.zeros((n, 1024), dtype=np.float32)
    f2 = np.zeros((n, 3072), dtype=np.float32)
    f3 = np.zeros((n, 1764), dtype=np.float32)
    f4_noclahe = np.zeros((n, 1860), dtype=np.float32)
    f4_clahe = np.zeros((n, 1860), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    brightness = np.zeros(n, dtype=np.float32)
    roi_ratio = np.zeros(n, dtype=np.float32)

    for idx, row in enumerate(records):
        full_path = os.path.join(data_root, str(row["Path"]))
        img_rgb = load_rgb_image(full_path)
        roi = (
            int(row["Roi.X1"]),
            int(row["Roi.Y1"]),
            int(row["Roi.X2"]),
            int(row["Roi.Y2"]),
        )
        h, w = img_rgb.shape[:2]
        roi_w = max(1, roi[2] - roi[0])
        roi_h = max(1, roi[3] - roi[1])
        roi_ratio[idx] = float((roi_w * roi_h) / max(1, h * w))

        no_roi = preprocess_image(img_rgb, roi=None, use_roi=False, use_clahe=False)
        roi_raw = preprocess_image(img_rgb, roi=roi, use_roi=True, use_clahe=False)
        roi_clahe = preprocess_image(img_rgb, roi=roi, use_roi=True, use_clahe=True)

        f0[idx] = extract_raw_gray(no_roi)
        f1[idx] = extract_raw_gray(roi_raw)
        f2[idx] = extract_raw_rgb(roi_raw)
        f3[idx] = extract_hog_features(roi_clahe)
        f4_noclahe[idx] = extract_hog_color_features(roi_raw)
        f4_clahe[idx] = extract_hog_color_features(roi_clahe)
        y[idx] = int(row["ClassId"])
        brightness[idx] = float(img_rgb.mean())

    return {
        "F0": f0,
        "F1": f1,
        "F2": f2,
        "F3_noCLAHE": f4_noclahe[:, :1764].copy(),
        "F3": f3,
        "F4_noCLAHE": f4_noclahe,
        "F4": f4_clahe,
        "y": y,
        "brightness": brightness,
        "roi_ratio": roi_ratio,
    }


def load_all_images_and_features(
    df: pd.DataFrame,
    data_root: str,
    cache_path: str | None = None,
    n_workers: int | None = None,
) -> dict[str, np.ndarray]:
    """Load or extract all feature sets (F0..F4) with optional .npz caching."""
    if cache_path and os.path.exists(cache_path):
        loaded = np.load(cache_path)
        raw = {key: loaded[key] for key in loaded.files}
        legacy_map = {
            "X_no_roi_gray": "F0",
            "X_roi_gray": "F1",
            "X_roi_rgb": "F2",
            "X_hog": "F3",
            "X_no_clahe_hog_color": "F4_noCLAHE",
            "X_hog_color": "F4",
            "roi_area_ratio": "roi_ratio",
        }
        out: dict[str, np.ndarray] = {}
        for k, v in raw.items():
            out[legacy_map.get(k, k)] = v
        if "F3_noCLAHE" not in out and "F4_noCLAHE" in out:
            out["F3_noCLAHE"] = out["F4_noCLAHE"][:, :1764]
        if "y" not in out:
            out["y"] = df["ClassId"].to_numpy(dtype=np.int32)
        return out

    records = df.to_dict("records")
    workers = n_workers or max(1, min(4, (os.cpu_count() or 2)))
    chunk_size = max(1, (len(records) + workers - 1) // workers)
    chunks = [
        (records[i : i + chunk_size], data_root)
        for i in range(0, len(records), chunk_size)
    ]

    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_extract_chunk, chunks))
    else:
        parts = [_extract_chunk(c) for c in chunks]

    merged: dict[str, np.ndarray] = {}
    for key in parts[0]:
        merged[key] = np.concatenate([p[key] for p in parts], axis=0)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, **merged)
    return merged


def prepare_scaled_split(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_te: np.ndarray,
    pca_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA | None, float]:
    """Fit StandardScaler and optional exact full-SVD PCA strictly on training split."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    if pca_dim is None:
        return X_tr_s, X_val_s, X_te_s, scaler, None, 1.0

    pca = PCA(n_components=pca_dim, svd_solver="full")
    X_tr_p = pca.fit_transform(X_tr_s).astype(np.float32)
    X_val_p = pca.transform(X_val_s).astype(np.float32)
    X_te_p = pca.transform(X_te_s).astype(np.float32)
    exp_var = float(np.sum(pca.explained_variance_ratio_))
    return X_tr_p, X_val_p, X_te_p, scaler, pca, exp_var


def fit_eval_model(
    model: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray | None = None,
    y_te: np.ndarray | None = None,
) -> tuple[Any, dict[str, float], dict[str, float] | None, np.ndarray, np.ndarray | None]:
    """Fit estimator on train split; evaluate GroupVal and optionally locked Test split."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    val_pred = model.predict(X_val)
    val_pred_sec = time.perf_counter() - t1
    m_val = compute_metrics(y_val, val_pred, fit_time=fit_sec, pred_time=val_pred_sec)

    if X_te is None or y_te is None:
        return model, m_val, None, val_pred, None

    t2 = time.perf_counter()
    te_pred = model.predict(X_te)
    te_pred_sec = time.perf_counter() - t2
    m_te = compute_metrics(y_te, te_pred, fit_time=fit_sec, pred_time=te_pred_sec)
    return model, m_val, m_te, val_pred, te_pred


# ---------------------------------------------------------------------------
# Publication-grade plotting functions (clean large fonts for 2-column IEEE)
# ---------------------------------------------------------------------------
def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
    )


def plot_class_distribution(df_train: pd.DataFrame, df_test: pd.DataFrame, out_path: str) -> None:
    _apply_pub_style()
    tr_cnt = df_train["ClassId"].value_counts().sort_index()
    te_cnt = df_test["ClassId"].value_counts().sort_index()
    x = np.arange(43)
    w = 0.42
    fig, ax = plt.subplots(figsize=(12.5, 4.8), dpi=200)
    ax.bar(x - w / 2, tr_cnt.values, w, color="#1f4e79", label=f"Train ({len(df_train):,} ảnh)")
    ax.bar(x + w / 2, te_cnt.values, w, color="#d95f02", label=f"Test ({len(df_test):,} ảnh)")
    ax.axhline(tr_cnt.mean(), color="#475569", linestyle="--", lw=1.4, label=f"Trung bình Train ({tr_cnt.mean():.1f})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(43)], fontsize=8.8)
    ax.set_xlabel("Mã lớp biển báo (ClassId: 0–42)")
    ax.set_ylabel("Số lượng ảnh")
    ax.set_title("Phân phối số lượng mẫu theo 43 lớp trên tập Train và Test (GTSRB)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_sample_grid_43classes(df_train: pd.DataFrame, data_root: str, out_path: str) -> None:
    _apply_pub_style()
    fig, axes = plt.subplots(5, 9, figsize=(14.5, 8.8), dpi=200)
    for cid in range(43):
        ax = axes.flatten()[cid]
        sub = df_train[df_train["ClassId"] == cid]
        row = sub.iloc[len(sub) // 2]
        img = load_rgb_image(os.path.join(data_root, str(row["Path"])))
        ax.imshow(cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA))
        ax.set_title(f"[{cid}] {CLASS_SHORT_VI.get(cid, str(cid))}", fontsize=9.2, pad=3)
        ax.axis("off")
    for idx in range(43, 45):
        axes.flatten()[idx].axis("off")
    fig.suptitle("Ảnh mẫu đại diện cho toàn bộ 43 lớp biển báo giao thông GTSRB", fontsize=14, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, hspace=0.42, wspace=0.14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_eda_dimensions_brightness(
    df_train: pd.DataFrame,
    roi_ratio_pct: np.ndarray,
    br_train: np.ndarray,
    br_test: np.ndarray,
    out_path: str,
) -> None:
    _apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), dpi=200)

    axes[0].hist(df_train["Width"].values, bins=30, alpha=0.72, color="#1f4e79", label="Chiều rộng (Width)")
    axes[0].hist(df_train["Height"].values, bins=30, alpha=0.55, color="#ea580c", label="Chiều cao (Height)")
    axes[0].axvline(df_train["Width"].median(), color="#111827", linestyle="--", lw=1.4, label=f"Trung vị ({df_train['Width'].median():.0f} px)")
    axes[0].set_title("(a) Phân bố kích thước ảnh gốc")
    axes[0].set_xlabel("Kích thước (pixel)")
    axes[0].set_ylabel("Số lượng ảnh")
    axes[0].legend(fontsize=9.2)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].hist(roi_ratio_pct, bins=30, color="#0d9488", alpha=0.82)
    axes[1].axvline(roi_ratio_pct.mean(), color="#991b1b", linestyle="--", lw=1.6, label=f"TB: {roi_ratio_pct.mean():.1f}%")
    axes[1].set_title("(b) Tỷ lệ diện tích vùng ROI")
    axes[1].set_xlabel("Diện tích ROI / Toàn khung (%)")
    axes[1].set_ylabel("Số lượng ảnh")
    axes[1].legend(fontsize=9.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    axes[2].hist(br_train, bins=35, density=True, alpha=0.72, color="#2563eb", label=f"Train (TB={br_train.mean():.1f})")
    axes[2].hist(br_test, bins=35, density=True, alpha=0.55, color="#dc2626", label=f"Test (TB={br_test.mean():.1f})")
    axes[2].set_title("(c) Phân bố độ sáng trung bình")
    axes[2].set_xlabel("Cường độ mức xám (0–255)")
    axes[2].set_ylabel("Mật độ xác suất")
    axes[2].legend(fontsize=9.2)
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_preprocessing_hog_pipeline(df_train: pd.DataFrame, data_root: str, out_path: str) -> None:
    _apply_pub_style()
    sample_cids = [19, 14, 2]
    titles_row = ["Ngoặt trái (Lớp 19)", "Biển STOP (Lớp 14)", "Tốc độ 50 tối (Lớp 2)"]
    col_titles = ["1. Ảnh gốc + khung ROI", "2. Cắt ROI & Resize 32×32", "3. Cân bằng CLAHE", "4. Ảnh xám sau CLAHE", "5. Bản đồ gradient HOG"]

    fig, axes = plt.subplots(3, 5, figsize=(13.2, 7.6), dpi=200)
    for r_idx, (cid, rlabel) in enumerate(zip(sample_cids, titles_row)):
        sub = df_train[df_train["ClassId"] == cid]
        row = sub.iloc[min(len(sub) - 1, 5)]
        img = load_rgb_image(os.path.join(data_root, str(row["Path"])))
        roi = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))
        img_box = img.copy()
        cv2.rectangle(img_box, (roi[0], roi[1]), (roi[2], roi[3]), (34, 197, 94), 2)
        roi_raw = preprocess_image(img, roi=roi, use_roi=True, use_clahe=False)
        roi_clahe = preprocess_image(img, roi=roi, use_roi=True, use_clahe=True)
        gray = cv2.cvtColor(roi_clahe, cv2.COLOR_RGB2GRAY)
        _, hog_vis = extract_hog_features(roi_clahe, visualize=True)
        hog_norm = cv2.normalize(hog_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        imgs = [img_box, roi_raw, roi_clahe, gray, hog_norm]
        cmaps = [None, None, None, "gray", "inferno"]
        for c_idx in range(5):
            ax = axes[r_idx, c_idx]
            ax.imshow(imgs[c_idx], cmap=cmaps[c_idx])
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=10.5, pad=6)
            if c_idx == 0:
                ax.set_ylabel(rlabel, fontsize=10.5, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Minh họa tiền xử lý (cắt ROI, Resize 32×32, CLAHE) và bản đồ gradient HOG", fontsize=13.5, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.89, bottom=0.03, hspace=0.14, wspace=0.10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_knn_hyperparams(
    exp2_rows: list[dict[str, Any]],
    out_path: str,
    five_fold_knn: list[dict[str, Any]] | None = None,
) -> None:
    _apply_pub_style()
    ks = sorted({int(r["k"]) for r in exp2_rows})
    x = np.arange(len(ks))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=200)

    for w_mode, style, col in [("uniform", "o--", "#64748b"), ("distance", "s-", "#1d4ed8")]:
        ys = [next(r["group_val"]["accuracy"] * 100 for r in exp2_rows if r["k"] == k and r["weights"] == w_mode and r["metric"] == "euclidean") for k in ks]
        ax1.plot(x, ys, style, lw=2.2, markersize=6.2, color=col, label=f"Fold 1 weights='{w_mode}'")
    if five_fold_knn:
        cv_map = {int(r["k"]): r for r in five_fold_knn}
        cv_x = [i for i, k in enumerate(ks) if k in cv_map]
        cv_y = [cv_map[ks[i]]["acc_mean"] * 100 for i in cv_x]
        cv_sd = [cv_map[ks[i]]["acc_std"] * 100 for i in cv_x]
        ax1.errorbar(cv_x, cv_y, yerr=cv_sd, fmt="D-", capsize=3, lw=2.0, color="#0f766e", label="5-Fold GroupCV Mean ± SD")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.set_xlabel("Số láng giềng k")
    ax1.set_ylabel("GroupVal Accuracy (%)")
    ax1.set_title("(a) Ảnh hưởng của k, weights & 5-Fold GroupCV (F5)")
    ax1.legend(fontsize=9.2)
    ax1.grid(True, linestyle="--", alpha=0.35)

    for metric, style, col in [("euclidean", "o-", "#1d4ed8"), ("manhattan", "s-", "#b91c1c"), ("cosine", "^--", "#0f766e")]:
        ys = [next(r["group_val"]["accuracy"] * 100 for r in exp2_rows if r["k"] == k and r["weights"] == "distance" and r["metric"] == metric) for k in ks]
        ax2.plot(x, ys, style, lw=2.2, markersize=6.2, color=col, label=f"metric='{metric}'")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("Số láng giềng k")
    ax2.set_ylabel("GroupVal Accuracy (%)")
    ax2.set_title("(b) Độ đo khoảng cách trên lưới mở rộng k=1..31 (F5)")
    ax2.legend(fontsize=9.2)
    ax2.grid(True, linestyle="--", alpha=0.35)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Tinh chỉnh K-NN trên GroupVal & 5-Fold GroupCV (lưới {len(exp2_rows)} cấu hình, k=1..31)", fontsize=13.0, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.14, wspace=0.22)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dt_hyperparams(exp3_rows: list[dict[str, Any]], out_path: str) -> None:
    _apply_pub_style()
    depths = [5, 10, 15, 20, 30, None]
    labels = [str(d) if d is not None else "None" for d in depths]
    x = np.arange(len(depths))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=200)

    tr_g = [next(r["train_accuracy"] * 100 for r in exp3_rows if r["criterion"] == "gini" and r["max_depth"] == d) for d in depths]
    va_g = [next(r["group_val"]["accuracy"] * 100 for r in exp3_rows if r["criterion"] == "gini" and r["max_depth"] == d) for d in depths]
    ax1.plot(x, tr_g, "o--", color="#1e3a8a", lw=2.2, markersize=6, label="Train Accuracy (%)")
    ax1.plot(x, va_g, "s-", color="#dc2626", lw=2.2, markersize=6, label="GroupVal Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Độ sâu tối đa (max_depth)")
    ax1.set_ylabel("Độ chính xác (%)")
    ax1.set_title("(a) Hiện tượng Overfitting theo max_depth (Gini)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.35)

    for crit, style, col in [("gini", "o--", "#475569"), ("entropy", "s-", "#0f766e")]:
        ys = [next(r["group_val"]["accuracy"] * 100 for r in exp3_rows if r["criterion"] == crit and r["max_depth"] == d) for d in depths]
        ax2.plot(x, ys, style, color=col, lw=2.2, markersize=6.5, label=f"criterion='{crit}'")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Độ sâu tối đa (max_depth)")
    ax2.set_ylabel("GroupVal Accuracy (%)")
    ax2.set_title("(b) So sánh tiêu chí Gini và Entropy trên GroupVal")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.35)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Tinh chỉnh Decision Tree trên GroupVal tách theo track (12 cấu hình, F5)", fontsize=13.2, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.14, wspace=0.22)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pca_analysis(exp4_rows: list[dict[str, Any]], out_path: str) -> None:
    """Render Figure 7 (PCA sweep vs GroupVal Accuracy, EVR, Inference Time)."""
    _apply_pub_style()
    dims = [int(r["dims"]) for r in exp4_rows]
    accs = [float(r["group_val"]["accuracy"]) * 100.0 for r in exp4_rows]
    evrs = [float(r["explained_variance"]) * 100.0 for r in exp4_rows]
    times = [float(r["group_val"]["pred_time_sec"]) for r in exp4_rows]
    x = np.arange(len(dims))

    fig, ax1 = plt.subplots(figsize=(10.2, 5.1), dpi=200)
    l1 = ax1.plot(x, accs, "o-", color="#1d4ed8", lw=2.5, markersize=7.5, label="GroupVal Accuracy (%)", zorder=4)
    l2 = ax1.plot(x, evrs, "s--", color="#c2410c", lw=2.2, markersize=6.5, label="Phương sai tích lũy PCA (%)", zorder=4)

    for xi, acc_v, evr_v in zip(x, accs, evrs):
        if xi in (2, 3):  # d=128, 256: place orange variance % clearly above the curve dot
            acc_off, evr_off = (0, 10), (0, 10)
        elif acc_v >= evr_v:
            acc_off, evr_off = (0, 10), (0, -16)
        else:
            acc_off, evr_off = (0, -16), (0, 10)
        ax1.annotate(f"{acc_v:.2f}%", (xi, acc_v), textcoords="offset points", xytext=acc_off, ha="center", fontsize=9.8, fontweight="bold", color="#1e3a8a")
        if xi < len(dims) - 1:
            ax1.annotate(f"{evr_v:.1f}%", (xi, evr_v), textcoords="offset points", xytext=evr_off, ha="center", fontsize=9.2, color="#9a3412")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d}\n(Gốc F4)" if d == dims[-1] else str(d) for d in dims])
    ax1.set_xlabel("Số chiều sau PCA (d)")
    ax1.set_ylabel("GroupVal Accuracy / Phương sai (%)")
    ax1.set_ylim(40, 108)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    bars = ax2.bar(x, times, width=0.35, alpha=0.25, color="#be123c", edgecolor="#881337", lw=0.9, label="Thời gian suy luận GroupVal (s)", zorder=2)
    max_t = max(times) * 1.90
    for bar, t_v in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, t_v + 0.18, f"{t_v:.2f}s", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#881337")
    ax2.set_ylabel("Thời gian dự đoán GroupVal (giây)", color="#881337")
    ax2.set_ylim(0, max_t)
    ax2.spines["top"].set_visible(False)

    handles = l1 + l2 + [bars]
    ax1.legend(handles, [h.get_label() for h in handles], loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True, fontsize=9.5)
    fig.suptitle("Đánh đổi Độ chính xác GroupVal, Phương sai và Tốc độ suy luận theo PCA (K-NN k=11, cosine)", fontsize=12.5, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.09, right=0.90, top=0.80, bottom=0.14)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_model_comparison(exp5_rows: list[dict[str, Any]], out_path: str) -> None:
    """Render Figure 8 (Test Accuracy and Macro-F1 with 95% Bootstrap CI error bars)."""
    _apply_pub_style()
    models = [str(r["model"]) for r in exp5_rows]
    accs = [float(r["test"]["accuracy"]) * 100.0 for r in exp5_rows]
    f1s = [float(r["test"]["f1_macro"]) * 100.0 for r in exp5_rows]
    acc_errs = [float(r["bootstrap_ci"]["accuracy_halfwidth"]) * 100.0 for r in exp5_rows]
    f1_errs = [float(r["bootstrap_ci"]["f1_macro_halfwidth"]) * 100.0 for r in exp5_rows]
    y_pos = np.arange(len(models))
    h = 0.36

    fig_h = max(6.0, 0.44 * len(models) + 1.4)
    fig, ax = plt.subplots(figsize=(11.8, fig_h), dpi=200)
    bars_acc = ax.barh(y_pos - h / 2.0, accs, h, xerr=acc_errs, capsize=3, color="#1d4ed8", error_kw={"ecolor": "#0f172a", "lw": 1.0}, label="Test Accuracy (% ± CI 95%)")
    bars_f1 = ax.barh(y_pos + h / 2.0, f1s, h, xerr=f1_errs, capsize=3, color="#d97706", error_kw={"ecolor": "#0f172a", "lw": 1.0}, label="Macro-F1 (% ± CI 95%)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 114)
    ax.set_xlabel("Điểm đánh giá trên tập Test độc lập (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for b, err in zip(bars_acc, acc_errs):
        w = b.get_width()
        ax.text(w + err + 0.8, b.get_y() + b.get_height() / 2.0, f"{w:.2f}%", va="center", fontsize=8.8, fontweight="bold", color="#1e3a8a")
    for b, err in zip(bars_f1, f1_errs):
        w = b.get_width()
        ax.text(w + err + 0.8, b.get_y() + b.get_height() / 2.0, f"{w:.2f}%", va="center", fontsize=8.8, fontweight="bold", color="#92400e")

    fig.suptitle("So sánh hiệu năng các mô hình trên tập Test (12.630 ảnh, Bootstrap 95% CI)", fontsize=12.5, fontweight="bold", y=0.985)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=2, frameon=True)
    fig.subplots_adjust(left=0.35, right=0.97, top=0.86, bottom=0.09)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix_fig(y_te: np.ndarray, knn_te_pred: np.ndarray, acc_pct: float, out_path: str) -> None:
    _apply_pub_style()
    cm = confusion_matrix(y_te, knn_te_pred, labels=list(range(43)))
    cm_norm = cm.astype(np.float64) / np.maximum(1, cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(11.5, 9.6), dpi=200)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cbar.ax.tick_params(labelsize=10.5)
    labels = [f"{i}: {CLASS_SHORT_VI[i]}" for i in range(43)]
    ax.set_xticks(np.arange(43))
    ax.set_yticks(np.arange(43))
    ax.set_xticklabels(labels, rotation=90, fontsize=7.6)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlabel("Nhãn dự đoán (Predicted ClassId)")
    ax.set_ylabel("Nhãn thực tế (True ClassId)")
    ax.set_title(f"Ma trận nhầm lẫn chuẩn hóa (43×43) của K-NN F5 tối ưu (Test Acc = {acc_pct:.2f}%)", fontsize=13.5, pad=10)
    # Highlight severe confusion cells (>= 0.05 off-diagonal)
    for r in range(43):
        for c in range(43):
            if r != c and cm_norm[r, c] >= 0.05:
                ax.text(c, r, f"{cm[r, c]}", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#991b1b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error_analysis_samples(
    error_samples: list[dict[str, Any]],
    df_test: pd.DataFrame,
    data_root: str,
    out_path: str,
) -> None:
    """Render Figure 10 (2x4 gallery of misclassified test signs)."""
    _apply_pub_style()
    samples = error_samples[:8]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.6), dpi=200)
    for ax, sample in zip(axes.flatten(), samples):
        idx = int(sample["test_index"])
        img_rgb = load_rgb_image(os.path.join(data_root, str(df_test.iloc[idx]["Path"])))
        t_id = int(sample["true_id"])
        p_id = int(sample["pred_id"])
        t_short = CLASS_SHORT_VI.get(t_id, str(sample["true_name"]))
        p_short = CLASS_SHORT_VI.get(p_id, str(sample["pred_name"]))
        ax.imshow(img_rgb)
        ax.set_title(
            f"File: {sample['filename']} ({sample['width']}×{sample['height']} px)\n"
            f"Thật: [{t_id}] {t_short}\n"
            f"Đoán: [{p_id}] {p_short}",
            fontsize=10.2,
            color="#991b1b",
            pad=6,
        )
        ax.axis("off")

    for ax in axes.flatten()[len(samples) :]:
        ax.axis("off")
    fig.suptitle("Minh họa 8 mẫu ảnh Test thuộc các cặp biển báo K-NN (F5) hay nhầm lẫn", fontsize=13.8, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.83, bottom=0.04, hspace=0.44, wspace=0.20)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error_diagnostics(
    width_bins: list[dict[str, Any]],
    bright_bins: list[dict[str, Any]],
    cov_knn: list[dict[str, Any]],
    cov_mlp: list[dict[str, Any]],
    out_path: str,
) -> None:
    """Render Figure 11: Quantitative error rate with 95% Wilson CI and Confidence-Coverage curves."""
    _apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), dpi=200)

    w_lbls = [f"{b['label']}\n(n={b['count']})" for b in width_bins]
    w_errs = [b["error_rate_pct"] for b in width_bins]
    w_yerr = [
        [max(0.0, b["error_rate_pct"] - b["ci95_wilson"][0]) for b in width_bins],
        [max(0.0, b["ci95_wilson"][1] - b["error_rate_pct"]) for b in width_bins],
    ]
    bars0 = axes[0].bar(
        np.arange(len(w_lbls)),
        w_errs,
        yerr=w_yerr,
        capsize=4,
        color="#1f4e79",
        width=0.55,
        error_kw={"ecolor": "#0f172a", "lw": 1.1},
    )
    for b, val, hi in zip(bars0, w_errs, w_yerr[1]):
        axes[0].text(b.get_x() + b.get_width() / 2.0, val + hi + 0.4, f"{val:.2f}%", ha="center", va="bottom", fontsize=9.6, fontweight="bold")
    axes[0].set_xticks(np.arange(len(w_lbls)))
    axes[0].set_xticklabels(w_lbls, fontsize=9.4)
    axes[0].set_ylim(0, max(v + h for v, h in zip(w_errs, w_yerr[1])) * 1.24)
    axes[0].set_title("(a) Lỗi theo kích thước (Wilson 95% CI)")
    axes[0].set_ylabel("Tỷ lệ dự đoán sai (%)")

    b_lbls = [b["label"].replace(" (", "\n(") + f"\n(n={b['count']})" for b in bright_bins]
    b_errs = [b["error_rate_pct"] for b in bright_bins]
    b_yerr = [
        [max(0.0, b["error_rate_pct"] - b["ci95_wilson"][0]) for b in bright_bins],
        [max(0.0, b["ci95_wilson"][1] - b["error_rate_pct"]) for b in bright_bins],
    ]
    bars1 = axes[1].bar(
        np.arange(len(b_lbls)),
        b_errs,
        yerr=b_yerr,
        capsize=4,
        color="#c2410c",
        width=0.55,
        error_kw={"ecolor": "#0f172a", "lw": 1.1},
    )
    for b, val, hi in zip(bars1, b_errs, b_yerr[1]):
        axes[1].text(b.get_x() + b.get_width() / 2.0, val + hi + 0.4, f"{val:.2f}%", ha="center", va="bottom", fontsize=9.6, fontweight="bold")
    axes[1].set_xticks(np.arange(len(b_lbls)))
    axes[1].set_xticklabels(b_lbls, fontsize=8.8)
    axes[1].set_ylim(0, max(v + h for v, h in zip(b_errs, b_yerr[1])) * 1.24)
    axes[1].set_title("(b) Lỗi theo độ sáng (Wilson 95% CI)")
    axes[1].set_ylabel("Tỷ lệ dự đoán sai (%)")

    axes[2].plot([r["coverage_pct"] for r in cov_knn], [r["accuracy_pct"] for r in cov_knn], "o-", color="#1d4ed8", lw=2.2, label="K-NN (F5)")
    axes[2].plot([r["coverage_pct"] for r in cov_mlp], [r["accuracy_pct"] for r in cov_mlp], "s-", color="#0f766e", lw=2.2, label="MLP (F5)")
    axes[2].set_title("(c) Đường cong Coverage–Accuracy")
    axes[2].set_xlabel("Độ phủ mẫu được chấp nhận (%)")
    axes[2].set_ylabel("Độ chính xác trên tập chấp nhận (%)")
    axes[2].legend(loc="lower left")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Kiểm chứng thống kê đặc điểm lỗi (95% Wilson CI) và ngưỡng từ chối (12.630 ảnh Test)", fontsize=13.0, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.83, bottom=0.18, wspace=0.26)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ablation_and_loss(
    alpha_rows: list[dict[str, Any]],
    loss_f3p: list[float],
    loss_f5: list[float],
    out_path: str,
) -> None:
    """Render Figure 13: Color block weight alpha ablation & MLP training loss convergence curves."""
    _apply_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=200)

    alphas = [float(r["alpha"]) for r in alpha_rows]
    ax1.plot(alphas, [r["gv_raw1860_acc"] * 100 for r in alpha_rows], "o--", color="#0f766e", lw=2.1, markersize=6, label="GroupVal (1860d không PCA)")
    ax1.plot(alphas, [r["gv_pca128_acc"] * 100 for r in alpha_rows], "s-", color="#1d4ed8", lw=2.2, markersize=6.5, label="GroupVal (128d PCA)")
    ax1.plot(alphas, [r["te_pca128_acc"] * 100 for r in alpha_rows], "^:", color="#c2410c", lw=2.1, markersize=6.2, label="Test (128d PCA đối chứng)")
    ax1.set_xticks(alphas)
    ax1.set_xlabel("Hệ số nhân α cho khối 96 chiều Màu (so với HOG)")
    ax1.set_ylabel("K-NN Accuracy (k=11, Euclidean) (%)")
    ax1.set_title("(a) Ảnh hưởng trọng số màu α lên K-NN (F4 / F5)")
    ax1.legend(fontsize=9.2)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2.plot(np.arange(1, len(loss_f3p) + 1), loss_f3p, "o-", markersize=3.5, lw=2.1, color="#0f766e", label=f"MLP F3-PCA (128d, {len(loss_f3p)} epochs)")
    ax2.plot(np.arange(1, len(loss_f5) + 1), loss_f5, "s--", markersize=3.5, lw=2.1, color="#1d4ed8", label=f"MLP F5 (128d, {len(loss_f5)} epochs)")
    ax2.set_xlabel("Epoch huấn luyện")
    ax2.set_ylabel("Cross-Entropy Training Loss")
    ax2.set_title("(b) Đường cong hội tụ Training Loss của MLP")
    ax2.legend(fontsize=9.2)
    ax2.grid(True, linestyle="--", alpha=0.35)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(r"Kiểm chứng giả thuyết suy giảm màu sắc ($\alpha \in [0; 1]$) và hội tụ mạng MLP", fontsize=13.0, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.14, wspace=0.22)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostics computation helpers
# ---------------------------------------------------------------------------
def _compute_bin_errors(values: np.ndarray, is_error: np.ndarray, specs: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    rows = []
    for label, lo, hi in specs:
        mask = (values >= lo) & (values < hi)
        cnt = int(mask.sum())
        errs = int(is_error[mask].sum())
        lo_ci, hi_ci = _wilson_ci_pct(errs, cnt)
        rows.append(
            {
                "label": label,
                "count": cnt,
                "errors": errs,
                "error_rate_pct": float(100.0 * errs / max(1, cnt)),
                "ci95_wilson": [lo_ci, hi_ci],
            }
        )
    return rows


def _count_speed_up_errors(
    y_true: np.ndarray,
    probs: np.ndarray,
    tau: float = 0.0,
) -> dict[str, Any]:
    """Count total errors and dangerous low-to-high speed-limit misclassifications."""
    max_p = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    mask = max_p >= tau
    y_t = y_true[mask]
    y_p = pred[mask]
    c30_to_high = int(np.sum((y_t == 1) & np.isin(y_p, [2, 3, 4, 5, 7, 8])))
    all_up = 0
    for yt_i, yp_i in zip(y_t, y_p):
        st = SPEED_SIGN_ORDER.get(int(yt_i))
        sp = SPEED_SIGN_ORDER.get(int(yp_i))
        if st is not None and sp is not None and sp > st:
            all_up += 1
    return {
        "threshold": float(tau),
        "accepted_count": int(mask.sum()),
        "coverage_pct": float(100.0 * mask.sum() / len(y_true)),
        "accuracy_pct": float(100.0 * accuracy_score(y_t, y_p)) if mask.sum() > 0 else 100.0,
        "speed_30_to_higher_count": c30_to_high,
        "speed_any_up_count": int(all_up),
    }


def _compute_coverage_curve(y_true: np.ndarray, probs: np.ndarray) -> list[dict[str, Any]]:
    max_p = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    rows = []
    for tau in [0.0, 0.30, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        mask = max_p >= tau
        cnt = int(mask.sum())
        acc = float(100.0 * accuracy_score(y_true[mask], pred[mask])) if cnt > 0 else 100.0
        rows.append(
            {
                "threshold": float(tau),
                "accepted_count": cnt,
                "coverage_pct": float(100.0 * cnt / len(y_true)),
                "accuracy_pct": acc,
            }
        )
    return rows


def _predict_from_nbrs(
    nbr_labels: np.ndarray,
    nbr_dists: np.ndarray,
    k: int,
    weights: str,
    n_classes: int = 43,
) -> np.ndarray:
    """Predict discrete labels from precomputed top-K neighbor indices and distances."""
    y_k = nbr_labels[:, :k]
    if weights == "distance":
        d_k = nbr_dists[:, :k]
        w_k = np.where(d_k <= 1e-12, 1e12, 1.0 / np.maximum(d_k, 1e-12))
    else:
        w_k = np.ones_like(y_k, dtype=np.float64)
    scores = np.zeros((y_k.shape[0], n_classes), dtype=np.float64)
    rows = np.arange(y_k.shape[0])[:, None]
    np.add.at(scores, (rows, y_k), w_k)
    return np.argmax(scores, axis=1).astype(np.int32)


def _eval_knn_grid_from_top_k(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ks: list[int],
    metrics: list[str],
    weights_list: list[str],
    compute_train_uniform: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate a full grid of (k, weights, metric) by computing top-max(ks) neighbors once per metric."""
    max_k = max(ks)
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        nn = NearestNeighbors(n_neighbors=max_k, metric=metric, algorithm="brute", n_jobs=-1)
        t0 = time.perf_counter()
        nn.fit(X_tr)
        fit_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        d_val, idx_val = nn.kneighbors(X_val, return_distance=True)
        pred_t = time.perf_counter() - t1
        nbr_labels_val = y_tr[idx_val]

        nbr_labels_tr = None
        d_tr = None
        if compute_train_uniform and "uniform" in weights_list:
            d_tr, idx_tr = nn.kneighbors(X_tr, return_distance=True)
            nbr_labels_tr = y_tr[idx_tr]

        for k in ks:
            for weights in weights_list:
                y_pred_val = _predict_from_nbrs(nbr_labels_val, d_val, k, weights)
                mv = compute_metrics(y_val, y_pred_val, fit_time=fit_t, pred_time=pred_t)
                if weights == "distance":
                    tr_acc = 1.0
                elif nbr_labels_tr is not None and d_tr is not None:
                    tr_pred = _predict_from_nbrs(nbr_labels_tr, d_tr, k, "uniform")
                    tr_acc = float(accuracy_score(y_tr, tr_pred))
                else:
                    tr_acc = 1.0
                rows.append(
                    {
                        "k": int(k),
                        "weights": weights,
                        "metric": metric,
                        "train_accuracy": tr_acc,
                        "group_val": mv,
                    }
                )
    return rows


def run_experiments(
    data_root: str,
    results_dir: str = "results",
    figures_dir: str = "report/figures",
    models_dir: str = "models",
    cache_dir: str | None = None,
    skip_plots: bool = False,
    n_workers: int | None = None,
) -> dict[str, Any]:
    """Run all five GTSRB experiments under StratifiedGroupKFold (track-disjoint) validation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print("[Step 0/7] Loading GTSRB metadata & cached features...", flush=True)

    train_csv, test_csv = prepare_gtsrb_csvs(data_root)
    df_train_full = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    train_cache = os.path.join(cache_dir, "cache_feats_TRAIN_FULL.npz") if cache_dir else None
    test_cache = os.path.join(cache_dir, "cache_feats_TEST.npz") if cache_dir else None
    feats_tr_full = load_all_images_and_features(df_train_full, data_root, cache_path=train_cache, n_workers=n_workers)
    feats_te = load_all_images_and_features(df_test, data_root, cache_path=test_cache, n_workers=n_workers)

    # Track-disjoint 80/20 split using StratifiedGroupKFold (zero shared tracks)
    track_ids = extract_track_ids(df_train_full)
    sgkf = StratifiedGroupKFold(n_splits=5)
    idx_tr, idx_val = next(sgkf.split(np.zeros(len(df_train_full)), feats_tr_full["y"], groups=track_ids))

    y_tr = feats_tr_full["y"][idx_tr]
    y_val = feats_tr_full["y"][idx_val]
    y_te = feats_te["y"]

    tr_tracks = int(len(np.unique(track_ids[idx_tr])))
    val_tracks = int(len(np.unique(track_ids[idx_val])))
    overlap_tracks = int(len(set(track_ids[idx_tr]).intersection(set(track_ids[idx_val]))))

    # Empirical leakage audit: random frame split vs track-disjoint GroupKFold on F5 (k=5, euclidean)
    all_idx = np.arange(len(df_train_full))
    r_tr, r_val = train_test_split(all_idx, test_size=0.2, random_state=RANDOM_STATE, stratify=feats_tr_full["y"])
    xr_tr, xr_val, xr_te, _, _, _ = prepare_scaled_split(feats_tr_full["F4"][r_tr], feats_tr_full["F4"][r_val], feats_te["F4"], pca_dim=128)
    _, r_val_m, r_te_m, _, _ = fit_eval_model(
        KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean", algorithm="brute", n_jobs=-1),
        xr_tr, feats_tr_full["y"][r_tr], xr_val, feats_tr_full["y"][r_val], xr_te, y_te,
    )

    xg_tr, xg_val, xg_te, _, _, _ = prepare_scaled_split(feats_tr_full["F4"][idx_tr], feats_tr_full["F4"][idx_val], feats_te["F4"], pca_dim=128)
    _, g_val_m, g_te_m, _, _ = fit_eval_model(
        KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean", algorithm="brute", n_jobs=-1),
        xg_tr, y_tr, xg_val, y_val, xg_te, y_te,
    )

    track_info = count_unique_tracks(df_train_full)
    cls_counts = df_train_full["ClassId"].value_counts()
    t_cls_counts = df_test["ClassId"].value_counts()

    dataset_summary = {
        "train_full_count": int(len(df_train_full)),
        "train_split_count": int(len(idx_tr)),
        "val_split_count": int(len(idx_val)),
        "train_tracks_count": tr_tracks,
        "val_tracks_count": val_tracks,
        "overlap_tracks": overlap_tracks,
        "test_count": int(len(df_test)),
        "total_images": int(len(df_train_full) + len(df_test)),
        "num_classes": 43,
        "unique_train_tracks": int(track_info["unique_tracks"]),
        "train_min_class_count": int(cls_counts.min()),
        "train_min_class_id": int(cls_counts.idxmin()),
        "train_max_class_count": int(cls_counts.max()),
        "train_max_class_id": int(cls_counts.idxmax()),
        "test_min_class_count": int(t_cls_counts.min()),
        "test_max_class_count": int(t_cls_counts.max()),
        "width_min": int(df_train_full["Width"].min()),
        "width_max": int(df_train_full["Width"].max()),
        "width_median": float(df_train_full["Width"].median()),
        "height_min": int(df_train_full["Height"].min()),
        "height_max": int(df_train_full["Height"].max()),
        "height_median": float(df_train_full["Height"].median()),
        "roi_area_ratio_mean_pct": float(feats_tr_full["roi_ratio"].mean() * 100.0),
        "brightness_train_mean": float(feats_tr_full["brightness"].mean()),
        "brightness_train_std": float(feats_tr_full["brightness"].std()),
        "brightness_test_mean": float(feats_te["brightness"].mean()),
        "missing_values": int(df_train_full.isna().sum().sum() + df_test.isna().sum().sum()),
        "duplicate_rows": int(df_train_full.duplicated().sum() + df_test.duplicated().sum()),
        "environment": {
            "python": platform.python_version(),
            "platform": "Google Colab CPU (2 vCPU Intel Xeon, 12.7 GB RAM)",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scikit_image": skimage.__version__,
            "opencv": cv2.__version__,
        },
        "leakage_control_comparison": {
            "random_frame_split_val_acc": r_val_m["accuracy"],
            "random_frame_split_test_acc": r_te_m["accuracy"],
            "track_group_split_val_acc": g_val_m["accuracy"],
            "track_group_split_test_acc": g_te_m["accuracy"],
        },
    }

    # Exp 1: 9 feature configs across K-NN (k=5), DT (depth=20), GNB
    feature_specs = [
        ("F0: No-ROI Raw Grayscale (1024)", "F0", None),
        ("F1: ROI Raw Grayscale - Baseline (1024)", "F1", None),
        ("F2: ROI Raw RGB (3072)", "F2", None),
        ("F3_noCLAHE: ROI + HOG Không CLAHE (1764)", "F3_noCLAHE", None),
        ("F3: ROI + CLAHE + HOG (1764)", "F3", None),
        ("F3_PCA: ROI + CLAHE + HOG + PCA (128)", "F3", 128),
        ("F4_noCLAHE: ROI + HOG + Color (1860)", "F4_noCLAHE", None),
        ("F4: ROI + CLAHE + HOG + Color (1860)", "F4", None),
        ("F5: ROI + CLAHE + HOG + Color + PCA (128)", "F4", 128),
    ]

    print("[Step 1/7] Running Exp 1 (9 feature sets x 3 core models)...", flush=True)
    prev_exp1: dict[tuple[str, int], dict[str, Any]] = {}
    summary_path = os.path.join(results_dir, "metrics_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as pf:
                for pr in json.load(pf).get("exp1_features", []):
                    prev_exp1[(pr["feature"].split(":")[0], int(pr["dims"]))] = pr
        except Exception:
            prev_exp1 = {}

    exp1_results = []
    cache_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA | None, float]] = {}
    exp1_preds: dict[str, np.ndarray] = {}
    dt_needed_for_mcnemar = {"F3_noCLAHE", "F3", "F4_noCLAHE", "F4"}
    for feat_label, src_key, pca_dim in feature_specs:
        t_feat0 = time.perf_counter()
        split_key = feat_label.split(":")[0]
        X_tr_s, X_val_s, X_te_s, sc, pca_obj, exp_var = prepare_scaled_split(
            feats_tr_full[src_key][idx_tr], feats_tr_full[src_key][idx_val], feats_te[src_key], pca_dim=pca_dim
        )
        cache_splits[split_key] = (X_tr_s, X_val_s, X_te_s, sc, pca_obj, exp_var)

        knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean", algorithm="brute", n_jobs=-1)
        _, knn_val, knn_te, _, knn_te_p = fit_eval_model(knn, X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te)
        cached_row = prev_exp1.get((split_key, int(X_tr_s.shape[1])))
        if cached_row is not None and split_key not in dt_needed_for_mcnemar:
            dt_val, dt_te = cached_row["dt_val"], cached_row["dt_test"]
        else:
            dt = DecisionTreeClassifier(criterion="gini", max_depth=20, random_state=RANDOM_STATE)
            _, dt_val, dt_te, _, dt_te_p = fit_eval_model(dt, X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te)
            exp1_preds[f"dt_{split_key}"] = dt_te_p
        gnb = GaussianNB()
        _, gnb_val, gnb_te, _, gnb_te_p = fit_eval_model(gnb, X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te)

        exp1_preds[f"knn_{split_key}"] = knn_te_p
        exp1_preds[f"gnb_{split_key}"] = gnb_te_p
        exp1_results.append(
            {
                "feature": feat_label,
                "dims": int(X_tr_s.shape[1]),
                "explained_variance": float(exp_var),
                "knn_val": knn_val,
                "knn_test": knn_te,
                "dt_val": dt_val,
                "dt_test": dt_te,
                "gnb_val": gnb_val,
                "gnb_test": gnb_te,
            }
        )
        print(
            f"  Exp 1 [{split_key:<10}] dims={X_tr_s.shape[1]:<4} KNN_Te={knn_te['accuracy']*100:.2f}% DT_Te={dt_te['accuracy']*100:.2f}% ({time.perf_counter()-t_feat0:.1f}s)",
            flush=True,
        )

    X_tr_f5, X_val_f5, X_te_f5, scaler_f5, pca_f5, _ = cache_splits["F5"]
    X_tr_f3p, X_val_f3p, X_te_f3p, scaler_f3p, pca_f3p, _ = cache_splits["F3_PCA"]
    X_tr_f4, X_val_f4, X_te_f4, _, _, _ = cache_splits["F4"]
    X_tr_f1, X_val_f1, X_te_f1, _, _, _ = cache_splits["F1"]

    # Color block weight alpha ablation: test [HOG_scaled, alpha * Color_scaled] for alpha in {0, 0.25, 0.5, 0.75, 1.0}
    print("[Step 2/7] Running Color-Weight alpha ablation (alpha in 0..1)...", flush=True)
    color_weight_ablation = []
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        Xw_tr = X_tr_f4.copy()
        Xw_val = X_val_f4.copy()
        Xw_te = X_te_f4.copy()
        Xw_tr[:, 1764:] *= alpha
        Xw_val[:, 1764:] *= alpha
        Xw_te[:, 1764:] *= alpha
        knn_raw_a = KNeighborsClassifier(n_neighbors=11, weights="distance", metric="euclidean", algorithm="brute", n_jobs=-1)
        _, mv_raw, mte_raw, _, _ = fit_eval_model(knn_raw_a, Xw_tr, y_tr, Xw_val, y_val, Xw_te, y_te)
        pca_a = PCA(n_components=128, svd_solver="full")
        Xwp_tr = pca_a.fit_transform(Xw_tr).astype(np.float32)
        Xwp_val = pca_a.transform(Xw_val).astype(np.float32)
        Xwp_te = pca_a.transform(Xw_te).astype(np.float32)
        knn_pca_a = KNeighborsClassifier(n_neighbors=11, weights="distance", metric="euclidean", algorithm="brute", n_jobs=-1)
        _, mv_pca, mte_pca, _, _ = fit_eval_model(knn_pca_a, Xwp_tr, y_tr, Xwp_val, y_val, Xwp_te, y_te)
        color_weight_ablation.append(
            {
                "alpha": float(alpha),
                "gv_raw1860_acc": float(mv_raw["accuracy"]),
                "te_raw1860_acc": float(mte_raw["accuracy"]),
                "gv_pca128_acc": float(mv_pca["accuracy"]),
                "te_pca128_acc": float(mte_pca["accuracy"]),
                "gv_pca128_f1": float(mv_pca["f1_macro"]),
                "te_pca128_f1": float(mte_pca["f1_macro"]),
            }
        )
        print(f"  alpha={alpha:4.2f}: Raw1860 Te={mte_raw['accuracy']*100:.2f}% | PCA128 Te={mte_pca['accuracy']*100:.2f}%", flush=True)

    # Exp 2: Expanded K-NN grid (k in {1,3,5,7,11,15,21,31}, 48 configs) via single-pass top-31 slicing
    print("[Step 3/7] Running Exp 2 (Fast top-31 K-NN grid k=1..31 on F5 & F3_PCA) & Exp 3/4...", flush=True)
    knn_ks = [1, 3, 5, 7, 11, 15, 21, 31]
    exp2_results = _eval_knn_grid_from_top_k(
        X_tr_f5,
        y_tr,
        X_val_f5,
        y_val,
        ks=knn_ks,
        metrics=["euclidean", "manhattan", "cosine"],
        weights_list=["uniform", "distance"],
        compute_train_uniform=True,
    )
    best_knn_cfg = max(exp2_results, key=lambda r: r["group_val"]["accuracy"])

    exp2_f3p_results = _eval_knn_grid_from_top_k(
        X_tr_f3p,
        y_tr,
        X_val_f3p,
        y_val,
        ks=[3, 5, 7, 11, 15, 21, 31],
        metrics=["euclidean", "manhattan", "cosine"],
        weights_list=["distance"],
        compute_train_uniform=False,
    )
    best_f3p_cfg = max(exp2_f3p_results, key=lambda r: r["group_val"]["accuracy"])
    print(f"  Best K-NN GroupVal: F3-PCA k={best_f3p_cfg['k']} ({best_f3p_cfg['group_val']['accuracy']*100:.2f}%) | F5 k={best_knn_cfg['k']} ({best_knn_cfg['group_val']['accuracy']*100:.2f}%)", flush=True)

    # Exp 3: Full 12-config Decision Tree grid on F5 — evaluated STRICTLY on GroupVal
    exp3_results = []
    for criterion in ["gini", "entropy"]:
        for max_depth in [5, 10, 15, 20, 30, None]:
            dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=RANDOM_STATE)
            dt, m_val, _, _, _ = fit_eval_model(dt, X_tr_f5, y_tr, X_val_f5, y_val)
            exp3_results.append(
                {
                    "criterion": criterion,
                    "max_depth": max_depth,
                    "train_accuracy": float(accuracy_score(y_tr, dt.predict(X_tr_f5))),
                    "group_val": m_val,
                }
            )
    best_dt_cfg = max(exp3_results, key=lambda r: r["group_val"]["accuracy"])

    # Exp 4: Deterministic PCA dimensionality sweep (full SVD) strictly on GroupVal
    scaler_f4_full = StandardScaler()
    X_tr_f4s = scaler_f4_full.fit_transform(feats_tr_full["F4"][idx_tr]).astype(np.float32)
    X_val_f4s = scaler_f4_full.transform(feats_tr_full["F4"][idx_val]).astype(np.float32)

    pca_512 = PCA(n_components=512, svd_solver="full")
    X_tr_p512 = pca_512.fit_transform(X_tr_f4s).astype(np.float32)
    X_val_p512 = pca_512.transform(X_val_f4s).astype(np.float32)
    cum_var = np.cumsum(pca_512.explained_variance_ratio_)

    exp4_results = []
    for dim in [32, 64, 128, 256, 512, 1860]:
        if dim == 1860:
            Xd_tr, Xd_val, evr = X_tr_f4s, X_val_f4s, 1.0
        else:
            Xd_tr, Xd_val, evr = X_tr_p512[:, :dim], X_val_p512[:, :dim], float(cum_var[dim - 1])
        knn_p = KNeighborsClassifier(n_neighbors=int(best_knn_cfg["k"]), weights="distance", metric="cosine", algorithm="brute", n_jobs=-1)
        _, m_val, _, _, _ = fit_eval_model(knn_p, Xd_tr, y_tr, Xd_val, y_val)
        exp4_results.append({"dims": dim, "explained_variance": evr, "group_val": m_val})

    # Full 5-Fold StratifiedGroupKFold Cross-Validation (mean +/- std across all 5 track-disjoint folds)
    print("[Step 4/7] Running Full 5-Fold StratifiedGroupKFold CV...", flush=True)
    cv_buckets: dict[str, list[dict[str, float]]] = {}
    for fold_idx, (ftr, fva) in enumerate(sgkf.split(np.zeros(len(df_train_full)), feats_tr_full["y"], groups=track_ids)):
        yf_tr, yf_va = feats_tr_full["y"][ftr], feats_tr_full["y"][fva]
        Xf3_tr, Xf3_va, _, _, _, _ = prepare_scaled_split(feats_tr_full["F3"][ftr], feats_tr_full["F3"][fva], feats_te["F3"][:1], pca_dim=128)
        Xf5_tr, Xf5_va, _, _, _, _ = prepare_scaled_split(feats_tr_full["F4"][ftr], feats_tr_full["F4"][fva], feats_te["F4"][:1], pca_dim=128)

        for r_cv in _eval_knn_grid_from_top_k(Xf5_tr, yf_tr, Xf5_va, yf_va, ks=[3, 5, 7, 11, 15, 21, 31], metrics=["euclidean"], weights_list=["distance"]):
            cv_buckets.setdefault(f"K-NN F5 (k={r_cv['k']}, euclidean)", []).append(
                {"acc": r_cv["group_val"]["accuracy"], "f1": r_cv["group_val"]["f1_macro"]}
            )
        for r_cv3 in _eval_knn_grid_from_top_k(Xf3_tr, yf_tr, Xf3_va, yf_va, ks=[5, 7, 11, 15, 21], metrics=["euclidean"], weights_list=["distance"]):
            cv_buckets.setdefault(f"K-NN F3_PCA (k={r_cv3['k']}, euclidean)", []).append(
                {"acc": r_cv3["group_val"]["accuracy"], "f1": r_cv3["group_val"]["f1_macro"]}
            )

        fold_models: list[tuple[str, Any, np.ndarray, np.ndarray]] = [
            (f"Decision Tree F5 (depth={best_dt_cfg['max_depth']}, {best_dt_cfg['criterion']})", DecisionTreeClassifier(criterion=str(best_dt_cfg["criterion"]), max_depth=best_dt_cfg["max_depth"], random_state=RANDOM_STATE), Xf5_tr, Xf5_va),
            ("Gaussian Naive Bayes (F3_PCA)", GaussianNB(), Xf3_tr, Xf3_va),
            ("Gaussian Naive Bayes (F5)", GaussianNB(), Xf5_tr, Xf5_va),
            ("Random Forest F5 (100t, depth=25, default)", RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=25, class_weight=None, random_state=RANDOM_STATE, n_jobs=-1), Xf5_tr, Xf5_va),
            ("Random Forest F5 (100t, depth=25, balanced_subsample)", RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=25, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1), Xf5_tr, Xf5_va),
            ("MLP F3_PCA ((256, 128), max_iter=35)", MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", solver="adam", alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=35, early_stopping=False, random_state=RANDOM_STATE), Xf3_tr, Xf3_va),
            ("MLP F5 ((256, 128), max_iter=35)", MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", solver="adam", alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=35, early_stopping=False, random_state=RANDOM_STATE), Xf5_tr, Xf5_va),
        ]
        for m_lbl, est, xm_tr, xm_va in fold_models:
            _, mv_fold, _, _, _ = fit_eval_model(est, xm_tr, yf_tr, xm_va, yf_va)
            cv_buckets.setdefault(m_lbl, []).append({"acc": mv_fold["accuracy"], "f1": mv_fold["f1_macro"]})
        print(f"  Completed Fold {fold_idx + 1}/5 (Train={len(ftr)}, Val={len(fva)})", flush=True)

    cv_5fold_summary = []
    for m_lbl, entries in cv_buckets.items():
        accs_k = np.array([e["acc"] for e in entries])
        f1s_k = np.array([e["f1"] for e in entries])
        row_cv = {
            "model": m_lbl,
            "acc_mean": float(accs_k.mean()),
            "acc_std": float(accs_k.std(ddof=1)),
            "f1_mean": float(f1s_k.mean()),
            "f1_std": float(f1s_k.std(ddof=1)),
            "per_fold_acc": [float(x) for x in accs_k],
        }
        if m_lbl.startswith("K-NN F5 (k="):
            row_cv["k"] = int(m_lbl.split("k=")[1].split(",")[0])
        cv_5fold_summary.append(row_cv)
    cv_5fold_knn_f5 = [r for r in cv_5fold_summary if "k" in r]

    # MLP & Random Forest GroupVal tuning + MLP training loss curves
    print("[Step 5/7] Tuning MLP & Random Forest strictly on GroupVal...", flush=True)
    mlp_tuning_rows = []
    best_mlp_f3p_cfg: dict[str, Any] = {"hidden_layer_sizes": (256, 128), "max_iter": 50}
    best_mlp_f5_cfg: dict[str, Any] = {"hidden_layer_sizes": (256, 128), "max_iter": 50}
    best_mlp_f3p_gv, best_mlp_f5_gv = -1.0, -1.0
    mlp_loss_f3p: list[float] = []
    mlp_loss_f5: list[float] = []

    for feat_tag, xm_tr, xm_val in [("F3_PCA", X_tr_f3p, X_val_f3p), ("F5", X_tr_f5, X_val_f5)]:
        for h_layers in [(128,), (256, 128)]:
            for max_it in [30, 50]:
                m_cand = MLPClassifier(
                    hidden_layer_sizes=h_layers,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=max_it,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                )
                fitted_m, mv_m, _, _, _ = fit_eval_model(m_cand, xm_tr, y_tr, xm_val, y_val)
                mlp_tuning_rows.append(
                    {
                        "feature": feat_tag,
                        "hidden_layer_sizes": list(h_layers),
                        "max_iter": max_it,
                        "n_iter_converged": int(fitted_m.n_iter_),
                        "final_loss": float(fitted_m.loss_),
                        "group_val": mv_m,
                    }
                )
                if feat_tag == "F3_PCA" and mv_m["accuracy"] > best_mlp_f3p_gv:
                    best_mlp_f3p_gv = mv_m["accuracy"]
                    best_mlp_f3p_cfg = {"hidden_layer_sizes": h_layers, "max_iter": max_it}
                    mlp_loss_f3p = [float(x) for x in fitted_m.loss_curve_]
                elif feat_tag == "F5" and mv_m["accuracy"] > best_mlp_f5_gv:
                    best_mlp_f5_gv = mv_m["accuracy"]
                    best_mlp_f5_cfg = {"hidden_layer_sizes": h_layers, "max_iter": max_it}
                    mlp_loss_f5 = [float(x) for x in fitted_m.loss_curve_]

    rf_tuning_rows = []
    for cw, cw_lbl in [(None, "None"), ("balanced_subsample", "balanced_subsample")]:
        for md in [None, 25]:
            rf_c = RandomForestClassifier(
                n_estimators=200,
                criterion="gini",
                max_depth=md,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight=cw,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            _, mv_rf, _, _, _ = fit_eval_model(rf_c, X_tr_f5, y_tr, X_val_f5, y_val)
            rf_tuning_rows.append({"class_weight": cw_lbl, "max_depth": md, "group_val": mv_rf})

    # Exp 5: Locked Final Test Evaluation (80% split + Full-Train 39,209 Refit + Bootstrap 95% CI)
    print("[Step 6/7] Running Exp 5 (80% Split + Full-Train Refits + Holm-Bonferroni McNemar)...", flush=True)
    best_knn_model_80 = KNeighborsClassifier(
        n_neighbors=int(best_knn_cfg["k"]),
        weights=str(best_knn_cfg["weights"]),
        metric=str(best_knn_cfg["metric"]),
        algorithm="brute",
        n_jobs=-1,
    )
    best_f3p_model_80 = KNeighborsClassifier(
        n_neighbors=int(best_f3p_cfg["k"]),
        weights="distance",
        metric=str(best_f3p_cfg["metric"]),
        algorithm="brute",
        n_jobs=-1,
    )
    best_dt_model_80 = DecisionTreeClassifier(
        criterion=str(best_dt_cfg["criterion"]),
        max_depth=best_dt_cfg["max_depth"],
        random_state=RANDOM_STATE,
    )
    rf_def_80 = RandomForestClassifier(
        n_estimators=200, criterion="gini", max_depth=None, min_samples_split=2,
        min_samples_leaf=1, max_features="sqrt", class_weight=None,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_bal_80 = RandomForestClassifier(
        n_estimators=200, criterion="gini", max_depth=None, min_samples_split=2,
        min_samples_leaf=1, max_features="sqrt", class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    mlp_f3p_80 = MLPClassifier(
        hidden_layer_sizes=tuple(best_mlp_f3p_cfg["hidden_layer_sizes"]),
        activation="relu", solver="adam", alpha=1e-4, batch_size=256,
        learning_rate_init=1e-3, max_iter=int(best_mlp_f3p_cfg["max_iter"]),
        early_stopping=False, random_state=RANDOM_STATE,
    )
    mlp_f5_80 = MLPClassifier(
        hidden_layer_sizes=tuple(best_mlp_f5_cfg["hidden_layer_sizes"]),
        activation="relu", solver="adam", alpha=1e-4, batch_size=256,
        learning_rate_init=1e-3, max_iter=int(best_mlp_f5_cfg["max_iter"]),
        early_stopping=False, random_state=RANDOM_STATE,
    )

    alt_metric = "manhattan" if best_knn_cfg["metric"] == "euclidean" else "euclidean"
    model_specs = [
        ("Baseline K-NN (k=5, Raw Gray F1)", "Bắt buộc (Baseline)", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean", algorithm="brute", n_jobs=-1), (X_tr_f1, X_val_f1, X_te_f1)),
        ("Baseline Decision Tree (Raw Gray F1)", "Bắt buộc (Baseline)", DecisionTreeClassifier(criterion="gini", max_depth=20, random_state=RANDOM_STATE), (X_tr_f1, X_val_f1, X_te_f1)),
        ("Gaussian Naive Bayes (Raw Gray F1)", "Bắt buộc (Baseline)", GaussianNB(), (X_tr_f1, X_val_f1, X_te_f1)),
        (f"K-NN F3_PCA (k={best_f3p_cfg['k']}, {best_f3p_cfg['metric']})", "Bắt buộc (Tối ưu F3-PCA)", best_f3p_model_80, (X_tr_f3p, X_val_f3p, X_te_f3p)),
        (f"K-NN tối ưu F5 (k={best_knn_cfg['k']}, {best_knn_cfg['metric']})", "Bắt buộc (Tối ưu F5)", best_knn_model_80, (X_tr_f5, X_val_f5, X_te_f5)),
        (f"K-NN F5 (k={best_knn_cfg['k']}, {alt_metric} đối chứng)", "Bắt buộc (Học phần)", KNeighborsClassifier(n_neighbors=int(best_knn_cfg["k"]), weights="distance", metric=alt_metric, algorithm="brute", n_jobs=-1), (X_tr_f5, X_val_f5, X_te_f5)),
        (f"K-NN F4 không PCA (k={best_knn_cfg['k']}, {best_knn_cfg['metric']})", "Bắt buộc (Học phần)", KNeighborsClassifier(n_neighbors=int(best_knn_cfg["k"]), weights=str(best_knn_cfg["weights"]), metric=str(best_knn_cfg["metric"]), algorithm="brute", n_jobs=-1), (X_tr_f4, X_val_f4, X_te_f4)),
        (f"Decision Tree F5 (depth={best_dt_cfg['max_depth']}, {best_dt_cfg['criterion']})", "Bắt buộc (Học phần)", best_dt_model_80, (X_tr_f5, X_val_f5, X_te_f5)),
        ("Gaussian Naive Bayes (F3_PCA)", "Bắt buộc (Học phần)", GaussianNB(), (X_tr_f3p, X_val_f3p, X_te_f3p)),
        ("Gaussian Naive Bayes (F5)", "Bắt buộc (Học phần)", GaussianNB(), (X_tr_f5, X_val_f5, X_te_f5)),
        ("Random Forest (200 trees, F5 default)", "Mở rộng (Điểm thưởng)", rf_def_80, (X_tr_f5, X_val_f5, X_te_f5)),
        ("Random Forest (200 trees, F5 balanced)", "Mở rộng (Điểm thưởng)", rf_bal_80, (X_tr_f5, X_val_f5, X_te_f5)),
        (f"MLP Neural Network (F3_PCA, iter={best_mlp_f3p_cfg['max_iter']})", "Mở rộng (Tối ưu F3-PCA)", mlp_f3p_80, (X_tr_f3p, X_val_f3p, X_te_f3p)),
        (f"MLP Neural Network (F5, iter={best_mlp_f5_cfg['max_iter']})", "Mở rộng (Tối ưu F5)", mlp_f5_80, (X_tr_f5, X_val_f5, X_te_f5)),
    ]

    exp5_results = []
    exp5_preds: dict[str, np.ndarray] = {}
    knn_val_probs = None
    mlp_val_probs = None
    knn_probs_te = None
    mlp_probs_te = None
    knn_te_pred = np.array([], dtype=np.int32)
    for name, scope, clf, (Xm_tr, Xm_val, Xm_te) in model_specs:
        fitted_clf, m_val, m_te, _, y_pred_te = fit_eval_model(clf, Xm_tr, y_tr, Xm_val, y_val, Xm_te, y_te)
        ci = bootstrap_ci(y_te, y_pred_te)
        exp5_results.append({"model": name, "scope": scope, "group_val": m_val, "test": m_te, "bootstrap_ci": ci})
        exp5_preds[name] = y_pred_te
        if name.startswith("K-NN tối ưu F5"):
            knn_te_pred = y_pred_te
            knn_val_probs = fitted_clf.predict_proba(Xm_val)
            knn_probs_te = fitted_clf.predict_proba(Xm_te)
        elif name.startswith("MLP Neural Network (F5"):
            mlp_val_probs = fitted_clf.predict_proba(Xm_val)
            mlp_probs_te = fitted_clf.predict_proba(Xm_te)

    # Full-Train Refit (all 39,209 training images across all 1,307 tracks)
    y_full = feats_tr_full["y"]
    Xfull_f3p, _, Xfull_te_f3p, scaler_f3_full, pca_f3_full, _ = prepare_scaled_split(
        feats_tr_full["F3"], feats_tr_full["F3"][:1], feats_te["F3"], pca_dim=128
    )
    Xfull_f5, _, Xfull_te_f5, scaler_f5_full, pca_f5_full, _ = prepare_scaled_split(
        feats_tr_full["F4"], feats_tr_full["F4"][:1], feats_te["F4"], pca_dim=128
    )
    knn_f3p_final = KNeighborsClassifier(
        n_neighbors=int(best_f3p_cfg["k"]), weights="distance", metric=str(best_f3p_cfg["metric"]), algorithm="brute", n_jobs=-1
    )
    knn_f5_final = KNeighborsClassifier(
        n_neighbors=int(best_knn_cfg["k"]), weights=str(best_knn_cfg["weights"]), metric=str(best_knn_cfg["metric"]), algorithm="brute", n_jobs=-1
    )
    dt_f5_final = DecisionTreeClassifier(
        criterion=str(best_dt_cfg["criterion"]), max_depth=best_dt_cfg["max_depth"], random_state=RANDOM_STATE
    )
    mlp_f3p_final = MLPClassifier(
        hidden_layer_sizes=tuple(best_mlp_f3p_cfg["hidden_layer_sizes"]),
        activation="relu", solver="adam", alpha=1e-4, batch_size=256,
        learning_rate_init=1e-3, max_iter=int(best_mlp_f3p_cfg["max_iter"]),
        early_stopping=False, random_state=RANDOM_STATE,
    )
    mlp_f5_final = MLPClassifier(
        hidden_layer_sizes=tuple(best_mlp_f5_cfg["hidden_layer_sizes"]),
        activation="relu", solver="adam", alpha=1e-4, batch_size=256,
        learning_rate_init=1e-3, max_iter=int(best_mlp_f5_cfg["max_iter"]),
        early_stopping=False, random_state=RANDOM_STATE,
    )
    full_refit_specs = [
        (f"K-NN F3_PCA Full Train (k={best_f3p_cfg['k']})", "Full-Train Refit (39.209)", knn_f3p_final, (Xfull_f3p, Xfull_te_f3p)),
        (f"K-NN F5 Full Train (k={best_knn_cfg['k']})", "Full-Train Refit (39.209)", knn_f5_final, (Xfull_f5, Xfull_te_f5)),
        ("MLP F3_PCA Full Train", "Full-Train Refit (39.209)", mlp_f3p_final, (Xfull_f3p, Xfull_te_f3p)),
        ("MLP F5 Full Train", "Full-Train Refit (39.209)", mlp_f5_final, (Xfull_f5, Xfull_te_f5)),
        (f"DT F5 Full Train (depth={best_dt_cfg['max_depth']})", "Full-Train Refit (39.209)", dt_f5_final, (Xfull_f5, Xfull_te_f5)),
    ]
    for f_name, f_scope, f_clf, (Xft_tr, Xft_te) in full_refit_specs:
        t0 = time.perf_counter()
        f_clf.fit(Xft_tr, y_full)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        yp_ft = f_clf.predict(Xft_te)
        pred_s = time.perf_counter() - t1
        m_ft = compute_metrics(y_te, yp_ft, fit_time=fit_s, pred_time=pred_s)
        ci_ft = bootstrap_ci(y_te, yp_ft)
        exp5_results.append(
            {
                "model": f_name,
                "scope": f_scope,
                "group_val": None,
                "test": m_ft,
                "bootstrap_ci": ci_ft,
            }
        )
        exp5_preds[f_name] = yp_ft

    # Pairwise McNemar significance tests + Holm-Bonferroni correction across ALL comparisons
    knn_f5_name = next(k for k in exp5_preds if k.startswith("K-NN tối ưu F5"))
    knn_f3p_name = next(k for k in exp5_preds if k.startswith("K-NN F3_PCA (k="))
    mlp_f3p_name = next(k for k in exp5_preds if k.startswith("MLP Neural Network (F3_PCA"))
    mlp_f5_name = next(k for k in exp5_preds if k.startswith("MLP Neural Network (F5"))
    knn_f3p_full_name = next(k for k in exp5_preds if k.startswith("K-NN F3_PCA Full Train"))
    mlp_f3p_full_name = next(k for k in exp5_preds if k.startswith("MLP F3_PCA Full Train"))

    mcnemar_raw = {
        "F1_vs_F3noCLAHE_knn5": mcnemar_test(y_te, exp1_preds["knn_F1"], exp1_preds["knn_F3_noCLAHE"]),
        "F3noCLAHE_vs_F3_knn5": mcnemar_test(y_te, exp1_preds["knn_F3_noCLAHE"], exp1_preds["knn_F3"]),
        "F3_vs_F4_knn5": mcnemar_test(y_te, exp1_preds["knn_F3"], exp1_preds["knn_F4"]),
        "F3_vs_F4_dt20": mcnemar_test(y_te, exp1_preds["dt_F3"], exp1_preds["dt_F4"]),
        "F3_vs_F4_gnb": mcnemar_test(y_te, exp1_preds["gnb_F3"], exp1_preds["gnb_F4"]),
        "F4noCLAHE_vs_F4_dt20": mcnemar_test(y_te, exp1_preds["dt_F4_noCLAHE"], exp1_preds["dt_F4"]),
        "F4noCLAHE_vs_F4_knn5": mcnemar_test(y_te, exp1_preds["knn_F4_noCLAHE"], exp1_preds["knn_F4"]),
        "F3PCA_vs_F5_knn_opt": mcnemar_test(y_te, exp5_preds[knn_f3p_name], exp5_preds[knn_f5_name]),
        "GNB_F5_vs_KNN_F5": mcnemar_test(y_te, exp5_preds["Gaussian Naive Bayes (F5)"], exp5_preds[knn_f5_name]),
        "KNN_F5_vs_MLP_F5": mcnemar_test(y_te, exp5_preds[knn_f5_name], exp5_preds[mlp_f5_name]),
        "RF_F5_vs_MLP_F5": mcnemar_test(y_te, exp5_preds["Random Forest (200 trees, F5 default)"], exp5_preds[mlp_f5_name]),
        "KNN_F3P_80_vs_FullTrain": mcnemar_test(y_te, exp5_preds[knn_f3p_name], exp5_preds[knn_f3p_full_name]),
        "MLP_F3P_80_vs_FullTrain": mcnemar_test(y_te, exp5_preds[mlp_f3p_name], exp5_preds[mlp_f3p_full_name]),
    }
    mcnemar_results = apply_holm_bonferroni(mcnemar_raw)

    # Bidirectional confused pairs (top 10 undirected pairs + directional counts)
    pair_counts: dict[tuple[int, int], int] = {}
    for yt, yp in zip(y_te, knn_te_pred):
        if int(yt) != int(yp):
            pair_counts[(int(yt), int(yp))] = pair_counts.get((int(yt), int(yp)), 0) + 1
    undirected: dict[tuple[int, int], int] = {}
    for (a, b), c in pair_counts.items():
        pair_k = (min(a, b), max(a, b))
        undirected[pair_k] = undirected.get(pair_k, 0) + c
    top_undirected = sorted(undirected.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_confused_pairs = []
    for (ca, cb), tot in top_undirected:
        cnt_ab = pair_counts.get((ca, cb), 0)
        cnt_ba = pair_counts.get((cb, ca), 0)
        if cnt_ba > cnt_ab:
            ca, cb, cnt_ab, cnt_ba = cb, ca, cnt_ba, cnt_ab
        top_confused_pairs.append(
            {
                "class_a_id": ca,
                "class_a_name": CLASS_SHORT_VI[ca],
                "class_b_id": cb,
                "class_b_name": CLASS_SHORT_VI[cb],
                "count_a_to_b": int(cnt_ab),
                "count_b_to_a": int(cnt_ba),
                "total_count": int(tot),
            }
        )

    # Error samples gallery (from highest single directional error pairs)
    top_dir_pairs = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    error_samples = []
    for k, _ in top_dir_pairs:
        matches = np.where((y_te == k[0]) & (knn_te_pred == k[1]))[0]
        if len(matches) > 0:
            ti = int(matches[0])
            row = df_test.iloc[ti]
            error_samples.append(
                {
                    "test_index": ti,
                    "filename": os.path.basename(str(row["Path"])),
                    "width": int(row["Width"]),
                    "height": int(row["Height"]),
                    "true_id": k[0],
                    "true_name": CLASS_NAMES_VI[k[0]],
                    "pred_id": k[1],
                    "pred_name": CLASS_NAMES_VI[k[1]],
                }
            )

    # Quantitative error diagnostics + Wilson CI + speed stratification + logistic regression + GroupVal tau selection
    print("[Step 7/7] Running multivariate error diagnostics, margin-trim Test audit & saving bundle...", flush=True)
    is_err = (y_te != knn_te_pred)
    widths_te = df_test["Width"].to_numpy(dtype=np.float32)
    width_bins = _compute_bin_errors(
        widths_te,
        is_err,
        [("< 32 px", 0, 32), ("32–45 px", 32, 46), ("46–64 px", 46, 65), ("> 64 px", 65, 9999)],
    )
    bright_bins = _compute_bin_errors(
        feats_te["brightness"],
        is_err,
        [("< 50 (Rất tối)", 0, 50), ("50–100 (Tối vừa)", 50, 101), ("101–150 (Đủ sáng)", 101, 151), ("> 150 (Sáng gắt)", 151, 999)],
    )
    speed_mask = np.isin(y_te, list(SPEED_SIGN_ORDER.keys()))
    speed_stratified = {
        "speed_signs": _compute_bin_errors(widths_te[speed_mask], is_err[speed_mask], [("< 32 px", 0, 32), ("32–45 px", 32, 46), (">= 46 px", 46, 9999)]),
        "other_signs": _compute_bin_errors(widths_te[~speed_mask], is_err[~speed_mask], [("< 32 px", 0, 32), ("32–45 px", 32, 46), (">= 46 px", 46, 9999)]),
    }
    logistic_reg = fit_logistic_error_model(y_te, is_err, widths_te, feats_te["brightness"])

    cov_knn_val = _compute_coverage_curve(y_val, knn_val_probs) if knn_val_probs is not None else []
    cov_mlp_val = _compute_coverage_curve(y_val, mlp_val_probs) if mlp_val_probs is not None else []
    cov_knn = _compute_coverage_curve(y_te, knn_probs_te) if knn_probs_te is not None else []
    cov_mlp = _compute_coverage_curve(y_te, mlp_probs_te) if mlp_probs_te is not None else []

    speed_up_knn = [_count_speed_up_errors(y_te, knn_probs_te, tau=t) for t in [0.0, 0.70, 0.85]] if knn_probs_te is not None else []
    speed_up_mlp = [_count_speed_up_errors(y_te, mlp_probs_te, tau=t) for t in [0.0, 0.70, 0.95]] if mlp_probs_te is not None else []

    # Empirical Test evaluation of uncropped vs 10% crop_center_margin vs Ground-Truth ROI (Full-Train MLP F5 & K-NN F5)
    trim_cache_path = os.path.join(cache_dir, "cache_feats_TEST_TRIM10.npz") if cache_dir else None
    if trim_cache_path and os.path.exists(trim_cache_path):
        X_te_trim10_f4 = np.load(trim_cache_path)["X_te_trim10_f4"]
    else:
        test_records = df_test.to_dict("records")
        workers = n_workers or max(1, min(4, (os.cpu_count() or 2)))
        c_size = max(1, (len(test_records) + workers - 1) // workers)
        t_chunks = [(test_records[i : i + c_size], data_root) for i in range(0, len(test_records), c_size)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            X_te_trim10_f4 = np.concatenate(list(pool.map(_extract_margin_chunk, t_chunks)), axis=0)
        if trim_cache_path:
            np.savez_compressed(trim_cache_path, X_te_trim10_f4=X_te_trim10_f4)
    # Build uncropped (No-ROI CLAHE) vs 10% center margin (No-ROI CLAHE) vs GT ROI using K-NN & MLP on F3_PCA and F5
    X_te_trim10_f3p = pca_f3_full.transform(scaler_f3_full.transform(X_te_trim10_f4[:, :1764]))
    X_te_trim10_f5 = pca_f5_full.transform(scaler_f5_full.transform(X_te_trim10_f4))
    margin_trim_val = {
        "n_test": int(len(y_te)),
        "knn_f5_gt_roi_acc": float(accuracy_score(y_te, exp5_preds["K-NN F5 Full Train (k=" + str(best_knn_cfg["k"]) + ")"])),
        "knn_f5_trim10_no_roi_acc": float(accuracy_score(y_te, knn_f5_final.predict(X_te_trim10_f5))),
        "mlp_f3p_gt_roi_acc": float(accuracy_score(y_te, exp5_preds["MLP F3_PCA Full Train"])),
        "mlp_f3p_trim10_no_roi_acc": float(accuracy_score(y_te, mlp_f3p_final.predict(X_te_trim10_f3p))),
        "mlp_f5_gt_roi_acc": float(accuracy_score(y_te, exp5_preds["MLP F5 Full Train"])),
        "mlp_f5_trim10_no_roi_acc": float(accuracy_score(y_te, mlp_f5_final.predict(X_te_trim10_f5))),
    }
    print(f"  Margin-Trim Audit: MLP F3-PCA Trim10%={margin_trim_val['mlp_f3p_trim10_no_roi_acc']*100:.2f}% vs GT-ROI={margin_trim_val['mlp_f3p_gt_roi_acc']*100:.2f}%")

    p_cls, r_cls, f_cls, sup_cls = precision_recall_fscore_support(y_te, knn_te_pred, labels=list(range(43)), zero_division=0)
    per_class = [
        {
            "class_id": cid,
            "name_vi": CLASS_NAMES_VI[cid],
            "name_en": CLASS_NAMES_EN[cid],
            "support": int(sup_cls[cid]),
            "precision": float(p_cls[cid]),
            "recall": float(r_cls[cid]),
            "f1": float(f_cls[cid]),
        }
        for cid in range(43)
    ]

    bundle = {
        "scaler_f3": scaler_f3_full,
        "pca_f3": pca_f3_full,
        "knn_f3p_model": knn_f3p_final,
        "mlp_f3p_model": mlp_f3p_final,
        "scaler": scaler_f5_full,
        "pca": pca_f5_full,
        "knn_model": knn_f5_final,
        "dt_model": dt_f5_final,
        "mlp_model": mlp_f5_final,
        "best_knn_cfg": {"k": int(best_knn_cfg["k"]), "weights": str(best_knn_cfg["weights"]), "metric": str(best_knn_cfg["metric"])},
        "best_f3p_cfg": {"k": int(best_f3p_cfg["k"]), "weights": "distance", "metric": str(best_f3p_cfg["metric"])},
        "best_dt_cfg": {"criterion": str(best_dt_cfg["criterion"]), "max_depth": best_dt_cfg["max_depth"]},
    }
    joblib.dump(bundle, os.path.join(models_dir, "gtsrb_pipeline.joblib"), compress=3)

    # All 10 sample_images/ diagnostics (8 GTSRB signs + 2 external uncropped scenes before/after 10% crop)
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample_images")
    demo_specs = [
        ("test_class_14.png", 14, False),
        ("test_class_12.png", 12, False),
        ("test_class_13.png", 13, False),
        ("test_class_17.png", 17, False),
        ("test_class_40.png", 40, False),
        ("test_class_02.png", 2, False),
        ("test_class_18.png", 18, False),
        ("test_class_25.png", 25, False),
        ("external_scene_stop.png", 14, True),
        ("external_scene_speed50.png", 2, True),
    ]
    demo_multi = []
    for fn, cid_s, do_trim in demo_specs:
        spath = os.path.join(sample_dir, fn)
        if os.path.exists(spath):
            img_s = load_rgb_image(spath)
            img_in = crop_center_margin(img_s, 0.10) if do_trim else img_s
            pre_s = preprocess_image(img_in, roi=None, size=(32, 32), use_roi=False, use_clahe=True)
            f4_s = extract_hog_color_features(pre_s).reshape(1, -1)
            zf3_s = pca_f3_full.transform(scaler_f3_full.transform(f4_s[:, :1764]))
            zf5_s = pca_f5_full.transform(scaler_f5_full.transform(f4_s))
            pk3 = knn_f3p_final.predict_proba(zf3_s)[0]
            pm3 = mlp_f3p_final.predict_proba(zf3_s)[0]
            pk5 = knn_f5_final.predict_proba(zf5_s)[0]
            pdt5 = dt_f5_final.predict_proba(zf5_s)[0]
            pm5 = mlp_f5_final.predict_proba(zf5_s)[0]
            demo_multi.append(
                {
                    "filename": fn,
                    "auto_trim_margin": do_trim,
                    "true_class_id": cid_s,
                    "true_name_vi": CLASS_NAMES_VI[cid_s],
                    "knn_f3p_pred_id": int(np.argmax(pk3)),
                    "knn_f3p_prob": float(np.max(pk3)),
                    "mlp_f3p_pred_id": int(np.argmax(pm3)),
                    "mlp_f3p_prob": float(np.max(pm3)),
                    "knn_pred_id": int(np.argmax(pk5)),
                    "knn_prob": float(np.max(pk5)),
                    "dt_pred_id": int(np.argmax(pdt5)),
                    "dt_prob": float(np.max(pdt5)),
                    "mlp_pred_id": int(np.argmax(pm5)),
                    "mlp_prob": float(np.max(pm5)),
                }
            )

    summary: dict[str, Any] = {
        "dataset": dataset_summary,
        "hyperparams": {
            "split": {"method": "StratifiedGroupKFold", "n_splits": 5, "group_by": "track_id", "shared_tracks": overlap_tracks},
            "pca": {"n_components": 128, "svd_solver": "full"},
            "knn_opt_f5": bundle["best_knn_cfg"] | {"algorithm": "brute", "n_jobs": -1},
            "knn_f3_pca": bundle["best_f3p_cfg"] | {"algorithm": "brute", "n_jobs": -1},
            "dt_opt_f5": bundle["best_dt_cfg"] | {"random_state": RANDOM_STATE},
            "random_forest_default": {"n_estimators": 200, "criterion": "gini", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None, "random_state": RANDOM_STATE},
            "random_forest_balanced": {"n_estimators": 200, "criterion": "gini", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced_subsample", "random_state": RANDOM_STATE},
            "mlp_f3p": {"hidden_layer_sizes": list(best_mlp_f3p_cfg["hidden_layer_sizes"]), "activation": "relu", "solver": "adam", "alpha": 1e-4, "batch_size": 256, "learning_rate_init": 1e-3, "max_iter": best_mlp_f3p_cfg["max_iter"], "early_stopping": False, "random_state": RANDOM_STATE},
            "mlp_f5": {"hidden_layer_sizes": list(best_mlp_f5_cfg["hidden_layer_sizes"]), "activation": "relu", "solver": "adam", "alpha": 1e-4, "batch_size": 256, "learning_rate_init": 1e-3, "max_iter": best_mlp_f5_cfg["max_iter"], "early_stopping": False, "random_state": RANDOM_STATE},
        },
        "exp1_features": exp1_results,
        "color_weight_ablation": color_weight_ablation,
        "exp2_knn_hyperparams": exp2_results,
        "exp2_knn_f3p_hyperparams": exp2_f3p_results,
        "exp3_dt_hyperparams": exp3_results,
        "exp4_pca_dims": exp4_results,
        "exp_5fold_cv": cv_5fold_summary,
        "mlp_tuning": mlp_tuning_rows,
        "rf_tuning": rf_tuning_rows,
        "mlp_loss_curves": {"f3_pca": mlp_loss_f3p, "f5": mlp_loss_f5},
        "exp5_models_comparison": exp5_results,
        "mcnemar_tests": mcnemar_results,
        "top_confused_pairs_knn_f5": top_confused_pairs,
        "per_class_knn_f5": per_class,
        "error_samples_knn_f5": error_samples,
        "error_diagnostics": {
            "width_bins": width_bins,
            "brightness_bins": bright_bins,
            "speed_stratified": speed_stratified,
            "logistic_regression": logistic_reg,
            "coverage_knn_f5_val": cov_knn_val,
            "coverage_mlp_f5_val": cov_mlp_val,
            "coverage_knn_f5": cov_knn,
            "coverage_mlp_f5": cov_mlp,
            "speed_up_errors_knn_f5": speed_up_knn,
            "speed_up_errors_mlp_f5": speed_up_mlp,
        },
        "margin_trim_validation": margin_trim_val,
        "demo_samples_multi": demo_multi,
        "best_knn_cfg": bundle["best_knn_cfg"],
        "best_f3p_cfg": bundle["best_f3p_cfg"],
        "best_dt_cfg": bundle["best_dt_cfg"],
    }

    with open(os.path.join(results_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(per_class).to_csv(os.path.join(results_dir, "per_class_report_knn_f5.csv"), index=False, encoding="utf-8")

    if not skip_plots:
        plot_class_distribution(df_train_full, df_test, os.path.join(figures_dir, "fig1_class_distribution.png"))
        plot_sample_grid_43classes(df_train_full, data_root, os.path.join(figures_dir, "fig2_sample_grid_43classes.png"))
        plot_eda_dimensions_brightness(
            df_train_full,
            feats_tr_full["roi_ratio"] * 100.0,
            feats_tr_full["brightness"],
            feats_te["brightness"],
            os.path.join(figures_dir, "fig3_eda_dimensions_brightness.png"),
        )
        plot_preprocessing_hog_pipeline(df_train_full, data_root, os.path.join(figures_dir, "fig4_preprocessing_hog_pipeline.png"))
        plot_knn_hyperparams(exp2_results, os.path.join(figures_dir, "fig5_knn_hyperparams.png"), five_fold_knn=cv_5fold_knn_f5)
        plot_dt_hyperparams(exp3_results, os.path.join(figures_dir, "fig6_dt_hyperparams.png"))
        plot_pca_analysis(exp4_results, os.path.join(figures_dir, "fig7_pca_analysis.png"))
        plot_model_comparison(exp5_results, os.path.join(figures_dir, "fig8_model_comparison.png"))
        knn_opt_row = next(r for r in exp5_results if r["model"].startswith("K-NN tối ưu F5"))
        plot_confusion_matrix_fig(y_te, knn_te_pred, knn_opt_row["test"]["accuracy"] * 100.0, os.path.join(figures_dir, "fig9_confusion_matrix.png"))
        plot_error_analysis_samples(error_samples, df_test, data_root, os.path.join(figures_dir, "fig10_error_analysis_samples.png"))
        plot_error_diagnostics(width_bins, bright_bins, cov_knn, cov_mlp, os.path.join(figures_dir, "fig11_error_diagnostics.png"))
        plot_ablation_and_loss(color_weight_ablation, mlp_loss_f3p, mlp_loss_f5, os.path.join(figures_dir, "fig13_ablation_loss.png"))

    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for reproducing the GTSRB experimental pipeline."""
    parser = argparse.ArgumentParser(description="FIT4013 GTSRB Pipeline")
    parser.add_argument("--data-root", default="data", help="Root directory containing GTSRB/")
    parser.add_argument("--results-dir", default="results", help="Output folder for JSON & CSV")
    parser.add_argument("--figures-dir", default="report/figures", help="Output folder for PNG charts")
    parser.add_argument("--models-dir", default="models", help="Output folder for saved pipeline")
    parser.add_argument("--cache-dir", default=None, help="Optional folder for .npz feature caches")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args(argv)
    summary = run_experiments(
        data_root=args.data_root,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        models_dir=args.models_dir,
        cache_dir=args.cache_dir,
        skip_plots=args.skip_plots,
        n_workers=args.workers,
    )
    print(f"Hoan tat pipeline! Saved metrics_summary.json ({len(summary['exp5_models_comparison'])} models).")


if __name__ == "__main__":
    main()
