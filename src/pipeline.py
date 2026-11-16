"""End-to-end GTSRB training, track-disjoint GroupKFold tuning, statistical testing, and plotting pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .constants import CLASS_NAMES_EN, CLASS_NAMES_VI, CLASS_SHORT_VI  # noqa: E402
from .data_loader import load_rgb_image, prepare_gtsrb_csvs  # noqa: E402
from .features import (  # noqa: E402
    extract_hog_color_features,
    extract_hog_features,
    extract_raw_gray,
    extract_raw_rgb,
    preprocess_image,
)

RANDOM_STATE = 42


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


def plot_knn_hyperparams(exp2_rows: list[dict[str, Any]], out_path: str) -> None:
    _apply_pub_style()
    ks = [1, 3, 5, 7, 11]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=200)

    for w_mode, style, col in [("uniform", "o--", "#64748b"), ("distance", "s-", "#1d4ed8")]:
        ys = [next(r["group_val"]["accuracy"] * 100 for r in exp2_rows if r["k"] == k and r["weights"] == w_mode and r["metric"] == "euclidean") for k in ks]
        ax1.plot(ks, ys, style, lw=2.2, markersize=6.5, color=col, label=f"weights='{w_mode}'")
    ax1.set_xticks(ks)
    ax1.set_xlabel("Số láng giềng k")
    ax1.set_ylabel("GroupVal Accuracy (%)")
    ax1.set_title("(a) Ảnh hưởng của k và weights (Euclidean, F5)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.35)

    for metric, style, col in [("euclidean", "o-", "#1d4ed8"), ("manhattan", "s-", "#b91c1c"), ("cosine", "^--", "#0f766e")]:
        ys = [next(r["group_val"]["accuracy"] * 100 for r in exp2_rows if r["k"] == k and r["weights"] == "distance" and r["metric"] == metric) for k in ks]
        ax2.plot(ks, ys, style, lw=2.2, markersize=6.5, color=col, label=f"metric='{metric}'")
    ax2.set_xticks(ks)
    ax2.set_xlabel("Số láng giềng k")
    ax2.set_ylabel("GroupVal Accuracy (%)")
    ax2.set_title("(b) Ảnh hưởng của độ đo khoảng cách (weights='distance')")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.35)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Tinh chỉnh K-NN trên GroupVal tách theo track (lưới đầy đủ 30 cấu hình, F5)", fontsize=13.2, fontweight="bold", y=0.98)
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

    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=200)
    bars_acc = ax.barh(y_pos - h / 2.0, accs, h, xerr=acc_errs, capsize=3, color="#1d4ed8", error_kw={"ecolor": "#0f172a", "lw": 1.0}, label="Test Accuracy (% ± CI 95%)")
    bars_f1 = ax.barh(y_pos + h / 2.0, f1s, h, xerr=f1_errs, capsize=3, color="#d97706", error_kw={"ecolor": "#0f172a", "lw": 1.0}, label="Macro-F1 (% ± CI 95%)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 114)
    ax.set_xlabel("Điểm đánh giá trên tập Test độc lập (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for b, err in zip(bars_acc, acc_errs):
        w = b.get_width()
        ax.text(w + err + 0.8, b.get_y() + b.get_height() / 2.0, f"{w:.2f}%", va="center", fontsize=9.2, fontweight="bold", color="#1e3a8a")
    for b, err in zip(bars_f1, f1_errs):
        w = b.get_width()
        ax.text(w + err + 0.8, b.get_y() + b.get_height() / 2.0, f"{w:.2f}%", va="center", fontsize=9.2, fontweight="bold", color="#92400e")

    fig.suptitle("So sánh hiệu năng các mô hình trên tập Test (12.630 ảnh, Bootstrap 95% CI)", fontsize=12.5, fontweight="bold", y=0.98)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=2, frameon=True)
    fig.subplots_adjust(left=0.32, right=0.97, top=0.81, bottom=0.11)
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
    """Render Figure 11: Quantitative error rate by Width & Brightness and Confidence-Coverage curves."""
    _apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), dpi=200)

    w_lbls = [f"{b['label']}\n(n={b['count']})" for b in width_bins]
    w_errs = [b["error_rate_pct"] for b in width_bins]
    bars0 = axes[0].bar(np.arange(len(w_lbls)), w_errs, color="#1f4e79", width=0.55)
    for b, val in zip(bars0, w_errs):
        axes[0].text(b.get_x() + b.get_width() / 2.0, val + 0.4, f"{val:.2f}%", ha="center", va="bottom", fontsize=9.8, fontweight="bold")
    axes[0].set_xticks(np.arange(len(w_lbls)))
    axes[0].set_xticklabels(w_lbls, fontsize=9.5)
    axes[0].set_ylim(0, max(w_errs) * 1.25)
    axes[0].set_title("(a) Tỷ lệ lỗi theo kích thước ảnh gốc")
    axes[0].set_ylabel("Tỷ lệ dự đoán sai (%)")

    b_lbls = [b["label"].replace(" (", "\n(") + f"\n(n={b['count']})" for b in bright_bins]
    b_errs = [b["error_rate_pct"] for b in bright_bins]
    bars1 = axes[1].bar(np.arange(len(b_lbls)), b_errs, color="#c2410c", width=0.55)
    for b, val in zip(bars1, b_errs):
        axes[1].text(b.get_x() + b.get_width() / 2.0, val + 0.4, f"{val:.2f}%", ha="center", va="bottom", fontsize=9.8, fontweight="bold")
    axes[1].set_xticks(np.arange(len(b_lbls)))
    axes[1].set_xticklabels(b_lbls, fontsize=9.0)
    axes[1].set_ylim(0, max(b_errs) * 1.25)
    axes[1].set_title("(b) Tỷ lệ lỗi theo độ sáng ảnh gốc")
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
    fig.suptitle("Kiểm chứng định lượng nguyên nhân lỗi và hiệu quả ngưỡng từ chối (12.630 ảnh Test)", fontsize=13.2, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.83, bottom=0.18, wspace=0.26)
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
        rows.append(
            {
                "label": label,
                "count": cnt,
                "errors": errs,
                "error_rate_pct": float(100.0 * errs / max(1, cnt)),
            }
        )
    return rows


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
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

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

    # Exp 1: Feature sets F0..F5 + F3_PCA across K-NN, Decision Tree, GaussianNB
    feature_specs = [
        ("F0: No-ROI Raw Grayscale (1024)", "F0", None),
        ("F1: ROI Raw Grayscale - Baseline (1024)", "F1", None),
        ("F2: ROI Raw RGB (3072)", "F2", None),
        ("F3: ROI + CLAHE + HOG (1764)", "F3", None),
        ("F3_PCA: ROI + CLAHE + HOG + PCA (128)", "F3", 128),
        ("F4_noCLAHE: ROI + HOG + Color (1860)", "F4_noCLAHE", None),
        ("F4: ROI + CLAHE + HOG + Color (1860)", "F4", None),
        ("F5: ROI + CLAHE + HOG + Color + PCA (128)", "F4", 128),
    ]

    exp1_results = []
    cache_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA | None, float]] = {}
    exp1_preds: dict[str, np.ndarray] = {}
    for feat_label, src_key, pca_dim in feature_specs:
        split_key = feat_label.split(":")[0]
        X_tr_s, X_val_s, X_te_s, sc, pca_obj, exp_var = prepare_scaled_split(
            feats_tr_full[src_key][idx_tr], feats_tr_full[src_key][idx_val], feats_te[src_key], pca_dim=pca_dim
        )
        cache_splits[split_key] = (X_tr_s, X_val_s, X_te_s, sc, pca_obj, exp_var)

        knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean", algorithm="brute", n_jobs=-1)
        _, knn_val, knn_te, _, knn_te_p = fit_eval_model(knn, X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te)
        dt = DecisionTreeClassifier(criterion="gini", max_depth=20, random_state=RANDOM_STATE)
        _, dt_val, dt_te, _, _ = fit_eval_model(dt, X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te)
        gnb = GaussianNB()
        _, gnb_val, gnb_te, _, _ = fit_eval_model(gnb, X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te)

        exp1_preds[f"knn_{split_key}"] = knn_te_p
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

    X_tr_f5, X_val_f5, X_te_f5, scaler_f5, pca_f5, _ = cache_splits["F5"]
    X_tr_f3p, X_val_f3p, X_te_f3p, _, _, _ = cache_splits["F3_PCA"]
    X_tr_f4, X_val_f4, X_te_f4, _, _, _ = cache_splits["F4"]
    X_tr_f1, X_val_f1, X_te_f1, _, _, _ = cache_splits["F1"]

    # Exp 2: Full 30-config K-NN grid on F5 — evaluated STRICTLY on GroupVal (no Test peeking)
    exp2_results = []
    for k in [1, 3, 5, 7, 11]:
        for weights in ["uniform", "distance"]:
            for metric in ["euclidean", "manhattan", "cosine"]:
                clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, algorithm="brute", n_jobs=-1)
                _, m_val, _, _, _ = fit_eval_model(clf, X_tr_f5, y_tr, X_val_f5, y_val)
                exp2_results.append({"k": k, "weights": weights, "metric": metric, "group_val": m_val})
    best_knn_cfg = max(exp2_results, key=lambda r: r["group_val"]["accuracy"])

    # Also find best F3_PCA K-NN strictly on GroupVal
    f3p_val_best = None
    best_f3p_cfg = {"k": 11, "weights": "distance", "metric": "euclidean"}
    for metric in ["euclidean", "manhattan", "cosine"]:
        for k in [3, 5, 7, 11]:
            clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric=metric, algorithm="brute", n_jobs=-1)
            _, mv, _, _, _ = fit_eval_model(clf, X_tr_f3p, y_tr, X_val_f3p, y_val)
            if f3p_val_best is None or mv["accuracy"] > f3p_val_best:
                f3p_val_best = mv["accuracy"]
                best_f3p_cfg = {"k": k, "weights": "distance", "metric": metric}

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

    # Exp 5: Locked Final Test Evaluation (evaluated once per selected/benchmark architecture + Bootstrap 95% CI)
    best_knn_model = KNeighborsClassifier(
        n_neighbors=int(best_knn_cfg["k"]),
        weights=str(best_knn_cfg["weights"]),
        metric=str(best_knn_cfg["metric"]),
        algorithm="brute",
        n_jobs=-1,
    )
    best_dt_model = DecisionTreeClassifier(
        criterion=str(best_dt_cfg["criterion"]),
        max_depth=best_dt_cfg["max_depth"],
        random_state=RANDOM_STATE,
    )
    rf_model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=30,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )

    alt_metric = "manhattan" if best_knn_cfg["metric"] == "euclidean" else "euclidean"
    model_specs = [
        ("Baseline K-NN (k=5, Raw Gray F1)", "Bắt buộc (Baseline)", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean", algorithm="brute", n_jobs=-1), (X_tr_f1, X_val_f1, X_te_f1)),
        ("Baseline Decision Tree (Raw Gray F1)", "Bắt buộc (Baseline)", DecisionTreeClassifier(criterion="gini", max_depth=20, random_state=RANDOM_STATE), (X_tr_f1, X_val_f1, X_te_f1)),
        ("Gaussian Naive Bayes (Raw Gray F1)", "Bắt buộc (Baseline)", GaussianNB(), (X_tr_f1, X_val_f1, X_te_f1)),
        (f"K-NN F3_PCA ({best_f3p_cfg['k']}, {best_f3p_cfg['metric']})", "Bắt buộc (Học phần)", KNeighborsClassifier(n_neighbors=best_f3p_cfg["k"], weights="distance", metric=best_f3p_cfg["metric"], algorithm="brute", n_jobs=-1), (X_tr_f3p, X_val_f3p, X_te_f3p)),
        (f"K-NN tối ưu F5 (k={best_knn_cfg['k']}, {best_knn_cfg['metric']})", "Bắt buộc (Học phần)", best_knn_model, (X_tr_f5, X_val_f5, X_te_f5)),
        (f"K-NN F5 ({best_knn_cfg['k']}, {alt_metric} đối chứng)", "Bắt buộc (Học phần)", KNeighborsClassifier(n_neighbors=int(best_knn_cfg["k"]), weights="distance", metric=alt_metric, algorithm="brute", n_jobs=-1), (X_tr_f5, X_val_f5, X_te_f5)),
        (f"K-NN F4 không PCA (k={best_knn_cfg['k']}, {best_knn_cfg['metric']})", "Bắt buộc (Học phần)", KNeighborsClassifier(n_neighbors=int(best_knn_cfg["k"]), weights=str(best_knn_cfg["weights"]), metric=str(best_knn_cfg["metric"]), algorithm="brute", n_jobs=-1), (X_tr_f4, X_val_f4, X_te_f4)),
        (f"Decision Tree F5 (depth={best_dt_cfg['max_depth']}, {best_dt_cfg['criterion']})", "Bắt buộc (Học phần)", best_dt_model, (X_tr_f5, X_val_f5, X_te_f5)),
        ("Gaussian Naive Bayes (F5)", "Bắt buộc (Học phần)", GaussianNB(), (X_tr_f5, X_val_f5, X_te_f5)),
        ("Random Forest (200 trees, F5)", "Mở rộng (Điểm thưởng)", rf_model, (X_tr_f5, X_val_f5, X_te_f5)),
        ("MLP Neural Network (F5)", "Mở rộng (Điểm thưởng)", mlp_model, (X_tr_f5, X_val_f5, X_te_f5)),
    ]

    exp5_results = []
    exp5_preds: dict[str, np.ndarray] = {}
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
            knn_probs_te = fitted_clf.predict_proba(Xm_te)
        elif name.startswith("MLP Neural Network"):
            mlp_probs_te = fitted_clf.predict_proba(Xm_te)

    # Pairwise McNemar significance tests on Test set (12,630 samples)
    knn_f5_name = next(k for k in exp5_preds if k.startswith("K-NN tối ưu F5"))
    knn_f3p_name = next(k for k in exp5_preds if k.startswith("K-NN F3_PCA"))
    mcnemar_results = {
        "F3_vs_F4_knn5": mcnemar_test(y_te, exp1_preds["knn_F3"], exp1_preds["knn_F4"]),
        "F4noCLAHE_vs_F4_knn5": mcnemar_test(y_te, exp1_preds["knn_F4_noCLAHE"], exp1_preds["knn_F4"]),
        "F3PCA_vs_F5_knn_opt": mcnemar_test(y_te, exp5_preds[knn_f3p_name], exp5_preds[knn_f5_name]),
        "GNB_F5_vs_KNN_F5": mcnemar_test(y_te, exp5_preds["Gaussian Naive Bayes (F5)"], exp5_preds[knn_f5_name]),
        "KNN_F5_vs_MLP_F5": mcnemar_test(y_te, exp5_preds[knn_f5_name], exp5_preds["MLP Neural Network (F5)"]),
        "RF_F5_vs_MLP_F5": mcnemar_test(y_te, exp5_preds["Random Forest (200 trees, F5)"], exp5_preds["MLP Neural Network (F5)"]),
    }

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

    # Quantitative error diagnostics by Width, Brightness & Confidence-Coverage
    is_err = (y_te != knn_te_pred)
    width_bins = _compute_bin_errors(
        df_test["Width"].to_numpy(dtype=np.float32),
        is_err,
        [("< 32 px", 0, 32), ("32–45 px", 32, 46), ("46–64 px", 46, 65), ("> 64 px", 65, 9999)],
    )
    bright_bins = _compute_bin_errors(
        feats_te["brightness"],
        is_err,
        [("< 50 (Rất tối)", 0, 50), ("50–100 (Tối vừa)", 50, 101), ("101–150 (Đủ sáng)", 101, 151), ("> 150 (Sáng gắt)", 151, 999)],
    )
    cov_knn = _compute_coverage_curve(y_te, knn_probs_te) if knn_probs_te is not None else []
    cov_mlp = _compute_coverage_curve(y_te, mlp_probs_te) if mlp_probs_te is not None else []

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
        "scaler": scaler_f5,
        "pca": pca_f5,
        "knn_model": best_knn_model,
        "dt_model": best_dt_model,
        "mlp_model": mlp_model,
        "best_knn_cfg": {"k": int(best_knn_cfg["k"]), "weights": str(best_knn_cfg["weights"]), "metric": str(best_knn_cfg["metric"])},
        "best_dt_cfg": {"criterion": str(best_dt_cfg["criterion"]), "max_depth": best_dt_cfg["max_depth"]},
    }
    joblib.dump(bundle, os.path.join(models_dir, "gtsrb_pipeline.joblib"), compress=3)

    # Multi-sample demo diagnostics matching sample_images/
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample_images")
    demo_multi = []
    for cid_s in [14, 12, 13, 17, 40]:
        spath = os.path.join(sample_dir, f"test_class_{cid_s:02d}.png")
        if os.path.exists(spath):
            img_s = load_rgb_image(spath)
            pre_s = preprocess_image(img_s, roi=None, size=(32, 32), use_roi=False, use_clahe=True)
            f_s = pca_f5.transform(scaler_f5.transform(extract_hog_color_features(pre_s).reshape(1, -1)))
            pk = best_knn_model.predict_proba(f_s)[0]
            pm = mlp_model.predict_proba(f_s)[0]
            demo_multi.append(
                {
                    "filename": f"test_class_{cid_s:02d}.png",
                    "true_class_id": cid_s,
                    "true_name_vi": CLASS_NAMES_VI[cid_s],
                    "knn_pred_id": int(np.argmax(pk)),
                    "knn_prob": float(np.max(pk)),
                    "mlp_pred_id": int(np.argmax(pm)),
                    "mlp_prob": float(np.max(pm)),
                }
            )

    summary: dict[str, Any] = {
        "dataset": dataset_summary,
        "hyperparams": {
            "split": {"method": "StratifiedGroupKFold", "n_splits": 5, "group_by": "track_id", "shared_tracks": overlap_tracks},
            "pca": {"n_components": 128, "svd_solver": "full"},
            "knn_opt_f5": bundle["best_knn_cfg"] | {"algorithm": "brute", "n_jobs": -1},
            "knn_f3_pca": best_f3p_cfg | {"algorithm": "brute", "n_jobs": -1},
            "dt_opt_f5": bundle["best_dt_cfg"] | {"random_state": RANDOM_STATE},
            "random_forest": {"n_estimators": 200, "criterion": "gini", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None, "random_state": RANDOM_STATE},
            "mlp": {"hidden_layer_sizes": [256, 128], "activation": "relu", "solver": "adam", "alpha": 1e-4, "batch_size": 256, "learning_rate_init": 1e-3, "max_iter": 30, "early_stopping": False, "random_state": RANDOM_STATE},
        },
        "exp1_features": exp1_results,
        "exp2_knn_hyperparams": exp2_results,
        "exp3_dt_hyperparams": exp3_results,
        "exp4_pca_dims": exp4_results,
        "exp5_models_comparison": exp5_results,
        "mcnemar_tests": mcnemar_results,
        "top_confused_pairs_knn_f5": top_confused_pairs,
        "per_class_knn_f5": per_class,
        "error_samples_knn_f5": error_samples,
        "error_diagnostics": {
            "width_bins": width_bins,
            "brightness_bins": bright_bins,
            "coverage_knn_f5": cov_knn,
            "coverage_mlp_f5": cov_mlp,
        },
        "demo_samples_multi": demo_multi,
        "best_knn_cfg": bundle["best_knn_cfg"],
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
        plot_knn_hyperparams(exp2_results, os.path.join(figures_dir, "fig5_knn_hyperparams.png"))
        plot_dt_hyperparams(exp3_results, os.path.join(figures_dir, "fig6_dt_hyperparams.png"))
        plot_pca_analysis(exp4_results, os.path.join(figures_dir, "fig7_pca_analysis.png"))
        plot_model_comparison(exp5_results, os.path.join(figures_dir, "fig8_model_comparison.png"))
        knn_opt_row = next(r for r in exp5_results if r["model"].startswith("K-NN tối ưu F5"))
        plot_confusion_matrix_fig(y_te, knn_te_pred, knn_opt_row["test"]["accuracy"] * 100.0, os.path.join(figures_dir, "fig9_confusion_matrix.png"))
        plot_error_analysis_samples(error_samples, df_test, data_root, os.path.join(figures_dir, "fig10_error_analysis_samples.png"))
        plot_error_diagnostics(width_bins, bright_bins, cov_knn, cov_mlp, os.path.join(figures_dir, "fig11_error_diagnostics.png"))

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
