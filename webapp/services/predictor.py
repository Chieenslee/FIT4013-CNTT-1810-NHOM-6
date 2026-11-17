"""Traffic sign inference service: F3/F5 feature transformation, consensus, and safety checks."""

import os
import time
from typing import Any

import numpy as np

from constants import CLASS_NAMES_EN, CLASS_NAMES_VI
from features import extract_hog_color_features
from webapp.config import (
    DEFAULT_MODEL_CHOICE,
    DEFAULT_TAU,
    MODEL_SPECS,
    SAMPLE_CATALOG,
    SAMPLE_DIR,
    get_sign_category,
)
from webapp.services.image_processor import build_visual_pipeline, read_image_unicode
from webapp.services.model_registry import get_model_bundle, get_per_class_map, get_test_benchmarks


def _resolve_model_spec(model_choice: str) -> dict[str, Any]:
    for spec in MODEL_SPECS:
        if spec["choice"] == model_choice:
            return spec
    for spec in MODEL_SPECS:
        if model_choice and model_choice in spec["choice"]:
            return spec
    return MODEL_SPECS[0]


def run_prediction(
    img_rgb: np.ndarray,
    model_choice: str = DEFAULT_MODEL_CHOICE,
    auto_trim_margin: bool = False,
    tau: float = DEFAULT_TAU,
) -> dict[str, Any]:
    """Execute preprocessing, F3/F5 PCA projections, and multi-model consensus inference."""
    bundle = get_model_bundle()
    visual = build_visual_pipeline(img_rgb, auto_trim_margin=auto_trim_margin)

    feat_1860 = extract_hog_color_features(visual["clahe_rgb"]).reshape(1, -1)
    feat_pca_f5 = bundle["pca"].transform(bundle["scaler"].transform(feat_1860))
    if "scaler_f3" in bundle and "pca_f3" in bundle:
        feat_pca_f3 = bundle["pca_f3"].transform(bundle["scaler_f3"].transform(feat_1860[:, :1764]))
    else:
        feat_pca_f3 = feat_pca_f5

    per_class = get_per_class_map()
    benchmarks = get_test_benchmarks()
    target_spec = _resolve_model_spec(model_choice)

    consensus_rows: list[dict[str, Any]] = []
    selected_probs = None
    selected_spec = MODEL_SPECS[0]
    selected_ms = 0.0

    for spec in MODEL_SPECS:
        bkey = spec["bundle_key"]
        if bkey not in bundle:
            continue
        x_in = feat_pca_f3 if spec.get("feature_space") == "f3" else feat_pca_f5
        t0 = time.perf_counter()
        probs = bundle[bkey].predict_proba(x_in)[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        cid = int(np.argmax(probs))
        conf = float(probs[cid] * 100.0)
        consensus_rows.append(
            {
                "choice": spec["choice"],
                "short_label": spec["short_label"],
                "bundle_key": bkey,
                "class_id": cid,
                "name_vi": CLASS_NAMES_VI[cid],
                "name_en": CLASS_NAMES_EN[cid],
                "category": get_sign_category(cid),
                "confidence": round(conf, 2),
                "latency_ms": round(dt_ms, 2),
                "score_type": spec["score_type"],
            }
        )
        if spec["choice"] == target_spec["choice"]:
            selected_probs = probs
            selected_spec = spec
            selected_ms = dt_ms

    if selected_probs is None:
        fallback_key = selected_spec["bundle_key"] if selected_spec["bundle_key"] in bundle else "knn_model"
        x_fb = feat_pca_f3 if (fallback_key == "knn_f3p_model") else feat_pca_f5
        selected_probs = bundle[fallback_key].predict_proba(x_fb)[0]

    top5_idx = np.argsort(selected_probs)[::-1][:5]
    top_predictions = []
    for rank, idx in enumerate(top5_idx, 1):
        cid = int(idx)
        cls_f1 = per_class.get(cid, {}).get("f1", 0.0) * 100.0
        top_predictions.append(
            {
                "rank": rank,
                "class_id": cid,
                "name_vi": CLASS_NAMES_VI[cid],
                "name_en": CLASS_NAMES_EN[cid],
                "category": get_sign_category(cid),
                "confidence": round(float(selected_probs[cid] * 100.0), 2),
                "knn_test_f1": round(cls_f1, 1),
            }
        )

    pred_ids = [r["class_id"] for r in consensus_rows]
    primary = top_predictions[0]
    agree_count = sum(1 for cid in pred_ids if cid == primary["class_id"])
    conf_ratio = primary["confidence"] / 100.0

    if conf_ratio >= tau:
        safety_level = "safe"
        safety_label = f"Đạt ngưỡng chấp nhận ADAS (τ = {tau:.2f})"
    else:
        safety_level = "reject"
        safety_label = (
            f"Cảnh báo tin cậy thấp ({conf_ratio:.2f} < τ = {tau:.2f}) — Cần kiểm tra lại / đối chiếu đồng thuận"
        )

    return {
        "model_choice": selected_spec["choice"],
        "score_type": selected_spec["score_type"],
        "latency_ms": round(selected_ms, 2),
        "auto_trim_margin": bool(auto_trim_margin),
        "tau": round(float(tau), 2),
        "safety_level": safety_level,
        "safety_label": safety_label,
        "agreement_count": agree_count,
        "agreement_total": len(consensus_rows),
        "primary": primary,
        "top_predictions": top_predictions,
        "consensus": consensus_rows,
        "stages": visual["stages"],
        "diagnostics": visual["diagnostics"],
        "benchmark": benchmarks.get(selected_spec["choice"], {}),
        "_clahe_disp": visual["clahe_disp"],
        "_hog_disp": visual["hog_disp"],
    }


def evaluate_sample_gallery() -> list[dict[str, Any]]:
    """Evaluate all curated GTSRB + external uncropped samples across saved F3-PCA and F5 models."""
    results: list[dict[str, Any]] = []
    for item in SAMPLE_CATALOG:
        path = os.path.join(SAMPLE_DIR, item["filename"])
        if not os.path.exists(path):
            continue
        rgb = read_image_unicode(path)
        do_trim = bool(item.get("auto_trim", False))
        res = run_prediction(rgb, model_choice=DEFAULT_MODEL_CHOICE, auto_trim_margin=do_trim)
        by_key = {c["bundle_key"]: c for c in res["consensus"]}
        results.append(
            {
                "filename": item["filename"],
                "true_id": item["true_id"],
                "true_name_vi": CLASS_NAMES_VI[item["true_id"]],
                "short_label": item["short_label"],
                "auto_trim": do_trim,
                "thumb_uri": res["stages"]["orig_uri"],
                "knn_f3p": by_key.get("knn_f3p_model", by_key["knn_model"]),
                "mlp_f3p": by_key.get("mlp_f3p_model", by_key["mlp_model"]),
                "knn": by_key["knn_model"],
                "dt": by_key["dt_model"],
                "mlp": by_key["mlp_model"],
                "size_str": f"{res['diagnostics']['width']}×{res['diagnostics']['height']}",
            }
        )
    return results


def format_prediction_markdown(result: dict[str, Any]) -> str:
    """Format prediction result dictionary into Markdown compatible with CLI & tests."""
    lines = [
        f"### Mô hình sử dụng: {result['model_choice']} ({result['latency_ms']:.2f} ms)",
        "",
    ]
    for item in result["top_predictions"][:3]:
        lines.append(
            f"**Top {item['rank']}: Lớp {item['class_id']} – {item['name_vi']}** (`{item['name_en']}`) — Độ tin cậy: **{item['confidence']:.2f}%**"
        )
    lines.extend(
        [
            "",
            f"#### Đối chiếu đồng thuận {len(result['consensus'])} mô hình đã lưu (F3-PCA & F5-PCA 128 chiều)",
            "| Mô hình | Dự đoán Top-1 | Độ tin cậy (chưa hiệu chuẩn) | Thời gian |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result["consensus"]:
        lines.append(
            f"| {row['choice']} | Lớp {row['class_id']} – {row['name_vi']} | {row['confidence']:.2f}% | {row['latency_ms']:.2f} ms |"
        )
    lines.append(
        "\n*Ghi chú: Giá trị phần trăm là điểm bỏ phiếu nghịch đảo (K-NN), độ tinh khiết nút lá (DT) hoặc softmax (MLP), chưa hiệu chuẩn xác suất hậu nghiệm.*"
    )
    return "\n".join(lines)
