"""Traffic sign inference service: feature transformation, multi-model consensus, and safety checks."""

import os
import time
from typing import Any

import numpy as np

from constants import CLASS_NAMES_EN, CLASS_NAMES_VI
from features import extract_hog_color_features
from webapp.config import DEFAULT_TAU, MODEL_SPECS, SAMPLE_CATALOG, SAMPLE_DIR, get_sign_category
from webapp.services.image_processor import build_visual_pipeline, read_image_unicode
from webapp.services.model_registry import get_model_bundle, get_per_class_map, get_test_benchmarks


def run_prediction(
    img_rgb: np.ndarray,
    model_choice: str = "K-NN (F5 - PCA 128)",
    auto_trim_margin: bool = False,
    tau: float = DEFAULT_TAU,
) -> dict[str, Any]:
    """Execute end-to-end preprocessing, F5 PCA projection, and multi-model inference."""
    bundle = get_model_bundle()
    visual = build_visual_pipeline(img_rgb, auto_trim_margin=auto_trim_margin)

    feat_1860 = extract_hog_color_features(visual["clahe_rgb"]).reshape(1, -1)
    feat_scaled = bundle["scaler"].transform(feat_1860)
    feat_pca = bundle["pca"].transform(feat_scaled)

    per_class = get_per_class_map()
    benchmarks = get_test_benchmarks()

    consensus_rows: list[dict[str, Any]] = []
    selected_probs = None
    selected_spec = MODEL_SPECS[0]
    selected_ms = 0.0

    for spec in MODEL_SPECS:
        t0 = time.perf_counter()
        probs = bundle[spec["bundle_key"]].predict_proba(feat_pca)[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        cid = int(np.argmax(probs))
        conf = float(probs[cid] * 100.0)
        consensus_rows.append(
            {
                "choice": spec["choice"],
                "class_id": cid,
                "name_vi": CLASS_NAMES_VI[cid],
                "name_en": CLASS_NAMES_EN[cid],
                "category": get_sign_category(cid),
                "confidence": round(conf, 2),
                "latency_ms": round(dt_ms, 2),
                "score_type": spec["score_type"],
            }
        )
        if spec["choice"] == model_choice or (
            model_choice in spec["choice"] and selected_probs is None
        ):
            selected_probs = probs
            selected_spec = spec
            selected_ms = dt_ms

    if selected_probs is None:
        selected_probs = bundle["knn_model"].predict_proba(feat_pca)[0]

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
    """Evaluate all 8 curated GTSRB test samples simultaneously across K-NN, DT, and MLP."""
    results: list[dict[str, Any]] = []
    for item in SAMPLE_CATALOG:
        path = os.path.join(SAMPLE_DIR, item["filename"])
        if not os.path.exists(path):
            continue
        rgb = read_image_unicode(path)
        res = run_prediction(rgb, model_choice="K-NN (F5 - PCA 128)", auto_trim_margin=False)
        by_model = {c["choice"]: c for c in res["consensus"]}
        results.append(
            {
                "filename": item["filename"],
                "true_id": item["true_id"],
                "true_name_vi": CLASS_NAMES_VI[item["true_id"]],
                "short_label": item["short_label"],
                "thumb_uri": res["stages"]["orig_uri"],
                "knn": by_model["K-NN (F5 - PCA 128)"],
                "dt": by_model["Decision Tree (F5 - PCA 128)"],
                "mlp": by_model["MLP Neural Network (Mở rộng)"],
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
            "#### Đối chiếu đồng thuận cả 3 mô hình đã lưu (F5 – PCA 128 chiều)",
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
