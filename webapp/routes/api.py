"""REST API endpoints for image inference, sample testing, and benchmark retrieval."""

import io
import os

import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image

from webapp.config import DEFAULT_TAU, SAMPLE_CATALOG, SAMPLE_DIR
from webapp.services import (
    evaluate_sample_gallery,
    get_metrics_summary,
    get_test_benchmarks,
    read_image_unicode,
    run_prediction,
)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _json_safe(result: dict) -> dict:
    """Strip internal numpy display buffers before JSON serialization."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint verifying pipeline availability."""
    return jsonify({"status": "ok", "benchmarks": get_test_benchmarks()})


@api_bp.route("/samples", methods=["GET"])
def list_samples():
    """Return 8 curated sample signs and live multi-model consensus evaluation."""
    return jsonify({"samples": evaluate_sample_gallery()})


@api_bp.route("/metrics", methods=["GET"])
def metrics():
    """Return dataset metadata and locked-Test model performance benchmarks."""
    summary = get_metrics_summary()
    return jsonify(
        {
            "dataset": summary.get("dataset", {}),
            "exp5_models_comparison": summary.get("exp5_models_comparison", []),
        }
    )


@api_bp.route("/predict", methods=["POST"])
def predict():
    """Run traffic sign inference from either an uploaded image file or a sample filename."""
    model_choice = request.form.get("model_choice") or (
        request.get_json(silent=True) or {}
    ).get("model_choice", "K-NN (F5 - PCA 128)")

    raw_trim = request.form.get("auto_trim")
    if raw_trim is None and request.is_json:
        raw_trim = (request.get_json(silent=True) or {}).get("auto_trim", False)
    auto_trim = str(raw_trim).lower() in {"1", "true", "yes", "on"}

    raw_tau = request.form.get("tau")
    if raw_tau is None and request.is_json:
        raw_tau = (request.get_json(silent=True) or {}).get("tau", DEFAULT_TAU)
    try:
        tau = max(0.0, min(0.99, float(raw_tau if raw_tau is not None else DEFAULT_TAU)))
    except (TypeError, ValueError):
        tau = DEFAULT_TAU

    img_rgb = None
    source_name = ""

    if "image" in request.files and request.files["image"].filename:
        file_obj = request.files["image"]
        source_name = file_obj.filename
        try:
            with Image.open(io.BytesIO(file_obj.read())) as pil:
                img_rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            return jsonify({"error": f"Không thể đọc tệp ảnh tải lên: {exc}"}), 400
    else:
        sample_name = request.form.get("sample_name") or (
            request.get_json(silent=True) or {}
        ).get("sample_name")
        allowed = {s["filename"]: s for s in SAMPLE_CATALOG}
        if sample_name not in allowed:
            return jsonify({"error": "Vui lòng tải ảnh lên hoặc chọn một ảnh mẫu hợp lệ."}), 400
        sample_path = os.path.join(SAMPLE_DIR, sample_name)
        if not os.path.exists(sample_path):
            return jsonify({"error": f"Không tìm thấy ảnh mẫu {sample_name}."}), 404
        img_rgb = read_image_unicode(sample_path)
        source_name = sample_name

    result = _json_safe(
        run_prediction(
            img_rgb,
            model_choice=model_choice,
            auto_trim_margin=auto_trim,
            tau=tau,
        )
    )
    result["source_name"] = source_name
    return jsonify(result)
