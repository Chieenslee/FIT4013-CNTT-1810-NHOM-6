"""HTML view routes serving the interactive GTSRB recognition web dashboard."""

import os

from flask import Blueprint, render_template, request

from webapp.config import DEFAULT_MODEL_CHOICE, DEFAULT_TAU, MODEL_SPECS, SAMPLE_CATALOG, SAMPLE_DIR
from webapp.services import (
    evaluate_sample_gallery,
    get_kpi_summary,
    read_image_unicode,
    run_prediction,
)

views_bp = Blueprint("views", __name__)


@views_bp.route("/", methods=["GET"])
def index():
    """Render main dashboard preloaded with default STOP sign or requested sample."""
    requested_sample = request.args.get("sample", "test_class_14.png")
    sample_map = {s["filename"]: s for s in SAMPLE_CATALOG}
    if requested_sample not in sample_map:
        requested_sample = "test_class_14.png"

    default_path = os.path.join(SAMPLE_DIR, requested_sample)
    default_trim = bool(sample_map[requested_sample].get("auto_trim", False))
    initial_result = None
    if os.path.exists(default_path):
        rgb = read_image_unicode(default_path)
        initial_result = run_prediction(
            rgb,
            model_choice=DEFAULT_MODEL_CHOICE,
            auto_trim_margin=default_trim,
            tau=DEFAULT_TAU,
        )

    gallery = evaluate_sample_gallery()
    kpis = get_kpi_summary()
    total = len(gallery)
    pass_counts = {
        "total": total,
        "knn_f3p": sum(1 for g in gallery if g["knn_f3p"]["class_id"] == g["true_id"]),
        "mlp_f3p": sum(1 for g in gallery if g["mlp_f3p"]["class_id"] == g["true_id"]),
        "knn": sum(1 for g in gallery if g["knn"]["class_id"] == g["true_id"]),
        "dt": sum(1 for g in gallery if g["dt"]["class_id"] == g["true_id"]),
        "mlp": sum(1 for g in gallery if g["mlp"]["class_id"] == g["true_id"]),
    }

    return render_template(
        "index.html",
        model_specs=MODEL_SPECS,
        default_tau=DEFAULT_TAU,
        default_model=DEFAULT_MODEL_CHOICE,
        selected_sample=requested_sample,
        initial_result=initial_result,
        gallery=gallery,
        kpis=kpis,
        pass_counts=pass_counts,
    )
