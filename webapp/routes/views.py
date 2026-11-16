"""HTML view routes serving the interactive GTSRB recognition web dashboard."""

import os

from flask import Blueprint, render_template, request

from webapp.config import DEFAULT_TAU, MODEL_SPECS, SAMPLE_CATALOG, SAMPLE_DIR
from webapp.services import (
    evaluate_sample_gallery,
    get_metrics_summary,
    read_image_unicode,
    run_prediction,
)

views_bp = Blueprint("views", __name__)


@views_bp.route("/", methods=["GET"])
def index():
    """Render main dashboard preloaded with default STOP sign or requested sample."""
    requested_sample = request.args.get("sample", "test_class_14.png")
    sample_names = {s["filename"] for s in SAMPLE_CATALOG}
    if requested_sample not in sample_names:
        requested_sample = "test_class_14.png"

    default_path = os.path.join(SAMPLE_DIR, requested_sample)
    initial_result = None
    if os.path.exists(default_path):
        rgb = read_image_unicode(default_path)
        initial_result = run_prediction(
            rgb,
            model_choice="K-NN (F5 - PCA 128)",
            auto_trim_margin=False,
            tau=DEFAULT_TAU,
        )

    gallery = evaluate_sample_gallery()
    summary = get_metrics_summary()
    dataset_meta = summary.get("dataset", {})
    return render_template(
        "index.html",
        model_specs=MODEL_SPECS,
        default_tau=DEFAULT_TAU,
        selected_sample=requested_sample,
        initial_result=initial_result,
        gallery=gallery,
        dataset_meta=dataset_meta,
    )
