"""Clean entry point for the GTSRB Traffic Sign Recognition Flask Web Application.

Run locally with: `python app_demo.py` (serves at http://127.0.0.1:7860).
All domain logic is modularized inside the `webapp/` package (`config`, `services/`, `routes/`).
"""

import numpy as np

from webapp import create_app
from webapp.config import DEFAULT_TAU
from webapp.services import (
    crop_center_margin,
    format_prediction_markdown,
    run_prediction,
)


def predict_traffic_sign(
    image_rgb: np.ndarray,
    model_choice: str = "K-NN (F5 - PCA 128)",
    auto_trim_margin: bool = False,
    tau: float = DEFAULT_TAU,
):
    """Run programmatic prediction returning (clahe_rgb_160, hog_rgb_160, summary_markdown)."""
    if image_rgb is None:
        return None, None, "Vui lòng tải lên một ảnh biển báo giao thông."
    result = run_prediction(
        image_rgb,
        model_choice=model_choice,
        auto_trim_margin=auto_trim_margin,
        tau=tau,
    )
    return result["_clahe_disp"], result["_hog_disp"], format_prediction_markdown(result)


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
