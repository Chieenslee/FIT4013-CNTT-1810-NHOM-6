"""Modular GTSRB Traffic Sign Recognition pipeline (FIT4013 - Nhom 6)."""

from .constants import CLASS_NAMES_EN, CLASS_NAMES_VI, CLASS_SHORT_VI
from .data_loader import load_rgb_image, prepare_gtsrb_csvs
from .features import (
    apply_clahe_rgb,
    crop_roi_image,
    extract_color_features,
    extract_hog_color_features,
    extract_hog_features,
    extract_raw_gray,
    extract_raw_rgb,
    preprocess_image,
)

__all__ = [
    "CLASS_NAMES_EN",
    "CLASS_NAMES_VI",
    "CLASS_SHORT_VI",
    "apply_clahe_rgb",
    "crop_roi_image",
    "extract_color_features",
    "extract_hog_color_features",
    "extract_hog_features",
    "extract_raw_gray",
    "extract_raw_rgb",
    "load_rgb_image",
    "prepare_gtsrb_csvs",
    "preprocess_image",
    "run_experiments",
]


def run_experiments(*args, **kwargs):
    """Lazily run src.pipeline.run_experiments without shadowing python -m src.pipeline."""
    from .pipeline import run_experiments as _run

    return _run(*args, **kwargs)
