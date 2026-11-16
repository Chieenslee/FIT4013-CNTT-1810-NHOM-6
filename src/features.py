"""Image preprocessing and classical feature extraction for GTSRB."""

from typing import Tuple
import cv2
import numpy as np
from skimage.feature import hog


def crop_roi_image(
    img_rgb: np.ndarray,
    roi: Tuple[int, int, int, int] = None,
    use_roi: bool = True,
) -> np.ndarray:
    """Crop the traffic sign Region of Interest (ROI) when valid bounds exist."""
    if not use_roi or roi is None:
        return img_rgb
    x1, y1, x2, y2 = roi
    h, w = img_rgb.shape[:2]
    x1 = max(0, min(int(x1), w - 2))
    y1 = max(0, min(int(y1), h - 2))
    x2 = max(x1 + 2, min(int(x2), w))
    y2 = max(y1 + 2, min(int(y2), h))
    return img_rgb[y1:y2, x1:x2]


def apply_clahe_rgb(img_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE on luminance channel in LAB space to reduce lighting variance."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge((l_eq, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_image(
    img_rgb: np.ndarray,
    roi: Tuple[int, int, int, int] = None,
    size: Tuple[int, int] = (32, 32),
    use_roi: bool = True,
    use_clahe: bool = True,
) -> np.ndarray:
    """Crop ROI, resize to uniform dimensions, and optionally balance contrast."""
    cropped = crop_roi_image(img_rgb, roi=roi, use_roi=use_roi)
    resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
    if use_clahe:
        resized = apply_clahe_rgb(resized)
    return resized


def extract_raw_gray(img_resized_rgb: np.ndarray) -> np.ndarray:
    """F1 Baseline: flattened normalized grayscale pixels (1024 dims for 32x32)."""
    gray = cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2GRAY)
    return (gray.astype(np.float32) / 255.0).flatten()


def extract_raw_rgb(img_resized_rgb: np.ndarray) -> np.ndarray:
    """F2 Baseline RGB: flattened normalized RGB pixels (3072 dims for 32x32)."""
    return (img_resized_rgb.astype(np.float32) / 255.0).flatten()


def extract_hog_features(
    img_resized_rgb: np.ndarray, visualize: bool = False
):
    """F3 HOG feature descriptor on grayscale image (1764 dims for 32x32)."""
    gray = cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2GRAY)
    if visualize:
        feat, hog_img = hog(
            gray,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True,
            visualize=True,
            feature_vector=True,
        )
        return feat.astype(np.float32), hog_img
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=False,
        feature_vector=True,
    )
    return feat.astype(np.float32)


def extract_color_features(img_resized_rgb: np.ndarray) -> np.ndarray:
    """Extract compact 96-dim color descriptor: HSV histogram (32) + 4x4 grid RGB (48) + channel stats (16)."""
    hsv = cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2HSV)

    # 32-bin HSV marginal histograms (16 H + 8 S + 8 V)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    h_hist /= (h_hist.sum() + 1e-6)
    s_hist /= (s_hist.sum() + 1e-6)
    v_hist /= (v_hist.sum() + 1e-6)

    # 4x4 spatial color grid in RGB normalized to [0, 1] -> 4 * 4 * 3 = 48 dims
    grid_4x4 = cv2.resize(img_resized_rgb, (4, 4), interpolation=cv2.INTER_AREA)
    grid_feat = (grid_4x4.astype(np.float32) / 255.0).flatten()

    # Global RGB & HSV channel mean/std + dominant color ratios -> 16 dims
    rgb_f = img_resized_rgb.astype(np.float32) / 255.0
    hsv_f = hsv.astype(np.float32)
    hsv_f[:, :, 0] /= 180.0
    hsv_f[:, :, 1:] /= 255.0

    rgb_mean = rgb_f.mean(axis=(0, 1))
    rgb_std = rgb_f.std(axis=(0, 1))
    hsv_mean = hsv_f.mean(axis=(0, 1))
    hsv_std = hsv_f.std(axis=(0, 1))

    # Color sign dominance indicators (red, blue, yellow/orange, white ratios)
    red_mask = ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 70)
    blue_mask = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 70)
    yellow_mask = (hsv[:, :, 0] >= 13) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 80)
    dark_mask = hsv[:, :, 2] <= 60
    ratios = np.array(
        [
            red_mask.mean(),
            blue_mask.mean(),
            yellow_mask.mean(),
            dark_mask.mean(),
        ],
        dtype=np.float32,
    )

    stats = np.concatenate([rgb_mean, rgb_std, hsv_mean, hsv_std, ratios]).astype(
        np.float32
    )
    return np.concatenate([h_hist, s_hist, v_hist, grid_feat, stats]).astype(
        np.float32
    )


def extract_hog_color_features(img_resized_rgb: np.ndarray) -> np.ndarray:
    """F4 Combined feature descriptor: HOG (1764 dims) + Color descriptor (96 dims) = 1860 dims."""
    hog_f = extract_hog_features(img_resized_rgb, visualize=False)
    col_f = extract_color_features(img_resized_rgb)
    return np.concatenate([hog_f, col_f]).astype(np.float32)
