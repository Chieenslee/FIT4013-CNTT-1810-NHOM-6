"""Unicode-safe image reading, ROI margin trimming, visual stage encoding, and diagnostics."""

import base64
import io
from typing import Any

import cv2
import numpy as np
from PIL import Image

from features import extract_color_features, extract_hog_features, preprocess_image


def read_image_unicode(path: str) -> np.ndarray:
    """Read an image into an RGB uint8 numpy array regardless of Unicode characters in path."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def crop_center_margin(img_rgb: np.ndarray, margin_ratio: float = 0.10) -> np.ndarray:
    """Trim outer margin ratio when an uploaded real-world photo includes surrounding background."""
    h, w = img_rgb.shape[:2]
    dx = max(1, int(w * margin_ratio))
    dy = max(1, int(h * margin_ratio))
    if w - 2 * dx >= 16 and h - 2 * dy >= 16:
        return img_rgb[dy : h - dy, dx : w - dx]
    return img_rgb


def encode_png_data_uri(img_rgb: np.ndarray, upscale_size: int = 160) -> str:
    """Convert an RGB uint8 array into an inline PNG base64 URI for clean browser rendering."""
    pil = Image.fromarray(img_rgb.astype(np.uint8))
    if upscale_size > 0:
        pil = pil.resize((upscale_size, upscale_size), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def inspect_image_diagnostics(raw_rgb: np.ndarray, clahe_rgb: np.ndarray) -> dict[str, Any]:
    """Compute image quality indicators linked to quantitative error analysis bins."""
    h, w = raw_rgb.shape[:2]
    gray_raw = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(gray_raw.mean())

    if w < 32:
        res_status = "Độ phân giải thấp (< 32 px — vùng rủi ro lỗi 16,58%)"
        res_level = "warn"
    elif w <= 45:
        res_status = "Độ phân giải trung bình (32–45 px — rủi ro 10,54%)"
        res_level = "mid"
    else:
        res_status = "Độ phân giải rõ nét (>= 46 px — vùng lỗi thấp ~7,4%)"
        res_level = "ok"

    if mean_brightness < 50:
        bright_status = "Ánh sáng rất tối (< 50 — CLAHE đang bù sáng, rủi ro 12,90%)"
        bright_level = "warn"
    elif mean_brightness > 150:
        bright_status = "Chói sáng cao (> 150 — dễ mất chi tiết biên, rủi ro 13,31%)"
        bright_level = "warn"
    else:
        bright_status = "Ánh sáng ổn định (50–150)"
        bright_level = "ok"

    color_feat = extract_color_features(clahe_rgb)
    red_pct, blue_pct, yellow_pct, dark_pct = [round(float(x) * 100.0, 1) for x in color_feat[-4:]]

    return {
        "width": int(w),
        "height": int(h),
        "mean_brightness": round(mean_brightness, 1),
        "resolution_status": res_status,
        "resolution_level": res_level,
        "brightness_status": bright_status,
        "brightness_level": bright_level,
        "color_ratios": {
            "red": red_pct,
            "blue": blue_pct,
            "yellow": yellow_pct,
            "dark": dark_pct,
        },
    }


def build_visual_pipeline(img_rgb: np.ndarray, auto_trim_margin: bool = False) -> dict[str, Any]:
    """Run standardized preprocessing stages and produce visual outputs plus HOG map."""
    trimmed = crop_center_margin(img_rgb, 0.10) if auto_trim_margin else img_rgb
    resized_rgb = preprocess_image(trimmed, roi=None, size=(32, 32), use_roi=False, use_clahe=False)
    clahe_rgb = preprocess_image(trimmed, roi=None, size=(32, 32), use_roi=False, use_clahe=True)

    _, hog_vis = extract_hog_features(clahe_rgb, visualize=True)
    hog_norm = cv2.normalize(hog_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hog_color = cv2.applyColorMap(
        cv2.resize(hog_norm, (160, 160), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_INFERNO,
    )
    hog_color = cv2.cvtColor(hog_color, cv2.COLOR_BGR2RGB)
    clahe_disp = cv2.resize(clahe_rgb, (160, 160), interpolation=cv2.INTER_NEAREST)

    return {
        "raw_rgb": img_rgb,
        "clahe_rgb": clahe_rgb,
        "clahe_disp": clahe_disp,
        "hog_disp": hog_color,
        "stages": {
            "orig_uri": encode_png_data_uri(img_rgb, upscale_size=160),
            "resized_uri": encode_png_data_uri(resized_rgb, upscale_size=160),
            "clahe_uri": encode_png_data_uri(clahe_rgb, upscale_size=160),
            "hog_uri": encode_png_data_uri(hog_color, upscale_size=0),
        },
        "diagnostics": inspect_image_diagnostics(img_rgb, clahe_rgb),
    }
