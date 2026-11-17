"""Automated unit and API integration tests for FIT4013 Group 6 GTSRB project."""

import io
import os
import sys
import unittest

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import app_demo  # noqa: E402
from src.constants import CLASS_NAMES_VI  # noqa: E402
from src.data_loader import (  # noqa: E402
    ERDA_BASE_URL,
    GTSRB_ARCHIVES,
    download_gtsrb_dataset,
    prepare_gtsrb_csvs,
)
from src.features import (  # noqa: E402
    crop_center_margin,
    extract_color_features,
    extract_hog_color_features,
    extract_hog_features,
    extract_raw_gray,
    extract_raw_rgb,
    preprocess_image,
)
from src.pipeline import _wilson_ci_pct, apply_holm_bonferroni, extract_track_ids  # noqa: E402
from webapp import create_app  # noqa: E402


class TestGtsrbCore(unittest.TestCase):
    def test_track_disjoint_split_has_zero_overlap(self) -> None:
        rows = []
        for cid in range(3):
            for trk in range(10):
                for frm in range(5):
                    rows.append({"ClassId": cid, "Path": f"Train/{cid}/{trk:05d}_{frm:05d}.png"})
        df = pd.DataFrame(rows)
        tracks = extract_track_ids(df)
        self.assertEqual(len(np.unique(tracks)), 30)
        sgkf = StratifiedGroupKFold(n_splits=5)
        for tr_idx, va_idx in sgkf.split(df, df["ClassId"], groups=tracks):
            shared = set(tracks[tr_idx]).intersection(set(tracks[va_idx]))
            self.assertEqual(len(shared), 0)

    def test_crop_center_margin_and_feature_dimensions(self) -> None:
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = crop_center_margin(dummy, margin_ratio=0.10)
        self.assertEqual(cropped.shape, (80, 80, 3))
        pre = preprocess_image(cropped, roi=None, size=(32, 32), use_roi=False, use_clahe=True)
        self.assertEqual(pre.shape, (32, 32, 3))
        self.assertEqual(extract_raw_gray(pre).shape, (1024,))
        self.assertEqual(extract_raw_rgb(pre).shape, (3072,))
        self.assertEqual(extract_hog_features(pre).shape, (1764,))
        self.assertEqual(extract_color_features(pre).shape, (96,))
        self.assertEqual(extract_hog_color_features(pre).shape, (1860,))

    def test_statistical_helpers_and_data_urls(self) -> None:
        self.assertIn("sid.erda.dk", ERDA_BASE_URL)
        self.assertEqual(len(GTSRB_ARCHIVES), 3)
        self.assertTrue(callable(download_gtsrb_dataset))
        self.assertTrue(callable(prepare_gtsrb_csvs))
        lo, hi = _wilson_ci_pct(10, 100)
        self.assertLess(lo, 10.0)
        self.assertGreater(hi, 10.0)
        adj = apply_holm_bonferroni({"a": {"p_value": 0.01}, "b": {"p_value": 0.04}})
        self.assertAlmostEqual(adj["a"]["p_value_holm"], 0.02, places=6)
        self.assertAlmostEqual(adj["b"]["p_value_holm"], 0.04, places=6)
        self.assertEqual(len(CLASS_NAMES_VI), 43)


class TestTrainedBundleAndFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_path = os.path.join(PROJECT_ROOT, "models", "gtsrb_pipeline.joblib")
        cls.sample_dir = os.path.join(PROJECT_ROOT, "sample_images")
        cls.bundle = joblib.load(cls.model_path)
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_bundle_contains_both_f3pca_and_f5_models(self) -> None:
        expected_keys = {
            "scaler_f3",
            "pca_f3",
            "knn_f3p_model",
            "mlp_f3p_model",
            "scaler",
            "pca",
            "knn_model",
            "dt_model",
            "mlp_model",
        }
        self.assertTrue(expected_keys.issubset(set(self.bundle.keys())))

    def test_external_scene_trim10_fixes_speed50(self) -> None:
        p = os.path.join(self.sample_dir, "external_scene_speed50.png")
        self.assertTrue(os.path.exists(p))
        for m_choice in ["K-NN (F5 - PCA 128)", "MLP Neural Network (Mở rộng)"]:
            r = self.client.post(
                "/api/predict",
                data={
                    "sample_name": "external_scene_speed50.png",
                    "model_choice": m_choice,
                    "auto_trim": "true",
                    "tau": "0.70",
                },
            )
            self.assertEqual(r.status_code, 200)
            data = r.get_json()
            self.assertEqual(data["primary"]["class_id"], 2)
            self.assertEqual(len(data["consensus"]), 5)

    def test_flask_views_and_rest_api_endpoints(self) -> None:
        r_home = self.client.get("/")
        self.assertEqual(r_home.status_code, 200)
        html = r_home.get_data(as_text=True)
        self.assertIn("Dừng lại (STOP)", html)
        self.assertIn("Lê Trọng Chiến", html)

        r_health = self.client.get("/api/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.get_json()["status"], "ok")

        r_samples = self.client.get("/api/samples")
        self.assertEqual(r_samples.status_code, 200)
        self.assertEqual(len(r_samples.get_json()["samples"]), 10)

        r_metrics = self.client.get("/api/metrics")
        self.assertEqual(r_metrics.status_code, 200)
        self.assertGreaterEqual(len(r_metrics.get_json()["exp5_models_comparison"]), 11)

        buf = io.BytesIO()
        Image.open(os.path.join(self.sample_dir, "test_class_40.png")).save(buf, format="PNG")
        buf.seek(0)
        r_up = self.client.post(
            "/api/predict",
            data={
                "image": (buf, "roundabout.png"),
                "model_choice": "MLP F3-PCA 128 (Tốt nhất toàn cục)",
                "auto_trim": "false",
                "tau": "0.70",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(r_up.status_code, 200)
        self.assertEqual(r_up.get_json()["primary"]["class_id"], 40)

        arr14 = np.asarray(Image.open(os.path.join(self.sample_dir, "test_class_14.png")).convert("RGB"))
        pre, hog, md = app_demo.predict_traffic_sign(arr14)
        self.assertEqual(pre.shape, (160, 160, 3))
        self.assertEqual(hog.shape, (160, 160, 3))
        self.assertIn("**Top 1: Lớp 14", md)


if __name__ == "__main__":
    unittest.main()
