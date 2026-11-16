"""Central configuration, file paths, model specifications, and GTSRB group metadata."""

import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gtsrb_pipeline.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "results", "metrics_summary.json")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample_images")

DEFAULT_TAU = 0.70
MAX_UPLOAD_MB = 10

MODEL_SPECS = [
    {
        "choice": "K-NN (F5 - PCA 128)",
        "bundle_key": "knn_model",
        "metrics_prefix": "K-NN tối ưu F5",
        "score_type": "Trọng số nghịch đảo khoảng cách",
    },
    {
        "choice": "Decision Tree (F5 - PCA 128)",
        "bundle_key": "dt_model",
        "metrics_prefix": "Decision Tree F5",
        "score_type": "Độ tinh khiết nút lá cây quyết định",
    },
    {
        "choice": "MLP Neural Network (Mở rộng)",
        "bundle_key": "mlp_model",
        "metrics_prefix": "MLP Neural Network",
        "score_type": "Xác suất Softmax đầu ra mạng MLP",
    },
]

SAMPLE_CATALOG = [
    {"filename": "test_class_14.png", "true_id": 14, "short_label": "Lớp 14: STOP"},
    {"filename": "test_class_12.png", "true_id": 12, "short_label": "Lớp 12: Đường ưu tiên"},
    {"filename": "test_class_13.png", "true_id": 13, "short_label": "Lớp 13: Nhường đường"},
    {"filename": "test_class_17.png", "true_id": 17, "short_label": "Lớp 17: Cấm ngược chiều"},
    {"filename": "test_class_40.png", "true_id": 40, "short_label": "Lớp 40: Vòng xuyến"},
    {"filename": "test_class_02.png", "true_id": 2, "short_label": "Lớp 2: 50 km/h"},
    {"filename": "test_class_18.png", "true_id": 18, "short_label": "Lớp 18: Nguy hiểm khác"},
    {"filename": "test_class_25.png", "true_id": 25, "short_label": "Lớp 25: Công trường"},
]


def get_sign_category(class_id: int) -> str:
    """Map a GTSRB ClassId (0..42) to its traffic sign functional category."""
    if class_id in {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17}:
        return "Biển cấm & Giới hạn tốc độ"
    if 18 <= class_id <= 31 or class_id == 11:
        return "Biển cảnh báo nguy hiểm"
    if 33 <= class_id <= 40:
        return "Biển hiệu lệnh bắt buộc"
    return "Biển ưu tiên / Chỉ dẫn / Hết lệnh"
