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
DEFAULT_MODEL_CHOICE = "K-NN F3-PCA 128 (Tối ưu K-NN)"

MODEL_SPECS = [
    {
        "choice": "K-NN F3-PCA 128 (Tối ưu K-NN)",
        "short_label": "K-NN F3-PCA",
        "bundle_key": "knn_f3p_model",
        "feature_space": "f3",
        "metrics_prefix": "K-NN F3_PCA (k=",
        "full_prefix": "K-NN F3_PCA Full Train",
        "score_type": "Trọng số nghịch đảo khoảng cách (HOG-CLAHE PCA-128)",
    },
    {
        "choice": "MLP F3-PCA 128 (Tốt nhất toàn cục)",
        "short_label": "MLP F3-PCA",
        "bundle_key": "mlp_f3p_model",
        "feature_space": "f3",
        "metrics_prefix": "MLP Neural Network (F3_PCA",
        "full_prefix": "MLP F3_PCA Full Train",
        "score_type": "Xác suất Softmax đầu ra mạng MLP (HOG-CLAHE PCA-128)",
    },
    {
        "choice": "K-NN (F5 - PCA 128)",
        "short_label": "K-NN F5",
        "bundle_key": "knn_model",
        "feature_space": "f5",
        "metrics_prefix": "K-NN tối ưu F5",
        "full_prefix": "K-NN F5 Full Train",
        "score_type": "Trọng số nghịch đảo khoảng cách (F5 PCA-128)",
    },
    {
        "choice": "Decision Tree (F5 - PCA 128)",
        "short_label": "DT F5",
        "bundle_key": "dt_model",
        "feature_space": "f5",
        "metrics_prefix": "Decision Tree F5 (",
        "full_prefix": "Decision Tree F5 Full Train",
        "score_type": "Độ tinh khiết nút lá cây quyết định (F5 PCA-128)",
    },
    {
        "choice": "MLP Neural Network (Mở rộng)",
        "short_label": "MLP F5",
        "bundle_key": "mlp_model",
        "feature_space": "f5",
        "metrics_prefix": "MLP Neural Network (F5",
        "full_prefix": "MLP F5 Full Train",
        "score_type": "Xác suất Softmax đầu ra mạng MLP (F5 PCA-128)",
    },
]

SAMPLE_CATALOG = [
    {"filename": "test_class_14.png", "true_id": 14, "short_label": "Lớp 14: STOP", "auto_trim": False},
    {"filename": "test_class_12.png", "true_id": 12, "short_label": "Lớp 12: Đường ưu tiên", "auto_trim": False},
    {"filename": "test_class_13.png", "true_id": 13, "short_label": "Lớp 13: Nhường đường", "auto_trim": False},
    {"filename": "test_class_17.png", "true_id": 17, "short_label": "Lớp 17: Cấm ngược chiều", "auto_trim": False},
    {"filename": "test_class_40.png", "true_id": 40, "short_label": "Lớp 40: Vòng xuyến", "auto_trim": False},
    {"filename": "test_class_02.png", "true_id": 2, "short_label": "Lớp 2: 50 km/h", "auto_trim": False},
    {"filename": "test_class_18.png", "true_id": 18, "short_label": "Lớp 18: Nguy hiểm khác", "auto_trim": False},
    {"filename": "test_class_25.png", "true_id": 25, "short_label": "Lớp 25: Công trường", "auto_trim": False},
    {"filename": "external_scene_stop.png", "true_id": 14, "short_label": "Ngoại cảnh: STOP", "auto_trim": True},
    {"filename": "external_scene_speed50.png", "true_id": 2, "short_label": "Ngoại cảnh: 50 km/h", "auto_trim": True},
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
