"""Data loading and automated ERDA download utilities for GTSRB."""

import os
import urllib.request
import zipfile

import cv2
import numpy as np
import pandas as pd

ERDA_BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
GTSRB_ARCHIVES = [
    ("GTSRB_Final_Training_Images.zip", os.path.join("GTSRB", "Final_Training", "Images")),
    ("GTSRB_Final_Test_Images.zip", os.path.join("GTSRB", "Final_Test", "Images")),
    ("GTSRB_Final_Test_GT.zip", "GT-final_test.csv"),
]


def download_gtsrb_dataset(data_root: str = "data") -> tuple[str, str]:
    """Download and extract official GTSRB archives from sid.erda.dk if missing."""
    os.makedirs(data_root, exist_ok=True)
    for zip_name, marker_rel in GTSRB_ARCHIVES:
        marker_path = os.path.join(data_root, marker_rel)
        if not os.path.exists(marker_path):
            zip_path = os.path.join(data_root, zip_name)
            if not os.path.exists(zip_path):
                url = f"{ERDA_BASE_URL}/{zip_name}"
                print(f"Downloading {url} -> {zip_path} ...")
                urllib.request.urlretrieve(url, zip_path)
            print(f"Extracting {zip_path} -> {data_root} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_root)
    return prepare_gtsrb_csvs(data_root)


def prepare_gtsrb_csvs(data_root: str) -> tuple[str, str]:
    """Build standard Train.csv and Test.csv from extracted sid.erda.dk GTSRB metadata."""
    train_csv_out = os.path.join(data_root, "Train.csv")
    test_csv_out = os.path.join(data_root, "Test.csv")

    if not os.path.exists(train_csv_out):
        train_img_root = os.path.join(data_root, "GTSRB", "Final_Training", "Images")
        if not os.path.exists(train_img_root):
            return download_gtsrb_dataset(data_root)
        rows = []
        for cls_id in range(43):
            cls_folder = f"{cls_id:05d}"
            gt_path = os.path.join(train_img_root, cls_folder, f"GT-{cls_folder}.csv")
            df_c = pd.read_csv(gt_path, sep=";")
            for _, r in df_c.iterrows():
                rows.append(
                    {
                        "Width": int(r["Width"]),
                        "Height": int(r["Height"]),
                        "Roi.X1": int(r["Roi.X1"]),
                        "Roi.Y1": int(r["Roi.Y1"]),
                        "Roi.X2": int(r["Roi.X2"]),
                        "Roi.Y2": int(r["Roi.Y2"]),
                        "ClassId": int(r["ClassId"]),
                        "Path": os.path.join(
                            "GTSRB", "Final_Training", "Images", cls_folder, r["Filename"]
                        ).replace("\\", "/"),
                    }
                )
        pd.DataFrame(rows).to_csv(train_csv_out, index=False)

    if not os.path.exists(test_csv_out):
        gt_test = os.path.join(data_root, "GT-final_test.csv")
        if not os.path.exists(gt_test):
            return download_gtsrb_dataset(data_root)
        df_t = pd.read_csv(gt_test, sep=";")
        rows = []
        for _, r in df_t.iterrows():
            rows.append(
                {
                    "Width": int(r["Width"]),
                    "Height": int(r["Height"]),
                    "Roi.X1": int(r["Roi.X1"]),
                    "Roi.Y1": int(r["Roi.Y1"]),
                    "Roi.X2": int(r["Roi.X2"]),
                    "Roi.Y2": int(r["Roi.Y2"]),
                    "ClassId": int(r["ClassId"]),
                    "Path": os.path.join(
                        "GTSRB", "Final_Test", "Images", r["Filename"]
                    ).replace("\\", "/"),
                }
            )
        pd.DataFrame(rows).to_csv(test_csv_out, index=False)

    return train_csv_out, test_csv_out


def load_rgb_image(full_path: str) -> np.ndarray:
    """Read image in RGB channel order safely across Unicode paths."""
    raw_bytes = np.fromfile(full_path, dtype=np.uint8)
    bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {full_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
