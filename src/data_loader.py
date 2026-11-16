"""Data loading utilities for official/Kaggle GTSRB CSV format."""

import os
import pandas as pd
import cv2
import numpy as np


def prepare_gtsrb_csvs(data_root: str) -> tuple:
    """Build standard Train.csv and Test.csv if extracted from official sid.erda.dk archives."""
    train_csv_out = os.path.join(data_root, "Train.csv")
    test_csv_out = os.path.join(data_root, "Test.csv")

    if not os.path.exists(train_csv_out):
        train_img_root = os.path.join(data_root, "GTSRB", "Final_Training", "Images")
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
