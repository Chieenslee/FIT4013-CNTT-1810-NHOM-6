# FIT4013 – Nhập môn Học máy | Nhóm 6 – Lớp CNTT 18-10

## Đề tài 6: Nhận dạng biển báo giao thông trên bộ dữ liệu GTSRB (43 lớp)

- **Repository GitHub:** https://github.com/Chieenslee/FIT4013-CNTT-1810-NHOM-6
- **Mở trực tiếp Notebook trên GitHub:** https://github.com/Chieenslee/FIT4013-CNTT-1810-NHOM-6/blob/main/notebooks/FIT4013_Nhom6_DeTai6_GTSRB_Colab.ipynb
- **Mở & chạy ngay trên Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chieenslee/FIT4013-CNTT-1810-NHOM-6/blob/main/notebooks/FIT4013_Nhom6_DeTai6_GTSRB_Colab.ipynb)
- **Giảng viên hướng dẫn:** ThS. Lê Thị Thùy Trang
- **Nhóm thực hiện:** Nhóm 6 – Lớp CNTT 18-10 (Trường Đại học Đại Nam)
- **Thành viên nhóm:**
  - **Lê Trọng Chiến** (`1871020089` – **Trưởng nhóm** | `1871020089@dnu.edu.vn`): Điều phối nhóm, thiết kế pipeline chống rò rỉ `StratifiedGroupKFold`, trích xuất đặc trưng, giảm chiều PCA, tối ưu K-NN, Colab notebook & quản trị repo.
  - **Đặng Việt Anh** (`1871020683` | `1871020683@dnu.edu.vn`): EDA trên 51.839 ảnh, tải dữ liệu ERDA (`download_gtsrb_dataset`), tiền xử lý (ROI + Resize 32×32 + CLAHE), Baseline F0–F2 và bộ 10 ảnh mẫu kiểm thử.
  - **Nguyễn Hữu Dũng** (`1871020172` | `1871020172@dnu.edu.vn`): Decision Tree, GNB, Random Forest (cân bằng trọng số lớp), MLP, Bootstrap 95% CI, kiểm định McNemar (Holm–Bonferroni), 5-fold GroupKFold CV và bộ `tests/`.
  - **Kiều Văn Nam** (`1871020416` | `1871020416@dnu.edu.vn`): Thiết kế giao diện Web Demo mô-đun Python Flask (`webapp/`), phân tích lỗi đa biến (Wilson 95% CI, Logistic Regression, ngưỡng ADAS $\tau$), Ma trận nhầm lẫn 43×43 và báo cáo chuẩn IEEE LaTeX.

> **Minh bạch đóng góp & AI:** Xem chi tiết trong [AI_USAGE.md](AI_USAGE.md) và [LICENSE](LICENSE).

---

## 1. Nguồn dữ liệu & Quy chuẩn chống rò rỉ (GroupKFold theo Track)
- Dữ liệu GTSRB chính thức (39.209 ảnh Train từ 1.307 track + 12.630 ảnh Test độc lập = **51.839 ảnh**, **43 lớp**) do **Institut für Neuroinformatik (INI), Ruhr-Universität Bochum** công bố cho nghiên cứu học thuật (không phải CC0):
  - Kho gốc (ERDA): https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/
  - Kaggle mirror: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- **Phân tách dữ liệu chuẩn:** `StratifiedGroupKFold(n_splits=5)` theo `track_id` (Fold 0: 31.349 ảnh / 1.045 track Train vs 7.860 ảnh / 262 track GroupVal; **0 track giao nhau**, kèm đánh giá trung bình ± SD qua đủ 5 fold). Toàn bộ siêu tham số được khóa trên GroupVal trước khi tái huấn luyện Full-Train (39.209 ảnh) cho ứng dụng Web Flask.

---

## 2. Cấu trúc Repository
```text
├── notebooks/FIT4013_Nhom6_DeTai6_GTSRB_Colab.ipynb   # Notebook Google Colab 8 bước (nbformat 4.5)
├── src/
│   ├── __init__.py                                      # Giao diện gói & run_experiments
│   ├── constants.py                                     # Ánh xạ 43 lớp Việt / Anh
│   ├── data_loader.py                                   # download_gtsrb_dataset (ERDA) & Unicode I/O
│   ├── features.py                                      # crop_center_margin, ROI, CLAHE, HOG+Màu 1860d
│   └── pipeline.py                                      # 5-fold GroupKFold, Alpha, Full-Train, Wilson/Holm/Logistic
├── tests/test_project.py                                # Bộ kiểm thử unittest (Core + Bundle + Flask API)
├── models/gtsrb_pipeline.joblib                         # 5 mô hình Full-Train (F3-PCA & F5-PCA)
├── results/
│   ├── metrics_summary.json                             # Kết quả thực nghiệm & kiểm định chi tiết
│   └── per_class_report_knn_f5.csv                      # Độ đo 43 lớp
├── report/
│   ├── main.tex                                         # Mã nguồn LaTeX IEEE (Times New Roman 100%, số trang & footer GitHub)
│   ├── figures/                                         # 13 biểu đồ phân tích & thiết kế hệ thống
│   └── colab_screenshots/                               # Ảnh chụp thực thi Google Colab
├── sample_images/                                       # 8 mẫu GTSRB Test + 2 ảnh ngoại cảnh
├── webapp/                                              # Ứng dụng Web Flask phân lớp mô-đun
├── app_demo.py                                          # Điểm khởi chạy Web Demo (cổng 7860)
├── requirements.txt                                     # Phiên bản thư viện cố định chính xác
├── LICENSE                                              # MIT (mã nguồn) & Lưu ý trích dẫn INI GTSRB
├── AI_USAGE.md                                          # Minh bạch sử dụng AI & đóng góp thành viên
├── FIT4013_CNTT18_Nhom6_DeTai6_HoanChinh.pdf            # Báo cáo PDF chuẩn bài báo IEEE 2 cột
└── README.md
```

---

## 3. Tóm tắt kết quả trên tập Test (12.630 ảnh, Bootstrap 95% CI)

| Mô hình / Cấu hình | GroupVal Acc | Test Acc (95% CI) | Macro-F1 (95% CI) | Fit (s) | Pred (s) |
|---|---:|---:|---:|---:|---:|
| Baseline K-NN (k=5, Raw Gray F1) | 50.84% | 50.59 ± 0.87% | 43.76 ± 0.98% | 0.58 | 7.24 |
| Baseline Decision Tree (Raw Gray F1) | 59.08% | 56.48 ± 0.84% | 47.07 ± 1.04% | 55.54 | 0.01 |
| Gaussian Naive Bayes (Raw Gray F1) | 14.95% | 14.18 ± 0.58% | 15.23 ± 0.72% | 0.25 | 2.66 |
| K-NN F3_PCA (k=21, manhattan) | 92.90% | 90.59 ± 0.49% | 87.65 ± 0.78% | 0.01 | 11.83 |
| K-NN tối ưu F5 (k=21, euclidean) | 92.49% | 89.88 ± 0.51% | 86.29 ± 0.82% | 0.01 | 1.25 |
| K-NN F5 (k=21, manhattan đối chứng) | 92.38% | 90.29 ± 0.54% | 87.04 ± 0.80% | 0.02 | 11.93 |
| K-NN F4 không PCA (k=21, euclidean) | 92.85% | 90.04 ± 0.51% | 86.77 ± 0.77% | 0.20 | 12.66 |
| Decision Tree F5 (depth=15, entropy) | 70.13% | 69.07 ± 0.78% | 59.45 ± 1.08% | 43.40 | 0.00 |
| Gaussian Naive Bayes (F3_PCA) | 91.35% | 88.37 ± 0.57% | 85.47 ± 0.88% | 0.05 | 0.31 |
| Gaussian Naive Bayes (F5) | 91.03% | 88.20 ± 0.58% | 85.57 ± 0.79% | 0.05 | 0.31 |
| Random Forest (200 trees, F5 default) | 89.80% | 87.73 ± 0.56% | 81.79 ± 0.95% | 22.85 | 0.33 |
| Random Forest (200 trees, F5 balanced) | 89.94% | 87.58 ± 0.58% | 82.55 ± 0.93% | 26.05 | 0.34 |
| MLP Neural Network (F3_PCA, iter=30) | 94.64% | 92.96 ± 0.44% | 90.09 ± 0.73% | 14.55 | 0.03 |
| MLP Neural Network (F5, iter=30) | 94.73% | 92.77 ± 0.46% | 89.96 ± 0.75% | 14.99 | 0.02 |
| K-NN F3_PCA Full Train (k=21) | Full Train | 91.00 ± 0.48% | 88.49 ± 0.76% | 0.02 | 14.78 |
| K-NN F5 Full Train (k=21) | Full Train | 90.58 ± 0.51% | 87.80 ± 0.78% | 0.01 | 1.54 |
| MLP F3_PCA Full Train | Full Train | 93.39 ± 0.44% | 91.01 ± 0.74% | 17.64 | 0.02 |
| MLP F5 Full Train | Full Train | 93.38 ± 0.44% | 90.89 ± 0.67% | 17.98 | 0.02 |
| DT F5 Full Train (depth=15) | Full Train | 69.53 ± 0.74% | 61.47 ± 1.03% | 56.63 | 0.00 |

---

## 4. Hướng dẫn tái lập & kiểm thử nhanh
```bash
pip install -r requirements.txt
python -m src.pipeline --data-root data
python -m unittest discover -s tests -v
python app_demo.py
```
