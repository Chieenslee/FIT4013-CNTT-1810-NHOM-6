# FIT4013 – Nhập môn Học máy | Nhóm 6 – Lớp CNTT 18-10

## Đề tài 6: Nhận dạng biển báo giao thông trên bộ dữ liệu GTSRB (43 lớp)

- **Giảng viên hướng dẫn:** ThS. Lê Thị Thùy Trang
- **Nhóm thực hiện:** Nhóm 6 – Lớp CNTT 18-10 (Trường Đại học Đại Nam)
- **Repository GitHub:** https://github.com/Chieenslee/FIT4013-CNTT-1810-NHOM-6
- **Thành viên nhóm:**
  - **Lê Trọng Chiến** (`1871020089` – **Trưởng nhóm** | `1871020089@dnu.edu.vn`): Điều phối nhóm, thiết kế pipeline chống rò rỉ `StratifiedGroupKFold`, trích xuất đặc trưng, giảm chiều PCA, tối ưu K-NN, Google Colab notebook.
  - **Đặng Việt Anh** (`1871020683` | `1871020683@dnu.edu.vn`): EDA trên 51.839 ảnh, tiền xử lý (ROI + Resize 32×32 + CLAHE), phương án Baseline (F0–F2) và bộ ảnh mẫu kiểm thử đa lớp.
  - **Nguyễn Hữu Dũng** (`1871020172` | `1871020172@dnu.edu.vn`): Huấn luyện Decision Tree, Gaussian Naive Bayes, Random Forest, MLP, Bootstrap 95% CI và kiểm định McNemar.
  - **Kiều Văn Nam** (`1871020416` | `1871020416@dnu.edu.vn`): Thiết kế giao diện Web Demo mô-đun Python Flask (`webapp/`), phân tích lỗi theo kích thước/độ sáng/Coverage–Accuracy, Ma trận nhầm lẫn 43×43 và báo cáo chuẩn IEEE LaTeX.

---

## 1. Nguồn dữ liệu & Quy chuẩn chống rò rỉ (GroupKFold theo Track)
- Dữ liệu GTSRB chính thức (39.209 ảnh Train từ 1.307 track + 12.630 ảnh Test độc lập = **51.839 ảnh**, **43 lớp**):
  - Kho gốc (ERDA TU Braunschweig / RUB): https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/
  - Kaggle mirror (CC0 / Public Benchmark): https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- **Phân tách dữ liệu chuẩn:** `StratifiedGroupKFold(n_splits=5)` theo `track_id` (31.349 ảnh / 1.045 track huấn luyện; 7.860 ảnh / 262 track kiểm định; **0 track giao nhau**). Tập Test chỉ đánh giá một lần cuối cùng.

---

## 2. Cấu trúc Repository
```text
├── notebooks/FIT4013_Nhom6_DeTai6_GTSRB_Colab.ipynb   # Notebook Google Colab tái lập 8 bước
├── src/
│   ├── __init__.py                                      # Giao diện gói & run_experiments
│   ├── constants.py                                     # Ánh xạ 43 lớp Việt / Anh
│   ├── data_loader.py                                   # Nạp dữ liệu GTSRB & hỗ trợ Unicode
│   ├── features.py                                      # ROI, CLAHE, Pixel, HOG + Màu sắc 1860d
│   └── pipeline.py                                      # GroupKFold, Bootstrap CI, McNemar & Vẽ 11 hình
├── models/gtsrb_pipeline.joblib                         # Pipeline đã huấn luyện (Scaler+PCA+KNN/DT/MLP)
├── results/
│   ├── metrics_summary.json                             # Toàn bộ kết quả thực nghiệm thật
│   └── per_class_report_knn_f5.csv                      # Độ đo từng lớp (43 lớp)
├── report/
│   ├── main.tex                                         # Mã nguồn LaTeX IEEE (Times New Roman 100%)
│   ├── figures/                                         # Hệ thống sơ đồ thiết kế & biểu đồ 200+ DPI
│   └── colab_screenshots/                               # 5 ảnh chụp thực thi Google Colab
├── sample_images/                                       # 8 ảnh mẫu đa lớp từ tập Test
├── webapp/                                              # Ứng dụng Web Flask phân lớp mô-đun
│   ├── __init__.py                                      # Application Factory (create_app)
│   ├── config.py                                        # Cấu hình đường dẫn, mẫu, ngưỡng ADAS
│   ├── services/                                        # Tầng dịch vụ (model_registry, image_processor, predictor)
│   ├── routes/                                          # Tầng điều tuyến (views.py & REST API api.py)
│   ├── templates/                                       # Giao diện HTML Jinja2 (base.html, index.html)
│   └── static/                                          # Tài nguyên CSS/JS tách bạch (main.css, app.js)
├── app_demo.py                                          # Điểm khởi chạy gọn nhẹ của Web Demo (port 7860)
├── requirements.txt                                     # Phiên bản thư viện tối thiểu
├── FIT4013_CNTT18_Nhom6_DeTai6_HoanChinh.pdf            # Báo cáo PDF chuẩn bài báo IEEE 2 cột
└── README.md
```

---

## 3. Tóm tắt kết quả trên tập Test khóa (12.630 ảnh, Bootstrap 95% CI)

| Mô hình / Cấu hình | GroupVal Acc | Test Acc (95% CI) | Macro-F1 (95% CI) | Fit (s) | Pred (s) |
|---|---:|---:|---:|---:|---:|
| Baseline K-NN (k=5, Raw Gray F1) | 50.84% | 50.59 ± 0.87% | 43.76 ± 0.98% | 0.07 | 9.75 |
| Baseline Decision Tree (Raw Gray F1) | 59.08% | 56.48 ± 0.84% | 47.07 ± 1.04% | 66.58 | 0.02 |
| Gaussian Naive Bayes (Raw Gray F1) | 14.95% | 14.18 ± 0.58% | 15.23 ± 0.72% | 0.23 | 2.85 |
| K-NN F3_PCA (11, euclidean) | 92.79% | 90.45 ± 0.49% | 87.48 ± 0.80% | 0.01 | 1.66 |
| K-NN tối ưu F5 (k=11, euclidean) | 92.24% | 89.79 ± 0.53% | 86.50 ± 0.85% | 0.02 | 1.72 |
| K-NN F5 (11, manhattan đối chứng) | 92.19% | 89.95 ± 0.54% | 86.97 ± 0.87% | 0.01 | 12.01 |
| K-NN F4 không PCA (k=11, euclidean) | 92.58% | 89.81 ± 0.53% | 86.60 ± 0.77% | 0.09 | 18.06 |
| Decision Tree F5 (depth=15, entropy) | 70.13% | 69.07 ± 0.78% | 59.45 ± 1.08% | 51.08 | 0.01 |
| Gaussian Naive Bayes (F5) | 91.03% | 88.20 ± 0.58% | 85.57 ± 0.79% | 0.04 | 0.35 |
| Random Forest (200 trees, F5) | 89.80% | 87.73 ± 0.56% | 81.79 ± 0.95% | 26.94 | 0.30 |
| MLP Neural Network (F5) | 94.73% | 92.77 ± 0.46% | 89.96 ± 0.75% | 17.16 | 0.02 |

---

## 4. Hướng dẫn tái lập nhanh
```bash
pip install -r requirements.txt
python -m src.pipeline --data-root data
python app_demo.py
```
