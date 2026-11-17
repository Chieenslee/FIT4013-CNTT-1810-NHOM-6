# Khai báo sử dụng Công cụ AI & Minh bạch Đóng góp Nhóm 6

Tài liệu này ghi nhận trung thực phạm vi sử dụng công cụ AI hỗ trợ lập trình/biên tập trong môn **FIT4013 – Nhập môn Học máy** (Nhóm 6 – Lớp CNTT 18-10, Trường Đại học Đại Nam), cùng bản kiểm chứng thủ công và ma trận đóng góp của các thành viên.

---

## 1. Công cụ & Phạm vi sử dụng AI

| Thành phần | Mức độ AI hỗ trợ | Kiểm chứng & hiệu đính của nhóm |
|---|---|---|
| Thiết kế cấu trúc module `src/` & `webapp/` | Gợi ý khung tách module Flask (`app_demo.py`, `webapp/`), hàm tiện ích `download_gtsrb_dataset` | Tự đối chiếu API, chạy thực nghiệm trực tiếp trên 51.839 ảnh GTSRB, kiểm tra rò rỉ track và viết bộ test tự động `tests/test_project.py`. |
| Mở rộng thực nghiệm thống kê (`pipeline.py`) | Hỗ trợ viết hàm phụ trợ Bootstrap 95% CI, Wilson 95% CI, Holm–Bonferroni, Logistic Regression | Tự thiết kế câu hỏi phản chứng (F3 không CLAHE, nhân trọng số màu $\alpha$, lưới $k=1..31$, Full-Train refit), chạy trực tiếp và kiểm tra 100% số liệu trong `results/metrics_summary.json`. |
| Trình bày báo cáo IEEE LaTeX & chuẩn hóa bảng | Gợi ý cấu trúc bảng biểu `booktabs`, định dạng `IEEEtran` | Tự rà soát từng con số trong PDF với `metrics_summary.json`, loại bỏ nhận định mâu thuẫn về màu sắc, bổ sung hạn chế thực nghiệm và tự biên dịch/kiểm tra từng trang. |

## 2. Một số câu lệnh / định hướng kiểm chứng đại diện (Representative Prompts)

1. *"Kiểm chứng giả thuyết đặc trưng màu bị pha loãng bởi HOG: thử nhân khối màu sau StandardScaler với hệ số $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}$ và tách cấu hình F3 không CLAHE."*
2. *"Mở rộng lưới siêu tham số K-NN đến $k \in \{1, 3, 5, 7, 11, 15, 21, 31\}$, tính đầy đủ StratifiedGroupKFold 5-fold CV (mean ± std), hiệu chỉnh Holm–Bonferroni cho các phép thử McNemar và đánh giá mô hình Full-Train refit."*
3. *"Tách riêng demo Web Flask theo cấu trúc config/services/routes/templates/static, hỗ trợ cả pipeline tối ưu F3-PCA và F5-PCA, bổ sung cơ chế ngưỡng an toàn $\tau$ và unit test kiểm chứng."*

## 3. Phân công thành viên & Minh bạch lịch sử Git

| Thành viên | Mã SV | Email | Nhiệm vụ chính |
|---|---|---|---|
| **Lê Trọng Chiến** *(Trưởng nhóm)* | `1871020089` | `1871020089@dnu.edu.vn` | Điều phối, thiết kế pipeline chống rò rỉ `StratifiedGroupKFold`, trích xuất đặc trưng HOG/Màu/PCA, tối ưu K-NN, Colab notebook & quản trị repo. |
| **Đặng Việt Anh** | `1871020683` | `1871020683@dnu.edu.vn` | EDA 51.839 ảnh, chuẩn hóa tải dữ liệu GTSRB, tiền xử lý (ROI + Resize 32×32 + CLAHE), Baseline F0–F2 và chuẩn bị 10 ảnh mẫu kiểm thử. |
| **Nguyễn Hữu Dũng** | `1871020172` | `1871020172@dnu.edu.vn` | Huấn luyện & tinh chỉnh Decision Tree, GNB, Random Forest, MLP; Bootstrap 95% CI, McNemar + Holm–Bonferroni, 5-fold GroupKFold & bộ kiểm thử `tests/`. |
| **Kiều Văn Nam** | `1871020416` | `1871020416@dnu.edu.vn` | Thiết kế giao diện Web Flask (`webapp/`), phân tích lỗi (Wilson 95% CI, Hồi quy Logistic, ngưỡng ADAS $\tau$), ma trận nhầm lẫn và hoàn thiện báo cáo IEEE LaTeX. |

**Lưu ý minh bạch lịch sử Git:** Trong giai đoạn tích hợp bài tập ban đầu, toàn bộ mã nguồn hoàn chỉnh của cả 4 thành viên đã được gộp chung vào commit `0224d5d` dưới tài khoản điều phối của trưởng nhóm (`ChieensLee`), sau đó các phần hoàn thiện/kiểm thử bổ sung mới được đẩy tiếp theo từng commit riêng (nhóm không tạo commit lùi ngày hay giả lập lịch sử).
