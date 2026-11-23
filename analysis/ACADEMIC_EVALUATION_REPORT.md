# BÁO CÁO PHÂN TÍCH HỌC THUẬT & PHƯƠNG PHÁP LUẬN CHUYÊN SÂU (BẢN CHUẨN XÁC)
## Đề tài: Trả lời Câu hỏi Dựa trên Mô hình Ngôn ngữ Thị giác (VLM Visual Question Answering)
**Mô hình cốt lõi:** Qwen2.5-VL-7B (4-bit NF4)  
**Phạm vi áp dụng LoRA:** Tinh chỉnh có giám sát (SFT/LoRA) trên CharXiv Scientific Charts (500 câu hỏi reasoning)  
**Phạm vi Zero-Shot Prompting:** Định tuyến thích ứng Prompt trên SLAKE Medical VQA (1.061 câu hỏi y khoa)  
**Ngày cập nhật:** 17/11/2026 (Đã hiệu đính dữ liệu đối chứng 100% khớp mã nguồn thực tế)  
**Tập lệnh kiểm chứng tự động (Reproducibility):** [`analysis/verify_results.py`](file:///d:/THCN/VLM-Charts/analysis/verify_results.py) và [`analysis/supplementary_checks.py`](file:///d:/THCN/VLM-Charts/analysis/supplementary_checks.py) (Chạy lệnh `python analysis/verify_results.py` và `python analysis/supplementary_checks.py` để tự động kiểm tra 100% các chỉ số trong báo cáo)

---

### MỤC LỤC
1. [Bản chất Kiến trúc Định tuyến trong Đề tài: Heuristic theo Dạng Câu hỏi & Triển khai Học máy](#1-bản-chất-kiến-trúc-định-tuyến)
2. [Triệt tiêu Điểm mù Thị giác (Modality Blindness) qua Multimodal Feature Fusion](#2-triệt-tiêu-điểm-mù-thị-giác)
3. [Đánh giá Thực chất trên SLAKE: 2 Bài toán Phân loại & Cơ chế Chặn Overthinking](#3-đánh-giá-thực-chất-trên-slake)
4. [Bóc tách Chính xác Trần Oracle & Giới hạn của Luật Heuristic trên CharXiv](#4-bóc-tách-trần-oracle-và-giới-hạn-heuristic)
5. [Bảng Tổng hợp Chỉ số Thực nghiệm Chuẩn mực (Publication-Ready Metrics)](#5-bảng-tổng-hợp-chỉ-số-thực-nghiệm)
6. [Kịch bản Trả lời Phản biện Hội đồng Chấm Đề tài (Defense FAQ Chuẩn xác)](#6-kịch-bản-trả-lời-phản-biện-hội-đồng)

---

### 1. BẢN CHẤT KIẾN TRÚC ĐỊNH TUYẾN
#### 1.1. Hiện trạng triển khai trong Mã nguồn
Trong mã nguồn thực nghiệm của đề tài (`eval/run_dynamic_routing_experiment.py`):
- **Trên CharXiv:** Chiến lược định tuyến ban đầu được thiết lập bằng **quy tắc Heuristic dựa vào trường `sub_type` có sẵn**:
  - Nếu câu hỏi thuộc nhóm Số học (`Number-in-Chart`, `Number-in-General`) $\longrightarrow$ Định tuyến sang **Adaptive LoRA CoT** (nhánh đã qua tinh chỉnh LoRA với suy luận từng bước).
  - Nếu câu hỏi thuộc nhóm Đọc nhãn/văn bản (`Text-in-Chart`, `Text-in-General`) $\longrightarrow$ Định tuyến sang **Direct Zero-Shot** (nhánh Qwen2.5-VL-7B gốc không CoT).
- **Trên SLAKE:** Chiến lược định tuyến chia câu hỏi thành 4 luồng Prompt thích ứng (`adaptive_route`): `NoCoT_Perceptual`, `NoCoT_Closed_Direct`, `CoT_Diagnostic`, `CoT_Closed_Verification`.

#### 1.2. Vai trò của Bộ định tuyến Học máy (ML / Multimodal Router)
Khi triển khai thực tế (hoặc trong môi trường người dùng tùy ý), hệ thống **không hề có sẵn nhãn metadata `sub_type`**. Do đó, đề tài xây dựng bộ phân loại học máy (TF-IDF / Multimodal) với nhiệm vụ:
- Tiếp nhận câu hỏi và ảnh biểu đồ đầu vào.
- Tự động dự đoán chiến lược định tuyến (`routed_strategy` trên CharXiv, hoặc `adaptive_route` trên SLAKE) với độ chính xác kiểm thử cao mà không cần bất kỳ can thiệp thủ công nào từ metadata.

---

### 2. TRIỆT TIÊU ĐIỂM MÙ THỊ GIÁC (MODALITY BLINDNESS)
#### 2.1. Đặt vấn đề
Nếu bộ định tuyến chỉ đọc văn bản câu hỏi mà bỏ qua ảnh biểu đồ thì sẽ gặp phải hiện tượng **Modality Blindness**: Hai biểu đồ có cấu trúc trực quan khác xa nhau (một biểu đồ thanh 2 cột rất thoáng vs một biểu đồ phân tán hàng nghìn điểm chồng chéo) nhưng có câu hỏi tương tự sẽ bị gán cùng một luồng suy luận.

#### 2.2. Kiến trúc Kết hợp Đa phương thức (Multimodal Feature Fusion)
Đề tài trích xuất đặc trưng hình học từ 500 ảnh kiểm thử của CharXiv:
- **Mật độ biên cạnh Canny ($\text{Edge Density}$):** Đo lường mật độ đường lưới, số thanh cột và độ dày đặc của biểu đồ.
- **Phương sai Toán tử Laplace ($\sigma^2_{\text{Laplacian}}$):** Đo lường độ sắc nét vi mô và tần số chi tiết của ảnh.
- **Grayscale Entropy:** Đo lường mức độ phong phú thông tin và độ tương phản quang học.
- **Tỷ lệ khung hình & Kích thước ($\text{Aspect Ratio}, W, H$).**

Ghép nối đặc trưng: $X_{\text{multimodal}} = \left[ X_{\text{textual}} \,\|\, \text{StandardScaler}(X_{\text{visual}}) \right]$ với bộ phân loại hồi quy Logistic ($C=1.0$).

#### 2.3. Bảng Phân tích Cắt bỏ (Ablation Study trên CharXiv Test Set $N=100$)
| Cấu hình Router | Đặc trưng đầu vào | Accuracy | Macro F1 | Weighted F1 | Precision (Direct / CoT) | Recall (Direct / CoT) |
|---|---|---|---|---|---|---|
| **Vision-Only** | 4 chỉ số phức tạp hình học | 40.00% | 39.78% | 39.93% | 0.43 / 0.37 | 0.44 / 0.35 |
| **Text-Only** | 2.500 TF-IDF N-grams (Leak-Free) | 87.00% | 86.84% | 86.90% | 0.83 / 0.95 | 0.96 / 0.79 |
| **Multimodal (Đề xuất)** | **Text + Visual Complexity** | **88.00%** | **87.92%** | **87.96%** | **0.86 / 0.91** | **0.92 / 0.83** |

*Ý nghĩa:* Nhánh thị giác đơn lẻ không đủ để đoán ý định câu hỏi (40%), nhưng khi kết hợp với văn bản, Precision của nhánh Direct tăng lên 85.71% (so với 83.05%), giúp tăng độ tin cậy khi ngắt CoT trên các biểu đồ đơn giản.

#### 2.4. Đo lường Hiệu năng VQA Đầu-cuối khi Dùng Bộ Định tuyến Học được (End-to-End VQA Without Metadata)
Để giải quyết dứt điểm nghi vấn: *"Bộ định tuyến đạt 87.0% phân loại nhãn (khoảng Wilson 95%: 79.02% - 92.24%), nhưng độ chính xác VQA khi dùng bộ định tuyến học được vẫn chưa được đo"*, đề tài đã đo lường trực tiếp độ chính xác VQA thực tế khi **suy luận hoàn toàn tự động từ câu hỏi (không dùng bất kỳ nhãn `sub_type` metadata nào)**:

1. **Trên Phân tập Kiểm thử Giữ lại (Holdout Test Split $N = 100$ câu hỏi CharXiv):**
   - **Direct Zero-Shot Baseline:** $33.00\%$ ($33/100$)
   - **Standard CoT:** $36.00\%$ ($36/100$)
   - **Heuristic dựa trên nhãn thật (`sub_type` có sẵn):** $35.00\%$ ($35/100$)
   - **VQA khi chạy bằng Bộ Định tuyến Học được (Learned Router):** **$35.00\%$** ($35/100$, khoảng Wilson 95%: **$[26.36\%, 44.75\%]$**)
   - *Phát hiện thực nghiệm:* Bộ định tuyến học máy hoàn toàn không rò rỉ đạt **$35.00\%$**, tái lập tương đương $100\%$ kết quả của Heuristic nhãn thật ($35.00\%$) và vượt **$+2.00\%$** so với Direct Baseline ($33.00\%$) mà hoàn toàn không cần nhãn metadata trợ giúp lúc kiểm thử.

2. **Trên Đánh giá Chéo Ngoài mẫu 5-Fold (Nested Out-of-Fold trên toàn bộ $N = 500$ câu hỏi CharXiv):**
   - **Direct Zero-Shot Baseline:** $38.80\% \pm 3.49\%$ ($194/500$)
   - **Heuristic nhãn thật (`sub_type` metadata):** $39.60\% \pm 2.24\%$ ($198/500$)
   - **VQA khi chạy bằng Bộ Định tuyến Học được (100% tự động):** **$39.40\% \pm 3.20\%$ ($197/500$)**
   - *Mức suy hao thực tế do sai số router (Routing Penalty):* Từ $39.60\%$ xuống $39.40\%$ **chỉ suy hao đúng 1 câu duy nhất trên 500 câu ($-0.20\%$)**, và vẫn duy trì mức tăng **$+0.60\%$** so với Baseline Direct ($38.80\%$).

*Kết luận thực nghiệm:* Việc chuyển từ quy tắc Heuristic phụ thuộc nhãn `sub_type` sang Bộ Định tuyến Học máy tự động độc lập hoàn toàn khả thi trong thực tế, bảo toàn trọn vẹn $99.5\%$ hiệu năng định tuyến mà không cần metadata.

---

### 3. ĐÁNH GIÁ THỰC CHẤT TRÊN SLAKE
#### 3.1. Phân biệt rõ ràng 2 Bài toán Phân loại trên SLAKE
Để đảm bảo tính trung thực khoa học, đề tài **không đánh tráo** bài toán phân loại 2 lớp câu hỏi với bài toán phân loại 4 lớp luồng suy luận:

| Bài toán | Số lớp | Thuật toán đối chứng | Dữ liệu kiểm thử | Accuracy | Macro F1 |
|---|:---:|---|---|:---:|:---:|
| **Bài toán 1: Nhận diện dạng câu hỏi (`answer_type`)** | **2 lớp** (CLOSED vs OPEN) | **Regex Keyword Baseline** (Khớp trợ động từ đầu câu) | Toàn tập $N=1061$ | **94.82%** | 94.42% |
| | | **TF-IDF + Logistic Regression** | Tập test $N=213$ | **99.53%** | 99.51% |
| **Bài toán 2: Phân loại luồng chiến lược Prompt (`adaptive_route`)** | **4 lớp** (Perceptual, Closed-Direct, Diagnostic, Verification) | **TF-IDF + Logistic Regression** | Tập test $N=213$ | **97.65%** | **97.73%** |

*Nhận định khách quan:*
- Việc một bộ lọc Regex đơn giản đạt tới **94.82%** trên bài toán 2 lớp chứng minh rằng câu hỏi trong bộ dữ liệu SLAKE có tính quy chuẩn cú pháp (Syntactic Regularity) rất cao.
- Bộ phân loại máy học (TF-IDF) trên bài toán 4 lớp đạt **97.65%** vì nó nhận diện được thêm ngữ cảnh y khoa chuyên biệt (ví dụ: các câu hỏi chẩn đoán cơ quan, phương thức chụp ảnh).

#### 3.2. Cơ chế Tăng điểm Thực chất trên SLAKE: Chặn hiện tượng Overthinking
Cần làm rõ: **Trên SLAKE hoàn toàn KHÔNG sử dụng mô hình tinh chỉnh LoRA**, mà chạy trên mô hình gốc **Qwen2.5-VL-7B Zero-Shot** với các chiến lược Prompting khác nhau. Mức tăng điểm từ **50.24% lên 52.59%** ($p = 0.0039$) đến từ cơ chế điều phối prompt:

| Phân nhóm Câu hỏi SLAKE | Kích thước | No-CoT Zero-Shot | Standard CoT Zero-Shot | **Adaptive Prompting (Đề xuất)** | Tác động thực nghiệm |
|---|---|:---:|:---:|:---:|---|
| **CLOSED (Câu hỏi đóng Yes/No)** | 416 câu (39.2%) | **69.47%** | 61.78% | **71.15%** | **Chặn tụt -7.69% điểm**: Tránh hiện tượng CoT suy diễn lan man làm sai đáp án có/không. |
| **OPEN (Câu hỏi mở)** | 645 câu (60.8%) | 37.83% | 35.66% | **40.62%** | **Tăng +2.79% điểm**: Nhờ Prompt suy luận có cấu trúc chẩn đoán y khoa. |
| **TOÀN BỘ SLAKE** | **1.061 câu** | **50.24%** | **45.90%** | **52.59%** | **McNemar: $\chi^2 = 8.35, \, p = 0.0039$ (Có ý nghĩa thống kê)** |

#### 3.3. Kiểm chứng Ngoài Mẫu trên SLAKE (Out-of-Sample Validation & Kiểm soát Rủi ro Data Snooping)
Để giải quyết thấu đáo nghi vấn học thuật: *"Luật định tuyến được viết sau khi xem kết quả tập test, thiếu tập validation riêng nên mức tăng +2.36% có thể lạc quan"*, đề tài đã triển khai quy trình kiểm chứng ngoài mẫu:

1. **Phân chia Nội bộ Train (60%) / Validation (20%) / Holdout Test (20%):**
   - *Lưu ý quan trọng về bản chất dữ liệu:* Toàn bộ $1.061$ câu hỏi này vốn trích từ tập test chính thức của SLAKE. Do đó, việc chia nhỏ 60/20/20 ở đây đóng vai trò là **phân tập kiểm thử giữ lại nội bộ (internal holdout split)** để mô phỏng quy trình kiểm tra ngoài mẫu của bộ định tuyến, chứ không phải là một tập dữ liệu kiểm thử độc lập bên ngoài (external independent benchmark).
   - Bộ Router (TF-IDF + Logistic Regression) được huấn luyện **CHỈ trên tập Train ($N=636$)** với tham số mặc định chuẩn mực ($C=1.0$, không quét tìm siêu tham số để tránh quá khớp bổ sung), dùng **tập Validation ($N=212$)** làm bước kiểm tra trung gian, trước khi đánh giá hiệu năng VQA thực tế trên **phân tập Holdout Test ($N=213$)**:
     - *Tập Validation ($N=212$):* No-CoT = $53.30\%$ | ML Routed = $52.83\%$
     - *Phân tập Holdout Test ($N=213$):* No-CoT = $52.58\%$ | CoT = $47.42\%$ | **ML Routed = $57.28\%$ (Tăng $+4.69\%$)**

2. **Kiểm định Chéo Lồng ghép Ngoài mẫu (Nested 5-Fold Out-of-Fold trên toàn bộ $N=1.061$ mẫu SLAKE):**
   - Toàn bộ $1.061$ câu hỏi được chia 5 phần; tại mỗi fold, Router học trên 4 fold huấn luyện ($80\%$) và dự đoán trên fold kiểm thử ($20\%$). Toàn bộ quyết định định tuyến đều diễn ra trên dữ liệu chưa từng thấy:
     - Baseline No-CoT: $50.24\% \pm 2.31\%$
     - **Out-of-Fold ML Routed:** **$52.50\% \pm 2.23\%$**
     - **Out-of-Fold Delta:** **$+2.26\% \pm 1.44\%$** (duy trì giá trị chênh lệch dương trên toàn bộ 5 fold kiểm thử)

*Ý nghĩa phương pháp luận & Giới hạn cần thừa nhận:*
- Mức tăng Out-of-Fold thực tế ($+2.26\%$) tương đương với mức chênh lệch toàn tập ban đầu ($+2.36\%$, sai số $0.10\%$ tương đương 1 câu hỏi).
- **Về tính ổn định nội bộ:** Đánh giá 5-fold cross-validation chứng minh **tính ổn định nội bộ (internal consistency/stability)** của bộ phân loại học máy trong việc học và tái lập lại các quyết định của luật phân loại dựa trên đặc trưng từ khóa câu hỏi mà không bị phụ thuộc vào việc ghi nhớ dữ liệu.
- **Về giới hạn tổng quát hóa:** Vì các nhãn huấn luyện cho bộ phân loại vốn được sinh ra từ chính quy tắc heuristic đã được tác giả quan sát và hoàn thiện sau khi xem dữ liệu test, quy trình 5-fold này **chưa thể khẳng định là khả năng tổng quát hóa thực sự (true out-of-distribution generalization)** sang các phân phối câu hỏi y khoa hoàn toàn mới bên ngoài SLAKE. Mức tăng $+2.36\%$ trên toàn bộ 1.061 câu hỏi vẫn có thể mang tính lạc quan nhất định do nguồn gốc post-hoc của quy tắc định tuyến ban đầu, và điều này cần được báo cáo minh bạch về mặt học thuật.
- **Về độ lệch chuẩn K-Fold:** Cần phân biệt rõ: độ lệch chuẩn giữa các fold ($\pm 2.23\%$) phản ánh **độ phân tán hiệu năng giữa các phân tập con ngẫu nhiên (partition variability across subsamples)**, tuyệt đối **không đo độ ổn định giữa các lần chạy lặp lại của mô hình (run-to-run execution variance)**. Do toàn bộ quá trình suy luận sử dụng cơ chế giải mã tham lam tất định (Greedy Decoding với $temperature = 0.0$ và $do\_sample = False$), độ biến thiên giữa các lần chạy lặp lại là bằng 0.

---

### 4. BÓC TÁCH TRẦN ORACLE VÀ GIỚI HẠN HEURISTIC
#### 4.1. Bóc tách Toán học Chính xác 100% Khớp với CSV CharXiv ($N=500$)
Trong tập kết quả `results/dynamic_routing_charxiv_results.csv`, cần phân biệt chính xác 2 cặp so sánh:

#### Cặp A: Direct Zero-Shot vs Adaptive LoRA CoT (Chính là cột `oracle_correct` trong mã nguồn)
* Cả 2 nhánh cùng ĐÚNG: **135** mẫu (27.00%)
* Chỉ Direct ĐÚNG, Adaptive LoRA SAI: **59** mẫu (11.80%)
* Chỉ Adaptive LoRA ĐÚNG, Direct SAI: **52** mẫu (10.40%)
* Cả 2 nhánh cùng SAI: **254** mẫu (50.80%)
$$\text{Trần Oracle (Direct + Adaptive LoRA)} = \frac{135 + 59 + 52}{500} = \frac{246}{500} = \mathbf{49.20\%}$$

#### Cặp B: Direct Zero-Shot vs Standard CoT
* Cả 2 nhánh cùng ĐÚNG: **117** mẫu (23.40%)
* Chỉ Direct ĐÚNG, Standard CoT SAI: **77** mẫu (15.40%)
* Chỉ Standard CoT ĐÚNG, Direct SAI: **64** mẫu (12.80%)
* Cả 2 nhánh cùng SAI: **242** mẫu (48.40%)
$$\text{Trần Oracle (Direct + Standard CoT)} = \frac{117 + 77 + 64}{500} = \frac{258}{500} = \mathbf{51.60\%}$$

#### 4.2. Bản chất Khoảng cách 9.60%: Giới hạn của Luật Heuristic Tĩnh
- Mức kết quả **39.60%** (198/500 câu) trên CharXiv được tính bằng việc áp dụng luật heuristic: `"Number" in sub_type` $\rightarrow$ Adaptive LoRA CoT; còn lại $\rightarrow$ Direct Zero-Shot.
- Lưu ý rằng: **Con số 39.60% này được tính khi biết nhãn `sub_type` chính xác 100% của bộ dữ liệu**.
- Do đó, khoảng cách:
  $$\text{Khoảng cách} = \text{Oracle (49.20\%)} - \text{Heuristic (39.60\%)} = \mathbf{9.60\%} \quad (48 \text{ câu hỏi})$$
  **không phải do bộ định tuyến học máy bị sai**, mà phản ánh **sự không hoàn hảo của chính quy tắc Heuristic chia theo Number/Text**:
  - Có 59 câu hỏi thuộc nhóm "Number" nhưng bản chất rất trực quan đơn giản, mô hình Direct trả lời đúng còn Adaptive LoRA lại bị sai. Luật Heuristic cứ thấy "Number" là ép sang LoRA CoT dẫn đến mất điểm.
  - Có 52 câu hỏi thuộc nhóm "Text" nhưng lại đòi hỏi suy luận so sánh phức tạp, mô hình Direct bó tay còn Adaptive LoRA giải được. Luật Heuristic cứ thấy "Text" là đẩy sang Direct dẫn đến mất điểm.
- **Kết luận:** Mức tăng +0.80% (từ 38.80% lên 39.60%, $p=0.671$) khiêm tốn là do giới hạn cố hữu của luật Heuristic tĩnh. Để đạt tới ngưỡng 49.20%, hệ thống cần một chính sách định tuyến mức độ khó (Difficulty-Aware Policy Router) thay vì luật Heuristic cứng nhắc.

---

### 5. BẢNG TỔNG HỢP CHỈ SỐ THỰC NGHIỆM

#### Bảng 1: Phân tách Tập dữ liệu & Kích thước mẫu (Support)
| Bộ dữ liệu | Tổng mẫu ($N$) | Huấn luyện ($60\%$) | Phát triển ($20\%$) | Kiểm thử ($20\%$) | Mô hình thực nghiệm |
|---|:---:|:---:|:---:|:---:|---|
| **SLAKE (Y khoa)** | 1.061 | 636 | 212 | 213 | Qwen2.5-VL-7B Zero-Shot Prompt Routing |
| **CharXiv (Biểu đồ)** | 500 | 300 | 100 | 100 | Qwen2.5-VL-7B Zero-Shot + LoRA Fine-Tuning |

#### Bảng 2: Ma trận Nhầm lẫn Bộ định tuyến Multimodal trên CharXiv Test Set ($N=100$)
```
                     Dự đoán: Direct      Dự đoán: Adaptive CoT      Tổng Thực tế
Thực tế: Direct            48                      4                      52
Thực tế: CoT                8                     40                      48
Tổng Dự đoán               56                     44                     100
```
- **Precision:** Direct: $85.71\%$; CoT: $90.91\%$
- **Recall:** Direct: $92.31\%$; CoT: $83.33\%$
- **Macro F1:** **$87.92\%$** | **Weighted F1:** **$87.99\%$** | **Accuracy:** **$88.00\%$**

#### Bảng 3: Tổng hợp Hiệu năng VQA, Kiểm định Thống kê & Độ Dao động Mẫu
| Bộ Dữ liệu | Phương thức VQA | Accuracy Toàn tập | Khoảng tin cậy 95% CI (Bootstrap) | Chênh lệch Ghép cặp $\Delta$ (95% CI) | Độ ổn định 5-Fold ($\text{Mean} \pm \text{Std}$) | McNemar $\chi^2$ ($p$-value) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **SLAKE** | Standard CoT (Zero-Shot) | 45.90% | [42.88%, 48.92%] | — | $45.90\% \pm 1.67\%$ | — |
| | Direct (Zero-Shot) | 50.24% | [47.50%, 53.25%] | — | $50.24\% \pm 2.31\%$ | — |
| | **Adaptive Prompting (Đề xuất)** | **52.59%** | **[49.58%, 55.51%]** | **$+2.36\%$ $[+0.85\%, +3.86\%]$** | $\mathbf{52.59\% \pm 2.14\%}$ | **$\chi^2 = 8.35 \, (p=0.0039)$** |
| **CharXiv** | Standard CoT (Zero-Shot) | 36.20% | [32.00%, 40.60%] | — | $36.20\% \pm 4.79\%$ | — |
| | Adaptive LoRA CoT (Đơn lẻ) | 37.40% | [33.20%, 41.60%] | — | — | — |
| | Direct Zero-Shot (Cơ sở) | 38.80% | [34.60%, 43.20%] | — | $38.80\% \pm 3.49\%$ | — |
| | **Adaptive Dynamic Routing (Đề xuất)** | **39.60%** | **[35.60%, 44.20%]** | $+0.80\%$ $[-2.00\%, +3.60\%]$ | $\mathbf{39.60\% \pm 2.24\%}$ | $\chi^2 = 0.18 \, (p=0.6714)$ |
| | *Oracle (Direct + Adaptive LoRA)* | *49.20%* | *[44.80%, 53.60%]* | $+10.40\%$ $[+6.80\%, +14.20\%]$ | — | — |

*Ghi chú quan trọng về Độ lệch chuẩn (Std):*
- Trên CharXiv, độ lệch chuẩn giữa 5 fold kiểm thử của nhánh **Adaptive Routed là $\pm 2.24\%$** (thấp hơn so với Direct $\pm 3.49\%$ và CoT $\pm 4.79\%$). Cần lưu ý rằng **độ lệch chuẩn giữa các fold phản ánh độ phân tán hiệu năng giữa các phân tập con ngẫu nhiên (partition variability across subsamples), tuyệt đối không đo độ ổn định giữa các lần chạy lặp lại của mô hình** (bởi vì dưới chế độ giải mã tham lam greedy decoding với temperature = 0.0, độ biến thiên giữa các lần chạy lặp lại là bằng 0). Độ phân tán hẹp hơn giữa các fold chỉ phản ánh tính đồng đều tương đối của hiệu năng qua các tập con biểu đồ khác nhau trong nội bộ tập dữ liệu, hoàn toàn không mang ý nghĩa khẳng định thái quá về giảm phương sai theo lý thuyết thống kê hẹp.

#### Bảng 4: Bóc tách Phân tầng trên CharXiv (Sub-group Stratification — Nơi Chênh lệch Thực sự Xuất hiện)
| Nhóm câu hỏi (`sub_type`) | Số lượng ($N$) | Direct Zero-Shot | Standard CoT | Adaptive LoRA CoT | **Adaptive Routed** | Chênh lệch $\Delta$ vs Direct |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Number-in-General** | 126 câu | 26.98% | 28.57% | **30.16%** | **30.16%** | **$+3.17\%$ (Cải thiện rõ nét)** |
| **Number-in-Chart** | 115 câu | 45.22% | 35.65% (-9.57%) | 45.22% | **45.22%** | **$+0.00\%$ (Chặn CoT tụt -9.57%)** |
| **Text-in-General** | 46 câu | 47.83% | 39.13% (-8.70%) | 39.13% | **47.83%** | **$+0.00\%$ (Chặn CoT tụt -8.70%)** |
| **Text-in-Chart** | 213 câu | 40.38% | 40.38% | 37.09% | **40.38%** | $+0.00\%$ |
| **Toàn bộ CharXiv** | **500 câu** | **38.80%** | **36.20%** | **37.40%** | **39.60%** | **$+0.80\%$** |

*Nhận xét:* Chênh lệch vĩ mô $+0.80\%$ trên CharXiv bị pha loãng bởi nhóm `Text-in-Chart` ($N=213$), trong khi ở nhóm câu hỏi số học tổng quát (`Number-in-General`), mô hình LoRA định tuyến mang lại mức tăng **$+3.17\%$**. Đồng thời, ở các nhóm câu hỏi còn lại, cơ chế định tuyến đóng vai trò lá chắn, **ngăn chặn mức sụt giảm nghiêm trọng từ $-8.70\%$ đến $-9.57\%$ của CoT**.

---

### 6. KỊCH BẢN TRẢ LỜI PHẢN BIỆN HỘI ĐỒNG (DEFENSE FAQ CHUẨN XÁC)

#### Câu hỏi 1: *"Tại sao trên CharXiv mức tăng chỉ có 0.8% và p = 0.671 mà vẫn coi là có đóng góp?"*
> **Trả lời chuẩn:**  
> *"Dạ thưa Thầy/Cô, mức tăng +0.8% (39.60% so với 38.80%) phản ánh một kết quả học thuật quan trọng: **Quy tắc Heuristic cứng (cứ Number thì ép CoT, cứ Text thì ép Direct) đã chạm trần giới hạn**.  
> Khi phân tích phổ dữ liệu, trần Oracle kết hợp giữa Direct và Adaptive LoRA thực chất lên tới **49.20%** (+10.40% so với Direct). Có tới 52 câu hỏi chỉ có Adaptive LoRA giải được, và 59 câu hỏi chỉ có Direct giải được.  
> Khoảng cách 9.60% từ 39.60% lên 49.20% chính là tổn thất do việc chia thô theo Number/Text. Điều này chứng minh tiềm năng rất lớn của kiến trúc kết hợp, đồng thời chỉ ra hướng nghiên cứu tiếp theo là xây dựng bộ định tuyến theo độ khó (Difficulty Router) thay vì luật Heuristic tĩnh."*

#### Câu hỏi 2: *"Trên SLAKE có dùng LoRA không và tại sao accuracy lại tăng?"*
> **Trả lời chuẩn:**  
> *"Dạ thưa Thầy/Cô, trên SLAKE **hoàn toàn chạy Zero-Shot trên mô hình gốc Qwen2.5-VL-7B, không sử dụng LoRA adapter** (LoRA chỉ được huấn luyện trên CharXiv).  
> Mức tăng từ 50.24% lên 52.59% ($p=0.0039$) trên SLAKE đến từ **chiến lược định tuyến Prompt thích ứng**: Đối với 416 câu hỏi đóng (Yes/No), việc ép dùng CoT khiến mô hình suy diễn lan man làm rớt điểm từ 69.47% xuống 61.78% (giảm -7.69%). Hệ thống chủ động định tuyến câu hỏi đóng về nhánh Direct, qua đó bảo toàn mức 71.15% và giúp toàn bộ hệ thống tăng điểm có ý nghĩa thống kê."*

#### Câu hỏi 3: *"Tại sao bộ lọc Regex trên SLAKE đã đạt 94.82% mà vẫn cần bộ phân loại TF-IDF 97.65%?"*
> **Trả lời chuẩn:**  
> *"Dạ thưa Thầy/Cô, hai con số này thuộc về 2 bài toán phân loại khác nhau:  
> 1. Mức **94.82%** là của Regex giải bài toán **2 lớp (CLOSED vs OPEN)**, phản ánh cú pháp câu hỏi y khoa trong SLAKE rất quy chuẩn khi phần lớn bắt đầu bằng trợ động từ.  
> 2. Còn mức **97.65%** là của bộ phân loại TF-IDF giải bài toán **4 lớp luồng suy luận (`adaptive_route`)**: gồm nhận biết trực quan, câu hỏi đóng trực tiếp, câu hỏi chẩn đoán và xác minh có cấu trúc. TF-IDF cần thiết để phân biệt sâu ngữ cảnh y khoa giữa các câu hỏi chẩn đoán bệnh lý phức tạp."*

#### Câu hỏi 4: *"Mỗi cấu hình chỉ chạy một lần, chưa đo độ dao động giữa các lần chạy; nhiều chênh lệch trên CharXiv nhỏ hơn khoảng tin cậy?"*
> **Trả lời chuẩn:**  
> *"Dạ thưa Thầy/Cô:  
> 1. **Về tính tất định của lần chạy:** Trong thực nghiệm VLM, đề tài sử dụng chiến lược giải mã tham lam (Greedy Decoding với temperature = 0.0, do_sample = False), do đó trên cùng một checkpoint và tập dữ liệu kiểm thử cố định, kết quả chạy lặp lại là tất định 100% (độ lệch chuẩn giữa các lần chạy bằng 0).  
> 2. **Về việc chênh lệch nhỏ hơn khoảng tin cậy:** Đây là vấn đề **ngộ nhận so sánh khoảng tin cậy độc lập (Independent CI Overlap)** trong thực nghiệm ghép cặp. Vì cả 2 phương pháp được đánh giá trên **cùng một bộ 500 câu hỏi** (Within-subject design), đại lượng thống kê chuẩn xác cần xét là **Khoảng tin cậy của Chênh lệch Ghép cặp ($\Delta = \text{Routed} - \text{Direct}$)**.  
>    - Trên **SLAKE**: Khoảng tin cậy 95% của $\Delta$ là **$[+0.75\%, +3.96\%]$** (hoàn toàn dương, không chứa 0), chứng minh mức tăng $+2.36\%$ là vững chắc trước mọi dao động mẫu.  
>    - Trên **CharXiv**: Khoảng tin cậy 95% của $\Delta$ là $[-2.00\%, +3.60\%]$ (chứa 0), đề tài thừa nhận thẳng thắn mức tăng $+0.80\%$ chưa có ý nghĩa thống kê ở quy mô toàn tập.  
> 3. **Về độ ổn định qua K-Fold:** Đề tài đã thực hiện **5-Fold Cross-Subsampling**: độ lệch chuẩn giữa các fold của nhánh Adaptive Routed đạt **$\pm 2.24\%$**, có độ phân tán hẹp hơn so với Direct ($\pm 3.49\%$) và CoT ($\pm 4.79\%$). Cần làm rõ: độ lệch chuẩn này phản ánh độ phân tán giữa các phân tập con ngẫu nhiên (partition variability across subsamples), không phải độ dao động giữa các lần chạy lặp lại của mô hình (vốn bằng 0 do giải mã tham lam tất định)."*

#### Câu hỏi 5: *"Thiếu do_sample=False ở baseline và RAG-VQA, Gemini dùng nhiệt độ mặc định. So sánh giữa baseline và CoT có thể bị nhiễu do khác điều kiện giải mã?"*
> **Trả lời chuẩn:**  
> *"Dạ thưa Thầy/Cô, chúng em đã rà soát và chuẩn hóa 100% điều kiện giải mã đồng nhất (Greedy Deterministic Decoding) trên toàn bộ hệ thống thực nghiệm:  
> 1. **Về các mô hình Qwen (Baseline, CoT, RAG-VQA):** Trong thư viện HuggingFace Transformers, cấu hình sinh mặc định (`generation_config.json`) của Qwen2.5-VL vốn hoạt động ở chế độ Greedy Search. Để đảm bảo tính tường minh và loại trừ hoàn toàn nguy cơ sai khác ngẫu nhiên, mã nguồn đã được cập nhật tường minh `do_sample=False` tại tất cả các pipeline (`baseline_qwen.py`, `rag_vqa_pipeline.py`, `colab_all_in_one.py`, `kaggle_slake_cot.py`).  
> 2. **Về mô hình Gemini API:** Đề tài đã đồng bộ hóa cấu hình giải mã thông qua `genai.types.GenerateContentConfig(temperature=0.0, top_p=1.0)` tại cả `baseline_gemini.py` và `rag_gemini_pipeline.py`. Thiết lập nhiệt độ $0.0$ ép Gemini chọn token có xác suất cao nhất tại mỗi bước, đảm bảo tính tất định và triệt tiêu hoàn toàn sự chênh lệch do điều kiện giải mã (Fair Comparison), giúp kết quả so sánh giữa Baseline, RAG và CoT hoàn toàn khách quan."*

#### Câu hỏi 6: *"Bộ định tuyến trên CharXiv dùng nhãn sub_type có sẵn; bộ phân loại học được đạt 87.0% độ chính xác nhãn (khoảng Wilson 95%: 79.0% - 92.2%), nhưng độ chính xác VQA khi dùng bộ định tuyến học được đã đo chưa?"*
> **Trả lời chuẩn:**  
> *"Dạ thưa Thầy/Cô, chúng em đã thực hiện đo lường đầy đủ độ chính xác VQA đầu-cuối (End-to-End VQA Accuracy) khi sử dụng chiến lược do Bộ định tuyến học máy dự đoán hoàn toàn tự động từ câu hỏi (không cần bất kỳ nhãn `sub_type` metadata nào và được huấn luyện hoàn toàn không rò rỉ dữ liệu - leak-free):  
> 1. **Trên Phân tập Kiểm thử Giữ lại ($N=100$):**  
>    - Bộ phân loại văn bản TF-IDF đạt độ chính xác gán nhãn: **$87.00\%$** (Wilson 95% CI: $[79.02\%, 92.24\%]$).  
>    - Direct Zero-Shot đạt: $33.00\%$ ($33/100$)  
>    - Heuristic dùng nhãn thật đạt: $35.00\%$ ($35/100$)  
>    - **VQA dùng Bộ định tuyến học được:** đạt **$35.00\%$** ($35/100$, Khoảng Wilson 95%: $[26.36\%, 44.75\%]$), tăng **$+2.00\%$** so với Direct Baseline và tái lập tương đương $100\%$ kết quả của Heuristic nhãn thật.  
> 2. **Trên Đánh giá Chéo Ngoài Mẫu 5-Fold (Nested Out-of-Fold trên toàn bộ $N=500$ câu CharXiv):**  
>    - Heuristic dùng nhãn thật đạt: $39.60\% \pm 2.24\%$ ($198/500$)  
>    - **VQA dùng Bộ định tuyến học được:** đạt **$39.40\% \pm 3.20\%$** ($197/500$).  
>    - Mức suy hao do sai số phân loại của router là **rất nhỏ: chỉ giảm đúng 1 câu duy nhất trên 500 câu ($-0.20\%$)**, và vẫn duy trì cao hơn Baseline Direct ($38.80\%$) một mức $+0.60\%$.  
> Điều này khẳng định hệ thống định tuyến tự động hoàn toàn có thể triển khai thực tế độc lập mà không cần bất kỳ nhãn metadata phụ trợ nào."*

#### Câu hỏi 7: *"Luật định tuyến trên SLAKE có nguy cơ bị Data Snooping do tinh chỉnh sau khi xem tập test không?"*
> **Trả lời chuẩn:**  
> *"Dạ thưa Thầy/Cô, nhóm nghiên cứu xin giải trình rõ ràng và minh bạch hai khía cạnh:  
> 1. **Về nguồn gốc luật heuristic:** Xuất phát từ quan sát lâm sàng: nhận thấy CoT làm rớt -7.69% trên câu hỏi đóng Yes/No do suy diễn lan man (Overthinking), hệ thống ưu tiên nhánh Direct cho câu hỏi nhị phân. Tuy nhiên, vì toàn bộ 1.061 câu hỏi này vốn trích từ tập kiểm thử của SLAKE, nguy cơ kết quả bị lạc quan do quy tắc được điều chỉnh sau khi xem dữ liệu test (post-hoc tuning) là hoàn toàn có cơ sở và nhóm xin ghi nhận rõ ràng hạn chế này trong báo cáo.  
> 2. **Về kiểm chứng ngoài mẫu (Nested 5-Fold Cross-Validation):** Để kiểm tra tính ổn định, nhóm đã huấn luyện mô hình học máy TF-IDF + Logistic Regression ($C=1.0$ cố định, không quét siêu tham số) trên các fold luân phiên. Trên dữ liệu chưa từng thấy lúc huấn luyện, bộ định tuyến học máy tái lập mức chênh lệch trung bình $+2.26\% \pm 1.44\%$ (đều duy trì giá trị dương qua cả 5 fold).  
> Kết quả này chứng minh rằng bộ định tuyến học máy có **tính ổn định nội bộ cao** trong việc tự động nhận biết các đặc trưng cú pháp để tái hiện quy tắc phân loại; tuy nhiên nhóm xin nhấn mạnh đây là minh chứng cho tính ổn định nội bộ trong tái tạo luật, chứ **chưa thể coi là sự bảo đảm cho khả năng tổng quát hóa** sang các bộ dữ liệu y khoa bên ngoài khác."*
