# 06_feedback — Error Analyzer (MathDial) và Feedback Generator với mastery CCMF

Toàn bộ kết quả trong thư mục này là lần chạy lại ngày 14/09/2026 trên Colab NVIDIA L4 (torch 2.11.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2, accelerate 1.15.0; Llama-3.1-8B-Instruct 4-bit NF4, greedy), với mastery, KC yếu nhất, band và KC kế tiếp lấy từ loop tích hợp chạy bằng CCMF (`../04_integrated_loop/`).

## Cấu trúc

| Thư mục / file | Nội dung |
|---|---|
| `01_inputs/inputs_ccmf.json` | 201 lượt Error Analyzer (200 lượt MathDial qua gate + ca 498/3 dùng làm chẩn đoán cho feedback) và 30 ca feedback (11 correct, 11 incorrect, 8 undetermined); `inputs_ccmf.sha256` |
| `02_generation/results_ccmf.b64` | Kết quả xuất từ Colab (gzip + base64); `results_ccmf.sha256` là sha256 của payload JSON |
| `02_generation/results.json` | Bản giải nén của file trên |
| `02_generation/error_analyzer_predictions_mathdial_ccmf.csv` | Output Error Analyzer từng lượt |
| `02_generation/feedback_generator_cases.csv` | Feedback từng ca, kiểm tra tự động và điểm GPT-4o (chỉ để tham chiếu) |
| `02_generation/automatic_checks.csv` | JSON hợp lệ (EA 99.5%, feedback 86.7%), khớp band 70.0%, 4 fallback, lộ đáp án tự động 15.8% |
| `03_human_rating/rater_A.xlsx`, `rater_B.xlsx` | Điểm của 2 người chấm độc lập, chấm mù |
| `03_human_rating/feedback_human_metrics.csv`, `feedback_human_report.txt` | Điểm trung bình, κw, tương quan và Wilcoxon với GPT-4o |
| `fb_rating.py` | Tạo file chấm (`make`) và phân tích điểm (`analyze`) |
| `HUONG_DAN_CHAM_FEEDBACK.md` | Hướng dẫn chấm gửi người chấm |

## Ánh xạ tới bài

- Mục 4.5: JSON hợp lệ 99.5% trên 200 lượt MathDial → `02_generation/automatic_checks.csv`.
- Mục 4.6 và Hình 2 → `03_human_rating/feedback_human_metrics.csv`, `feedback_human_report.txt`; 86.7% / 70.0% / 15.8% → `02_generation/automatic_checks.csv`.

## Tái lập

1. Colab: `../08_scripts/regenerate_downstream_ccmf_colab.py` với `01_inputs/inputs_ccmf.json` (docstring của script vẫn ghi đường dẫn cũ `regenerated_cases/ccmf_full_rerun/`; không sửa để giữ nguyên hash script đã chạy, sha256 98dc50dc…).
2. Máy local: `python3 ../08_scripts/apply_ccmf_full_rerun.py 02_generation/results_ccmf.b64 $(cat 02_generation/results_ccmf.sha256)`.
3. Phân tích điểm (cần một thư mục bundle tạm trỏ tới dữ liệu run):
```bash
B=/tmp/ccmf_feedback_bundle; mkdir -p $B/mathdial/qlora/seed221
ln -sfn "$(cd ../.. && pwd)/20260912T221028Z/mathdial/qlora/seed221/test" $B/mathdial/qlora/seed221/test
cp 02_generation/feedback_generator_cases.csv $B/
python3 fb_rating.py analyze 03_human_rating/rater_A.xlsx 03_human_rating/rater_B.xlsx --bundle $B --out 03_human_rating
```
