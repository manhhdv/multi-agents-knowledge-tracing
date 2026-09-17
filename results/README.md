# Kết quả bài báo SoICT 2026 — bộ dữ liệu thống nhất

Mọi bảng và con số trong `sn-article.tex` được lấy từ các file trong thư mục này. Ngày 14/09/2026 thư mục đã được dọn và `_archive/` đã bị xoá. Hai input mà script tái lập vẫn cần nay nằm trong gói: `03_ccmf/table5_reference_inputs/cell26.txt` (mã CCMF gốc cho `08_scripts/ccmf_redesign.py`) và `04_integrated_loop/sava_verdicts_test.csv` (cột verdict SAVA cho `08_scripts/rerun_integrated_loop_ccmf.py`). Mọi thay đổi đường dẫn và sha256 ghi trong `03_ccmf/PATH_CHANGE_20260914.md`. Dữ liệu thô của lần chạy (logit gap, KC, file `qual_*`, `command.log`) nằm ở `../raw_runs/` (thư mục chạy gốc `20260912T221028Z`). Số bảng dưới đây theo bản thảo hiện tại: Bảng 1 backbone, Bảng 2 SAVA, Bảng 3 CCMF, Bảng 4–5 Error Analyzer, Hình 1 kiến trúc; bài không còn bảng nào khác.

Quy ước chung: CoMTA chỉ dùng để đánh giá. Ngưỡng và toàn bộ tham số Platt/CCMF được fit trên 1.755 lượt validation MathDial rồi áp nguyên cho CoMTA.

## Bảng/số trong bài → file

| Trong bài | File |
|---|---|
| Bảng 1, khối dưới (rerun zero-shot, QLoRA) | `01_backbone/metrics_summary.csv` (cột `official_*`) |
| Mục 3.1: tỉ lệ lượt đa-KC 58.75% (CoMTA, 366/623 lượt có nhãn) và 82.94% (MathDial, 10.948/13.200 lượt có nhãn) | `00_run_info/split_counts.csv` (dòng `split=all`, `skip_first_labelled_turn=False`); script `08_scripts/split_counts.py` |
| Mục 4.1: 8.425 lượt train; 1.755 lượt val và 1.985 lượt test sau khi bỏ lượt có nhãn đầu tiên của mỗi hội thoại (log huấn luyện in 2.202 lượt val vì không bỏ lượt này) | `00_run_info/split_counts.csv`; khớp `00_run_info/manifest.json` (`split_sizes`) và `command.log` của run |
| Mục 4.2: CI 95% của mức tăng AUC QLoRA so với zero-shot [+0.189, +0.271] | `paired_bootstrap` trong `03_ccmf/table5_reference_inputs/cell26.txt` (bootstrap theo hội thoại, 1.000 lần, seed 221) trên `raw_logit_gaps.json` của hai backbone |
| Bảng 1, khối trên | Số trích từ Table 2 của bài LLMKT (LAK'25), không phải dữ liệu của nhóm |
| Mục 4.2 (Acc/AUC/F1 và CI của backbone; file còn Loss, Prec., Rec. không báo trong bài) | `01_backbone/metrics_summary.csv`; CI ở `../supplementary/revision/cpu/out_03_table1_cis.csv` |
| F1 với ngưỡng chọn trên MathDial val (không còn báo trong bài) | `01_backbone/threshold_calibrated_metrics.csv` |
| Kiểm tra tái tạo evaluation gốc | `01_backbone/golden_test.csv` |
| Bảng 2 (SAVA trên benchmark dựng sẵn, gồm ablation không có conclusion cue) | `02_sava/constructed_benchmark_summary.csv`, `constructed_benchmark_per_transformation.csv`, từng item ở `constructed_benchmark_items.csv` |
| Mục 4.3: McNemar SAVA vs regex ($p=1.7\times10^{-136}$); file còn CI bootstrap của conclusion cue (+3.8 điểm), không báo trong bài | `02_sava/tests.json` |
| Mục 4.3: cue path quyết định 6.5% verdict, prose guard 20.7% (bảng verdict theo stage không còn trong bài) | `02_sava/verdicts_by_stage.csv` |
| Độ phủ 78.4%, khớp nhãn 74.4% / 84.6% (lượt cuối), regex 97.7% / 69.2% | `02_sava/real_turn_agreement.csv`; từng lượt ở `real_turns_mathdial_test.csv` (và `_val.csv`) |
| Baseline LLM judge cho SAVA (75.4% nhị phân, 73.0% ba lựa chọn, CI chênh lệch so với SAVA và regex, gate 61.4% / 27.2%) | `02_sava/llm_judge_agreement.csv`, `llm_judge_tests.json`; verdict từng lượt ở `llm_judge_verdicts_test.csv`; code Colab ở `08_scripts/llm_judge_colab_cells.py`, phân tích ở `08_scripts/analyze_llm_judge_baseline.py` |
| Gate Error Analyzer 60.6% lượt, 29.0% có nhãn đúng | `04_integrated_loop/gate_summary.csv`; từng lượt ở `integrated_run_mathdial_qlora_seed221.csv` |
| Bảng 3 (CCMF: phương pháp chính, cập nhật bằng SAVA/nhãn, ablation) và mục 4.4 | `03_ccmf/test_results.csv` (AUC, CI, log-loss `ll_test`, AUC theo số KC), `test_contrasts.csv` (tương phản với không cập nhật), `posthoc_label_hist_tie_aware.csv` (dòng "Last SAVA verdict per KC" và tương phản +0.012 [−0.000, +0.024]), `gamma_profile_qlora_seed221.csv` (γ), `params_<tag>.json` (λ = 0.90 …) |
| Quy trình đăng ký trước của CCMF | `03_ccmf/PREREGISTRATION.md`, `preregistration.json`, `manifest.json`, `independent_verification.json`; script `08_scripts/ccmf_redesign.py`, `08_scripts/verify_ccmf_redesign.py` |
| Loop tích hợp (MathDial) chạy lại bằng CCMF; gate 60.6% / 29.0% không đổi | `04_integrated_loop/integrated_run_mathdial_qlora_seed221.csv`, `gate_summary.csv`; script `08_scripts/rerun_integrated_loop_ccmf.py` (kiểm tra khớp `mean_bkt_sava`); verdict SAVA đầu vào ở `04_integrated_loop/sava_verdicts_test.csv` |
| Chạy lại Error Analyzer (200 lượt MathDial) và Feedback 30 ca với mastery CCMF trên Colab L4, 14/09/2026; chấm lại bởi 2 người | `06_feedback/` (cấu trúc và lệnh tái lập trong `06_feedback/README.md`) |
| Bảng 4 (Error Analyzer vs nhãn người) | `05_error_analyzer/error_analyzer_metrics.csv`, `agreement_report.txt`, nhãn ở `error_analyzer_gold.csv`, `annotator_A/B.xlsx`, `adjudication.xlsx` |
| JSON hợp lệ 91.3% (CoMTA, `05_error_analyzer/error_analyzer_predictions.csv`) / 99.5% (199/200 lượt MathDial qua gate, cột `valid_json` với `set=mathdial_sava_gated` trong `06_feedback/02_generation/error_analyzer_predictions_mathdial_ccmf.csv`; `automatic_checks.csv` ghi 0.99 vì bị `../supplementary/revision/cpu/05_regen_automatic_checks.py` ghi đè từ `05_error_analyzer/error_analyzer_predictions.csv` — lần chạy trước CCMF, 198/200; `08_scripts/apply_ccmf_full_rerun.py` gốc ghi 0.995) |
| Bảng 5 (ma trận nhầm lẫn Error Analyzer) | `../supplementary/revision/cpu/out_04_ea_confusion.csv`, `.tex` |
| Mục 4.6 (Feedback: người chấm, κw 0.85/0.67/0.61, GPT-4o, ρ, Wilcoxon) | `06_feedback/03_human_rating/feedback_human_metrics.csv`, `feedback_human_report.txt`, điểm ở `rater_A.xlsx`, `rater_B.xlsx` |
| JSON hợp lệ 86.7%, chiến lược khớp band 70.0%, 4 fallback, lộ đáp án tự động 15.8% | `06_feedback/02_generation/feedback_generator_cases.csv`, `automatic_checks.csv` |
| Mục 4.7: mô phỏng, ablation, grid 144 cấu hình của planner (không có bảng) | `07_planner/planner_simulation.csv`, `planner_ablation.csv`, `planner_sensitivity_grid.csv`, `planner_grid_wins.csv` |
| Môi trường, phiên bản thư viện, commit LLMKT | `00_run_info/` |

## Nguồn gốc dữ liệu

- **SAVA** là phiên bản trong `colab_full_framework_replication.ipynb` (cell 22), có quy tắc conclusion cue. Cột `sava_no_conclusion_cue` chỉ là ablation.
- **CCMF trong bài** là biến thể `mean_bkt` của `03_ccmf/`. Loop tích hợp (`04_integrated_loop`) đã chạy lại bằng CCMF.
- **Feedback (30 ca: 11 correct, 11 incorrect, 8 undetermined)** và **Error Analyzer MathDial (200 lượt)**: sinh lại toàn bộ bằng mastery CCMF ngày 14/09/2026, chấm lại bởi 2 người; chi tiết ở `06_feedback/README.md`. Các lần sinh lại từng phần trước đó (2 ca, 15 ca, kiểm tra L4 so với A100) đã bị thay thế và được dọn.
- **Error Analyzer (CoMTA, Bảng 4):** mastery trong prompt lấy trực tiếp từ logit, không phụ thuộc CCMF, nên không chạy lại.

## Ghi chú về `03_ccmf/`

Ngày 14/09/2026 thư mục `03_ccmf/redesign/` được làm phẳng thành `03_ccmf/` (nội dung file không đổi). Đường dẫn trong `08_scripts/ccmf_redesign.py`, `08_scripts/verify_ccmf_redesign.py`, `03_ccmf/manifest.json` và `spec.json` đã được sửa theo; sha256 trước và sau, cùng diff của hai script, ghi trong `03_ccmf/PATH_CHANGE_20260914.md`. `PREREGISTRATION.md`, văn bản đăng ký trước nhúng trong script (sha256 e17bd98c…) và các bản ghi chỉ-ghi-thêm `TEST_EVALUATED`, `test_evaluations.log` giữ nguyên, nên vẫn nhắc thư mục gốc và sha256 script lúc đánh giá (f8284c19…). Các input tham chiếu của Bảng 5 cũ (tham số, dự đoán và bảng ablation của head tích) mà hai script dùng cho các bước kiểm tra hồi quy được giữ nguyên vẹn trong `03_ccmf/table5_reference_inputs/`; chúng không phải kết quả báo cáo trong bài. `ccmf_redesign.py evaluate` đã chạy đúng một lần và không được chạy lại.

## Scripts (`08_scripts/`)

- `build_results_package.py`: dựng thư mục này. Script lịch sử; các nguồn nó đọc (`results_sava_conclusion/`, file top-level của run, `feedback_rating/`) đã được dọn nên không chạy lại được, kết quả của nó đã nằm trong thư mục này.
- `ccmf_refit_sava_update.py`: fit lại CCMF SAVA-update.
- `regenerate_downstream_ccmf_colab.py` (Colab) và `apply_ccmf_full_rerun.py` (local): sinh lại Error Analyzer + Feedback với mastery CCMF, kiểm tra hash, tạo file chấm. `make_rater_files_regenerated_cases.py`, `apply_regenerated_15_cases.py` là script lịch sử của các lần sinh lại từng phần đã bị thay thế.
- `06_feedback/fb_rating.py`: tạo file chấm và phân tích điểm người chấm (`analyze`).
- `05_error_analyzer/ea_annotation.py`: quy trình gán nhãn Error Analyzer.
