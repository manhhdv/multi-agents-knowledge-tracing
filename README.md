# MathKT-Agent — gói nộp SoICT 2026

**Cập nhật:** 15/09/2026. **Hạn nộp full paper:** 20/09/2026.

---

## 1. Trạng thái bản nộp

**File nộp:** `MathKT-Agent_SoICT2026/paper/sn-article.pdf`, compile từ đúng mã nguồn hiện tại.

| Yêu cầu SoICT 2026 | Bản hiện tại |
|---|---|
| Tối đa 12 trang, không tính tài liệu tham khảo | 12 trang nội dung (Mục 1–6), tài liệu tham khảo ở trang 13–15, tổng 15 trang |
| Định dạng Springer CCIS (`llncs`) | `\documentclass[a4paper]{llncs}` |
| Single-blind: có tên và đơn vị tác giả | có |
| PDF không đánh số trang | `llncs` không in số trang |
| Không lỗi LaTeX | 0 tham chiếu undefined, 0 overfull box |

Build lại:

```bash
cd MathKT-Agent_SoICT2026/paper
pdflatex sn-article && bibtex sn-article && pdflatex sn-article && pdflatex sn-article
```

`llncs.cls` có sẵn trong TeX Live. Hình vẽ lại bằng `pdflatex model_flow.tex` trong `paper/figures/`.
Sau khi build, xoá các file trung gian (`.aux .bbl .blg .log .fls .fdb_latexmk`).

`paper/` gồm:

| File | Vai trò |
|---|---|
| `sn-article.tex`, `sn-article.pdf` | bản thảo và PDF nộp |
| `ref.bib` | tài liệu tham khảo |
| `splncs04-citeorder.bst` | `splncs04.bst` của Springer, tắt `ITERATE {presort}` và `SORT` để đánh số theo thứ tự trích dẫn; **phải gửi kèm khi nộp mã nguồn camera-ready** |
| `figures/model_flow.tex`, `.pdf` | Hình 1 (TikZ standalone) |

---

## 2. Vòng hoàn thiện 15/09/2026

| Mục | Thay đổi | Lý do |
|---|---|---|
| Trích dẫn | Đánh số theo thứ tự xuất hiện; `natbib` `sort&compress` | Style cũ sắp alphabet nên trích dẫn đầu tiên là [4] và có `[26, 29, 23]`. Springer yêu cầu trích dẫn nhiều mục theo thứ tự số, dạng [4–6, 9]; nay là `[24–26]` |
| Trích dẫn | Sửa 3 chỗ dùng số làm danh từ: "values from LLMKT [5]", "as released in MathFish [31]", "following McNichols et al. [16]" | Văn phong trích dẫn số |
| `ref.bib` | Yosef et al.: sửa tác giả cuối Ben-Ari → **Kviatkovsky** | Sai so với arXiv 2604.22597 |
| `ref.bib` | Chu et al. → Findings EMNLP 2025 (sửa 6 tên riêng); Huang et al. 2026 → BEA 2026; RPKT → IEEE FMLDS 2025; MathFish → Findings EMNLP 2024 | Đã xuất bản chính thức, bib còn trích arXiv |
| `ref.bib` | Thêm DOI cho 14 mục; Corbett & Anderson → 1995 theo DOI; ghi chú "To appear in EMNLP 2026"; thống nhất tên publisher | Đối chiếu Crossref, ACL Anthology, arXiv |
| Mục 4.5 | Tỉ lệ JSON hợp lệ trên MathDial **99.0% → 99.5%** (199/200) | Xem §3 |
| Hình 1 | Vẽ lại (xem dưới) | Hình cũ thiếu luồng dữ liệu và bị thu nhỏ |
| Caption hình/bảng | Cùng khuôn "tiêu đề ngắn → chi tiết → ghi chú ký hiệu": Hình 1 *Architecture of MathKT-Agent*; Bảng 1 *Knowledge tracing results…*; Bảng 2 *Answer verification on the constructed benchmark*; Bảng 3 *Mastery aggregation on real MathDial logits*; Bảng 4 *Error classification against adjudicated human labels*; Bảng 5 *Confusion matrix of the Error Analyzer*. Tên trong bảng khớp Setup (*Exact string match*, *without conclusion cue*, *(a) Fine-tuned backbone (4-bit QLoRA)* / *(b) Zero-shot backbone*). Nhãn thống nhất: `fig:architecture`, `tab:kt`, `tab:sava`, `tab:ccmf`, `tab:ea`, `tab:ea-confusion` | Caption Bảng 2–3 bắt đầu bằng tên phương pháp thay vì nói bảng đo gì; nhãn `tab2`, `tab:equiv`, `tab:ea-cm` không theo quy tắc |
| Bảng 3 | Tách thành hai khối **(a) Fine-tuned backbone (4-bit QLoRA)** và **(b) Zero-shot backbone** trong cùng một bảng; bỏ `\resizebox`, chữ `\scriptsize` (7 pt, trước khoảng 6 pt); hàng ablation ngăn bằng đường kẻ mảnh; văn bản trỏ tới Table 3a/3b | Bảng 7 cột bị thu nhỏ khó đọc |
| Bù trang cho Bảng 3 | Hình 1 thấp hơn 8 mm; caption Hình 1, Bảng 1, Bảng 3 viết gọn (giữ n, Δ, †, ‡); cắt 9 câu/cụm lặp ý hoặc chi tiết phụ (xem dưới) | Tách bảng làm nội dung tràn sang trang 13 và tổng 16 trang |
| README | `MathKT-Agent_SoICT2026/README.md`, `results/README.md` cập nhật theo bản thảo | Còn số cũ (84.6/83.0/−1.5, McNemar, 72.4%, 1e-4) và số bảng/hình cũ |
| Dọn dẹp | `latex_notes/`, file build trung gian, `splncs04.bst` chuyển vào Thùng rác; hai `MANIFEST.sha256` sinh lại | |

**Hình 1 — chỉnh sửa.**

- Kích thước: vẽ đúng bề rộng 122 mm và chèn với `\textwidth` (trước là `0.8\textwidth`, chữ bị thu còn khoảng 6 pt). Khoảng cách dọc giữa các hộp 7 mm để bù trang cho Bảng 3.
- Tách hai khung. Khung **Offline (uses correctness labels)** gồm huấn luyện estimator và fit head CCMF trên validation. Khung **Inference (no correctness labels)** là các bước còn lại. Cách chia này khớp đúng câu trong Limitations: nhãn chỉ được bỏ ở inference.
- Bổ sung luồng dữ liệu còn thiếu:
  - CCMF ↔ learner memory (prior $m_k$ / update $m$)
  - memory → Feedback/MTA (mastery)
  - Error Analyzer → Feedback (diagnosis)
  - dialogue → SAVA (student answer, reference)
- Bỏ mũi tên sai nghĩa "learner profile → new dialogue" và mũi tên "then" CCMF → SAVA.
- Learner memory (đóng góp 1) vẽ theo style "added in this paper".

**Các câu, cụm đã cắt để bù trang cho Bảng 3 (đã duyệt):**

1. Caption Bảng 5: "Calculation and careless are never predicted." (Mục 4.5 đã nói)
2. Mục 4.5: "Offering an abstention class is not enough to obtain one."
3. Mục 4.2: "(around 63.71 from exported predictions; 63.65 in LLMKT's code)"
4. Conclusion: "The results are mixed."
5. Mục 4.6: "This round shows only the feedback text (no scaffolding question, strategy or hint), so scores are not comparable across rounds." (Limitations đã nói)
6. Mục 4.6: "Correctness is lower for incorrect than correct turns (3.09 vs. 4.36), and the four rule-based fallbacks receive higher Correctness (5.00 vs. 3.38) but lower Relevance (3.25 vs. 4.10) and Clarity (3.50 vs. 4.46) than LLM outputs."
7. Limitations: "With 104 SAVA-judged turns, human verification cannot exclude the judge's advantage measured against GPT-4o labels." (Mục 4.3 đã nói khoảng cách 3.9 điểm "is not confirmed")
8. Mục 4.3: "(78.7% after reweighting without the 4 unclear turns, against 78.4% overall)"
9. Mục 4.4: ", in line with LLMKT's observation that averaging outperforms a product [5]"

**Các sửa đổi nội dung của vòng trước vẫn còn trong bản thảo:**

1. Abstract: SAVA "on par with the regex and an 8B LLM judge" trên các lượt nó chấm.
2. Mục 4.4: "No AUC improvement over mean pooling is therefore claimed."
3. Conclusion: luật tiên quyết của MTA chỉ tác động trên 3.9% lượt logged.
4. Mục 4.3: 84.55% / 83.02% / −1.53 (tránh trùng số với 84.6% đồng thuận GPT-4o trên lượt cuối).
5. Mục 4.6: Wilcoxon "over the 13 untied of 30 pairs".
6. Mục 4.5: rerun có lớp `unclear` chỉ báo định tính; exact match nằm trong khoảng tin cậy của Bảng 4.
7. Limitations: bỏ 9 cạnh đáng ngờ nhất làm tỉ lệ tăng 3.9% → 7.9%.

---

## 3. Điểm cần nắm khi trả lời phản biện

- **99.5% hay 99.0%.** Có hai lần chạy Error Analyzer trên cùng 200 lượt MathDial qua gate.
  - `results/06_feedback/02_generation/error_analyzer_predictions_mathdial_ccmf.csv` là lần chạy lại với mastery CCMF, cũng là nguồn của 30 ca feedback (30/30 `raw_output` khớp). File này có **199/200**.
  - `results/05_error_analyzer/error_analyzer_predictions.csv` là lần chạy trước CCMF, có 198/200.
  - `supplementary/revision/cpu/05_regen_automatic_checks.py` đã ghi đè `automatic_checks.csv` thành 0.990 từ file thứ hai, nên file đó hiện lệch với bài.
- **Bảng 4 (CoMTA, 46 lượt) và 91.3%** lấy từ lần chạy trước CCMF. Điều này hợp lệ vì trên CoMTA mastery trong prompt lấy trực tiếp từ logit, không qua CCMF (`results/README.md`).
- **7.9% trong Limitations** chưa có file kết quả gắn nhãn trong gói. `cpu/out_08_endtoend_mta.json` ở thư mục gốc (không gắn nhãn) cho luật tiên quyết 7.86% và khác bản trong gói. Bản trong gói, `supplementary/revision/cpu/out_08_*`, là bản cho 3.9% như trong bài.
- **Macro-F1 của rule-based analyzer** là 0.1125, bài ghi 0.112 (làm tròn về số chẵn; làm tròn thông thường ra 0.113).
- **Người gán nhãn và người chấm là đồng tác giả.** Bài đã nêu ở Mục 4.1 và Limitations.

---

## 4. Cấu trúc thư mục

```
MathKT-Agent_SoICT2026/   gói phát hành: paper/, results/, raw_runs/, replication/, supplementary/
                          (README.md riêng, MANIFEST.sha256 riêng)
cpu/                      9 script CPU của vòng revision + out_0* (bản làm việc)
colab/                    3 notebook GPU (A đồ thị · B feedback đối chứng · C abstention)
colab_out/                đầu ra Colab: phiếu chấm, KEY, ea_abstention_*
labelling/                phiếu gán nhãn + sổ thống nhất 139 lượt
out/                      đồ thị tiên quyết + nguồn Achieve the Core + NGUON.md (ghi công ODC-BY)
```

Bản dùng cho bài của `cpu/ colab/ colab_out/ labelling/ out/` nằm trong
`MathKT-Agent_SoICT2026/supplementary/revision/`.

Kiểm toàn vẹn (không tính `.DS_Store` và chính file MANIFEST):

```bash
shasum -a 256 -c MANIFEST.sha256
```

Chạy lệnh này ở thư mục gốc, và chạy lại trong `MathKT-Agent_SoICT2026/`.

---

## 5. Đã xác minh

**Vòng 15/09/2026.** Đối chiếu khoảng 240 con số trong bài với file kết quả.

- Gần như toàn bộ khớp; các điểm lệch đã nêu ở §3.
- Các tổng khớp: 104 + 31 + 4 = 139; ma trận nhầm lẫn 46; 11 + 11 + 8 = 30; tỉ lệ luật MTA cộng lại 100.0%; 0.613 / 0.233 = 2.6.
- Tài liệu tham khảo: đối chiếu cả 31 mục với Crossref, ACL Anthology và arXiv.

**Vòng trước.** Tính lại độc lập từ file gốc.

- **Xác minh người (139 lượt):**
  - Đồng thuận 0.8489, κ 0.7137, 21 bất đồng, 4 unclear, 135 = 104 + 31.
  - Độ phủ sau tái cân bằng 0.7867 (không còn báo trong bài). GPT-4o 0.8991.
  - SAVA/regex 0.84552, judge 0.83023, hiệu ghép cặp −1.5284.
  - Trên 31 lượt SAVA từ chối: judge 0.6842, regex 0.3850 (28 lượt), GPT-4o 0.9514, lớp đa số 0.7682.
- **Rerun có lớp abstention:**
  - Validator 0.8478, tỉ lệ abstain 0.0000.
  - Exact 0.5000, macro-F1 0.2751, κ 0.2416.
  - Procedural 35/46; không bao giờ đoán calculation hay careless.
- **Vòng chấm ghép cặp:**
  - κ 0.5637 / 0.6865 / 0.5649; 75.0% điểm bằng 5.
  - 4.683 so với 4.233: +0.450 [+0.13, +0.82], p 0.0131.
  - 17/30 cặp hoà; mức tăng trên lượt incorrect +0.591.
- **Đồ thị và chạy end-to-end:**
  - Đồ thị: 103 KC (91 khớp chính xác + 12 khớp tiền tố); 238 cạnh co lập → 145, trong đó 109 trực tiếp; 0 chu trình; 0 cạnh ngược lớp.
  - Chạy lại MTA: sẵn sàng 18.4%, tiên quyết 3.9%, Commit 60.9%, fallback 21.3%, Maintain 8.0%, không KC 5.9%, khác đồ thị rỗng 39.6%.

Ba luật Maintain / Commit / no-KC **không phụ thuộc đồ thị**, nên việc chúng khớp chính xác xác nhận chuỗi
SAVA → CCMF → MTA chạy đúng engine. Self-check của `08_endtoend_mta.py` trùng cột `mean_bkt_sava`
ở mức 1.1e-16.
