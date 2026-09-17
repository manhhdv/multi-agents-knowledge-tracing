#!/usr/bin/env python3
"""Blind human annotation workflow for the Error Analyzer (46 incorrect CoMTA test turns).

    make     template CSV              -> annotator_A.xlsx, annotator_B.xlsx, HUONG_DAN_GAN_NHAN.md
    agree    annotator_A.xlsx B.xlsx   -> agreement_report.txt, adjudication.xlsx
    gold     adjudication.xlsx         -> error_analyzer_gold.csv (format read by notebook cell 33)
    metrics  gold CSV + predictions    -> error_analyzer_metrics.csv

The annotator and adjudication workbooks never contain model predictions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from sklearn.metrics import cohen_kappa_score, f1_score

CLASSES = ["conceptual", "procedural", "calculation", "careless", "other"]
LABEL_SHEET = "Gan nhan"
GUIDE_SHEET = "Huong dan"
ADJ_SHEET = "Thong nhat"
TAIL_LIMIT = 800
SEED = 221

GUIDELINE = """HƯỚNG DẪN GÁN NHÃN LOẠI LỖI — 46 lượt học sinh (CoMTA)

MỤC TIÊU
Mỗi dòng là một lượt trả lời của học sinh mà bộ dữ liệu đánh giá là SAI. Chọn MỘT loại lỗi chính cho lượt đó.

QUY TẮC BẮT BUỘC
1. Gán độc lập: không trao đổi với người gán còn lại cho đến khi cả hai nộp file.
2. Không dùng ChatGPT/Claude/công cụ AI nào để gợi ý nhãn.
3. Chỉ điền cột "gold", "gold_secondary", "ghi_chu". Không sửa, không xóa, không sắp xếp lại dòng.
4. Chỉ dựa vào ngữ cảnh và lượt học sinh. "[…]" ở đầu ngữ cảnh nghĩa là phần đầu hội thoại đã bị lược bớt.

5 LOẠI LỖI (định nghĩa giống hệt định nghĩa đưa cho mô hình)
• conceptual — hiểu sai một khái niệm hoặc quan hệ giữa các đại lượng.
  Ví dụ: "Hình chữ nhật 3×5 có diện tích 16" (nhầm diện tích với nửa chu vi);
         "1/2 + 1/3 = 2/5" (cộng tử với tử, mẫu với mẫu).
• procedural — một bước của phương pháp bị sai, bị thiếu hoặc sai thứ tự;
  học sinh biết cần làm gì nhưng thực hiện quy trình không đúng.
  Ví dụ: "2 + 3 × 4 = 20" (làm phép cộng trước); "x² − 5x + 6 = 0 ⇒ (x−2)(x−3) = 0 ⇒ x = 2" (bỏ sót nghiệm).
• calculation — phương pháp đúng nhưng tính toán số học sai.
  Ví dụ: "6 × 9 = 56"; "2x = 18 ⇒ x = 8".
• careless — sơ suất: chép sai số, sai dấu, đọc nhầm đề, gõ nhầm; học sinh gần như chắc chắn tự sửa được nếu nhìn lại.
  Ví dụ: đề cho 45 nhưng học sinh viết 54; "x − 5 = 3 ⇒ x = −8" khi các bước khác đều đúng.
• other — lượt không chứa lời giải hay câu trả lời toán học để phân loại: hỏi lại, xin giải thích, phàn nàn,
  trả lời "yes"/"không biết", lạc đề; hoặc có lỗi nhưng không thuộc 4 loại trên.
  Ví dụ: "Bạn giải thích lại được không?"; "tôi không hiểu".

THỨ TỰ QUYẾT ĐỊNH (xét lần lượt, dừng ở loại đầu tiên phù hợp)
  1) Lượt không có câu trả lời/lời giải toán học?  → other
  2) Chỉ là sơ suất chép/dấu/đọc đề, phần còn lại đúng?  → careless
  3) Phương pháp đúng, chỉ sai phép tính số học?  → calculation
  4) Đúng hướng nhưng sai/thiếu/sai thứ tự một bước?  → procedural
  5) Học sinh tin vào một quy tắc hoặc quan hệ sai?  → conceptual

PHÂN BIỆT KHI PHÂN VÂN
• conceptual vs procedural: nếu cho làm lại bài tương tự, học sinh có lặp đúng lỗi đó vì tin là đúng không?
  Có → conceptual. Không, chỉ thực hiện hỏng một bước → procedural.
• calculation vs careless: sai một phép tính số học → calculation; sai do chép, dấu, đọc đề → careless.
• Lượt có nhiều lỗi: chọn lỗi xảy ra SỚM NHẤT làm hỏng câu trả lời.

CỘT gold_secondary (không bắt buộc)
Chỉ điền khi một loại thứ hai cũng thực sự hợp lý. Để trống nếu bạn khá chắc chắn.

CỘT ghi_chu (không bắt buộc)
Ghi ngắn lý do nếu phân vân. Ghi chú giúp buổi thống nhất nhãn nhanh hơn.

KHI XONG
Kiểm tra ô "Đã gán" bên dưới báo 46/46, lưu file giữ nguyên tên, gửi lại cho người điều phối.
"""

ADJUDICATION_NOTE = """Thống nhất nhãn: hai người gán cùng xem các dòng tô vàng (bất đồng), thảo luận theo hướng dẫn,
điền cột "final". Dòng đồng thuận đã được điền sẵn. Nếu nhãn bị loại vẫn hợp lý, ghi nó vào "final_secondary".
Không xem dự đoán của mô hình trong buổi thống nhất."""


def short_kc(kc, limit=200):
    kc = str(kc)
    return kc if len(kc) <= limit else kc[:limit - 3] + "..."


def class_validation(ws, column, first_row, last_row, allow_blank):
    dv = DataValidation(type="list", formula1='"' + ",".join(CLASSES) + '"', allow_blank=allow_blank,
                        showErrorMessage=True, errorTitle="Nhãn không hợp lệ",
                        error="Chọn một trong: " + ", ".join(CLASSES))
    ws.add_data_validation(dv)
    dv.add(f"{column}{first_row}:{column}{last_row}")


def style_sheet(ws, widths, label_columns):
    header_fill = PatternFill("solid", fgColor="DDDDDD")
    label_fill = PatternFill("solid", fgColor="E8F5E9")
    for column, width in widths.items():
        ws.column_dimensions[column].width = width
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if cell.column_letter in label_columns:
                cell.fill = label_fill
    ws.freeze_panes = "A2"
    fit_to_page_width(ws)


def fit_to_page_width(ws):
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = "1:1"


def cmd_make(args):
    template = pd.read_csv(args.template)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "HUONG_DAN_GAN_NHAN.md").write_text(GUIDELINE, encoding="utf-8")
    n = len(template)
    for annotator in args.annotators:
        wb = Workbook()
        guide = wb.active
        guide.title = GUIDE_SHEET
        for i, line in enumerate(GUIDELINE.splitlines(), start=1):
            guide.cell(row=i, column=1, value=line).alignment = Alignment(wrap_text=True, vertical="top")
        guide.column_dimensions["A"].width = 125
        guide.page_setup.paperSize = guide.PAPERSIZE_A4
        guide.page_setup.fitToWidth = 1
        guide.page_setup.fitToHeight = 0
        guide.sheet_properties.pageSetUpPr.fitToPage = True
        guide["A1"].font = Font(bold=True, size=13)
        tail = len(GUIDELINE.splitlines()) + 2
        guide.cell(row=tail, column=1, value=f"Người gán: {annotator}").font = Font(bold=True)
        guide.cell(row=tail + 1, column=1,
                   value=f'="Đã gán: "&COUNTA(\'{LABEL_SHEET}\'!G2:G{n + 1})&"/{n}"').font = Font(bold=True)

        ws = wb.create_sheet(LABEL_SHEET)
        ws.append(["stt", "dialogue", "turn_id", "ngu_canh (hội thoại trước lượt này)", "luot_hoc_sinh",
                   "kcs", "gold", "gold_secondary", "ghi_chu"])
        for i, row in enumerate(template.itertuples(index=False), start=1):
            context = str(row.history_tail)
            if len(context) >= TAIL_LIMIT:
                context = "[…] " + context
            kcs = "\n".join("• " + short_kc(k) for k in json.loads(row.kcs))
            ws.append([i, int(row.dialogue), int(row.turn_id), context, str(row.student_text), kcs, None, None, None])
        style_sheet(ws, {"A": 5, "B": 9, "C": 8, "D": 75, "E": 40, "F": 40, "G": 15, "H": 15, "I": 30}, {"G", "H", "I"})
        class_validation(ws, "G", 2, n + 1, allow_blank=False)
        class_validation(ws, "H", 2, n + 1, allow_blank=True)
        wb.active = 1
        path = out / f"annotator_{annotator}.xlsx"
        wb.save(path)
        print("wrote", path)


def read_labels(path):
    frame = pd.read_excel(path, sheet_name=LABEL_SHEET)
    frame = frame.rename(columns=lambda c: str(c).split(" ")[0])
    for column in ("gold", "gold_secondary"):
        frame[column] = frame[column].fillna("").astype(str).str.strip().str.lower()
    missing = frame.loc[frame.gold == "", "stt"].tolist()
    invalid = frame.loc[(frame.gold != "") & ~frame.gold.isin(CLASSES), "stt"].tolist()
    invalid += frame.loc[(frame.gold_secondary != "") & ~frame.gold_secondary.isin(CLASSES), "stt"].tolist()
    if missing or invalid:
        sys.exit(f"{path}: rows without gold {missing}; rows with an invalid class {sorted(set(invalid))}")
    return frame


def cmd_agree(args):
    a, b = read_labels(args.a), read_labels(args.b)
    keys = ["dialogue", "turn_id"]
    merged = a.merge(b[keys + ["gold", "gold_secondary", "ghi_chu"]], on=keys, suffixes=("_A", "_B"))
    assert len(merged) == len(a) == len(b), "The two workbooks do not cover the same turns."
    kappa = cohen_kappa_score(merged.gold_A, merged.gold_B, labels=CLASSES)
    agreement = float((merged.gold_A == merged.gold_B).mean())
    relaxed = float(((merged.gold_A == merged.gold_B) | (merged.gold_A == merged.gold_secondary_B)
                     | (merged.gold_B == merged.gold_secondary_A)).mean())
    crosstab = pd.crosstab(merged.gold_A, merged.gold_B, rownames=["A"], colnames=["B"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)
    report = (f"n = {len(merged)}\nCohen's kappa (primary labels) = {kappa:.3f}\n"
              f"Raw agreement = {agreement:.3f}\nRelaxed agreement (either primary matches the other's "
              f"primary or secondary) = {relaxed:.3f}\n\nConfusion (rows A, columns B):\n{crosstab}\n\n"
              f"Label distribution A: {a.gold.value_counts().to_dict()}\n"
              f"Label distribution B: {b.gold.value_counts().to_dict()}\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement_report.txt").write_text(report, encoding="utf-8")
    print(report)

    wb = Workbook()
    ws = wb.active
    ws.title = ADJ_SHEET
    ws.append(["stt", "dialogue", "turn_id", "ngu_canh", "luot_hoc_sinh", "label_A", "secondary_A", "ghi_chu_A",
               "label_B", "secondary_B", "ghi_chu_B", "trang_thai", "final", "final_secondary"])
    disagree_fill = PatternFill("solid", fgColor="FFF3B0")
    for r in merged.itertuples(index=False):
        agreed = r.gold_A == r.gold_B
        secondary = ""
        if agreed:
            options = {s for s in (r.gold_secondary_A, r.gold_secondary_B) if s and s != r.gold_A}
            secondary = sorted(options)[0] if len(options) == 1 else ""
        ws.append([r.stt, r.dialogue, r.turn_id, r.ngu_canh, r.luot_hoc_sinh, r.gold_A, r.gold_secondary_A,
                   "" if pd.isna(r.ghi_chu_A) else r.ghi_chu_A, r.gold_B, r.gold_secondary_B,
                   "" if pd.isna(r.ghi_chu_B) else r.ghi_chu_B, "dong_thuan" if agreed else "BAT_DONG",
                   r.gold_A if agreed else None, secondary or None])
    n = len(merged)
    style_sheet(ws, {"A": 5, "B": 9, "C": 8, "D": 70, "E": 40, "F": 13, "G": 13, "H": 22, "I": 13, "J": 13,
                     "K": 22, "L": 12, "M": 15, "N": 15}, {"M", "N"})
    for row in ws.iter_rows(min_row=2):
        if row[11].value == "BAT_DONG":
            for cell in row[:12]:
                cell.fill = disagree_fill
    class_validation(ws, "M", 2, n + 1, allow_blank=False)
    class_validation(ws, "N", 2, n + 1, allow_blank=True)
    note = wb.create_sheet("Cach thong nhat")
    for i, line in enumerate(ADJUDICATION_NOTE.splitlines(), start=1):
        note.cell(row=i, column=1, value=line)
    note.column_dimensions["A"].width = 120
    path = out / "adjudication.xlsx"
    wb.save(path)
    print("wrote", path, f"({int((merged.gold_A != merged.gold_B).sum())} rows to adjudicate)")


def cmd_gold(args):
    frame = pd.read_excel(args.adjudication, sheet_name=ADJ_SHEET)
    for column in ("final", "final_secondary"):
        frame[column] = frame[column].fillna("").astype(str).str.strip().str.lower()
    bad = frame.loc[~frame.final.isin(CLASSES) | ((frame.final_secondary != "") & ~frame.final_secondary.isin(CLASSES)),
                    "stt"].tolist()
    if bad:
        sys.exit(f"Rows with a missing or invalid final label: {bad}")
    frame.loc[frame.final_secondary == frame.final, "final_secondary"] = ""
    gold = pd.DataFrame({"dialogue": frame.dialogue, "turn_id": frame.turn_id, "gold": frame.final,
                         "gold_secondary": frame.final_secondary})
    gold.to_csv(args.out, index=False)
    print("wrote", args.out, gold.gold.value_counts().to_dict())


def bootstrap_ci(correct, resamples=2000):
    rng = np.random.default_rng(SEED)
    values = np.asarray(correct, dtype=float)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cmd_metrics(args):
    gold = pd.read_csv(args.gold, dtype={"dialogue": str})
    gold["gold"] = gold.gold.astype(str).str.strip().str.lower()
    gold["gold_secondary"] = gold.gold_secondary.fillna("").astype(str).str.strip().str.lower()
    assert set(gold.gold) <= set(CLASSES), set(gold.gold) - set(CLASSES)
    predictions = pd.read_csv(args.predictions, dtype={"dialogue": str})
    predictions = predictions[predictions.set == "comta_incorrect_label"]
    merged = predictions.merge(gold, on=["dialogue", "turn_id"])
    assert len(merged) == len(gold), "Every gold row must match an analysed CoMTA turn."
    majority = merged.gold.value_counts().idxmax()
    valid = merged.valid_json.astype(str).str.lower() == "true"
    present = sorted(set(merged.gold))
    rows = []
    for method, prediction in [("llm_error_analyzer", merged.category),
                               ("llm_valid_json_only", merged.category.where(valid)),
                               ("rule_based", merged.category_rule_based),
                               ("majority_class", pd.Series(majority, index=merged.index))]:
        keep = prediction.notna()
        p, g, s = prediction[keep], merged.gold[keep], merged.gold_secondary[keep]
        exact = (p == g)
        low, high = bootstrap_ci(exact)
        rows.append({"method": method, "n": int(keep.sum()), "exact_match": float(exact.mean()),
                     "exact_match_ci_low": low, "exact_match_ci_high": high,
                     "macro_f1": float(f1_score(g, p, labels=CLASSES, average="macro", zero_division=0)),
                     "macro_f1_gold_classes": float(f1_score(g, p, labels=present, average="macro", zero_division=0)),
                     "cohen_kappa_vs_gold": float(cohen_kappa_score(g, p, labels=CLASSES)),
                     "relaxed_agreement": float(((p == g) | (p == s)).mean())})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out, index=False)
    print(f"gold distribution: {merged.gold.value_counts().to_dict()}  (majority = {majority})")
    print(metrics.round(3).to_string(index=False))
    print("\nConfusion (rows gold, columns LLM):")
    print(pd.crosstab(merged.gold, merged.category).reindex(index=CLASSES, columns=CLASSES, fill_value=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("--template", required=True)
    make.add_argument("--out", default=".")
    make.add_argument("--annotators", nargs="+", default=["A", "B"])
    agree = sub.add_parser("agree")
    agree.add_argument("a")
    agree.add_argument("b")
    agree.add_argument("--out", default=".")
    gold = sub.add_parser("gold")
    gold.add_argument("adjudication")
    gold.add_argument("--out", default="error_analyzer_gold.csv")
    metrics = sub.add_parser("metrics")
    metrics.add_argument("gold")
    metrics.add_argument("predictions")
    metrics.add_argument("--out", default="error_analyzer_metrics.csv")
    args = parser.parse_args()
    {"make": cmd_make, "agree": cmd_agree, "gold": cmd_gold, "metrics": cmd_metrics}[args.command](args)


if __name__ == "__main__":
    main()
