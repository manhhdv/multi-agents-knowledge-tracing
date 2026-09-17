"""Build blind rater workbooks for the two regenerated feedback cases, in the same format as the original 30.

Input : results.json (Error Analyzer + feedback records exported from the Colab run)
Output: <out>/feedback_generator_cases.csv, rater_A.xlsx, rater_B.xlsx, HUONG_DAN_CHAM_FEEDBACK.md
"""
import hashlib
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

PAPER = Path("/Users/manhnv/Downloads/SoICT2026_Paper")
results_path, out, expected_hash = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

results = json.loads(results_path.read_text())
canonical = json.dumps(results, ensure_ascii=True, sort_keys=True, allow_nan=True)
digest = hashlib.sha256(canonical.encode()).hexdigest()
assert digest == expected_hash, f"transcription mismatch: {digest} != {expected_hash}"
print("results verified against the Colab hash")

out.mkdir(parents=True, exist_ok=True)
bundle = out / "_bundle"
(bundle / "mathdial/qlora/seed221").mkdir(parents=True, exist_ok=True)
link = bundle / "mathdial/qlora/seed221/test"
if not link.exists():
    os.symlink(PAPER / "20260912T221028Z/mathdial/qlora/seed221/test", link)
cases = pd.DataFrame(results["feedback"])
cases.to_csv(bundle / "feedback_generator_cases.csv", index=False)
cases.to_csv(out / "feedback_generator_cases_2.csv", index=False)
pd.DataFrame([results["error_analysis"]]).to_csv(out / "error_analyzer_498_3.csv", index=False)

sys.path.insert(0, str(PAPER / "feedback_rating"))
import fb_rating  # noqa: E402

fb_rating.GUIDELINE = fb_rating.GUIDELINE.replace("30 trường hợp", f"{len(cases)} trường hợp").replace(
    "30/30", f"{len(cases)}/{len(cases)}")
fb_rating.cmd_make(Namespace(bundle=str(bundle), out=str(out), raters=["A", "B"]))
for rater in ["A", "B"]:
    frame = pd.read_excel(out / f"rater_{rater}.xlsx", sheet_name=fb_rating.RATE_SHEET)
    print(rater, len(frame), "rows:", frame[["dialogue", "turn_id"]].values.tolist(),
          "| rating columns empty:", bool(frame[["correctness", "relevance", "clarity", "lo_dap_an"]].isna().all().all()))
