"""Merge the Colab re-run of the Error Analyzer (200 + 1 MathDial turns) and Feedback Generator (30 cases) that use
CCMF mastery, verify it, compute the automatic checks and build blind rater workbooks.

usage: python apply_ccmf_full_rerun.py <results.b64> <expected sha256 printed by cell R5>
Inputs : 06_feedback/01_inputs/inputs_ccmf.sha256
Outputs: 06_feedback/02_generation/ (results.json, error_analyzer_predictions_mathdial_ccmf.csv,
         feedback_generator_cases.csv with GPT-4o scores, automatic_checks.csv)
         06_feedback/03_human_rating/ (blank rater_A.xlsx, rater_B.xlsx)
After both raters return their workbooks:
  python ../../fb_rating.py analyze rater_A.xlsx rater_B.xlsx --bundle <bundle> --out .   (see printed command)
"""
import base64, gzip, hashlib, json, os, shutil, sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parents[1]
PAPER = PKG.parent
FEEDBACK = PKG / "06_feedback"
INPUTS_DIR = FEEDBACK / "01_inputs"
base = Path(os.environ.get("CCMF_RERUN_APPLY_OUT", FEEDBACK))  # override only for local pipeline tests
OUT, RATING = base / "02_generation", base / "03_human_rating"
OUT.mkdir(parents=True, exist_ok=True)
RATING.mkdir(parents=True, exist_ok=True)
b64_path, expected = Path(sys.argv[1]), sys.argv[2].strip()

raw = gzip.decompress(base64.b64decode(b64_path.read_text().strip()))
digest = hashlib.sha256(raw).hexdigest()
assert digest == expected, f"transcription mismatch: {digest} != {expected}"
results = json.loads(raw)
inputs_sha = (INPUTS_DIR / "inputs_ccmf.sha256").read_text().strip()
assert results["inputs_sha256"] == inputs_sha, "Colab used a different inputs_ccmf.json"
assert len(results["error_analysis"]) == 201 and len(results["feedback"]) == 30
print("results verified:", digest)
(OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

ea = pd.DataFrame(results["error_analysis"])
ea.to_csv(OUT / "error_analyzer_predictions_mathdial_ccmf.csv", index=False)
fb = pd.DataFrame(results["feedback"])
fb.to_csv(OUT / "feedback_generator_cases.csv", index=False)

gated = ea[ea.set == "mathdial_sava_gated"]
checks = {
    "ea_mathdial_n": len(gated), "ea_mathdial_valid_json": float(gated.valid_json.mean()),
    "feedback_n": len(fb), "feedback_valid_json": float(fb.valid_json.mean()),
    "feedback_strategy_band_match": float(fb.strategy_band_match.mean()),
    "feedback_answer_leak_not_correct": float(fb[fb.verdict != "correct"].answer_leak.astype(float).mean()),
    "feedback_rule_fallback": int((fb.source == "rule_fallback").sum()),
}
for c in ["correctness", "relevance", "clarity"]:
    if c in fb:
        checks[f"judge_{c}_mean"] = float(fb[c].mean())
pd.DataFrame([checks]).to_csv(OUT / "automatic_checks.csv", index=False)
print(pd.Series(checks).to_string())

bundle = Path(os.environ.get("TMPDIR", "/tmp")) / "ccmf_feedback_bundle"
(bundle / "mathdial/qlora/seed221").mkdir(parents=True, exist_ok=True)
link = bundle / "mathdial/qlora/seed221/test"
if not link.exists():
    os.symlink(PAPER / "20260912T221028Z/mathdial/qlora/seed221/test", link)
fb.to_csv(bundle / "feedback_generator_cases.csv", index=False)
sys.path.insert(0, str(PKG / "06_feedback"))
import fb_rating  # noqa: E402

fb_rating.cmd_make(Namespace(bundle=str(bundle), out=str(RATING), raters=["A", "B"]))
for rater in ["A", "B"]:
    frame = pd.read_excel(RATING / f"rater_{rater}.xlsx", sheet_name=fb_rating.RATE_SHEET)
    print(rater, len(frame), "rows | rating columns empty:",
          bool(frame[["correctness", "relevance", "clarity", "lo_dap_an"]].isna().all().all()))
print("\nWhen both workbooks are rated, run:\n"
      f"  python {FEEDBACK / 'fb_rating.py'} analyze {RATING / 'rater_A.xlsx'} {RATING / 'rater_B.xlsx'} "
      f"--bundle {bundle} --out {RATING}")
