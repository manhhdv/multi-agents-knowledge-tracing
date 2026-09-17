"""Decode the Colab results for the 15 regenerated feedback cases, verify them, and build blind rater workbooks.

usage: apply_regen15.py <results.b64> <expected sha256> <out dir>
"""
import base64, gzip, hashlib, json, os, shutil, sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

PAPER = Path("/Users/manhnv/Downloads/SoICT2026_Paper")
b64_path, expected, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

raw = gzip.decompress(base64.b64decode(b64_path.read_text().strip()))
digest = hashlib.sha256(raw).hexdigest()
assert digest == expected, f"transcription mismatch: {digest} != {expected}"
results = json.loads(raw)
assert len(results["feedback"]) == 15, len(results["feedback"])
print("results verified against the Colab hash:", len(results["feedback"]), "feedback,", len(results["error_analysis"]), "error analyses")

out.mkdir(parents=True, exist_ok=True)
(out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
cases = pd.DataFrame(results["feedback"])
cases.to_csv(out / "feedback_generator_cases_15.csv", index=False)
pd.DataFrame(results["error_analysis"]).to_csv(out / "error_analyzer_15.csv", index=False)

bundle = out / "_bundle"
(bundle / "mathdial/qlora/seed221").mkdir(parents=True, exist_ok=True)
os.symlink(PAPER / "20260912T221028Z/mathdial/qlora/seed221/test", bundle / "mathdial/qlora/seed221/test")
cases.to_csv(bundle / "feedback_generator_cases.csv", index=False)
sys.path.insert(0, str(PAPER / "feedback_rating"))
import fb_rating  # noqa: E402

n = len(cases)
fb_rating.GUIDELINE = fb_rating.GUIDELINE.replace("30 trường hợp", f"{n} trường hợp").replace("30/30", f"{n}/{n}")
fb_rating.cmd_make(Namespace(bundle=str(bundle), out=str(out), raters=["A", "B"]))
shutil.rmtree(bundle)
for rater in ["A", "B"]:
    frame = pd.read_excel(out / f"rater_{rater}.xlsx", sheet_name=fb_rating.RATE_SHEET)
    print(rater, len(frame), "rows | rating columns empty:",
          bool(frame[["correctness", "relevance", "clarity", "lo_dap_an"]].isna().all().all()))
print(cases[["dialogue", "turn_id", "verdict", "band", "source", "valid_json", "answer_leak", "strategy_band_match"]].to_string(index=False))
