"""05 - Sinh lai results/06_feedback/02_generation/automatic_checks.csv.
File tom tat hien ghi ea_mathdial_valid_json = 0.995 trong khi file bang chung theo dong
(error_analyzer_predictions.csv, set='mathdial_sava_gated') cho 198/200 = 0.990.
Script tinh lai MOI o tu cac file theo dong va ghi ra ban da sua, kem bao cao sai lech."""
import os, sys, json, math
import numpy as np, pandas as pd

def _find_root():
    env = os.environ.get("MATHKT_ROOT")
    if env and os.path.isdir(os.path.join(env, "results")):
        return os.path.abspath(env)
    seeds = []
    try: seeds.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError: pass
    seeds.append(os.getcwd())
    for s in seeds:
        p = s
        for _ in range(6):
            for q in (p, os.path.join(p, "MathKT-Agent_SoICT2026")):   # goi tai lap chep san trong revision_kit
                if os.path.isdir(os.path.join(q, "results")) and os.path.isdir(os.path.join(q, "raw_runs")):
                    return q
            p = os.path.dirname(p)
    raise SystemExit("Khong tim thay goc goi (can thu muc chua ca results/ va raw_runs/). "
                     "Dat bien moi truong MATHKT_ROOT.")

def _outdir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.getcwd()

ROOT = _find_root(); OUT = _outdir()
def rp(*a): return os.path.join(ROOT, *a)
def op(*a): return os.path.join(OUT, *a)

pred = pd.read_csv(rp("results", "05_error_analyzer", "error_analyzer_predictions.csv"))
cases = pd.read_csv(rp("results", "06_feedback", "02_generation", "feedback_generator_cases.csv"))
old_p = rp("results", "06_feedback", "02_generation", "automatic_checks.csv")
old = pd.read_csv(old_p).iloc[0]

ea = pred[pred["set"] == "mathdial_sava_gated"]
notc = cases[cases.verdict != "correct"]
new = dict(
    ea_mathdial_n=int(len(ea)),
    ea_mathdial_valid_json=float(ea.valid_json.mean()),
    feedback_n=int(len(cases)),
    feedback_valid_json=float(cases.valid_json.mean()),
    feedback_strategy_band_match=float(cases.strategy_band_match.mean()),
    feedback_answer_leak_not_correct=float(notc.answer_leak.mean()),
    feedback_rule_fallback=int((cases.source == "rule_fallback").sum()),
    judge_correctness_mean=float(cases.correctness.mean()),
    judge_relevance_mean=float(cases.relevance.mean()),
    judge_clarity_mean=float(cases.clarity.mean()),
)
pd.DataFrame([new]).to_csv(op("out_05_automatic_checks_FIXED.csv"), index=False)

print("%-34s %12s %12s  %s" % ("o", "cu", "tinh lai", ""))
for k, v in new.items():
    o = old.get(k, float("nan"))
    flag = "" if (isinstance(v, (int, float)) and abs(float(o) - float(v)) < 1e-9) else "  <-- LECH"
    print("%-34s %12.6f %12.6f%s" % (k, float(o), float(v), flag))
bad = [r["dialogue"] for _, r in ea[~ea.valid_json.astype(bool)].iterrows()]
print()
print("Hai dong khong hop le trong 200 luot MathDial (deu roi ve rule_fallback): dialogue %s" % bad)
print("-> out_05_automatic_checks_FIXED.csv  (chep de len results/06_feedback/02_generation/automatic_checks.csv)")
