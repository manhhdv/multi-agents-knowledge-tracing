"""07 - CHAY SAU COLAB B VA SAU KHI CHAM. Feedback co dieu kien doi chung chua?
Dong y kien #5. Ghep nhan tu hai nguoi cham tren 60 muc tron lan, tach lai hai dieu kien
bang feedback_baseline_KEY.csv, roi kiem dinh Wilcoxon GHEP CAP theo tung case (30 cap).
Doc tu revision_kit/colab_out/ (tai tu MyDrive/revision_kit/colab_out): feedback_baseline_KEY.csv,
feedback_baseline_rater_A.xlsx, ..._B.xlsx"""
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

from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score

IN = os.path.join(os.path.dirname(OUT), "colab_out")
def ip(*a): return os.path.join(IN, *a)
KEY = ip("feedback_baseline_KEY.csv")
for p in (KEY, ip("feedback_baseline_rater_A.xlsx"), ip("feedback_baseline_rater_B.xlsx")):
    if not os.path.exists(p):
        raise SystemExit("Thieu %s. Chay colab/B_feedback_baseline.ipynb va cham diem truoc."
                         % os.path.basename(p))
key = pd.read_csv(KEY)
CRIT = ["correctness", "relevance", "clarity"]

def load(who):
    d = pd.read_excel(ip("feedback_baseline_rater_%s.xlsx" % who), sheet_name="Cham diem")
    return d[["item_id"] + CRIT].rename(columns={c: "%s_%s" % (c, who) for c in CRIT})

d = key.merge(load("A"), on="item_id").merge(load("B"), on="item_id")
for c in CRIT:
    d[c] = (pd.to_numeric(d["%s_A" % c]) + pd.to_numeric(d["%s_B" % c])) / 2

print("Do dong thuan giua hai nguoi cham (quadratic-weighted kappa):")
for c in CRIT:
    print("  %-12s %.3f" % (c, cohen_kappa_score(d["%s_A" % c].astype(int), d["%s_B" % c].astype(int),
                                                 weights="quadratic")))

sysd = d[d.condition == "system_mastery_conditioned"].set_index(["dialogue", "turn_id"])
base = d[d.condition == "baseline_llmkt_only"].set_index(["dialogue", "turn_id"])
common = sysd.index.intersection(base.index)
assert len(common) == 30, "Ky vong 30 cap, nhan duoc %d" % len(common)

rows = []
for c in CRIT:
    x, y = sysd.loc[common, c].values, base.loc[common, c].values
    diff = x - y
    W = wilcoxon(x, y, zero_method="wilcox") if np.any(diff != 0) else None
    rng = np.random.default_rng(221)
    bs = [np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append(dict(criterion=c, system_mean=float(x.mean()), baseline_mean=float(y.mean()),
                     mean_difference=float(diff.mean()), ci_low=lo, ci_high=hi,
                     wilcoxon_p=float(W.pvalue) if W else float("nan"),
                     n_pairs=len(common), n_ties=int((diff == 0).sum())))
out = pd.DataFrame(rows)
out.to_csv(op("out_07_feedback_baseline.csv"), index=False)
print()
print("He co dieu kien hoa mastery/chan doan vs doi chung LLMKT-only (30 cap, cung 2 nguoi cham):")
print(out.round(4).to_string(index=False))
print("\nDoc ket qua: khoang tin cay chua 0 => KHONG chung minh duoc dieu kien hoa lam tot hon;")
print("do la ket qua am hop le va nen bao cao dung nhu vay, thay cho cau 'no baseline was rated'.")
print("-> out_07_feedback_baseline.csv")
