"""03 - Khoang tin cay bootstrap cho khoi rerun cua Bang 1.
Dong y kien R-b 3.4: Bang 1 la bang duy nhat khong co CI, trong khi CoMTA chi n=78.
Bootstrap theo HOI THOAI (khong theo luot) de ton trong cau truc phu thuoc, 2000 lan lap.
Tu kiem: diem uoc luong phai khop metrics_summary.csv da phat hanh."""
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

from scipy.special import expit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

B, THR = 2000, 0.5
RUNS = [("mathdial", "zero_shot", "seed-na", "mathdial/zero_shot/seed-na/test/raw_logit_gaps.json"),
        ("mathdial", "qlora", "seed221", "mathdial/qlora/seed221/test/raw_logit_gaps.json")]
_c = "comta/zero_shot/seed-na/test/raw_logit_gaps.json"
if os.path.exists(rp("raw_runs", _c)):
    RUNS.append(("comta", "zero_shot", "seed-na", _c))

ms = pd.read_csv(rp("results", "01_backbone", "metrics_summary.csv"))
rows = []
for ds, model, seed, rel in RUNS:
    recs = json.load(open(rp("raw_runs", rel)))
    y = np.asarray([r["label"] for r in recs], float)
    p = np.asarray([float(np.mean(expit(np.asarray(r["logit_gaps"], float)))) for r in recs])
    d = np.asarray([r["dialogue"] for r in recs])
    ref = ms[(ms.dataset == ds) & (ms.model == model) & (ms.seed == seed) & (ms.split == "test")]
    pt = dict(accuracy=100 * accuracy_score(y, p > THR), auc=100 * roc_auc_score(y, p),
              f1=100 * f1_score(y, p > THR))
    if len(ref):
        for k in pt:
            assert abs(pt[k] - float(ref.iloc[0][k])) < 5e-2, (ds, model, k, pt[k], ref.iloc[0][k])
    ud = np.unique(d); idx = {u: np.where(d == u)[0] for u in ud}
    rng = np.random.default_rng(221); acc = {k: [] for k in pt}
    for _ in range(B):
        ii = np.concatenate([idx[u] for u in rng.choice(ud, len(ud), replace=True)])
        if len(np.unique(y[ii])) < 2: continue
        acc["accuracy"].append(100 * accuracy_score(y[ii], p[ii] > THR))
        acc["auc"].append(100 * roc_auc_score(y[ii], p[ii]))
        acc["f1"].append(100 * f1_score(y[ii], p[ii] > THR))
    r = dict(dataset=ds, model=model, seed=seed, n=len(y), n_dialogues=len(ud))
    for k, v in pt.items():
        lo, hi = np.percentile(acc[k], [2.5, 97.5]); r[k] = v; r[k + "_lo"] = lo; r[k + "_hi"] = hi
    rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(op("out_03_table1_cis.csv"), index=False)
print(df.round(2).to_string(index=False))
print()
print("%% Dan vao khoi duoi cua Bang 1 (thay the ba dong rerun). CI bootstrap theo hoi thoai, 2000 lan lap.")
name = {("mathdial", "zero_shot"): "LLMKT, zero-shot (rerun)", ("mathdial", "qlora"): "LLMKT, 4-bit QLoRA (rerun)",
        ("comta", "zero_shot"): "LLMKT, zero-shot (rerun), CoMTA"}
for _, r in df.iterrows():
    cells = " & ".join("%.2f \\tiny[%.1f, %.1f]" % (r[k], r[k + "_lo"], r[k + "_hi"])
                       for k in ("accuracy", "auc", "f1"))
    print("%% %-34s & %s \\\\" % (name[(r.dataset, r.model)], cells))
print("-> out_03_table1_cis.csv")
