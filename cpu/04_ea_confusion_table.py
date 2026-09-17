"""04 - Ma tran nham lan 5x3 cho Error Analyzer (thay doan van dem theo lop).
Dong y kien R-b 4.5. Xuat thang LaTeX. Chay CPU.
Tu kiem: tai lap exact match / macro-F1 / kappa trong error_analyzer_metrics.csv."""
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

from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
CLASSES = ["conceptual", "procedural", "calculation", "careless", "other"]
gold = pd.read_csv(rp("results", "05_error_analyzer", "error_analyzer_gold.csv"))
pred = pd.read_csv(rp("results", "05_error_analyzer", "error_analyzer_predictions.csv"))
df = pred[pred.dataset == "comta"].merge(gold, on=["dialogue", "turn_id"])
assert len(df) == 46, len(df)
ref = pd.read_csv(rp("results", "05_error_analyzer", "error_analyzer_metrics.csv"))
r = ref[ref.method == "llm_error_analyzer"].iloc[0]
assert abs(accuracy_score(df.gold, df.category) - r.exact_match) < 1e-3
assert abs(f1_score(df.gold, df.category, average="macro", labels=CLASSES, zero_division=0) - r.macro_f1) < 1e-3
assert abs(cohen_kappa_score(df.gold, df.category) - r.cohen_kappa_vs_gold) < 1e-3
print("Tu kiem OK: exact %.4f, macro-F1 %.4f, kappa %.4f" % (r.exact_match, r.macro_f1, r.cohen_kappa_vs_gold))

cm = pd.crosstab(pd.Categorical(df.gold, CLASSES), pd.Categorical(df.category, CLASSES), dropna=False)
cm = cm.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
cm.to_csv(op("out_04_ea_confusion.csv"))
print(); print(cm.to_string())
used = [c for c in CLASSES if cm[c].sum() > 0]
tex = ["\\begin{table}[t]", "\\centering",
       "\\caption{Error Analyzer against adjudicated human labels on 46 incorrect CoMTA turns: "
       "predicted class (columns) by human label (rows). The analyzer never predicts "
       "\\emph{calculation} or \\emph{careless}.}\\label{tab:ea-cm}", "\\small",
       "\\begin{tabular}{l" + "c" * len(used) + "c}", "\\toprule",
       "Human label & " + " & ".join("\\emph{%s}" % c for c in used) + " & Total \\\\", "\\midrule"]
for c in CLASSES:
    tex.append("\\emph{%s} & " % c + " & ".join(str(int(cm.loc[c, u])) for u in used)
               + " & %d \\\\" % int(cm.loc[c].sum()))
tex += ["\\midrule", "Total & " + " & ".join(str(int(cm[u].sum())) for u in used)
        + " & %d \\\\" % int(cm.values.sum()), "\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(op("out_04_ea_confusion.tex"), "w").write("\n".join(tex) + "\n")
print(); print("\n".join(tex))
print("-> out_04_ea_confusion.csv / .tex")
