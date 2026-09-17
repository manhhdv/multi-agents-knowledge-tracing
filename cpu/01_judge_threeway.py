"""01 - Bien the LLM judge CO tuy chon abstain (three-way).
Dong y kien: 'SAVA abstain la nhuong bo do phu'. Chay CPU, khong GPU, khong API key.
Doc duy nhat results/02_sava/llm_judge_verdicts_test.csv (da phat hanh).
Tu kiem: tai lap 75.4%% cua judge binary truoc khi bao so moi."""
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

JUDGED = ("correct", "incorrect")
jv = pd.read_csv(rp("results", "02_sava", "llm_judge_verdicts_test.csv"))
gold = jv.label.map({1: "correct", 0: "incorrect"})
sava = jv.sava.astype(str)
sj = sava.isin(JUDGED)                      # SAVA co phan quyet
abst = ~sj                                   # SAVA tu choi

def stats(col, mask=None):
    v = jv[col].astype(str); judged = v.isin(JUDGED)
    m = judged if mask is None else (judged & mask)
    return dict(coverage=float(judged.mean()),
                agreement_when_judging=float((v[m] == gold[m]).mean()),
                n_judged=int(m.sum()))

# --- tu kiem truoc khi bao so moi ---
binary = stats("llm_judge_binary")
assert abs(binary["agreement_when_judging"] - 0.754) < 1e-3, binary
assert abs(binary["coverage"] - 1.0) < 1e-6, binary

three = stats("llm_judge_three_way")
three_on_abst = stats("llm_judge_three_way", abst)
bin_on_abst = stats("llm_judge_binary", abst)
regex_on_abst = stats("numeric_regex", abst)
maj = float(gold[abst].value_counts(normalize=True).max())
n_decl = int((jv.llm_judge_three_way.astype(str) == "undetermined").sum())

res = dict(
    n_turns=int(len(jv)),
    n_sava_abstain=int(abst.sum()),
    judge_binary=binary,
    judge_three_way=three,
    judge_three_way_n_declined=n_decl,
    on_sava_abstained_turns=dict(judge_three_way=three_on_abst["agreement_when_judging"],
                                 judge_binary=bin_on_abst["agreement_when_judging"],
                                 numeric_regex=regex_on_abst["agreement_when_judging"],
                                 majority_class=maj),
)
with open(op("out_01_judge_threeway.json"), "w") as f: json.dump(res, f, indent=1)

print("So luot: %d | SAVA tu choi: %d" % (res["n_turns"], res["n_sava_abstain"]))
print("judge binary   : do phu %.4f | dong thuan %.4f" % (binary["coverage"], binary["agreement_when_judging"]))
print("judge three-way: do phu %.4f | dong thuan %.4f | chi tu choi %d/%d luot"
      % (three["coverage"], three["agreement_when_judging"], n_decl, len(jv)))
print("Tren %d luot SAVA tu choi: three-way %.4f | binary %.4f | regex %.4f | lop da so %.4f"
      % (abst.sum(), three_on_abst["agreement_when_judging"], bin_on_abst["agreement_when_judging"],
         regex_on_abst["agreement_when_judging"], maj))
print("-> out_01_judge_threeway.json")
