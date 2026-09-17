"""08 - CHAY SAU cpu/09. Vong end-to-end THAT: SAVA -> CCMF -> MTA voi do thi tien quyet.
Dong y kien #1 va #4: hien MTA chay tren mastery mo phong, va tren MathDial thi do thi rong
nen co che cot loi (Maintain / Commit / Advance-theo-tien-quyet) chua he duoc kich hoat tren
du lieu that. Script nay chay dung ba luat cua Eq.(5) tren mastery THAT do CCMF uoc luong,
voi do thi tien quyet Achieve the Core (cpu/09), va bao cao co che nao thuc su duoc dung.
Dau vao: out/prereq_graph_mathdial_atc.json (do thi ATC dung bang quy tac, cpu/09), hoac tro bang --graph.
Tu kiem: tai lap AUC/log-loss test da phat hanh cua CCMF truoc khi chay planner."""
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

import argparse
from scipy.special import expit
from sklearn.metrics import roc_auc_score, log_loss

ap = argparse.ArgumentParser()
ap.add_argument("--graph", default=os.path.join(os.path.dirname(OUT), "out", "prereq_graph_mathdial_atc.json"))
ap.add_argument("--tau", type=float, default=0.8)
args = ap.parse_args()
TAU = args.tau

tr = pd.read_csv(rp("results", "03_ccmf", "test_results.csv"))
row = tr[(tr.model == "qlora") & (tr.variant == "mean_bkt_sava")].iloc[0]
P = json.loads(row.params_natural_json)
a, b, lam, pL, m0, s, g = P["a"], P["b"], P["lambda"], P["p_L"], P["m0"], P["s"], P["g"]
mu = P.get("mu") or 0.0
raw = json.load(open(rp("raw_runs", "mathdial", "qlora", "seed221", "test", "raw_logit_gaps.json")))
_sv = pd.read_csv(rp("results", "02_sava", "real_turns_mathdial_test.csv"))
# id hoi thoai la chuoi trong raw_logit_gaps.json nhung la so nguyen trong CSV -> phai chuan hoa,
# neu khong moi tra cuu deu truot va cap nhat SAVA khong bao gio duoc ap dung.
sava = {(str(d), int(t)): str(v) for d, t, v in zip(_sv.dialogue, _sv.turn_id, _sv.sava)}

if not os.path.exists(args.graph):
    raise SystemExit("Chua co do thi tien quyet: %s\nChay cpu/09_build_atc_graph.py truoc, "
                     "hoac tro --graph toi file JSON {KC: [cac KC tien quyet]}." % args.graph)
G = json.load(open(args.graph))
G = G.get("prerequisites", G)
pre = {k: set(v) for k, v in G.items()}

preds, log = [], []
cur, state, E, committed = None, {}, set(), None
for r in raw:
    if r["dialogue"] != cur:
        cur, state, E, committed = r["dialogue"], {}, set(), None
    for k, d in zip(r["kcs"], r["logit_gaps"]):
        mk = state.get(k, m0)
        state[k] = (1 - lam) * (mk + (1 - mk) * pL) + lam * expit(a * d + b)
    mbar = float(np.mean([state[k] for k in r["kcs"]]))
    preds.append((1 - s) * mbar + g * (1 - mbar))
    v = sava.get((str(r["dialogue"]), int(r["turn_id"])), "undetermined")     # cap nhat sau khi da ghi y_hat
    if mu and v in ("correct", "incorrect"):
        yv = 1.0 if v == "correct" else 0.0
        for k in r["kcs"]: state[k] = (1 - mu) * state[k] + mu * yv
    E |= {k for k, m in state.items() if m >= TAU}
    slipped = {k: state[k] for k in E if state.get(k, 0.0) < TAU}
    if slipped:
        rule, choice = "maintain", max(slipped, key=slipped.get)
    elif committed is not None and state.get(committed, 1.0) < TAU:
        rule, choice = "commit", committed
    else:
        ready = {k: state[k] for k in state if k not in E and pre.get(k, set()) <= E}
        if ready: rule, choice = "advance_ready", max(ready, key=ready.get)
        else:
            rest = {k: state[k] for k in state if k not in E}
            rule, choice = ("advance_fallback", max(rest, key=rest.get)) if rest else ("none", None)
    committed = choice
    unm = {k: state[k] for k in state if state[k] < TAU}
    empty_graph_choice = max(unm, key=unm.get) if unm else None
    log.append(dict(dialogue=r["dialogue"], turn_id=r["turn_id"], rule=rule, next_kc=choice,
                    next_kc_empty_graph=empty_graph_choice, n_known_kcs=len(state), n_mastered=len(E),
                    n_ready=len([k for k in state if k not in E and pre.get(k, set()) <= E]),
                    constrained=bool(choice != empty_graph_choice)))

y = np.asarray([r["label"] for r in raw], float); p = np.asarray(preds)
rel = pd.read_csv(rp("results", "03_ccmf", "test_predictions_qlora_seed221.csv"))
dmax = float(np.abs(p - rel.mean_bkt_sava.values).max())
assert dmax < 1e-9, ("replay lech %.2e so voi cot mean_bkt_sava da phat hanh; "
                     "kiem tra lai nguon verdict SAVA va kieu id hoi thoai" % dmax)
auc, ll = roc_auc_score(y, p), log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))
print("Tu kiem OK: chuoi SAVA->CCMF trung cot mean_bkt_sava da phat hanh (max|diff| = %.1e); "
      "AUC %.4f / log-loss %.4f" % (dmax, auc, ll))

L = pd.DataFrame(log); L.to_csv(op("out_08_endtoend_mta.csv"), index=False)
cov = float(np.mean([len(set(r["kcs"]) & set(pre)) / max(1, len(r["kcs"])) for r in raw]))
summ = dict(n_turns=len(L), tau=TAU, kc_covered_by_graph=cov,
            rule_share={k: float(v) for k, v in L.rule.value_counts(normalize=True).items()},
            share_choice_differs_from_empty_graph=float(L.constrained.mean()),
            mean_ready_set_size=float(L.n_ready.mean()),
            share_turns_with_nonempty_ready_set=float((L.n_ready > 0).mean()),
            mean_mastered_kcs_at_turn=float(L.n_mastered.mean()))
json.dump(summ, open(op("out_08_endtoend_mta.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
print("-> out_08_endtoend_mta.csv / .json")
