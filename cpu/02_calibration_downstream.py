"""02 - Calibration co doi quyet dinh ha nguon khong?
Dong y kien R-b W3.1: 'calibration chua bao gio duoc chung minh la co y nghia'.
Chay lai bo loc CCMF voi tham so DA PHAT HANH tren logit that, roi so quyet dinh cua
MTA (do thi rong -> chon KC chua thanh thao gan nguong nhat) giua ba cau hinh mastery:
  A. CCMF day du (Platt + bo loc BKT)         <- he thong trong bai
  B. Bo loc BKT nhung BO Platt (z = sigmoid(d)) <- tach rieng anh huong cua calibration
  C. Xac suat tho cuoi cung moi KC, khong bo loc <- gan LLMKT mean pooling nhat
Chay CPU, khong GPU. Tu kiem: tai lap AUC/log-loss test da phat hanh cua CCMF truoc."""
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
from sklearn.metrics import roc_auc_score, log_loss

TAU = 0.8
tr = pd.read_csv(rp("results", "03_ccmf", "test_results.csv"))
row = tr[(tr.model == "qlora") & (tr.variant == "mean_bkt_none")].iloc[0]
P = json.loads(row.params_natural_json)
a, b, lam, pL, m0, s, g = P["a"], P["b"], P["lambda"], P["p_L"], P["m0"], P["s"], P["g"]
raw = json.load(open(rp("raw_runs", "mathdial", "qlora", "seed221", "test", "raw_logit_gaps.json")))

def replay(calibrated=True, filtered=True):
    """Tra ve (pred_turn, mastery_truoc_moi_luot) theo Eq.(1)-(3) cua bai."""
    preds, states = [], []
    cur_dlg, state = None, {}
    for r in raw:
        if r["dialogue"] != cur_dlg:
            cur_dlg, state = r["dialogue"], {}
        for k, d in zip(r["kcs"], r["logit_gaps"]):
            z = expit(a * d + b) if calibrated else expit(d)
            if filtered:
                mk = state.get(k, m0)
                state[k] = (1 - lam) * (mk + (1 - mk) * pL) + lam * z
            else:
                state[k] = z
        mbar = float(np.mean([state[k] for k in r["kcs"]]))
        preds.append((1 - s) * mbar + g * (1 - mbar))
        states.append(dict(state))
    return np.asarray(preds), states

y = np.asarray([r["label"] for r in raw], float)
pA, sA = replay(True, True)

# --- tu kiem o muc do chinh xac may: phai trung tung du doan da phat hanh ---
rel = pd.read_csv(rp("results", "03_ccmf", "test_predictions_qlora_seed221.csv"))
assert [(str(d), int(t)) for d, t in zip(rel.dialogue, rel.turn_id)] == \
       [(str(r["dialogue"]), int(r["turn_id"])) for r in raw], "thu tu dong khong khop"
dmax = float(np.abs(pA - rel.mean_bkt_none.values).max())
assert dmax < 1e-9, "replay lech %.2e so voi cot mean_bkt_none da phat hanh" % dmax
auc, ll = roc_auc_score(y, pA), log_loss(y, np.clip(pA, 1e-12, 1 - 1e-12))
print("Tu kiem OK: replay trung cot mean_bkt_none da phat hanh (max|diff| = %.1e); "
      "AUC %.4f / log-loss %.4f" % (dmax, auc, ll))

pB, sB = replay(False, True)
pC, sC = replay(True, False)
pC2, sC2 = replay(False, False)

def mta_choice(state):
    """MTA voi do thi rong = Advance den KC chua dat nguong, gan nguong nhat."""
    unm = {k: v for k, v in state.items() if v < TAU}
    return max(unm, key=unm.get) if unm else None

def compare(sX, sY, labX, labY):
    cx = [mta_choice(t) for t in sX]; cy = [mta_choice(t) for t in sY]
    both = [(u, v) for u, v in zip(cx, cy) if u is not None and v is not None]
    same = float(np.mean([u == v for u, v in both]))
    # quyet dinh thu hai: phan loai da/chua thanh thao tung KC o nguong tau
    tot = agr = 0
    for tx, ty in zip(sX, sY):
        for k in tx:
            if k in ty:
                tot += 1; agr += int((tx[k] >= TAU) == (ty[k] >= TAU))
    return dict(comparison="%s vs %s" % (labX, labY), n_turns_both_define_target=len(both),
                mta_same_kc=same, mta_differs=1 - same,
                n_kc_turn_pairs=tot, mastered_flag_agreement=agr / tot)

cmps = [compare(sA, sB, "CCMF day du", "bo Platt (chi bo loc)"),
        compare(sA, sC, "CCMF day du", "xac suat tho cuoi cung moi KC"),
        compare(sA, sC2, "CCMF day du", "sigmoid tho cuoi cung, khong bo loc")]
res = dict(tau=TAU, params_used=P, self_check=dict(auc=auc, auc_released=float(row.auc),
           ll=ll, ll_released=float(row.ll_test)), comparisons=cmps)
with open(op("out_02_calibration_downstream.json"), "w") as f: json.dump(res, f, indent=1)
for c in cmps:
    print("%-45s | MTA chon cung KC %.1f%% (%d luot) | co/chua thanh thao khop %.1f%% (%d cap KC-luot)"
          % (c["comparison"], 100 * c["mta_same_kc"], c["n_turns_both_define_target"],
             100 * c["mastered_flag_agreement"], c["n_kc_turn_pairs"]))
print("-> out_02_calibration_downstream.json")
