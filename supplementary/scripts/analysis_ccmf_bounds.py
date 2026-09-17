"""CCMF Nelder-Mead starts + box-boundary sensitivity (VALIDATION ONLY).

The paper notes that p_L, m0 and s reach their box bounds and are not
interpreted. This checks whether that box is binding: refit the primary variant
(mean_bkt, mode none) from the six pre-registered starts under the original raw
box and under a widened box, and compare validation log-loss / AUC.

Test predictions are never touched: the released protocol pre-registered a
single test evaluation, which has already been spent (03_ccmf/TEST_EVALUATED).
The engine below follows the pre-registered formula and is verified against the
released validation log-loss and AUC at the released raw vector before any refit.
"""
import os as _os
def _find_root():
    """Locate the package root (the directory holding results/ and raw_runs/).

    Works when run as a script, exec'd, or pasted into a notebook cell.
    Override with the MATHKT_ROOT environment variable.
    """
    env = _os.environ.get("MATHKT_ROOT")
    if env:
        return _os.path.abspath(env)
    seeds = [_os.getcwd()]
    try:
        seeds.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    except NameError:
        pass
    for seed in seeds:
        d = seed
        for _ in range(6):
            if _os.path.isdir(_os.path.join(d, "results")) and _os.path.isdir(_os.path.join(d, "raw_runs")):
                return d
            d = _os.path.dirname(d)
    raise RuntimeError("package root not found; set MATHKT_ROOT to the folder containing results/ and raw_runs/")
_ROOT = _find_root()
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score, log_loss

B   = _os.path.join(_ROOT, "results")
RUN = _os.path.join(_ROOT, "raw_runs")
CLIP = (1e-6, 1 - 1e-6)

# the six pre-registered starts, verbatim from the PREREGISTRATION text embedded
# in 08_scripts/ccmf_redesign.py (raw order [r_a,r_b,r_lambda,r_mu,r_pL,r_m0,r_s,r_g])
CCMF_STARTS_8 = [
    [0, 0, 0, -2, -3, -0.4, -2, -2],
    [0.3, -0.2, -0.5, -1, -2.5, 0, -1.5, -2.5],
    [-0.3, 0.2, 0.5, -2.5, -3.5, -1, -2.5, -1.5],
    [0, 0, -1, -0.5, -2, -0.8, -1, -3],
    [0.6, -0.4, 1, -3, -4, 0.5, -3, -1],
    [0, 0, 6, 0, -3, 0, -8, -8],   # S6: nested mean-pooling start
]
FREE = [0, 1, 2, 4, 5, 6, 7]        # mode "none" drops index 3 (r_mu)

def unpack(raw8):
    r = np.asarray(raw8, float)
    a = float(np.exp(np.clip(r[0], -3, 3)))
    b = float(4 * np.tanh(r[1]))
    lam, mu, pL, m0 = (float(expit(x)) for x in r[2:6])
    s, g = (float(0.49 * expit(x)) for x in r[6:8])
    return a, b, lam, mu, pL, m0, s, g

def load_val():
    rows = json.load(open(f"{RUN}/mathdial/qlora/seed221/val/raw_logit_gaps.json"))
    return rows, np.array([int(r["label"]) for r in rows])

def predict(raw7, rows):
    raw8 = list(raw7[:3]) + [0.0] + list(raw7[3:])
    a, b, lam, mu, pL, m0, s, g = unpack(raw8)
    state, cur, out = {}, None, []
    for r in rows:
        if r["dialogue"] != cur:
            cur, state = r["dialogue"], {}
        z = expit(a * np.asarray(r["logit_gaps"], float) + b)
        mt = []
        for kc, zk in zip(r["kcs"], z):
            m = state.get(kc, m0)
            prior = m + (1 - m) * pL
            mtk = (1 - lam) * prior + lam * zk
            mt.append(mtk)
            state[kc] = mtk           # mode none: commit the blended state
        P = float(np.mean(mt))
        out.append((1 - s) * P + g * (1 - P))
    return np.clip(np.array(out), *CLIP)

def objective(raw7, rows, y, box):
    if np.any(raw7 < box[:, 0] - 1e-12) or np.any(raw7 > box[:, 1] + 1e-12):
        return 1e6
    return float(log_loss(y, predict(raw7, rows), labels=[0, 1]))

def make_box(expit_lim):
    lo = [-3, -3] + [-expit_lim] * 5
    hi = [3, 3] + [expit_lim] * 5
    return np.array(list(zip(lo, hi)), float)

def fit(rows, y, box, starts):
    recs = []
    for i, s8 in enumerate(starts, 1):
        x0 = np.array([s8[j] for j in FREE], float)
        x0 = np.clip(x0, box[:, 0], box[:, 1])
        simplex = [x0] + [
            (x0 + 0.5 * np.eye(len(x0))[k]) if (x0 + 0.5 * np.eye(len(x0))[k])[k] <= box[k, 1]
            else (x0 - 0.5 * np.eye(len(x0))[k]) for k in range(len(x0))]
        best = minimize(objective, x0, args=(rows, y, box), method="Nelder-Mead",
                        bounds=[tuple(b) for b in box],
                        options={"maxiter": 4000, "maxfev": 8000, "xatol": 1e-4,
                                 "fatol": 1e-7, "adaptive": True,
                                 "initial_simplex": np.array(simplex)})
        # pre-registered polish: fresh simplex of step 0.05, accept iff it improves
        for _ in range(2):
            sx = [best.x] + [
                (best.x + 0.05 * np.eye(len(x0))[k]) if (best.x + 0.05 * np.eye(len(x0))[k])[k] <= box[k, 1]
                else (best.x - 0.05 * np.eye(len(x0))[k]) for k in range(len(x0))]
            pol = minimize(objective, best.x, args=(rows, y, box), method="Nelder-Mead",
                           bounds=[tuple(b) for b in box],
                           options={"maxiter": 4000, "maxfev": 8000, "xatol": 1e-4,
                                    "fatol": 1e-7, "adaptive": True,
                                    "initial_simplex": np.array(sx)})
            if pol.fun < best.fun - 1e-7:
                best = pol
            else:
                break
        recs.append({"start": i, "ll_val": float(best.fun), "x": best.x.copy()})
    best = min(recs, key=lambda r: r["ll_val"])
    n_agree = sum(1 for r in recs if r["ll_val"] <= best["ll_val"] + 1e-4)
    return best, recs, n_agree
