"""MTA planner under noise calibrated to CCMF's real estimation error.

Reuses the paper's own planner dynamics (notebook cell 28) verbatim; the only
change is the observation model: instead of obs = mastery + N(0, sigma) i.i.d.,
we inject the three properties measured on CCMF's real MathDial estimates.
Validation/test predictions are read from the released run; nothing is refitted.
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
from dataclasses import dataclass, asdict, replace
from scipy.stats import wilcoxon

B   = _os.path.join(_ROOT, "results")
PRIMARY_SEED, PLANNER_LEARNERS, PLANNER_GRID_LEARNERS = 221, 500, 200

# ---------------------------------------------------------------- measurement
def measure_ccmf_error():
    """Three structural properties of CCMF's per-turn / per-KC estimates."""
    pred = pd.read_csv(f"{B}/03_ccmf/table5_reference_inputs/"
                       "ccmf_test_predictions_mathdial_qlora_seed221.csv")
    y, p = pred["label"].to_numpy(float), pred["ccmf_no_update"].to_numpy(float)

    # (1) noise scale: excess Brier over the model's own predicted Bernoulli
    # variance. If p_hat equalled true mastery m*, E[(p-y)^2] = E[p(1-p)].
    brier = float(np.mean((p - y) ** 2))
    bern = float(np.mean(p * (1 - p)))
    sigma = float(np.sqrt(max(0.0, brier - bern)))

    # (2) temporal dependence: lag-1 autocorrelation of a KC's mastery series
    # within a dialogue (CCMF keeps 1-lambda of the stored state each turn).
    il = pd.read_csv(f"{B}/04_integrated_loop/"
                     "integrated_run_mathdial_qlora_seed221.csv")
    series = {}
    for _, r in il.iterrows():
        for kc, m in json.loads(r["kc_mastery_before"]).items():
            series.setdefault((r["dialogue"], kc), []).append(float(m))
    pairs = [(s[i], s[i + 1]) for s in series.values() if len(s) > 1
             for i in range(len(s) - 1)]
    a, b = np.array([x for x, _ in pairs]), np.array([y_ for _, y_ in pairs])
    rho = float(np.corrcoef(a, b)[0, 1])

    # (3) first-occurrence behaviour. Parameters come from the PRIMARY variant
    # of the released redesign (test_results.csv, is_primary=True), i.e. the row
    # reported as "CCMF (ours)" in the paper -- not the superseded design in
    # table5_reference_inputs. At a KC's first occurrence in a dialogue the
    # stored state is m0, which sits at its upper box bound, so
    #   m~ = (1-lambda)*prior + lambda*z  ==  z + w*(prior - z),  w = 1-lambda
    # i.e. the first estimate is shrunk TOWARDS ~1.0 by weight w, an upward
    # bias that zero-mean Gaussian noise cannot represent.
    tr = pd.read_csv(f"{B}/03_ccmf/test_results.csv")
    row = tr[(tr.dataset == "mathdial") & (tr.model == "qlora")
             & (tr.variant == "mean_bkt_none")].iloc[0]
    cc = json.loads(row["params_natural_json"])
    lam, m0, pL = float(cc["lambda"]), float(cc["m0"]), float(cc["p_L"])
    prior = m0 + (1 - m0) * pL
    w = 1 - lam

    first, later = [], []
    for s in series.values():
        first.append(s[0])
        later.extend(s[1:])
    n_series = len(series)
    n_single = sum(1 for s in series.values() if len(s) == 1)
    return {"brier": brier, "bernoulli_var": bern, "sigma_hat": sigma,
            "estimate_lag1_autocorr": rho,
            "lambda": lam, "m0": m0, "p_learn": pL, "first_prior": prior,
            "first_shrink_weight": w,
            "error_persistence_coef": w,
            "n_kc_series": n_series,
            "n_single_observation_series": n_single,
            "single_observation_share": n_single / n_series,
            "mean_first_estimate": float(np.mean(first)),
            "mean_later_estimate": float(np.mean(later)),
            "at_bound_flags": row["at_bound"]}

# ------------------------------------------------------- paper's planner code
@dataclass(frozen=True)
class SimConfig:
    n_kcs: int = 12
    turns: int = 120
    tau: float = 0.8
    gain: float = 0.185
    decay: float = 0.004
    readiness: float = 0.8
    noise: float = 0.0
    graph: str = "chains"
    prior: str = "uniform"
    noise_model: str = "iid"      # added: iid | ar1 | ar1_firstbias
    rho: float = 0.0              # added
    first_bias: float = 0.0       # added

PAPER_CONFIG = SimConfig()

def make_prerequisites(cfg, rng):
    if cfg.graph == "chains":
        return {k: ([] if k % 4 == 0 else [k - 1]) for k in range(cfg.n_kcs)}
    prerequisites = {0: []}
    for j in range(1, cfg.n_kcs):
        size = min(int(rng.integers(0, 3)), j)
        prerequisites[j] = sorted(int(x) for x in rng.choice(j, size=size, replace=False))
    return prerequisites

def random_topological_order(prerequisites, rng):
    remaining, order = set(prerequisites), []
    while remaining:
        ready = sorted(k for k in remaining if all(p not in remaining for p in prerequisites[k]))
        k = int(rng.choice(ready))
        order.append(k); remaining.remove(k)
    return order

def mta_choose(obs, prerequisites, memory, tau, maintain=True, nearest=True, use_prerequisites=True):
    ever = memory.setdefault("ever", set())
    ever.update(int(k) for k in np.flatnonzero(obs >= tau))
    if maintain:
        slipped = [k for k in sorted(ever) if obs[k] < tau]
        if slipped:
            memory["current"] = max(slipped, key=lambda k: obs[k]); return memory["current"]
    current = memory.get("current")
    if current is not None and current not in ever:
        return current
    todo = [k for k in range(len(obs)) if k not in ever]
    if not todo:
        below = [k for k in range(len(obs)) if obs[k] < tau]
        return max(below, key=lambda k: obs[k]) if below else None
    ready = [k for k in todo if not use_prerequisites or all(p in ever for p in prerequisites.get(k, []))]
    pool = ready or todo
    memory["current"] = max(pool, key=lambda k: obs[k]) if nearest else min(pool, key=lambda k: obs[k])
    return memory["current"]

def unmastered(obs, tau):
    return np.flatnonzero(obs < tau)

POLICIES = {
    "greedy_lowest_first": lambda o, pre, mem, rng, t, cfg: (lambda u: None if len(u) == 0 else int(u[np.argmin(o[u])]))(unmastered(o, cfg.tau)),
    "random_unmastered": lambda o, pre, mem, rng, t, cfg: (lambda u: None if len(u) == 0 else int(rng.choice(u)))(unmastered(o, cfg.tau)),
    "fixed_curriculum_no_skip": lambda o, pre, mem, rng, t, cfg: None if len(unmastered(o, cfg.tau)) == 0 else int(t % cfg.n_kcs),
    "prereq_lowest_first_previous": lambda o, pre, mem, rng, t, cfg: (lambda u: None if len(u) == 0 else (lambda pool: int(pool[np.argmin(o[pool])]))(
        [k for k in u if all(o[p] >= cfg.tau for p in pre[k])] or list(u)))(unmastered(o, cfg.tau)),
    "filtered_curriculum_index_order": lambda o, pre, mem, rng, t, cfg: (lambda u: None if len(u) == 0 else int(u[0]))(unmastered(o, cfg.tau)),
    "filtered_curriculum_random_topo": lambda o, pre, mem, rng, t, cfg: next((k for k in mem["order"] if o[k] < cfg.tau), None),
    "mta": lambda o, pre, mem, rng, t, cfg: mta_choose(o, pre, mem, cfg.tau),
}
MAIN_POLICIES = list(POLICIES)

def simulate_planner(policy, learner_seed, cfg=PAPER_CONFIG):
    rng = np.random.default_rng(learner_seed)
    observation_rng = np.random.default_rng(learner_seed + 1_000_000)
    graph_rng = np.random.default_rng(learner_seed + 2_000_000)
    if cfg.prior == "uniform":
        mastery = rng.uniform(0.05, 0.40, cfg.n_kcs)
    else:
        mastery = np.where(rng.random(cfg.n_kcs) < 0.3, rng.uniform(0.60, 0.95, cfg.n_kcs), rng.uniform(0.05, 0.40, cfg.n_kcs))
    difficulty = rng.uniform(0.7, 1.3, cfg.n_kcs)
    prerequisites = make_prerequisites(cfg, graph_rng)
    memory = {"order": random_topological_order(prerequisites, np.random.default_rng(learner_seed + 3_000_000))}
    eps = np.zeros(cfg.n_kcs)          # AR(1) state
    seen = np.zeros(cfg.n_kcs, bool)   # for the first-occurrence offset
    for turn in range(cfg.turns):
        if cfg.noise:
            if cfg.noise_model == "iid":
                err = observation_rng.normal(0, cfg.noise, cfg.n_kcs)
            else:  # AR(1) with matched marginal sd
                innov = observation_rng.normal(0, cfg.noise * np.sqrt(1 - cfg.rho ** 2), cfg.n_kcs)
                eps = cfg.rho * eps + innov
                err = eps.copy()
            obs = mastery + err
            if cfg.noise_model == "ar1_firstbias":
                obs = np.where(seen, obs, obs + cfg.first_bias * (1.0 - obs))
            obs = np.clip(obs, 0, 1)
        else:
            obs = mastery.copy()
        kc = POLICIES[policy](obs, prerequisites, memory, rng, turn, cfg)
        if kc is None:
            break
        seen[kc] = True
        readiness = 1.0 if all(mastery[p] >= cfg.tau for p in prerequisites[kc]) else cfg.readiness
        gain = (cfg.gain / difficulty[kc]) * readiness * (1 - mastery[kc])
        mastery *= (1 - cfg.decay)
        mastery[kc] = min(1.0, mastery[kc] + gain)
    return float(np.mean(mastery >= cfg.tau))

EVAL_SEEDS = [PRIMARY_SEED + i for i in range(PLANNER_LEARNERS)]
GRID_SEEDS = [PRIMARY_SEED + i for i in range(PLANNER_GRID_LEARNERS)]

def evaluate_policies(policies, cfg, seeds):
    return {p: np.array([simulate_planner(p, s, cfg) for s in seeds]) for p in policies}

def summarize(values, cfg_label, reference="mta"):
    rng = np.random.default_rng(PRIMARY_SEED)
    rows = []
    for policy, v in values.items():
        boot = [rng.choice(v, len(v), replace=True).mean() for _ in range(1000)]
        diff = values[reference] - v
        p_value = wilcoxon(values[reference], v).pvalue if policy != reference and np.any(diff) else float("nan")
        rows.append({"observation_model": cfg_label, "policy": policy, "n_learners": len(v),
                     "fraction_mastered": v.mean(), "ci_low": np.quantile(boot, 0.025),
                     "ci_high": np.quantile(boot, 0.975), "mta_minus_policy": diff.mean(),
                     "wilcoxon_p_vs_mta": p_value})
    return pd.DataFrame(rows)
