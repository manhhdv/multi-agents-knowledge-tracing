"""(a) SAVA cascade latency on the real MathDial test turns, measured by rerunning
the released cascade on the released turn text; the rerun's verdicts are checked
against the recorded ones first, so the timing is for the exact evaluated workload.
(b) Blind human-verification package anchoring the agreement figures to truth.
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
import json, re, time, numpy as np, pandas as pd
B   = _os.path.join(_ROOT, "results")
RUN = _os.path.join(_ROOT, "raw_runs")

SAVA_QUESTION_GUARD_GLOBAL = True
# released cascade, verbatim: the SAVA cell of the replication notebook
_NB = json.load(open(_os.path.join(_ROOT, "replication", "colab_full_framework_replication.ipynb")))
exec(next("".join(c["source"]) for c in _NB["cells"]
          if c["cell_type"] == "code" and "def sava_verify" in "".join(c["source"])))

turns = json.load(open(f"{RUN}/mathdial/qlora/seed221/test/raw_logit_gaps.json"))
rec   = pd.read_csv(f"{B}/02_sava/llm_judge_verdicts_test.csv")
rec["dialogue"] = rec["dialogue"].astype(str); rec["turn_id"] = rec["turn_id"].astype(int)
idx = rec.set_index(["dialogue","turn_id"])

# ---- (a) rerun + timing -----------------------------------------------------
out, times = [], []
for t in turns:
    st, rf = t["student_text"], t["reference"]
    t0 = time.perf_counter()
    v = sava_verify(st, rf, question_guard_global=True)
    times.append(time.perf_counter() - t0)
    out.append({"dialogue": str(t["dialogue"]), "turn_id": int(t["turn_id"]),
                "label": int(t["label"]), "final_turn": bool(t["final_turn"]),
                "student_text": st, "reference": rf,
                "sava_rerun": v["verdict"], "stage_rerun": v["stage"]})
df = pd.DataFrame(out)
m = df.merge(rec[["dialogue","turn_id","sava","sava_stage","numeric_regex",
                  "llm_judge_binary","llm_judge_three_way"]], on=["dialogue","turn_id"], how="left")
agree_verdict = float((m.sava_rerun == m.sava).mean())
agree_stage   = float((m.stage_rerun == m.sava_stage).mean())
ms = np.array(times) * 1000.0
cost = {"n_turns": int(len(df)), "rerun_matches_recorded_verdict": agree_verdict,
        "rerun_matches_recorded_stage": agree_stage,
        "mean_ms_per_turn": float(ms.mean()), "median_ms_per_turn": float(np.median(ms)),
        "p95_ms_per_turn": float(np.percentile(ms, 95)), "max_ms_per_turn": float(ms.max()),
        "total_s_all_turns": float(ms.sum()/1000.0),
        "note": "single CPU core, no GPU, no network; SymPy cascade only"}
by_stage = (pd.DataFrame({"stage": m.stage_rerun, "ms": ms})
            .groupby("stage")["ms"].agg(["size","mean","median"]).reset_index()
            .sort_values("size", ascending=False))

# ---- (b) disagreement structure + blind sample ------------------------------
m["sava3"] = m.sava_rerun
m["judge"] = m.llm_judge_binary
def pat(r):
    parts = []
    parts.append("SAVA=" + ("abstain" if r.sava3 == "undetermined" else ("hit" if
                 (r.sava3 == "correct") == (r.label == 1) else "miss")))
    parts.append("JUDGE=" + ("hit" if (r.judge == "correct") == (r.label == 1) else "miss"))
    return " | ".join(parts)
m["pattern"] = m.apply(pat, axis=1)
patterns = m.pattern.value_counts().rename_axis("pattern").reset_index(name="n_turns")
patterns["share"] = patterns.n_turns / len(m)

# stratified blind sample: every pattern, proportional but with a floor, so the
# cells that decide the conclusions (judge/SAVA disagreeing with the reference)
# are all represented.
N_TARGET, FLOOR = 120, 15
rng = np.random.default_rng(221)
picks = []
for p, grp in m.groupby("pattern"):
    k = max(FLOOR, int(round(N_TARGET * len(grp) / len(m))))
    k = min(k, len(grp))
    picks.append(grp.sample(k, random_state=221))
samp = pd.concat(picks).sample(frac=1.0, random_state=221).reset_index(drop=True)
samp.insert(0, "item_id", [f"HV{i:03d}" for i in range(1, len(samp)+1)])

sheet = samp[["item_id","reference","student_text"]].copy()
sheet["annotator_verdict (correct/incorrect/undetermined)"] = ""
sheet["annotator_notes"] = ""
key = samp[["item_id","dialogue","turn_id","final_turn","label","sava_rerun",
            "stage_rerun","numeric_regex","llm_judge_binary","llm_judge_three_way","pattern"]]

# precision of the anchor: binomial half-width for the reference label's own accuracy
def halfwidth(n, p=0.75):
    return float(1.96*np.sqrt(p*(1-p)/n))
sizing = {"sample_n": int(len(samp)),
          "ci_halfwidth_at_sample_n": halfwidth(len(samp)),
          "n_for_5pp_halfwidth": int(np.ceil(1.96**2*0.75*0.25/0.05**2)),
          "n_for_1pp_halfwidth": int(np.ceil(1.96**2*0.75*0.25/0.01**2))}
