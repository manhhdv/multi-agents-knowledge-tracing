"""Quantify the overlap between SAVA's constructed benchmark and the cascade's
own conclusion-cue extraction rules (the circularity the Limitations flags).

The cue regex is taken verbatim from the released cascade (notebook cell 22).
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
import re, json, pandas as pd, numpy as np
B   = _os.path.join(_ROOT, "results")

# verbatim from the released SAVA cascade
_CONCL_CUE = re.compile(r"\b(so|therefore|thus|hence|in total|altogether|the answer|"
                        r"answer is|total is|which is|that (?:is|means)|equals)\b")

def run():
    it = pd.read_csv(f"{B}/02_sava/constructed_benchmark_items.csv")
    it["cand"] = it["candidate"].astype(str)
    it["has_cue"] = it["cand"].str.lower().apply(lambda s: bool(_CONCL_CUE.search(s)))
    it["correct"] = (it["sava"] == it["gold"])
    it["correct_noCue"] = (it["sava_no_conclusion_cue"] == it["gold"])
    it["correct_regex"] = (it["numeric_regex"] == it["gold"])

    overall = {
        "n_items": int(len(it)),
        "n_references": int(it["reference_id"].nunique()),
        "n_transformations": int(it["transformation"].nunique()),
        "share_items_containing_a_cascade_cue": float(it["has_cue"].mean()),
        "sava_acc_all": float(it["correct"].mean()),
        "sava_acc_cue_items": float(it.loc[it.has_cue, "correct"].mean()),
        "sava_acc_noncue_items": float(it.loc[~it.has_cue, "correct"].mean()),
        "n_cue_items": int(it.has_cue.sum()),
        "n_noncue_items": int((~it.has_cue).sum()),
        "sava_noCueRule_acc_cue_items": float(it.loc[it.has_cue, "correct_noCue"].mean()),
        "sava_noCueRule_acc_noncue_items": float(it.loc[~it.has_cue, "correct_noCue"].mean()),
        "regex_acc_cue_items": float(it.loc[it.has_cue, "correct_regex"].mean()),
        "regex_acc_noncue_items": float(it.loc[~it.has_cue, "correct_regex"].mean()),
    }

    # per-transformation: which families are cue-bearing by construction
    per = (it.groupby("transformation")
             .agg(n=("cand", "size"), share_with_cue=("has_cue", "mean"),
                  sava_acc=("correct", "mean"),
                  sava_noCueRule_acc=("correct_noCue", "mean"),
                  regex_acc=("correct_regex", "mean"))
             .sort_values("share_with_cue", ascending=False).reset_index())

    # how often the cue PATH is what actually decides a verdict, benchmark vs real
    stage = pd.read_csv(f"{B}/02_sava/verdicts_by_stage.csv")
    rt = pd.read_csv(f"{B}/02_sava/real_turns_mathdial_test.csv")
    bench_cue_path = float(it["sava_stage"].astype(str).str.startswith("conclusion_").mean())
    real_cue_path = float(rt["sava_stage"].astype(str).str.startswith("conclusion_").mean())
    # real turns where the cue rule changes the verdict vs the ablated cascade
    changed = (rt["sava"] != rt["sava_no_conclusion_cue"])
    overall.update({
        "benchmark_share_decided_by_cue_path": bench_cue_path,
        "real_turn_share_decided_by_cue_path": real_cue_path,
        "real_turn_share_cue_rule_changes_verdict": float(changed.mean()),
        "n_real_turns": int(len(rt)),
    })
    return overall, per, stage
