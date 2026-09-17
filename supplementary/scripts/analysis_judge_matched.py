"""Coverage-matched comparison of SAVA, the numeric regex and the 8B LLM judge.

Reproduces the matched-turn figures reported in Sec. 4.3, the Abstract and the
Conclusion of the revised manuscript:

  * SAVA vs regex on the turns BOTH judge   -> 74.4% vs 74.4%, one discordant turn
  * LLM judge vs SAVA on the turns SAVA judges -> +3.9 points, McNemar p=8.8e-4,
    dialogue-level bootstrap 95% CI [+1.3, +6.2]
  * decomposition of the judge's 75.4% into 78.3% (SAVA-judged) and 65.0%
    (SAVA-declined), where the regex reaches 48.0% and neither beats the 66.6%
    majority class; a judge fallback for full coverage gives 72.4%

Input: results/02_sava/llm_judge_verdicts_test.csv (released with the paper).
CPU only, no GPU and no API key. Runtime ~2 min (the bootstrap dominates).
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
import os, json
import numpy as np, pandas as pd
from scipy.stats import binomtest

RESULTS = _os.path.join(_ROOT, "results")
N_BOOT, SEED = 2000, 221


def load():
    df = pd.read_csv(os.path.join(RESULTS, "02_sava", "llm_judge_verdicts_test.csv"))
    df["abstain"] = df["sava"] == "undetermined"
    for col, name in [("sava", "sava_ok"), ("numeric_regex", "regex_ok"), ("llm_judge_binary", "judge_ok")]:
        df[name] = ((df[col] == "correct").astype(int) == df["label"]).astype(int)
        df[name + "_judged"] = df[col].isin(["correct", "incorrect"])
    return df


def run():
    df = load()
    judged, declined = df[~df.abstain], df[df.abstain]

    # (1) SAVA vs regex on the turns both judge
    both = judged[judged.regex_ok_judged]
    b1 = int(((both.sava_ok == 1) & (both.regex_ok == 0)).sum())
    c1 = int(((both.sava_ok == 0) & (both.regex_ok == 1)).sum())

    # (2) judge vs SAVA on the turns SAVA judges
    b2 = int(((judged.judge_ok == 1) & (judged.sava_ok == 0)).sum())
    c2 = int(((judged.judge_ok == 0) & (judged.sava_ok == 1)).sum())
    rng = np.random.default_rng(SEED)
    dlg = judged.dialogue.unique()
    groups = {d: g for d, g in judged.groupby("dialogue")}
    boot = []
    for _ in range(N_BOOT):
        s = pd.concat([groups[d] for d in rng.choice(dlg, len(dlg), replace=True)])
        boot.append(s.judge_ok.mean() - s.sava_ok.mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])

    # (3) decomposition and the cost of full coverage
    regex_declined = declined[declined.regex_ok_judged]
    maj = float(max(declined.label.mean(), 1 - declined.label.mean()))
    blend = (judged.sava_ok.mean() * len(judged) + declined.judge_ok.mean() * len(declined)) / len(df)

    out = {
        "n_turns": int(len(df)),
        "sava_coverage": float(len(judged) / len(df)),
        "matched_sava_vs_regex": {
            "n": int(len(both)), "sava": float(both.sava_ok.mean()), "regex": float(both.regex_ok.mean()),
            "diff": float(both.sava_ok.mean() - both.regex_ok.mean()),
            "mcnemar_b_sava_only": b1, "mcnemar_c_regex_only": c1,
            "mcnemar_p": float(binomtest(b1, b1 + c1, 0.5).pvalue) if b1 + c1 else float("nan")},
        "matched_judge_vs_sava": {
            "n": int(len(judged)), "judge": float(judged.judge_ok.mean()), "sava": float(judged.sava_ok.mean()),
            "diff": float(judged.judge_ok.mean() - judged.sava_ok.mean()),
            "mcnemar_b_judge_only": b2, "mcnemar_c_sava_only": c2,
            "mcnemar_p": float(binomtest(b2, b2 + c2, 0.5).pvalue),
            "bootstrap_ci": [float(lo), float(hi)], "n_boot": N_BOOT},
        "declined_turns": {
            "n": int(len(declined)), "majority_class": maj,
            "judge": float(declined.judge_ok.mean()),
            "regex": float(regex_declined.regex_ok.mean()), "n_regex_judged": int(len(regex_declined)),
            "share_containing_a_digit": None},
        "judge_overall_decomposition": {
            "on_sava_judged": float(judged.judge_ok.mean()),
            "on_sava_declined": float(declined.judge_ok.mean()),
            "weighted_overall": float((judged.judge_ok.mean() * len(judged)
                                       + declined.judge_ok.mean() * len(declined)) / len(df))},
        "sava_with_judge_fallback_full_coverage": float(blend),
    }
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=1))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_matched_results.json"), "w") as f:
        json.dump(r, f, indent=1)
