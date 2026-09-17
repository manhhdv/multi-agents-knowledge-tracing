"""LLM-judge baseline for SAVA on the 1,985 MathDial test turns.

usage: analyze_llm_judge_baseline.py <export.b64> <expected sha256>
Writes 02_sava/llm_judge_verdicts_test.csv, 02_sava/llm_judge_agreement.csv and 02_sava/llm_judge_tests.json.
"""
import base64, gzip, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
b64_path, expected = Path(sys.argv[1]), sys.argv[2]
raw = gzip.decompress(base64.b64decode(b64_path.read_text().strip()))
assert hashlib.sha256(raw).hexdigest() == expected, "export does not match the Colab hash"
export = json.loads(raw)
assert export["turn_key_sha256"] == "8bc0df3ece2c38a8dbf75eb63e13b1da9e8885e122f709326448f530fb073c66"

real = pd.read_csv(PKG / "02_sava" / "real_turns_mathdial_test.csv", dtype={"dialogue": str})
assert len(real) == len(export["three_way"]) == len(export["binary"])
real["llm_judge_three_way"] = [v if v != "unparsed" else "undetermined" for v in export["three_way"]]
real["llm_judge_binary"] = [v if v != "unparsed" else "undetermined" for v in export["binary"]]
real["llm_judge_three_way_raw"], real["llm_judge_binary_raw"] = export["raw_three_way"], export["raw_binary"]
real.to_csv(PKG / "02_sava" / "llm_judge_verdicts_test.csv", index=False)
unparsed = {k: int(sum(v == "unparsed" for v in export[k])) for k in ["three_way", "binary"]}

METHODS = ["sava", "numeric_regex", "llm_judge_three_way", "llm_judge_binary"]


def stats(part, m):
    spoken = part[part[m] != "undetermined"]
    agree = (spoken[m] == "correct").astype(int) == spoken.label
    inc = part[part[m] == "incorrect"]
    return {"coverage": len(spoken) / len(part), "agreement_when_judging": agree.mean() if len(spoken) else np.nan,
            "agreeing_share": agree.sum() / len(part), "gate_share": len(inc) / len(part),
            "gate_with_correct_label": (inc.label == 1).mean() if len(inc) else np.nan}


rows = []
for subset, part in [("all", real), ("final", real[real.final_turn]), ("non_final", real[~real.final_turn])]:
    for m in METHODS:
        rows.append({"subset": subset, "method": m, "n": len(part), **stats(part, m)})
table = pd.DataFrame(rows)
table.to_csv(PKG / "02_sava" / "llm_judge_agreement.csv", index=False)

rng = np.random.default_rng(221)
groups = [g.index.to_numpy() for _, g in real.groupby("dialogue")]
tests = {"unparsed_outputs": unparsed}
for judge in ["llm_judge_three_way", "llm_judge_binary"]:
    for ref in ["sava", "numeric_regex"]:
        point = {k: stats(real, judge)[k] - stats(real, ref)[k] for k in ["agreement_when_judging", "agreeing_share"]}
        boot = {k: [] for k in point}
        for _ in range(1000):
            part = real.loc[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])]
            a, b = stats(part, judge), stats(part, ref)
            for k in point:
                boot[k].append(a[k] - b[k])
        tests[f"{judge}_minus_{ref}"] = {k: {"delta": point[k], "ci95": np.quantile(boot[k], [0.025, 0.975]).tolist()} for k in point}
(PKG / "02_sava" / "llm_judge_tests.json").write_text(json.dumps(tests, indent=2))

pd.set_option("display.width", 200)
print(table.round(4).to_string(index=False))
print(json.dumps(tests, indent=1))
