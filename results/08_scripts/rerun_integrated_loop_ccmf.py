"""Re-run the MathDial integrated loop with CCMF (redesign `mean_bkt_sava`).

Same control flow as notebook cell 30: CCMF predicts the turn, the stored SAVA verdict updates mastery for later
turns, MTA (empty prerequisite graph, tau = 0.8) picks the next KC. Checks: predictions equal
03_ccmf/test_predictions_qlora_seed221.csv[mean_bkt_sava]; SAVA verdicts, stages and gate are unchanged.
Writes 04_integrated_loop/integrated_run_mathdial_qlora_seed221.csv and gate_summary.csv.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

PKG = Path(__file__).resolve().parents[1]
PAPER = PKG.parent
OLD = PKG / "04_integrated_loop/sava_verdicts_test.csv"
ROWS = PAPER / "20260912T221028Z/mathdial/qlora/seed221/test/raw_logit_gaps.json"
PARAMS = PKG / "03_ccmf/params_qlora_seed221.json"
PRED = PKG / "03_ccmf/test_predictions_qlora_seed221.csv"
TAU = 0.8
Y = {"correct": 1, "incorrect": 0, "undetermined": None}


def mta_choose(obs, prerequisites, memory, tau, maintain=True, nearest=True, use_prerequisites=True):
    """Verbatim copy of notebook cell 28."""
    ever = memory.setdefault("ever", set())
    ever.update(int(k) for k in np.flatnonzero(obs >= tau))
    if maintain:
        slipped = [k for k in sorted(ever) if obs[k] < tau]
        if slipped:
            memory["current"] = max(slipped, key=lambda k: obs[k])
            return memory["current"]
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


def main():
    p = json.loads(PARAMS.read_text())["variants"]["mean_bkt_sava"]["natural"]
    a, b, lam, mu, p_l, m0, s, g = (p[k] for k in ["a", "b", "lambda", "mu", "p_L", "m0", "s", "g"])
    rows = json.loads(ROWS.read_text())
    old = pd.read_csv(OLD, dtype={"dialogue": str})
    assert len(old) == len(rows)
    records, state, planner_memory, current_dialogue = [], {}, {}, None
    for row, (_, o) in zip(rows, old.iterrows()):
        assert str(row["dialogue"]) == o.dialogue and int(row["turn_id"]) == int(o.turn_id)
        if str(row["dialogue"]) != current_dialogue:
            state, planner_memory, current_dialogue = {}, {}, str(row["dialogue"])
        obs = expit(a * np.asarray(row["logit_gaps"], float) + b)
        current = {}
        for kc, z in zip(row["kcs"], obs):
            prev = state.get(kc, m0)
            current[kc] = (1 - lam) * (prev + (1 - prev) * p_l) + lam * z
        pooled = float(np.mean(list(current.values())))
        predicted = (1 - s) * pooled + g * (1 - pooled)
        u = Y[o.sava_verdict]
        for kc, v in current.items():
            state[kc] = v if u is None else (1 - mu) * v + mu * u
        names = sorted(state)
        position = {kc: i for i, kc in enumerate(names)}
        memory = {"ever": {position[kc] for kc in planner_memory.get("ever", set())},
                  "current": position.get(planner_memory.get("current"))}
        next_index = mta_choose(np.array([state[k] for k in names]), {}, memory, TAU)
        planner_memory = {"ever": {names[i] for i in memory["ever"]},
                          "current": names[memory["current"]] if memory.get("current") is not None else None}
        records.append({"dialogue": o.dialogue, "turn_id": int(o.turn_id), "label": int(o.label), "final_turn": o.final_turn,
                        "predicted_correctness": float(np.clip(predicted, 1e-6, 1 - 1e-6)),
                        "sava_verdict": o.sava_verdict, "sava_stage": o.sava_stage, "diagnosis_gate": o.diagnosis_gate,
                        "n_kcs": len(row["kcs"]), "kcs": json.dumps(row["kcs"]), "kc_mastery_before": json.dumps(current),
                        "sava_extracted": o.sava_extracted, "mean_mastery_after": float(np.mean([state[k] for k in current])),
                        "planner_next_kc": names[next_index] if next_index is not None else None})
    new = pd.DataFrame(records)
    reference = pd.read_csv(PRED)["mean_bkt_sava"].to_numpy()
    assert np.allclose(new.predicted_correctness.to_numpy(), reference, atol=1e-9)
    assert (new.diagnosis_gate == old.diagnosis_gate).all() and (new.sava_verdict == old.sava_verdict).all()
    new.to_csv(PKG / "04_integrated_loop/integrated_run_mathdial_qlora_seed221.csv", index=False)
    gated = new[new.diagnosis_gate == "run_error_analyzer"]
    pd.DataFrame([{"turns": len(new), "gate_open_share": len(gated) / len(new), "gated_turns": len(gated),
                   "gated_with_correct_label_share": float(gated.label.mean()),
                   "verdict_incorrect": int((new.sava_verdict == "incorrect").sum()),
                   "verdict_undetermined": int((new.sava_verdict == "undetermined").sum()),
                   "verdict_correct": int((new.sava_verdict == "correct").sum())}]
                 ).to_csv(PKG / "04_integrated_loop/gate_summary.csv", index=False)
    print("integrated loop re-run with CCMF: predictions match mean_bkt_sava; gate and verdicts unchanged")
    return old, new


if __name__ == "__main__":
    main()
