"""Independent verification of the pre-registered CCMF re-specification (run after `ccmf_redesign.py evaluate`).

Re-implements every candidate predictor from the frozen raw parameters without importing ccmf_redesign.py or the
notebook cell, recomputes all test AUCs from the written predictions, re-runs the dialogue-level paired bootstrap
with its own code, and checks the protocol artefacts (single evaluation, sentinel, fits hashes, primary variant,
decision rule). Writes 03_ccmf/independent_verification.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path("/Users/manhnv/Downloads/SoICT2026_Paper")
RUN = ROOT / "20260912T221028Z"
KQ = ROOT / "ket_qua_bai_bao"
OUT = KQ / "03_ccmf"
BACKBONES = [("qlora", "seed221", "qlora_seed221"), ("zero_shot", "seed-na", "zero_shot_seed-na")]
CLIP = (1e-6, 1 - 1e-6)
CANDIDATES = ["geo", "min", "tempered_and", "mean_bkt_none", "mean_bkt_model", "mean_bkt_sava", "mean_bkt_gold"]
TABLE5 = ["mean_raw", "mean_platt", "noisy_and_platt", "ccmf_no_update", "ccmf_model_update",
          "ccmf_sava_update", "ccmf_gold_update"]


def natural(raw8):
    r = np.asarray(raw8, float)
    return (float(np.exp(np.clip(r[0], -3, 3))), float(4 * np.tanh(r[1])), *[float(expit(v)) for v in r[2:6]],
            float(0.49 * expit(r[6])), float(0.49 * expit(r[7])))


def static(rows, kind, raw8, gamma=None):
    a, b, _, _, _, _, s, g = natural(raw8)
    out = []
    for r in rows:
        d = np.asarray(r["logit_gaps"], float)
        if kind == "min":
            P = expit(a * d.min() + b)
        else:
            logz = -np.logaddexp(0, -(a * d + b))
            e = 1.0 / len(d) if kind == "geo" else float(len(d)) ** (-gamma)
            P = np.exp(e * logz.sum())
        out.append((1 - s) * P + g * (1 - P))
    return np.clip(np.array(out), *CLIP)


def mean_bkt(rows, raw8, mode, verdicts):
    a, b, lam, mu, pL, m0, s, g = natural(raw8)
    out, state, cur = [], {}, None
    for i, r in enumerate(rows):
        if r["dialogue"] != cur:
            state, cur = {}, r["dialogue"]
        vals = {}
        for kc, d in zip(r["kcs"], r["logit_gaps"]):
            prev = state.get(kc, m0)
            vals[kc] = (1 - lam) * (prev + (1 - prev) * pL) + lam * expit(a * d + b)
        P = sum(vals.values()) / len(vals)
        yhat = (1 - s) * P + g * (1 - P)
        out.append(yhat)
        u = {"none": None, "model": yhat, "gold": r["label"], "sava": verdicts[i]}[mode]
        for kc, v in vals.items():
            state[kc] = v if u is None else (1 - mu) * v + mu * u
    return np.clip(np.array(out), *CLIP)


def label_hist(rows, mode, verdicts, m0_ref):
    out, hist, cur = [], {}, None
    for i, r in enumerate(rows):
        if r["dialogue"] != cur:
            hist, cur = {}, r["dialogue"]
        out.append(float(np.mean([hist.get(k, m0_ref) for k in r["kcs"]])))
        u = r["label"] if mode == "gold" else verdicts[i]
        if u is not None:
            for k in r["kcs"]:
                hist[k] = float(u)
    return np.clip(np.array(out), *CLIP)


def bootstrap(y, preds, dialogues, reference, n_boot=1000, seed=221):
    rng = np.random.default_rng(seed)
    groups = {d: np.flatnonzero(dialogues == d) for d in np.unique(dialogues)}
    ids = list(groups)
    deltas = {k: [] for k in preds}
    for _ in range(n_boot):
        idx = np.concatenate([groups[ids[j]] for j in rng.integers(0, len(ids), len(ids))])
        if len(np.unique(y[idx])) < 2:
            continue
        base = roc_auc_score(y[idx], preds[reference][idx])
        for k, p in preds.items():
            deltas[k].append(roc_auc_score(y[idx], p[idx]) - base)
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in deltas.items()}


def decide(L, U):
    if L > 0:
        return "improves"
    if U < 0:
        return "harms (within margin)" if L > -0.010 else "harms"
    return "non-inferior" if L > -0.010 else "inconclusive"


def main():
    report, problems = {}, []
    prereg = json.loads((OUT / "preregistration.json").read_text())
    log_lines = [l for l in (OUT / "test_evaluations.log").read_text().splitlines() if l.strip()]
    report["n_test_evaluations_logged"] = len(log_lines)
    report["sentinel_exists"] = (OUT / "TEST_EVALUATED").exists()
    if len(log_lines) != 1:
        problems.append(f"test_evaluations.log has {len(log_lines)} lines")
    results = pd.read_csv(OUT / "test_results.csv")
    table5 = pd.read_csv(KQ / "03_ccmf" / "table5_reference_inputs" / "ccmf_real_logits_ablation.csv")
    for model, seed, tag in BACKBONES:
        fits_path = OUT / f"fits_{tag}.json"
        sha = hashlib.sha256(fits_path.read_bytes()).hexdigest()
        if sha != prereg["fits_sha256"][tag]:
            problems.append(f"{tag}: fits sha256 differs from preregistration.json")
        fits = json.loads(fits_path.read_text())
        rows = json.loads((RUN / "mathdial" / model / seed / "test" / "raw_logit_gaps.json").read_text())
        for r in rows:
            r["label"], r["dialogue"], r["turn_id"] = int(r["label"]), str(r["dialogue"]), str(r["turn_id"])
        ver = pd.read_csv(KQ / "02_sava" / "real_turns_mathdial_test.csv", dtype={"dialogue": str, "turn_id": str})
        assert list(zip(ver.dialogue, ver.turn_id)) == [(r["dialogue"], r["turn_id"]) for r in rows]
        verdicts = [{"correct": 1, "incorrect": 0}.get(v) for v in ver.sava]
        y = np.array([r["label"] for r in rows])
        dialogues = np.array([r["dialogue"] for r in rows])
        # round_trip parsing: pandas' default float parser can move values by one ULP, which breaks exact ties
        written = pd.read_csv(OUT / f"test_predictions_{tag}.csv", dtype={"dialogue": str, "turn_id": str},
                              float_precision="round_trip")
        assert list(zip(written.dialogue, written.turn_id)) == [(r["dialogue"], r["turn_id"]) for r in rows]

        # 1. predictions re-derived from the frozen raw parameters with independent code
        mine = {}
        for name in CANDIDATES:
            f = fits["variants"][name]
            if name in ("geo", "min"):
                mine[name] = static(rows, name, f["raw8"])
            elif name == "tempered_and":
                mine[name] = static(rows, "tempered_and", f["raw8"], gamma=float(expit(f["raw_free"][2])))
            else:
                mine[name] = mean_bkt(rows, f["raw8"], f["mode"], verdicts)
        mine["mean_raw"] = np.clip(np.array([expit(np.asarray(r["logit_gaps"], float)).mean() for r in rows]), *CLIP)
        mine["label_hist_gold"] = label_hist(rows, "gold", verdicts, fits["m0_ref"])
        mine["label_hist_sava"] = label_hist(rows, "sava", verdicts, fits["m0_ref"])
        pred_diff = {k: float(np.max(np.abs(v - written[k].values))) for k, v in mine.items()}
        for k, d in pred_diff.items():
            if d > 1e-9:
                problems.append(f"{tag}/{k}: re-derived predictions differ by {d:.2e}")

        # 2. AUCs recomputed from the written predictions
        res = results[results.model == model].set_index("variant")
        cols = [c for c in written.columns if c not in ("dialogue", "turn_id", "label")]
        preds = {c: written[c].values.astype(float) for c in cols}
        auc = {c: float(roc_auc_score(y, preds[c])) for c in cols}
        auc_diff = {c: abs(auc[c] - float(res.loc[c, "auc"])) for c in cols}
        for c, d in auc_diff.items():
            if d > 1e-9:
                problems.append(f"{tag}/{c}: AUC differs from test_results.csv by {d:.2e}")
        t5 = table5[(table5.dataset == "mathdial") & (table5.model == model)].set_index("variant")
        t5_diff = {c: abs(auc[c] - float(t5.loc[c, "auc"])) for c in TABLE5}
        for c, d in t5_diff.items():
            if d > 1e-3:
                problems.append(f"{tag}/{c}: AUC differs from Table 5 by {d:.2e}")

        # 3. bootstrap with independent code (protocol seed) and a robustness seed
        ci = bootstrap(y, preds, dialogues, "mean_raw")
        ci_diff = {c: max(abs(ci[c][0] - float(res.loc[c, "delta_ci_low"])), abs(ci[c][1] - float(res.loc[c, "delta_ci_high"])))
                   for c in cols}
        for c, d in ci_diff.items():
            if d > 1e-9:
                problems.append(f"{tag}/{c}: delta CI differs from test_results.csv by {d:.2e}")
        ci_alt = bootstrap(y, preds, dialogues, "mean_raw", seed=2026)

        # 4. primary variant and decision rule
        primary = prereg["primary_variant"][tag]
        flagged = res.index[res.is_primary.astype(bool)].tolist()
        if flagged != [primary]:
            problems.append(f"{tag}: primary in test_results {flagged} != preregistration {primary}")
        L, U = ci[primary]
        decision = decide(L, U)
        if decision != res.loc[primary, "decision"]:
            problems.append(f"{tag}: decision {res.loc[primary, 'decision']} != recomputed {decision}")
        report[tag] = {"primary": primary, "primary_auc": auc[primary], "primary_delta": auc[primary] - auc["mean_raw"],
                       "primary_delta_ci": [L, U], "decision_recomputed": decision,
                       "decision_under_seed_2026": decide(*ci_alt[primary]), "primary_delta_ci_seed_2026": list(ci_alt[primary]),
                       "max_pred_diff": max(pred_diff.values()), "max_auc_diff_vs_results": max(auc_diff.values()),
                       "max_auc_diff_vs_table5": max(t5_diff.values()), "max_delta_ci_diff_vs_results": max(ci_diff.values()),
                       "auc": auc, "delta_ci": ci, "delta_ci_seed_2026": ci_alt}
    report["problems"] = problems
    report["verdict"] = "pass" if not problems else "fail"
    (OUT / "independent_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in BACKBONES and not isinstance(v, dict)}, indent=1))
    for _, _, tag in BACKBONES:
        r = report[tag]
        print(tag, {k: r[k] for k in ["primary", "primary_auc", "primary_delta", "primary_delta_ci", "decision_recomputed",
                                      "decision_under_seed_2026", "max_pred_diff", "max_auc_diff_vs_results",
                                      "max_auc_diff_vs_table5", "max_delta_ci_diff_vs_results"]})


if __name__ == "__main__":
    main()
