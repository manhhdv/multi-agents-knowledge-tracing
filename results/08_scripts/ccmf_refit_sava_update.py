import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
S = Path(sys.argv[1]); RUN = Path("/Users/manhnv/Downloads/SoICT2026_Paper/20260912T221028Z")
OUT = Path("/Users/manhnv/Downloads/SoICT2026_Paper/results_sava_conclusion")
src = (S / "cell26.txt").read_text().split("ccmf_rows, CCMF_FITS")[0]
g = {"np": np, "roc_auc_score": roc_auc_score, "CCMF_MAXITER": 1500, "CCMF_RESTARTS": 3, "PRIMARY_SEED": 221, "N_BOOT": 1000}
exec(src, g)
Y = {"correct": 1, "incorrect": 0, "undetermined": None}
def load(model, seed, split):
    rows = json.loads((RUN / f"mathdial/{model}/{seed}/{split}/raw_logit_gaps.json").read_text())
    for r in rows:
        r["gaps"] = np.asarray(r["logit_gaps"], dtype=float); r["label"] = int(r["label"])
    return rows
ver = {s: pd.read_csv(S / f"sava_v2/real_{s}_verdicts.csv", dtype={"dialogue": str}) for s in ("val", "test")}
ablation = pd.read_csv(RUN / "ccmf_real_logits_ablation.csv")
report = {}
for model, seed, tag in [("qlora", "seed221", "qlora_seed221"), ("zero_shot", "seed-na", "zero_shot_seed-na")]:
    val, test = load(model, seed, "val"), load(model, seed, "test")
    for rows, s in ((val, "val"), (test, "test")):
        assert [(r["dialogue"], int(r["turn_id"])) for r in rows] == list(zip(ver[s].dialogue, ver[s].turn_id))
    v1_test = [Y[v] for v in ver["test"].v1]; a_val = [Y[v] for v in ver["val"].A]; a_test = [Y[v] for v in ver["test"].A]
    saved = json.loads((RUN / f"ccmf_parameters_mathdial_{tag}.json").read_text())["parameters"]["ccmf_sava_update"]["raw"]
    y = np.array([r["label"] for r in test]); dialogues = np.array([r["dialogue"] for r in test])
    p_v1 = g["predict_ccmf"](test, np.array(saved), "sava", v1_test)
    ref_auc = ablation.query("dataset=='mathdial' and model==@model and variant=='ccmf_sava_update'").auc.iloc[0]
    assert abs(roc_auc_score(y, p_v1) - ref_auc) < 1e-9, (roc_auc_score(y, p_v1), ref_auc)
    print(tag, "v1 SAVA-update AUC reproduced:", round(ref_auc, 6), flush=True)
    raw, fits = g["fit_ccmf"](val, "sava", a_val)
    p_a = g["predict_ccmf"](test, raw, "sava", a_test)
    preds = {"mean_raw": g["mean_prediction"](test) if "mean_prediction" in g else np.array([np.mean(1/(1+np.exp(-r["gaps"]))) for r in test]),
             "ccmf_sava_update_v1": p_v1, "ccmf_sava_update_A": p_a}
    boot = g["paired_bootstrap"](y, preds, dialogues)
    keys = ["a", "b", "lambda", "mu", "p_learn", "m0", "slip", "guess"]
    report[tag] = {"auc_A": roc_auc_score(y, p_a), "auc_mean_raw": roc_auc_score(y, preds["mean_raw"]),
                   "delta_A": roc_auc_score(y, p_a) - roc_auc_score(y, preds["mean_raw"]),
                   "delta_ci_A": boot["ccmf_sava_update_A"]["delta_ci"], "ci_A": boot["ccmf_sava_update_A"]["ci"],
                   "delta_ci_v1_recomputed": boot["ccmf_sava_update_v1"]["delta_ci"],
                   "params_A": dict(zip(keys, g["unpack_ccmf"](raw))), "raw_A": [float(x) for x in raw], "fits_A": fits}
    print(tag, json.dumps({k: v for k, v in report[tag].items() if k not in ("raw_A",)}, default=float), flush=True)
    pd.DataFrame({"dialogue": dialogues, "turn_id": [r["turn_id"] for r in test], "label": y, **preds}).to_csv(OUT / f"ccmf_sava_update_A_predictions_{tag}.csv", index=False)
(OUT / "ccmf_sava_update_A.json").write_text(json.dumps(report, indent=2, default=float))
print("done")
