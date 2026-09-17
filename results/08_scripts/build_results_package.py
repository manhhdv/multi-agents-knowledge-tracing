"""Assemble ket_qua_bai_bao/: one results folder whose files match every number in the paper.

Sources: the original run folder (raw logits, benchmark candidates, EA predictions, planner), the notebook's final SAVA
(cell 22), the refitted CCMF SAVA-update head, the regenerated integrated loop, the CoMTA transfer recomputed with
MathDial-fitted parameters, the error-analyzer annotation set and the merged 30-case feedback ratings.
"""
import json, re, shutil
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

ROOT = Path("/Users/manhnv/Downloads/SoICT2026_Paper")
RUN = ROOT / "20260912T221028Z"
SAVA_NEW = ROOT / "results_sava_conclusion"
SCRATCH = Path(__file__).parent
OUT = ROOT / "ket_qua_bai_bao"
assert not OUT.exists(), f"{OUT} already exists"
d = {k: OUT / k for k in ["00_run_info", "01_backbone", "02_sava", "03_ccmf", "04_integrated_loop", "05_error_analyzer",
                          "06_feedback", "07_planner", "08_scripts"]}
for p in d.values():
    p.mkdir(parents=True)
(d["06_feedback"] / "regenerated_cases").mkdir()

# ---------------- 00 run info ----------------
for f in ["manifest.json", "runtime.json", "pip_freeze.txt", "repository_commit.txt"]:
    shutil.copy2(RUN / f, d["00_run_info"] / f)

# ---------------- 01 backbone (CoMTA used for evaluation only: no CoMTA-trained rows) ----------------
ms = pd.read_csv(RUN / "metrics_summary.csv")
ms[~((ms.dataset == "comta") & (ms.model == "qlora"))].to_csv(d["01_backbone"] / "metrics_summary.csv", index=False)
gt = pd.read_csv(RUN / "golden_test.csv")
gt[~((gt.dataset == "comta") & (gt.model == "qlora"))].to_csv(d["01_backbone"] / "golden_test.csv", index=False)
th = pd.read_csv(RUN / "threshold_calibrated_metrics.csv")
th = th[th.dataset == "mathdial"].assign(threshold_fit_on="mathdial")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
cz = pd.read_csv(SCRATCH / "comta_transfer_predictions.csv")
t = float(th[(th.model == "zero_shot")].threshold.iloc[0]); hard = (cz.mean_raw >= t).astype(int)
th = pd.concat([th, pd.DataFrame([{"dataset": "comta", "model": "zero_shot", "seed": "seed-na", "threshold": t, "n": len(cz),
      "accuracy": 100 * accuracy_score(cz.label, hard), "auc": 100 * roc_auc_score(cz.label, cz.mean_raw),
      "precision": 100 * precision_score(cz.label, hard), "recall": 100 * recall_score(cz.label, hard),
      "f1": 100 * f1_score(cz.label, hard), "threshold_fit_on": "mathdial"}])], ignore_index=True)
th.to_csv(d["01_backbone"] / "threshold_calibrated_metrics.csv", index=False)
for f in ["confusion_mathdial_qlora_seed221.png", "confusion_mathdial_zero_shot_seed-na.png"]:
    shutil.copy2(RUN / f, d["01_backbone"] / f)

# ---------------- 02 SAVA ----------------
nb = json.loads((ROOT / "colab_full_framework_replication.ipynb").read_text())
final_ns = {"re": re, "SAVA_QUESTION_GUARD_GLOBAL": True}; exec("".join(nb["cells"][22]["source"]), final_ns)
nocue_ns = {"re": re, "SAVA_QUESTION_GUARD_GLOBAL": True}; exec((SAVA_NEW / "cell22.txt").read_text(), nocue_ns)
bench = pd.read_csv(RUN / "sava_constructed_benchmark.csv", dtype=str)
res = [final_ns["sava_verify"](c, r) for c, r in zip(bench.candidate, bench.reference)]
nocue = [nocue_ns["sava_verify"](c, r) for c, r in zip(bench.candidate, bench.reference)]
assert [x["verdict"] for x in nocue] == bench.sava.tolist() and [x["stage"] for x in nocue] == bench.sava_stage.tolist()
items = bench.drop(columns=["sava", "sava_stage"]).assign(
    sava=[x["verdict"] for x in res], sava_stage=[x["stage"] for x in res],
    sava_no_conclusion_cue=[x["verdict"] for x in nocue])
items = items[["reference_id", "reference", "transformation", "candidate", "gold", "sava", "sava_stage",
               "sava_no_conclusion_cue", "numeric_regex", "exact_string", "always_correct"]]
items.to_csv(d["02_sava"] / "constructed_benchmark_items.csv", index=False)
methods = ["sava", "sava_no_conclusion_cue", "numeric_regex", "exact_string", "always_correct"]
summ = []
for m in methods:
    right, spoken = items[m] == items.gold, items[m] != "undetermined"
    summ.append({"method": m, "n": len(items), "coverage": spoken.mean(), "accuracy_all": right.mean(),
                 "balanced_accuracy": (right[items.gold == "correct"].mean() + right[items.gold == "incorrect"].mean()) / 2,
                 "accuracy_when_judging": right[spoken].mean()})
pd.DataFrame(summ).to_csv(d["02_sava"] / "constructed_benchmark_summary.csv", index=False)
items.assign(**{m: items[m] == items.gold for m in methods}).groupby(["gold", "transformation"])[methods].mean() \
    .to_csv(d["02_sava"] / "constructed_benchmark_per_transformation.csv")
ok_s, ok_r = (items.sava == items.gold).to_numpy(), (items.numeric_regex == items.gold).to_numpy()
only_s, only_r = int((ok_s & ~ok_r).sum()), int((~ok_s & ok_r).sum())
real = {}
for split in ["test", "val"]:
    v = pd.read_csv(SAVA_NEW / f"real_{split}_verdicts.csv", dtype={"dialogue": str})
    real[split] = v.rename(columns={"A": "sava", "stage_A": "sava_stage", "v1": "sava_no_conclusion_cue",
                                    "stage_v1": "sava_no_conclusion_cue_stage", "regex": "numeric_regex"})[
        ["dialogue", "turn_id", "label", "final_turn", "sava", "sava_stage", "sava_no_conclusion_cue",
         "sava_no_conclusion_cue_stage", "numeric_regex"]]
    real[split].to_csv(d["02_sava"] / f"real_turns_mathdial_{split}.csv", index=False)
stages = pd.concat([items.sava_stage.value_counts().rename("benchmark"),
                    real["test"].sava_stage.value_counts().rename("real_test_turns")], axis=1).fillna(0).astype(int)
stages["real_test_share"] = stages.real_test_turns / len(real["test"])
stages.to_csv(d["02_sava"] / "verdicts_by_stage.csv")
agr = pd.read_csv(SAVA_NEW / "real_turn_summary.csv")
agr = agr[agr.method.isin(["A", "v1", "regex"])].replace({"method": {"A": "sava", "v1": "sava_no_conclusion_cue", "regex": "numeric_regex"}})
agr.to_csv(d["02_sava"] / "real_turn_agreement.csv", index=False)
boot = json.loads((SAVA_NEW / "bootstrap_vs_v1_test.json").read_text())["A_vs_v1"]
(d["02_sava"] / "tests.json").write_text(json.dumps({
    "benchmark_mcnemar_sava_vs_regex": {"only_sava_right": only_s, "only_regex_right": only_r,
                                        "exact_p": binomtest(only_r, only_s + only_r, 0.5).pvalue,
                                        "accuracy_delta": float(ok_s.mean() - ok_r.mean())},
    "real_test_conclusion_cue_ablation_dialogue_bootstrap": boot}, indent=2))

# ---------------- 03 CCMF ----------------
abl = pd.read_csv(RUN / "ccmf_real_logits_ablation.csv")
abl = abl[abl.dataset == "mathdial"].assign(params_fit_on="mathdial")
refit = json.loads((SAVA_NEW / "ccmf_sava_update_A.json").read_text())
for model, tag in [("qlora", "qlora_seed221"), ("zero_shot", "zero_shot_seed-na")]:
    r = refit[tag]; mask = (abl.model == model) & (abl.variant == "ccmf_sava_update")
    assert mask.sum() == 1
    abl.loc[mask, ["auc", "ci_low", "ci_high", "delta_vs_mean_raw", "delta_ci_low", "delta_ci_high"]] = \
        [r["auc_A"], r["ci_A"][0], r["ci_A"][1], r["delta_A"], r["delta_ci_A"][0], r["delta_ci_A"][1]]
    params = json.loads((RUN / f"ccmf_parameters_mathdial_{tag}.json").read_text())
    params["parameters"]["fit_on"] = "mathdial"
    params["parameters"]["ccmf_sava_update"] = {"raw": r["raw_A"], **r["params_A"]}
    params["fits"]["ccmf_sava_update"] = r["fits_A"]
    (d["03_ccmf"] / f"ccmf_parameters_mathdial_{tag}.json").write_text(json.dumps(params, indent=2))
    pred = pd.read_csv(RUN / f"ccmf_test_predictions_mathdial_{tag}.csv", dtype={"dialogue": str})
    newp = pd.read_csv(SAVA_NEW / f"ccmf_sava_update_A_predictions_{tag}.csv", dtype={"dialogue": str})
    assert (pred.dialogue.values == newp.dialogue.values).all() and (pred.turn_id.values == newp.turn_id.values).all()
    pred["ccmf_sava_update"] = newp["ccmf_sava_update_A"].values
    pred.to_csv(d["03_ccmf"] / f"ccmf_test_predictions_mathdial_{tag}.csv", index=False)
comta = pd.read_csv(SCRATCH / "comta_transfer_ccmf.csv")
abl = pd.concat([abl, comta], ignore_index=True)
abl.to_csv(d["03_ccmf"] / "ccmf_real_logits_ablation.csv", index=False)
shutil.copy2(SCRATCH / "comta_transfer_predictions.csv", d["03_ccmf"] / "ccmf_test_predictions_comta_zero_shot_seed-na_mathdial_params.csv")

# ---------------- 04 integrated loop ----------------
integ = pd.read_csv(SCRATCH / "integrated_run_final.csv", dtype={"dialogue": str})
integ.to_csv(d["04_integrated_loop"] / "integrated_run_mathdial_qlora_seed221.csv", index=False)
gated = integ.diagnosis_gate == "run_error_analyzer"
pd.DataFrame([{"turns": len(integ), "gate_open_share": gated.mean(), "gated_turns": int(gated.sum()),
               "gated_with_correct_label_share": (integ[gated].label == 1).mean(),
               **{f"verdict_{k}": int(v) for k, v in integ.sava_verdict.value_counts().items()}}]) \
    .to_csv(d["04_integrated_loop"] / "gate_summary.csv", index=False)

# ---------------- 05 error analyzer ----------------
shutil.copy2(RUN / "error_analyzer_predictions.csv", d["05_error_analyzer"] / "error_analyzer_predictions.csv")
for f in ["error_analyzer_gold.csv", "error_analyzer_metrics.csv", "agreement_report.txt", "annotator_A.xlsx",
          "annotator_B.xlsx", "adjudication.xlsx", "HUONG_DAN_GAN_NHAN.md", "ea_annotation.py"]:
    shutil.copy2(ROOT / "error_analyzer_annotation" / f, d["05_error_analyzer"] / f)

# ---------------- 06 feedback (30 cases) ----------------
FM = SCRATCH / "fb_merge" / "out"
for f in ["rater_A.xlsx", "rater_B.xlsx", "feedback_human_metrics.csv", "feedback_human_report.txt", "HUONG_DAN_CHAM_FEEDBACK.md"]:
    shutil.copy2(FM / f, d["06_feedback"] / f)
shutil.copy2(ROOT / "feedback_rating" / "fb_rating.py", d["06_feedback"] / "fb_rating.py")
cases = pd.read_csv(ROOT / "feedback_rating" / "feedback_generator_cases_n30.csv")
cases.to_csv(d["06_feedback"] / "feedback_generator_cases.csv", index=False)
agg = {"n": ("valid_json", "size"), "valid_json_rate": ("valid_json", "mean"), "answer_leak_rate": ("answer_leak", "mean"),
       "strategy_band_match": ("strategy_band_match", "mean"), "mean_feedback_words": ("feedback_words", "mean"),
       **{f"judge_{c}_mean": (c, "mean") for c in ["correctness", "relevance", "clarity"]}}
pd.concat([cases.groupby("verdict").agg(**agg), cases.assign(verdict="all").groupby("verdict").agg(**agg)]) \
    .to_csv(d["06_feedback"] / "feedback_generator_summary.csv")
B2 = ROOT / "feedback_rating" / "bo_sung_2_ca"
for f in ["results.json", "error_analyzer_498_3.csv"]:
    shutil.copy2(B2 / f, d["06_feedback"] / "regenerated_cases" / f)
pd.DataFrame({"control_case_index": [24, 25, 26, 27, 28, 29], "common_prefix_chars": [22, 181, 1075, 924, 960, 154],
              "original_chars": [938, 991, 1075, 924, 960, 717], "identical": [False, False, True, True, True, False],
              "valid_json_regenerated": [True] * 6}).to_csv(d["06_feedback"] / "regenerated_cases" / "control_batch_check_L4_vs_A100.csv", index=False)

# ---------------- 07 planner ----------------
for f in ["planner_simulation.csv", "planner_ablation.csv", "planner_sensitivity_grid.csv", "planner_grid_wins.csv"]:
    shutil.copy2(RUN / f, d["07_planner"] / f)

# ---------------- 08 scripts ----------------
shutil.copy2(Path(__file__), d["08_scripts"] / "build_results_package.py")
shutil.copy2(SAVA_NEW / "ccmf_refit_A.py", d["08_scripts"] / "ccmf_refit_sava_update.py")
shutil.copy2(B2 / "make_rater_files.py", d["08_scripts"] / "make_rater_files_regenerated_cases.py")

# ---------------- consistency printout against the paper ----------------
s = pd.DataFrame(summ).set_index("method")
print("SAVA benchmark coverage/acc:", s.loc["sava", ["coverage", "accuracy_all"]].round(3).tolist(), "| no cue:",
      s.loc["sava_no_conclusion_cue", "accuracy_all"].round(3), "| regex:", s.loc["numeric_regex", "accuracy_all"].round(3),
      "| McNemar p:", binomtest(only_r, only_s + only_r, 0.5).pvalue)
print(stages.to_string())
a = agr[agr.subset.isin(["all", "final"]) & (agr.split == "test")][["subset", "method", "coverage", "agreement_when_judging"]]
print(a.round(4).to_string(index=False))
print(abl[["dataset", "model", "variant", "auc", "delta_vs_mean_raw", "delta_ci_low", "delta_ci_high"]].round(4).to_string(index=False))
print("CoMTA mean_platt AUC (MathDial params), full precision:", comta.set_index("variant").loc["mean_platt", "auc"])
print(pd.read_csv(d["04_integrated_loop"] / "gate_summary.csv").round(4).to_string(index=False))
print(th.round(2).to_string(index=False))
print("files:", sum(1 for _ in OUT.rglob("*") if _.is_file()))
