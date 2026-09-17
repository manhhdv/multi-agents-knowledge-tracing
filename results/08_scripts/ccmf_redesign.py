#!/usr/bin/env python
"""Pre-registered CCMF re-specification on real MathDial logits (QLoRA + zero-shot).

Subcommands
  check     validation-only regression checks C1/C2/C3/C5/C6/C7 (never opens a test path)
  fit       validation-only fitting of every pre-registered family for both backbones
            (leakage guard: builtins.open / io.open raise on any "/test/" or "_test" path)
  evaluate  single-shot test evaluation (sentinel-protected; no force flag)

Reused code: the notebook cell `ket_qua_bai_bao/03_ccmf/table5_reference_inputs/cell26.txt` is exec'd
verbatim (text before the marker "ccmf_rows, CCMF_FITS") and provides CCMF_STARTS,
unpack_ccmf, ccmf_predict_turn, ccmf_commit, predict_ccmf, fit_ccmf, fit_platt,
fit_noisy_and, paired_bootstrap.  Nothing of it is rewritten here.
"""
import argparse
import builtins
import datetime
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("once", category=ConvergenceWarning)

# --------------------------------------------------------------------------------------
# Paths, constants
# --------------------------------------------------------------------------------------
ROOT = Path("/Users/manhnv/Downloads/SoICT2026_Paper")
RUN = ROOT / "20260912T221028Z"
KQ = ROOT / "ket_qua_bai_bao"
OUT = KQ / "03_ccmf"
CKPT = OUT / "checkpoints"
CELL26 = KQ / "03_ccmf" / "table5_reference_inputs" / "cell26.txt"
SCRIPT_PATH = Path(__file__).resolve()
PYTHON_PATH = ("/private/tmp/claude-502/-Users-manhnv-Downloads-SoICT2026-Paper/"
               "1ff52e8a-83db-4302-977b-dc1205fa3006/scratchpad/venv/bin/python")

BACKBONES = [("qlora", "seed221", "qlora_seed221"), ("zero_shot", "seed-na", "zero_shot_seed-na")]
TAGS = [t for _, _, t in BACKBONES]
N_VAL, D_VAL, N_TEST, D_TEST = 1755, 415, 1985, 515
CLIP = (1e-6, 1 - 1e-6)
SEED = 221
N_FALLBACK = 6

FIT_ORDER = ["geo", "min", "tempered_and", "mean_bkt_none", "mean_bkt_model", "mean_bkt_sava", "mean_bkt_gold"]
PRIMARY_SET = ["geo", "min", "mean_bkt_none", "mean_bkt_model"]  # order breaks ties
REFERENCE_ROWS = ["mean_raw", "mean_platt", "noisy_and_platt", "ccmf_no_update", "ccmf_model_update",
                  "ccmf_sava_update", "ccmf_gold_update", "label_hist_gold", "label_hist_sava"]
TABLE5_ROWS = REFERENCE_ROWS[:7]
CCMF_MODE_NAME = {"none": "ccmf_no_update", "model": "ccmf_model_update",
                  "gold": "ccmf_gold_update", "sava": "ccmf_sava_update"}
CONTRAST_REFS = {"noisy_and_platt": ["geo", "min", "tempered_and"],
                 "mean_bkt_none": ["mean_bkt_model", "mean_bkt_sava", "mean_bkt_gold"],
                 "label_hist_gold": ["mean_bkt_gold"],
                 "label_hist_sava": ["mean_bkt_sava"]}
NON_INFERIORITY_MARGIN = 0.010

NM_OPTIONS = {"maxiter": 4000, "maxfev": 8000, "xatol": 1e-4, "fatol": 1e-7, "adaptive": True}
MAX_CONTINUATIONS = 2
MAX_POLISH = 2
POLISH_STEP = 0.05
SIMPLEX_STEP = 0.5

# --------------------------------------------------------------------------------------
# Verbatim pre-registration text (written to PREREGISTRATION.md; sha256 in manifest.json)
# --------------------------------------------------------------------------------------
PREREG_TEXT = r"""# FROZEN PRE-REGISTRATION — CCMF re-specification on real logits (MathDial, QLoRA + zero-shot)

Status: frozen before any fit of a new family and before any new script opens a test path. The `fit` command writes a verbatim copy of this document to `ket_qua_bai_bao/03_ccmf/redesign/PREREGISTRATION.md` and its sha256 into `manifest.json` before fitting starts. Any deviation must be logged in `manifest.json` and stated in the paper.

## 0. Inputs, notation, reused code (verified on disk)

- Backbones (both are run for everything below): `qlora/seed221` (tag `qlora_seed221`) and `zero_shot/seed-na` (tag `zero_shot_seed-na`).
- Rows: `/Users/manhnv/Downloads/SoICT2026_Paper/20260912T221028Z/mathdial/<model>/<seed>/<split>/raw_logit_gaps.json`; val = 1,755 turns / 415 dialogues; test = 1,985 turns / 515 dialogues. Rows are ordered by dialogue then turn; temporal state resets whenever `row["dialogue"]` changes. The JSON stores `turn_id` and `label` as ints; the loader always applies `r["gaps"] = np.asarray(r["logit_gaps"], float)`, `r["label"] = int(r["label"])`, `r["turn_id"] = str(r["turn_id"])`, `r["dialogue"] = str(r["dialogue"])`.
- Notation: turn j, KC set C_j, n_j = |C_j| >= 1, gaps d_jk = logit(True) - logit(False), label y_j in {0,1}, sigma = expit, z_jk = sigma(a*d_jk + b). Val n_j histogram {1:303, 2:919, 3:364, 4:120, 5:24, 6:15, 7:6, 8:2, 10:2}; no KC repeats inside a turn (asserted on both splits).
- SAVA verdicts: `ket_qua_bai_bao/02_sava/real_turns_mathdial_{val,test}.csv`, read with `pd.read_csv(path, dtype={"dialogue": str, "turn_id": str})`; column `sava` mapped correct->1, incorrect->0, undetermined->None (no update). Alignment is positional and is asserted: `[(r["dialogue"], r["turn_id"]) for r in rows] == list(zip(csv.dialogue, csv.turn_id))` and `csv.label.astype(int).tolist() == [r["label"] for r in rows]`. Val verdicts: 265 correct / 1,072 incorrect / 418 undetermined.
- Reused code, never rewritten: exec the text of `_archive/results_sava_conclusion/cell26.txt` before the marker `"ccmf_rows, CCMF_FITS"` into a globals dict `{np, roc_auc_score, CCMF_MAXITER: 1500, CCMF_RESTARTS: 3, PRIMARY_SEED: 221, N_BOOT: 1000}` exactly as `ket_qua_bai_bao/08_scripts/ccmf_refit_sava_update.py` does. This provides `CCMF_STARTS` (5 vectors), `unpack_ccmf`, `ccmf_predict_turn`, `ccmf_commit`, `predict_ccmf`, `fit_ccmf` (reference only, not called for new fits), `fit_platt`, `fit_noisy_and`, `paired_bootstrap`. `sava_labels`/`sava_verify` are never called. `mean_raw(rows) = np.array([expit(r["gaps"]).mean() for r in rows])` is defined locally.
- Stored Table-5 parameters: `ket_qua_bai_bao/03_ccmf/ccmf_parameters_mathdial_<tag>.json` with `parameters.platt.{a,b}`, `parameters.noisy_and.{s,g}`, `parameters.ccmf_{no,model,gold,sava}_update.raw` (8-vectors) and `fits.<same key>` = list of `{success, loss, iterations}` per restart. Table 5: `ket_qua_bai_bao/03_ccmf/ccmf_real_logits_ablation.csv` (rows with dataset == "mathdial"); stored test predictions: `ket_qua_bai_bao/03_ccmf/ccmf_test_predictions_mathdial_<tag>.csv`.
- Objective everywhere: mean binary log-loss `sklearn.metrics.log_loss(y, np.clip(yhat, 1e-6, 1-1e-6), labels=[0,1])` (identical to `fit_ccmf`). Every prediction of every variant and reference is clipped to [1e-6, 1-1e-6] before log-loss and before being written.
- Environment recorded in `manifest.json`: python path `/private/tmp/claude-502/-Users-manhnv-Downloads-SoICT2026-Paper/1ff52e8a-83db-4302-977b-dc1205fa3006/scratchpad/venv/bin/python`, versions of numpy, scipy (>= 1.7 required for Nelder-Mead `bounds`), scikit-learn, pandas, sha256 of the script text.
- Script: `/Users/manhnv/Downloads/SoICT2026_Paper/ket_qua_bai_bao/08_scripts/ccmf_redesign.py` with subcommands `check`, `fit`, `evaluate`. Output directory `/Users/manhnv/Downloads/SoICT2026_Paper/ket_qua_bai_bao/03_ccmf/redesign/` (created by the script).

## 1. Shared components (identical for every family)

- Transforms (reuse `unpack_ccmf` verbatim): a = exp(clip(r_a, -3, 3)); b = 4*tanh(r_b); lambda, mu, p_L, m0 = expit(raw[2..5]); s, g = 0.49*expit(raw[6..7]). Static families build the 8-vector `raw8 = [r_a, r_b, 0, 0, 0, 0, r_s, r_g]` and read (a, b, s, g) from `unpack_ccmf(raw8)`; the tempered family additionally computes gamma = expit(r_gamma) locally. No other transform code exists.
- Raw box (passed to Nelder-Mead as `bounds` and asserted in the objective): r_a in [-3, 3], r_b in [-3, 3] (|b| <= 3.98), every expit-type raw (r_lambda, r_mu, r_pL, r_m0, r_gamma, r_s, r_g) in [-8, 8] (expit in [3.4e-4, 0.99966]; s, g in [1.6e-4, 0.4898]). Consequence: 1 - s - g >= 0.02 > 0, so the slip/guess wrapper yhat = g + (1-s-g)*P is strictly increasing in P and can never change an AUC; it only sets the floor g / ceiling 1-s and therefore log-loss. a > 0 always, so sigma(a*d+b) is strictly increasing in d.
- Numerics: log z_jk is computed as `-np.logaddexp(0, -(a*d_jk + b))` (never `log(expit(.))`); geo and tempered_and are computed in log space.
- Raw orders: static (geo, min) = [r_a, r_b, r_s, r_g]; tempered_and = [r_a, r_b, r_gamma, r_s, r_g]; mean_bkt = [r_a, r_b, r_lambda, r_mu, r_pL, r_m0, r_s, r_g] (same order as `unpack_ccmf`, so `CCMF_STARTS` are reusable).

## 2. Candidate families (all pre-registered; every one is reported)

### F1 `geo` (mandatory) — geometric mean of calibrated per-KC probabilities + slip/guess; 4 parameters (a, b, s, g)
P_j = (prod_{k in C_j} z_jk)^(1/n_j) = exp((1/n_j) * sum_k log z_jk); yhat_j = (1-s)*P_j + g*(1-P_j).
Mode: none. Rationale: log P_geo = (1/n_j) log P_prod removes exactly the n_j-fold counting of one turn label in the noisy-AND while keeping the product link. AUC depends on (a, b) only through comparisons across turns with different n_j / spread; invariant to (s, g). Not parameter-free.

### F2 `min` (mandatory) — weakest KC + slip/guess; 4 parameters (a, b, s, g)
P_j = min_k z_jk = sigma(a * min_k d_jk + b) (exact because sigma is increasing and a > 0); yhat_j = (1-s)*P_j + g*(1-P_j).
Mode: none. AUC is parameter-free: yhat is strictly increasing in min_k d_jk for every admissible parameter (clipping cannot create ties because g >= 1.6e-4 > 1e-6). The script asserts `roc_auc_score(y, yhat_min) == roc_auc_score(y, min_k d_jk)` to 1e-12 on val (fit stage) and test (evaluate stage). The fit determines only calibration (val/test log-loss), which is why log-loss is reported next to AUC.

### F3 `mean_bkt` (mandatory) — Eq. 2 and Eq. 4 unchanged, Eq. 3 product replaced by the mean; 8 parameters (a, b, lambda, mu, p_L, m0, s, g)
For each k in C_j (state m_k initialised to m0 at the first occurrence of KC k in the dialogue; state dict reset when the dialogue changes):
  prior_k = m_k + (1 - m_k)*p_L;  m~_k = (1 - lambda)*prior_k + lambda*z_jk                       (Eq. 2)
  P_j = (1/n_j) * sum_{k in C_j} m~_k;  yhat_j = (1-s)*P_j + g*(1-P_j)                            (Eq. 3 with mean)
  after yhat_j is recorded: m_k <- m~_k if u_j is None else (1-mu)*m~_k + mu*u_j for k in C_j    (Eq. 4, via `ccmf_commit`)
with u_j = None (mode none) / the UNCLIPPED yhat_j (mode model, exactly as `predict_ccmf`) / SAVA verdict, undetermined -> None (mode sava) / y_j (mode gold). Modes: none, model, sava, gold (sava is fitted with validation verdicts and evaluated with test verdicts; gold is an oracle). In mode none mu is unused: r_mu is fixed at 0 and removed from the optimised vector (7 free parameters); in the other modes all 8 are free.
Implementation: `predict_agg(rows, raw8, mode, verdicts, agg)` = copy of `predict_ccmf` whose turn function is `ccmf_predict_turn` with `agg(values)` (agg in {np.mean, np.prod}) in place of `np.prod`; `ccmf_commit` reused verbatim. Nesting: lambda -> 1, s = g -> 0 is mean pooling of Platt-calibrated probabilities with jointly fitted (a, b), so the anchor A_bkt (Sec. 6) must hold for every mode (at lambda = 1 the commit does not affect predictions). Identifiability: lambda -> 1 makes p_L, m0, mu drop out; lambda -> 0 with mu -> 1 turns the row into a label carry-over (compare with the label_hist references). AUC depends on (a, b, lambda, p_L, m0) and on mu in update modes; in mode model also on (s, g) through the committed prediction. Not parameter-free.

### F4 `tempered_and` (the single extra family; justified by two proposals from the double-counting diagnosis) — tempered conjunction; 5 parameters (a, b, gamma, s, g)
log P_j = n_j^(-gamma) * sum_{k in C_j} log z_jk, i.e. P_j = (prod_k z_jk)^(n_j^-gamma), gamma = expit(r_gamma) in (0, 1); yhat_j = (1-s)*P_j + g*(1-P_j).
Mode: none. gamma -> 0 recovers the paper's noisy-AND over z (Eq. 3 with lambda = 1, the head that lost 0.024-0.049 AUC); gamma -> 1 recovers F1 (geo). If the n_j per-KC readings of one turn label carry n_j^(1-gamma) effective independent pieces of evidence, this is the correct pooling, so the fitted gamma is a one-number measurement of the redundancy diagnosed in the paper. AUC depends on (a, b, gamma); invariant to (s, g). Not parameter-free. gamma is deliberately not allowed above 1 (that would be an ad-hoc KC-count bonus). Diagnostic (validation only, never on test): gamma profile — for gamma in {0, 0.25, 0.5, 0.75, 1} fixed exactly (not via r_gamma), refit (a, b, s, g) with the static protocol of Sec. 5 and report LL_val(gamma) and AUC_val(gamma) to `gamma_profile_<tag>.csv`. tempered_and is a diagnostic candidate: reported in full, excluded from the primary selection set (Sec. 8).

Reading: link in {conjunctive: geo, min, tempered_and; compensatory: mean_bkt} x counting in {once: geo, min, mean_bkt, tempered_and(gamma=1); n-fold: noisy_and_platt / stored CCMF, tempered_and(gamma=0)}.

## 3. Reference rows (reported, never candidates, never refitted with the new optimiser)

- `mean_raw`: yhat_j = (1/n_j) sum_k sigma(d_jk). Zero parameters; the Table 5 reference and the bootstrap reference.
- `mean_platt`: yhat_j = (1/n_j) sum_k sigma(a_platt d_jk + b_platt) with the STORED `parameters.platt`. `noisy_and_platt`: (1-s) prod_k sigma(a_platt d + b_platt) + g (1 - prod) with the STORED `parameters.noisy_and`.
- `ccmf_no_update`, `ccmf_model_update`, `ccmf_sava_update`, `ccmf_gold_update`: `predict_ccmf(rows, stored_raw, mode, verdicts)` with the stored raw 8-vectors and the original `unpack_ccmf` (no box, no refit).
- `label_hist_gold`, `label_hist_sava` (0 parameters; label-history floors for the update modes): per dialogue keep l_k = most recently committed label of KC k; yhat_j = (1/n_j) sum_{k in C_j} [l_k if k seen earlier in the dialogue else m0_ref], m0_ref = validation label mean (0.4735, computed in the fit stage and stored in the fits JSON); after the prediction, if u_j is not None: l_k <- u_j for all k in C_j (u_j = y_j for gold; SAVA verdict for sava, undetermined = no update). This is exactly mean_bkt at lambda = 0, mu = 1, p_L = 0, m0 = m0_ref, s = g = 0. Reported with AUC only (values are 0/1/m0_ref; no log-loss).

## 4. Label-update modes per variant
geo: none. min: none. tempered_and: none. mean_bkt: none, model, sava, gold. label_hist: gold, sava. References: none. Candidate fits per backbone: 7 (geo, min, tempered_and, mean_bkt x 4); 14 in total. Causality: a label/verdict of turn j is committed only after yhat_j is recorded and affects later turns of the same dialogue only (verified by check C7).

## 5. Fitting protocol (`fit` command; validation only; deterministic)

- Objective: validation log-loss (Sec. 0) of the variant's clipped predictions on the 1,755 val turns, with the mode's update rule applied during fitting (model: own unclipped prediction; sava: validation verdicts; gold: validation labels). Backbones fitted independently.
- Optimiser: `scipy.optimize.minimize(objective, x0, method="Nelder-Mead", bounds=box, options={"maxiter": 4000, "maxfev": 8000, "xatol": 1e-4, "fatol": 1e-7, "adaptive": True, "initial_simplex": S})` for every family/mode/backbone, where box is the raw box of Sec. 1 restricted to the free coordinates and S = [x0] + [x0 + 0.5 e_i for each free coordinate i] (if x0 + 0.5 e_i leaves the box, use x0 - 0.5 e_i). Nelder-Mead is the only optimiser (no gradient polish).
- Continuation: if a run ends with nit >= 4000 (hit_maxiter), restart from its final point with a fresh simplex (same options) at most 2 more times; totals nit_total / nfev_total are recorded.
- Polish (fixes premature simplex collapse): from the best point of the start, run Nelder-Mead again with a fresh simplex of step 0.05 (same bounds/options); accept iff it lowers the loss; repeat until the improvement is < 1e-7 or 2 polishes have run.
- Starts (fixed lists; index order breaks ties):
  - mean_bkt (raw order of Sec. 1): S1-S5 = `CCMF_STARTS[0..4]` verbatim ([0,0,0,-2,-3,-0.4,-2,-2], [0.3,-0.2,-0.5,-1,-2.5,0,-1.5,-2.5], [-0.3,0.2,0.5,-2.5,-3.5,-1,-2.5,-1.5], [0,0,-1,-0.5,-2,-0.8,-1,-3], [0.6,-0.4,1,-3,-4,0.5,-3,-1]); S6 = [0, 0, 6, 0, -3, 0, -8, -8] (nested mean-pooling start: a = 1, b = 0, lambda = 0.9975, s = g = 1.6e-4). Mode none drops index 3 (r_mu) from every start. 6 starts.
  - geo and min (raw order [r_a, r_b, r_s, r_g]): T1 = [0, 0, -2, -2]; T2 = [0.3, -0.2, -1.5, -2.5]; T3 = [-0.3, 0.2, -2.5, -1.5]; T4 = [0, 0, -1, -3]; T5 = [0.6, -0.4, -3, -1] (the (0,1,6,7) projection of CCMF_STARTS); T6 = [0, 0, -8, -8]. 6 starts.
  - tempered_and (raw order [r_a, r_b, r_gamma, r_s, r_g]): T1-T6 with r_gamma = 0 inserted at index 2, plus U7 = [0, 0, -4, -2, -2] (gamma = 0.018, near noisy-AND) and U8 = [0, 0, 4, -2, -2] (gamma = 0.982, near geo). 8 starts.
  - gamma-profile refits: the 6 static starts.
- Restart selection: lowest validation log-loss (raw, unpenalised) after continuation and polish; ties within 1e-10 broken by the lowest start index. All per-start records are stored: loss, nit_total, nfev_total, hit_maxiter, success of the last run, number of polishes accepted, final raw vector.
- Convergence flags per fit: `converged` := final run success == True and not hit_maxiter after continuation; `n_starts_agree` := number of starts within 1e-4 of the best loss; `restart_sensitive` := best two starts differ by > 1e-3; `at_bound` per parameter := |r_i - bound| < 1e-3.
- Pre-specified fallback (runs at most once per fit, only when triggered): if the selected start is not `converged`, or an anchor of Sec. 6 fails, run 6 extra starts drawn uniformly in the raw box from `np.random.default_rng(221)` (drawn in the fixed order geo, min, tempered_and, mean_bkt none/model/sava/gold, backbone qlora then zero_shot, so the draws are reproducible), same protocol, selection over all starts; `fallback_used` = True. If the condition persists the fit is reported as "optimiser failure"; nothing else is tuned.
- Implementation and budget: reference implementation is the loop (`predict_agg`); static families are vectorised with flattened gaps, `np.add.reduceat` / `np.minimum.reduceat` over per-turn offsets. A vectorised time-step engine for mean_bkt is permitted only after the equality gates of Sec. 6 (C5) pass; otherwise the loop is used (worst case ~4-5 h; run `fit` with run_in_background and checkpoint `fits_<tag>.json` after every (family, mode)). `fit` is resumable from checkpoints.
- Determinism check: geo on QLoRA is fitted twice in the same process and the two raw vectors must be bit-identical.
- Leakage guard: all file access in `fit` goes through one loader that raises if "/test/" or "_test" occurs in the path; additionally `builtins.open` and `io.open` are wrapped for the duration of `fit` to raise on such paths and to append every opened path to `manifest.json["fit_opened_paths"]`. `fit` never reads `real_turns_mathdial_test.csv`.
- Frozen outputs of `fit`: `fits_<tag>.json` (per variant: raw, natural parameters, per-start records, flags, LL_val, AUC_val, identifiability flags, m0_ref), `gamma_profile_<tag>.csv`, `validation_report.csv`, `regression_checks_val.json`, `preregistration.json` (variant list and modes, optimiser settings, starts, primary variant per backbone, decision rules of Sec. 8, package versions, script sha256, sha256 of each fits JSON, timestamp). The primary variant per backbone (Sec. 8) is written here, before any test file exists in the output directory.

## 6. Validation-only diagnostics, anchors and gates (`check` and `fit`)

- Identifiability (mean_bkt): `lambda_saturated` := lambda > 0.999 -> p_L, m0, mu are reported as "n.i." (not identified) and never interpreted. Perturbation test for every free raw of mean_bkt: replacing r_i by 0 (parameter = 0.5, a = 1, b = 0) changes LL_val by < 1e-4 -> `unidentified_i` = True, reported as "n.i.".
- geo sensitivity grid (validation, descriptive): AUC_val of geo for a in {0.5, 1, 2, 4} x b in {-2, 0, 2} (s = g = 0.1); if the range is < 0.002 the geo AUC is declared effectively parameter-free on that backbone.
- Convex anchors (nesting; must hold, else the fallback of Sec. 5, then "optimiser failure" if still violated). A_min: LL_val(min) <= LL_val(logistic regression y ~ min_k d_jk, C = 1e6, max_iter 2000) + 5e-4. A_bkt: LL_val(mean_bkt, every mode) <= LL_val(mean_platt with stored Platt parameters) + 5e-4. A_temp: LL_val(tempered_and) <= LL_val(geo) + 5e-4. A_min and A_bkt are enforced only when the anchor's slope (LR slope / stored a_platt) is positive and its (a, b) lie inside the box; otherwise (zero-shot: a_platt = -0.040) they are reported only.
- Also reported: share of validation KC slots with a prior occurrence in the dialogue (61.2%) — the maximal leverage of the temporal filter.

## 7. Test evaluation (`evaluate` command; exactly once)

- Preconditions: `fits_qlora_seed221.json` and `fits_zero_shot_seed-na.json` exist with a selected start for all 14 fits and `preregistration.json` names a primary variant per backbone; the sha256 of each fits JSON matches `preregistration.json`; the sentinel `redesign/TEST_EVALUATED` does not exist. There is no force/override flag. `evaluate` appends one line (timestamp, script sha256, fits sha256s) to the append-only `test_evaluations.log` and creates the sentinel before writing results. Deleting the sentinel or re-running is a protocol deviation that must be disclosed in the paper together with the numbers of every run.
- One pass per backbone: load test rows and test verdicts once; compute predictions for every candidate (mean_bkt modes with test verdicts / test labels / own predictions, causal) and every reference row of Sec. 3, all in one function; the variant list is read from `preregistration.json`, not from code.
- Bootstrap: `paired_bootstrap(y_test, predictions, dialogues, reference="mean_raw")` called ONCE per backbone with ALL candidate and reference predictions in one dict (N_BOOT = 1000, seed 221, dialogue-level; identical resamples across calls). Three further descriptive contrast calls per backbone with the same dict: reference = "noisy_and_platt" (counting effect for geo, min, tempered_and), reference = "mean_bkt_none" (label-update effect for mean_bkt model/sava/gold), reference = "label_hist_gold" (for mean_bkt_gold) and reference = "label_hist_sava" (for mean_bkt_sava). The mean_raw CI must be identical in every call (assertion).
- Test log-loss is reported for every fitted variant and reference (except label_hist); the calibration reference is mean_platt, never mean_raw (uncalibrated).
- Descriptive stratum AUCs by n_j in {1, 2, 3, >= 4} for every variant on test (computed inside the single pass).
- Outputs: `test_results.csv` (one row per backbone x variant x mode, columns: dataset, model, seed, variant, family, mode, k_free, ll_val, auc_val, selection_score, n, auc, ci_low, ci_high, delta_vs_mean_raw, delta_ci_low, delta_ci_high, ll_test, auc_n1, auc_n2, auc_n3, auc_n4plus, converged, hit_maxiter, fallback_used, n_starts_agree, restart_sensitive, at_bound, not_identified, params_natural_json, raw_json, is_primary, decision, is_reference, table5_reproduced), `test_contrasts.csv`, `test_predictions_<tag>.csv` (dialogue, turn_id, label, every variant), `regression_checks_test.json`, `manifest.json`, `PREREGISTRATION.md`.

## 8. Primary selection and improvement criterion (exact)

- Primary candidate per backbone B, decided in `fit` before any test access: V*_B = argmin over E = {geo, min, mean_bkt_none, mean_bkt_model} of selection_score = LL_val + k/N_val with k = 4, 4, 7, 8 and N_val = 1,755 (the k/N term removes the differential in-sample optimism of different parameter counts). Ties within 1e-10 are broken in the order listed. Excluded from E: tempered_and (diagnostic; nests geo), mean_bkt_sava and mean_bkt_gold (use verifier/oracle information that mean_raw does not have).
- Statistic: Delta_V = AUC_test(V) - AUC_test(mean_raw); [L, U] = `paired_bootstrap(...)["delta_ci"]` (2.5%/97.5% percentiles over 1,000 dialogue resamples, seed 221).
- Decision for V*_B (mutually exclusive, tested in this order): "improves" iff L > 0; "harms" iff U < 0 (sub-flag "within margin" iff L > -0.010); "non-inferior" iff L > -0.010 (and neither of the above); "inconclusive" otherwise.
- Claim rule: the paper may state that the re-specified CCMF improves on LLMKT mean pooling only if V*_qlora is "improves" (delta CI excluding 0 on the QLoRA backbone). If it improves with Delta < 0.005 the paper must state the magnitude explicitly. Zero-shot is a robustness replicate (mean_raw test AUC 0.517 is at chance): its decision is computed and reported identically but supports no claim. All other rows (6 other candidates per backbone, references, contrasts) are descriptive secondary results with CIs; no improvement claim is made from a secondary CI whatever its sign; a mean_bkt_gold/sava row is called a "label-history effect" unless its delta vs label_hist_* has L > 0 as well. Log-loss carries no claim. No multiplicity correction is applied because exactly one confirmatory test per backbone exists.

## 9. Regression checks tying the script to Table 5 (must pass; tolerances are frozen)

- C1 (check/fit, val): 1,755 turns / 415 dialogues, contiguous dialogue blocks, all gaps finite, no duplicate KC within a turn, verdict CSV alignment and label equality; same on test inside `evaluate` (1,985 / 515).
- C2 (fit, val): for each backbone and mode, `log_loss(y_val, predict_ccmf(val_rows, stored_raw, mode, val_verdicts))` equals `min(f["loss"] for f in fits[mode])` in the stored JSON to 1e-8 (QLoRA: none 0.6007795927779107, model 0.599818498201986, gold 0.5998184982019859, sava 0.5998184982019863; zero-shot: none 0.690159266702512, model 0.6889917959138967, gold 0.680288906210362, sava 0.6820732051313364). The zero-shot sava check (lambda = 0, mu = 1) validates that the `sava` column is the verdict version used for Table 5; on failure the pipeline stops before any test access (no column switching).
- C3 (fit, val, informational): `fit_platt(val_rows)` vs stored (a, b) (QLoRA 0.7441414919162463 / 0.28250141263034617; zero-shot -0.03962046987209394 / -0.10515468426417889) and `fit_noisy_and(val_rows, a, b)` vs stored (s, g) (QLoRA 7.33e-15 / 0.2623607935075742; zero-shot 0.48999999999999455 / 0.4716476250264483): |diff| > 1e-6 is logged as a library-version warning; the stored values are what the reference rows use.
- C4 (evaluate, test, inside the single pass; MANDATED gate): mean_raw, mean_platt, noisy_and_platt and the four ccmf_* rows recomputed from the stored parameters must match Table 5 (`ccmf_real_logits_ablation.csv`, dataset == mathdial) within 0.001 AUC (QLoRA 0.7453415453527434, 0.745318130917235, 0.69595337473277, 0.7214980148630764, 0.7238572737452917, sava 0.7238582917642268, gold 0.7238582917642268; zero-shot 0.5169413621093352, 0.4863188435304897, 0.4634042553191489, 0.5347327700295226, 0.5448971800875496, sava 0.5488806881808002, gold 0.6073236282194849). Expected agreement is 1e-9; the max |diff| is written to `regression_checks_test.json` and any |diff| > 1e-9 is disclosed. Also: per-turn predictions vs `ccmf_test_predictions_mathdial_<tag>.csv` (max |diff| reported, gate 1e-6) and bootstrap `ci` / `delta_ci` of the seven reproduced rows vs the CSV to 1e-9 (e.g. QLoRA ccmf_no_update delta CI [-0.0319051456654427, -0.0160316886303503], noisy_and_platt [-0.0632739220019048, -0.0354255335519407], mean_raw CI [0.7219641455999233, 0.7687610206560572]; zero-shot ccmf_gold_update delta CI [0.0531225351405406, 0.13334767688395]) — verifies seed, N_BOOT and dialogue grouping.
- C5 (check, val, engine equality, 1e-12 on predictions): `predict_agg(agg=np.prod)` equals `predict_ccmf` for the 4 stored raw vectors and all 6 mean_bkt starts in all 4 modes; any vectorised mean_bkt engine equals `predict_agg(agg=np.mean)` on the same inputs; vectorised static predictors equal a naive per-row loop for all starts; tempered_and with gamma = 1 equals geo, with gamma = 0 and the stored Platt (a, b), (s, g) equals noisy_and_platt (val LL QLoRA 0.62426); mean_bkt mode none with lambda = 1 (natural-parameter call) equals mean of sigma(a d + b) with the same (a, b, s, g).
- C6 (check): transforms respect the boxes for r in {-1e6, -8, 0, 8, 1e6}; 1 - s - g >= 0.02; min AUC invariance (Sec. 2) on val; geo/tempered_and/mean_bkt(none) AUC unchanged when (s, g) are replaced by (0.1, 0.1).
- C7 (check, val): state isolation — predictions of 20 dialogues (first 20 in file order) are identical (1e-12) when all other dialogues are removed; causality — permuting labels/verdicts of turns >= j within a dialogue leaves yhat_j unchanged (modes sava/gold, label_hist).
- C8: `manifest.json["fit_opened_paths"]` contains no "/test/" or "_test" path; sentinel created exactly once; fits sha256 verified; environment logged.
- C9: anchors A_min, A_bkt, A_temp (Sec. 6) evaluated and logged.

## 10. Expected behaviour (pre-registered predictions, falsifiable)
- QLoRA: geo, min, tempered_and, mean_bkt_none within +-0.005 of mean_raw (0.745): expected decision "non-inferior", not "improves" (within-turn gap spread median 0.06 logits, Spearman(min, mean) = 0.999; the noisy-AND loss came from n_j scaling of the product while the label rate rises with n_j). gamma_hat >= 0.9 with LL_val decreasing in gamma. lambda_hat near 1 again (lambda_saturated), so model/sava/gold rows coincide with none within 0.003. label_hist_gold ~ 0.57.
- Zero-shot: every label-free variant inside mean_raw's CI [0.483, 0.549]; mean_bkt gold/sava track label_hist within +-0.02 with lambda -> 0, mu -> 1 (label-history effect).
- Convergence: >= 90% of starts converged; no selected fit non-converged after continuation/fallback.

## 11. Reporting in the paper
Extend Table 5 with the 7 candidate rows and the 2 label_hist rows per backbone (Table-5 rows re-derived by the script), add LL_val and AUC_val columns, mark V* and its decision, print natural parameters with at_bound / "n.i." flags, state the number of converged starts, fallbacks, and the count of test evaluations from `test_evaluations.log`.
"""

VARIANT_SPEC_JSON = r"""[{"id":"geo","family":"F1 (mandatory): geometric mean of Platt-calibrated per-KC probabilities, then slip/guess (stateless)","formula":"z_jk = sigmoid(a*d_jk + b); P_j = (prod_{k in C_j} z_jk)^(1/n_j) = exp((1/n_j) * sum_k log z_jk) with log z = -logaddexp(0, -(a*d+b)); yhat_j = (1-s)*P_j + g*(1-P_j); clip to [1e-6, 1-1e-6]; transforms via unpack_ccmf([r_a, r_b, 0,0,0,0, r_s, r_g]): a = exp(clip(r_a,-3,3)), b = 4*tanh(r_b), s = 0.49*expit(r_s), g = 0.49*expit(r_g); raw box r_a in [-3,3], r_b in [-3,3], r_s, r_g in [-8,8]","params":["a","b","s","g"],"label_update_modes":["none"],"auc_parameter_free":false},{"id":"min","family":"F2 (mandatory): weakest KC (min over KCs of Platt-calibrated probabilities), then slip/guess (stateless)","formula":"P_j = min_k sigmoid(a*d_jk + b) = sigmoid(a * min_k d_jk + b) (a > 0); yhat_j = (1-s)*P_j + g*(1-P_j); clip to [1e-6, 1-1e-6]; same transforms and raw box as geo; script asserts AUC(yhat_min) == AUC(min_k d_jk) to 1e-12 on val and test","params":["a","b","s","g"],"label_update_modes":["none"],"auc_parameter_free":true},{"id":"mean_bkt","family":"F3 (mandatory): BKT-style temporal filter (Eq. 2) + label update (Eq. 4) with the MEAN over the turn's KCs replacing the product in Eq. 3, then slip/guess","formula":"for k in C_j (m_k init m0 at first occurrence in the dialogue; state reset at dialogue change): prior_k = m_k + (1-m_k)*p_L; m~_k = (1-lambda)*prior_k + lambda*sigmoid(a*d_jk + b); P_j = (1/n_j) * sum_{k in C_j} m~_k; yhat_j = (1-s)*P_j + g*(1-P_j); after yhat_j is recorded: m_k <- m~_k if u_j is None else (1-mu)*m~_k + mu*u_j (ccmf_commit), u_j = None (none) | unclipped yhat_j (model) | SAVA verdict correct=1/incorrect=0/undetermined=None (sava) | y_j (gold); clip to [1e-6, 1-1e-6]; transforms = unpack_ccmf verbatim on [r_a, r_b, r_lambda, r_mu, r_pL, r_m0, r_s, r_g]; raw box r_a, r_b in [-3,3], others in [-8,8]; in mode none r_mu is fixed at 0 and removed from the optimised vector (7 free)","params":["a","b","lambda","mu","p_L","m0","s","g"],"label_update_modes":["none","model","sava","gold"],"auc_parameter_free":false},{"id":"tempered_and","family":"F4 (single extra family, justified by two proposals from the double-counting diagnosis; diagnostic candidate excluded from the primary set): tempered conjunction nesting the noisy-AND (gamma=0) and geo (gamma=1)","formula":"log P_j = n_j^(-gamma) * sum_{k in C_j} log sigmoid(a*d_jk + b), i.e. P_j = (prod_k z_jk)^(n_j^-gamma); yhat_j = (1-s)*P_j + g*(1-P_j); clip to [1e-6, 1-1e-6]; gamma = expit(r_gamma) in (0,1), r_gamma in [-8,8]; a, b, s, g as in geo; raw order [r_a, r_b, r_gamma, r_s, r_g]; validation-only gamma profile at gamma in {0, 0.25, 0.5, 0.75, 1} with (a,b,s,g) refit","params":["a","b","gamma","s","g"],"label_update_modes":["none"],"auc_parameter_free":false}]"""

# --------------------------------------------------------------------------------------
# Leakage guard (installed for `check` and `fit`)
# --------------------------------------------------------------------------------------
# The frozen pre-registration has no trailing newline; drop the one added by the closing quotes so that
# PREREGISTRATION.md is byte-identical to it.
if PREREG_TEXT.endswith("\n"):
    PREREG_TEXT = PREREG_TEXT[:-1]

_ORIG_OPEN = builtins.open
OPENED_PATHS = []
GUARD_ACTIVE = False


def _is_test_path(p):
    return "/test/" in p or "_test" in p


def _guarded_open(file, *args, **kwargs):
    if isinstance(file, (str, bytes, os.PathLike)):
        p = os.fspath(file)
        p = p.decode(errors="replace") if isinstance(p, bytes) else p
        if _is_test_path(p):
            raise RuntimeError(f"LEAKAGE GUARD: attempt to open a test path during fit/check: {p}")
        OPENED_PATHS.append(p)
    return _ORIG_OPEN(file, *args, **kwargs)


def install_guard():
    global GUARD_ACTIVE
    builtins.open = _guarded_open
    io.open = _guarded_open
    GUARD_ACTIVE = True


def remove_guard():
    global GUARD_ACTIVE
    builtins.open = _ORIG_OPEN
    io.open = _ORIG_OPEN
    GUARD_ACTIVE = False


def guarded_path(path, stage):
    """The single loader gate: every path opened in check/fit passes through here."""
    p = str(path)
    if stage in ("check", "fit") and _is_test_path(p):
        raise RuntimeError(f"LEAKAGE GUARD (loader): {p}")
    return Path(p)


def read_text(path, stage):
    return guarded_path(path, stage).read_text()


def read_json(path, stage):
    return json.loads(read_text(path, stage))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------
# Reused code: exec cell26.txt exactly as ccmf_refit_sava_update.py does
# --------------------------------------------------------------------------------------
G = {"np": np, "roc_auc_score": roc_auc_score, "CCMF_MAXITER": 1500, "CCMF_RESTARTS": 3,
     "PRIMARY_SEED": SEED, "N_BOOT": 1000}
_CELL26_SRC = CELL26.read_text().split("ccmf_rows, CCMF_FITS")[0]
exec(_CELL26_SRC, G)
CCMF_STARTS = G["CCMF_STARTS"]
unpack_ccmf = G["unpack_ccmf"]
ccmf_predict_turn = G["ccmf_predict_turn"]
ccmf_commit = G["ccmf_commit"]
predict_ccmf = G["predict_ccmf"]
fit_platt = G["fit_platt"]
fit_noisy_and = G["fit_noisy_and"]
paired_bootstrap = G["paired_bootstrap"]
assert len(CCMF_STARTS) == 5


def mean_raw(rows):
    return np.array([expit(r["gaps"]).mean() for r in rows])


# --------------------------------------------------------------------------------------
# Variant definitions (raw orders, boxes, starts)
# --------------------------------------------------------------------------------------
STATIC_NAMES = ["r_a", "r_b", "r_s", "r_g"]
TEMP_NAMES = ["r_a", "r_b", "r_gamma", "r_s", "r_g"]
BKT_NAMES = ["r_a", "r_b", "r_lambda", "r_mu", "r_pL", "r_m0", "r_s", "r_g"]
BKT_NONE_FREE = [0, 1, 2, 4, 5, 6, 7]
STATIC_LO, STATIC_HI = np.array([-3., -3., -8., -8.]), np.array([3., 3., 8., 8.])
TEMP_LO, TEMP_HI = np.array([-3., -3., -8., -8., -8.]), np.array([3., 3., 8., 8., 8.])
BKT_LO, BKT_HI = np.array([-3., -3.] + [-8.] * 6), np.array([3., 3.] + [8.] * 6)

STATIC_STARTS = [np.array(v, float) for v in
                 [[0, 0, -2, -2], [0.3, -0.2, -1.5, -2.5], [-0.3, 0.2, -2.5, -1.5],
                  [0, 0, -1, -3], [0.6, -0.4, -3, -1], [0, 0, -8, -8]]]
for _s, _c in zip(STATIC_STARTS[:5], CCMF_STARTS):
    assert np.array_equal(_s, _c[[0, 1, 6, 7]]), "T1-T5 must be the (0,1,6,7) projection of CCMF_STARTS"
TEMP_STARTS = [np.insert(v, 2, 0.0) for v in STATIC_STARTS] + \
              [np.array([0, 0, -4, -2, -2], float), np.array([0, 0, 4, -2, -2], float)]
BKT_STARTS = [np.array(v, float) for v in CCMF_STARTS] + [np.array([0, 0, 6, 0, -3, 0, -8, -8], float)]
GAMMA_PROFILE = [0.0, 0.25, 0.5, 0.75, 1.0]

VARIANTS = {
    "geo": dict(family="geo", mode="none", k_free=4, names=STATIC_NAMES, lo=STATIC_LO, hi=STATIC_HI,
                starts=STATIC_STARTS),
    "min": dict(family="min", mode="none", k_free=4, names=STATIC_NAMES, lo=STATIC_LO, hi=STATIC_HI,
                starts=STATIC_STARTS),
    "tempered_and": dict(family="tempered_and", mode="none", k_free=5, names=TEMP_NAMES, lo=TEMP_LO, hi=TEMP_HI,
                         starts=TEMP_STARTS),
    "mean_bkt_none": dict(family="mean_bkt", mode="none", k_free=7, names=[BKT_NAMES[i] for i in BKT_NONE_FREE],
                          lo=BKT_LO[BKT_NONE_FREE], hi=BKT_HI[BKT_NONE_FREE],
                          starts=[s[BKT_NONE_FREE] for s in BKT_STARTS]),
}
for _m in ["model", "sava", "gold"]:
    VARIANTS[f"mean_bkt_{_m}"] = dict(family="mean_bkt", mode=_m, k_free=8, names=BKT_NAMES, lo=BKT_LO, hi=BKT_HI,
                                      starts=BKT_STARTS)
assert list(VARIANTS) == FIT_ORDER


def to_raw8(family, mode, x):
    """Free raw vector -> (8-vector for unpack_ccmf, gamma or None)."""
    x = np.asarray(x, float)
    if family in ("geo", "min"):
        return np.array([x[0], x[1], 0., 0., 0., 0., x[2], x[3]]), None
    if family == "tempered_and":
        return np.array([x[0], x[1], 0., 0., 0., 0., x[3], x[4]]), float(expit(x[2]))
    if family == "mean_bkt":
        if mode == "none":
            assert len(x) == 7
            return np.insert(x, 3, 0.0), None
        assert len(x) == 8
        return x.copy(), None
    raise ValueError(family)


def natural_params(family, mode, x):
    raw8, gamma = to_raw8(family, mode, x)
    a, b, lam, mu, pL, m0, s, g = unpack_ccmf(raw8)
    if family in ("geo", "min"):
        return {"a": a, "b": b, "s": s, "g": g}
    if family == "tempered_and":
        return {"a": a, "b": b, "gamma": gamma, "s": s, "g": g}
    out = {"a": a, "b": b, "lambda": lam, "mu": mu, "p_L": pL, "m0": m0, "s": s, "g": g}
    if mode == "none":
        out["mu"] = None  # unused in mode none (r_mu fixed at 0, not optimised)
    return out


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def load_rows(model, seed, split, stage):
    path = RUN / "mathdial" / model / seed / split / "raw_logit_gaps.json"
    if stage in ("check", "fit"):
        assert split == "val", "check/fit may only load the validation split"
    rows = read_json(path, stage)
    for r in rows:
        r["gaps"] = np.asarray(r["logit_gaps"], float)
        r["label"] = int(r["label"])
        r["turn_id"] = str(r["turn_id"])
        r["dialogue"] = str(r["dialogue"])
    return rows


def load_verdicts(split, rows, stage):
    path = KQ / "02_sava" / f"real_turns_mathdial_{split}.csv"
    if stage in ("check", "fit"):
        assert split == "val"
    csv = pd.read_csv(guarded_path(path, stage), dtype={"dialogue": str, "turn_id": str})
    assert [(r["dialogue"], r["turn_id"]) for r in rows] == list(zip(csv.dialogue, csv.turn_id)), "verdict alignment"
    assert csv.label.astype(int).tolist() == [r["label"] for r in rows], "verdict label equality"
    Y = {"correct": 1, "incorrect": 0, "undetermined": None}
    return [Y[v] for v in csv.sava]


def load_stored(tag, stage):
    return read_json(KQ / "03_ccmf" / "table5_reference_inputs" / f"ccmf_parameters_mathdial_{tag}.json", stage)


class Prep:
    """Flattened / time-stepped views of a row list (validation or test)."""

    def __init__(self, rows, verdicts):
        self.rows = rows
        self.verdicts = verdicts
        self.y = np.array([r["label"] for r in rows], int)
        self.n = np.array([len(r["gaps"]) for r in rows], int)
        self.off = np.concatenate([[0], np.cumsum(self.n)[:-1]])
        self.G = np.concatenate([r["gaps"] for r in rows])
        self.mind = np.minimum.reduceat(self.G, self.off)
        self.dialogues = np.array([r["dialogue"] for r in rows])
        pair_ids, pair, seen_before = {}, np.empty(len(self.G), int), np.zeros(len(self.G), bool)
        slot = 0
        for r in rows:
            for kc in r["kcs"]:
                key = (r["dialogue"], kc)
                if key in pair_ids:
                    seen_before[slot] = True
                pair[slot] = pair_ids.setdefault(key, len(pair_ids))
                slot += 1
        self.pair, self.n_pairs, self.seen_before = pair, len(pair_ids), seen_before
        pos, cur, t = [], None, 0
        for r in rows:
            if r["dialogue"] != cur:
                cur, t = r["dialogue"], 0
            pos.append(t)
            t += 1
        pos = np.array(pos)
        self.steps = []
        for t in range(pos.max() + 1):
            rows_t = np.flatnonzero(pos == t)
            slots_t = np.concatenate([np.arange(self.off[i], self.off[i] + self.n[i]) for i in rows_t])
            n_t = self.n[rows_t]
            red = np.concatenate([[0], np.cumsum(n_t)[:-1]])
            self.steps.append((rows_t, slots_t, red, n_t.astype(float), n_t))
        self.u_gold = self.y.astype(float)
        self.u_sava = (np.array([np.nan if v is None else float(v) for v in verdicts])
                       if verdicts is not None else None)


# --------------------------------------------------------------------------------------
# Predictors
# --------------------------------------------------------------------------------------
def agg_predict_turn(state, kcs, gaps, params, agg):
    """ccmf_predict_turn with agg(values) in place of np.prod (otherwise verbatim)."""
    a, b, lam, mu, p_learn, m0, s, g = params
    observations = expit(a * np.asarray(gaps, dtype=float) + b)
    current, values = {}, []
    for kc, obs in zip(kcs, observations):
        previous = state.get(kc, m0)
        prior = previous + (1 - previous) * p_learn
        value = (1 - lam) * prior + lam * obs
        current[kc] = value
        values.append(value)
    pooled = float(agg(values))
    return (1 - s) * pooled + g * (1 - pooled), current


def predict_agg(rows, raw, mode, verdicts=None, agg=np.mean, params=None):
    """Copy of predict_ccmf with agg_predict_turn; ccmf_commit reused verbatim."""
    params = unpack_ccmf(raw) if params is None else params
    state, current_dialogue, predictions = {}, None, []
    for i, row in enumerate(rows):
        if row["dialogue"] != current_dialogue:
            state, current_dialogue = {}, row["dialogue"]
        prediction, current = agg_predict_turn(state, row["kcs"], row["gaps"], params, agg)
        predictions.append(prediction)
        label = {"none": None, "model": prediction, "gold": row["label"],
                 "sava": verdicts[i] if verdicts is not None else None}[mode]
        ccmf_commit(state, current, label, params[3])
    return np.clip(np.array(predictions), CLIP[0], CLIP[1])


def bkt_engine(prep, params, mode, agg="mean"):
    """Vectorised time-step engine (one numpy pass per within-dialogue position)."""
    a, b, lam, mu, pL, m0, s, g = params
    obs = expit(a * prep.G + b)
    m = np.full(prep.n_pairs, m0, float)
    yhat = np.empty(len(prep.y))
    u_all = {"none": None, "model": None, "gold": prep.u_gold, "sava": prep.u_sava}[mode]
    for rows_t, slots_t, red, n_tf, n_ti in prep.steps:
        pidx = prep.pair[slots_t]
        prev = m[pidx]
        prior = prev + (1 - prev) * pL
        val = (1 - lam) * prior + lam * obs[slots_t]
        P = np.add.reduceat(val, red) / n_tf if agg == "mean" else np.multiply.reduceat(val, red)
        pred = (1 - s) * P + g * (1 - P)
        yhat[rows_t] = pred
        if mode == "none":
            new = val
        else:
            u = pred if mode == "model" else u_all[rows_t]
            us = np.repeat(u, n_ti)
            new = np.where(np.isnan(us), val, (1 - mu) * val + mu * us)
        m[pidx] = new
    return np.clip(yhat, CLIP[0], CLIP[1])


USE_VEC_BKT = False  # switched on only after C5 passes


def predict_bkt(prep, params, mode):
    if USE_VEC_BKT:
        return bkt_engine(prep, params, mode, "mean")
    raw_dummy = None
    return predict_agg(prep.rows, raw_dummy, mode, prep.verdicts, np.mean, params=params)


def static_predict(prep, kind, a, b, s, g, gamma=None):
    if kind == "min":
        P = expit(a * prep.mind + b)
    else:
        logz = -np.logaddexp(0, -(a * prep.G + b))
        sums = np.add.reduceat(logz, prep.off)
        if kind == "geo":
            P = np.exp(sums / prep.n)
        elif kind == "tempered_and":
            P = np.exp(sums * prep.n.astype(float) ** (-gamma))
        else:
            raise ValueError(kind)
    return np.clip((1 - s) * P + g * (1 - P), CLIP[0], CLIP[1])


def static_naive(rows, kind, a, b, s, g, gamma=None):
    out = []
    for r in rows:
        z = expit(a * r["gaps"] + b)
        n = len(z)
        if kind == "geo":
            P = np.prod(z) ** (1.0 / n)
        elif kind == "min":
            P = z.min()
        else:
            P = np.prod(z) ** (float(n) ** (-gamma))
        out.append((1 - s) * P + g * (1 - P))
    return np.clip(np.array(out), CLIP[0], CLIP[1])


def predict_variant(prep, family, mode, x):
    raw8, gamma = to_raw8(family, mode, x)
    params = unpack_ccmf(raw8)
    a, b, lam, mu, pL, m0, s, g = params
    if family in ("geo", "min", "tempered_and"):
        return static_predict(prep, family, a, b, s, g, gamma)
    return predict_bkt(prep, params, mode)


def predict_label_hist(rows, mode, verdicts, m0_ref):
    hist, cur, preds = {}, None, []
    for i, r in enumerate(rows):
        if r["dialogue"] != cur:
            hist, cur = {}, r["dialogue"]
        preds.append(float(np.mean([hist.get(k, m0_ref) for k in r["kcs"]])))
        u = r["label"] if mode == "gold" else verdicts[i]
        if u is not None:
            for k in r["kcs"]:
                hist[k] = float(u)
    return np.array(preds)


def ll(y, p):
    return float(log_loss(y, np.clip(p, CLIP[0], CLIP[1]), labels=[0, 1]))


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def reference_predictions(prep, stored, m0_ref):
    """All reference rows of Sec. 3 (stored parameters, original predict_ccmf)."""
    rows, verdicts = prep.rows, prep.verdicts
    ap, bp = stored["parameters"]["platt"]["a"], stored["parameters"]["platt"]["b"]
    sp, gp = stored["parameters"]["noisy_and"]["s"], stored["parameters"]["noisy_and"]["g"]
    products = np.array([np.prod(expit(ap * r["gaps"] + bp)) for r in rows])
    preds = {"mean_raw": mean_raw(rows),
             "mean_platt": np.array([expit(ap * r["gaps"] + bp).mean() for r in rows]),
             "noisy_and_platt": (1 - sp) * products + gp * (1 - products)}
    for mode, name in CCMF_MODE_NAME.items():
        preds[name] = predict_ccmf(rows, np.array(stored["parameters"][name]["raw"], float), mode, verdicts)
    preds["label_hist_gold"] = predict_label_hist(rows, "gold", verdicts, m0_ref)
    preds["label_hist_sava"] = predict_label_hist(rows, "sava", verdicts, m0_ref)
    return preds


def all_predictions(prep, stored, m0_ref, variant_fits):
    """Single pass: references + every fitted candidate (variant_fits: name -> raw_free list)."""
    preds = reference_predictions(prep, stored, m0_ref)
    for name, x in variant_fits.items():
        v = VARIANTS[name]
        preds[name] = predict_variant(prep, v["family"], v["mode"], np.asarray(x, float))
    return {k: np.clip(np.asarray(p, float), CLIP[0], CLIP[1]) for k, p in preds.items()}


# --------------------------------------------------------------------------------------
# Optimiser (Nelder-Mead with bounds, continuation, polish)
# --------------------------------------------------------------------------------------
def make_simplex(x0, lo, hi, step):
    S = [np.array(x0, float)]
    for i in range(len(x0)):
        v = np.array(x0, float)
        v[i] += step
        if v[i] > hi[i] or v[i] < lo[i]:
            v[i] = x0[i] - step
        S.append(v)
    return np.array(S)


def nm_run(obj, x0, lo, hi, step):
    return minimize(obj, np.asarray(x0, float), method="Nelder-Mead", bounds=list(zip(lo, hi)),
                    options={**NM_OPTIONS, "initial_simplex": make_simplex(x0, lo, hi, step)})


def _rec(res, stage):
    return {"stage": stage, "success": bool(res.success), "loss": float(res.fun), "iterations": int(res.nit),
            "nfev": int(res.nfev), "message": str(res.message)}


def run_start(obj, x0, lo, hi):
    t0 = time.perf_counter()
    runs, x = [], np.asarray(x0, float)
    for c in range(1 + MAX_CONTINUATIONS):
        res = nm_run(obj, x, lo, hi, SIMPLEX_STEP)
        runs.append(_rec(res, "main" if c == 0 else "continuation"))
        x = res.x
        hit = res.nit >= NM_OPTIONS["maxiter"] or res.nfev >= NM_OPTIONS["maxfev"]
        if not hit:
            break
    hit_maxiter, last = hit, res
    best_x, best_f = np.array(res.x, float), float(res.fun)
    n_pol = 0
    for _ in range(MAX_POLISH):
        r2 = nm_run(obj, best_x, lo, hi, POLISH_STEP)
        runs.append(_rec(r2, "polish"))
        last = r2
        if r2.fun < best_f:
            imp = best_f - float(r2.fun)
            best_x, best_f, n_pol = np.array(r2.x, float), float(r2.fun), n_pol + 1
            if imp < 1e-7:
                break
        else:
            break
    return {"loss": best_f, "x": best_x.tolist(), "x0": np.asarray(x0, float).tolist(), "runs": runs,
            "nit_total": int(sum(r["iterations"] for r in runs)), "nfev_total": int(sum(r["nfev"] for r in runs)),
            "hit_maxiter": bool(hit_maxiter), "success": bool(last.success), "n_polish_accepted": int(n_pol),
            "elapsed_s": time.perf_counter() - t0}


def make_objective(prep, family, mode, lo, hi, gamma_fixed=None):
    y = prep.y
    lo_t, hi_t = lo - 1e-12, hi + 1e-12

    def obj(x):
        x = np.asarray(x, float)
        assert np.all(x >= lo_t) and np.all(x <= hi_t), ("raw vector outside the box", x)
        if gamma_fixed is not None:
            raw8 = np.array([x[0], x[1], 0., 0., 0., 0., x[2], x[3]])
            a, b, _, _, _, _, s, g = unpack_ccmf(raw8)
            yhat = static_predict(prep, "tempered_and", a, b, s, g, gamma_fixed)
        else:
            yhat = predict_variant(prep, family, mode, x)
        return log_loss(y, np.clip(yhat, CLIP[0], CLIP[1]), labels=[0, 1])
    return obj


# --------------------------------------------------------------------------------------
# Multiprocessing worker (fork context; DATA populated in the parent before the pool)
# --------------------------------------------------------------------------------------
DATA = {}


def _worker(task):
    kind, tag = task["kind"], task["tag"]
    prep = DATA[tag]["prep"]
    if kind in ("fit", "dup"):
        v = VARIANTS[task["variant"]]
        obj = make_objective(prep, v["family"], v["mode"], v["lo"], v["hi"])
        rec = run_start(obj, np.array(task["x0"], float), v["lo"], v["hi"])
    elif kind == "gamma":
        obj = make_objective(prep, "tempered_and", "none", STATIC_LO, STATIC_HI, gamma_fixed=task["gamma"])
        rec = run_start(obj, np.array(task["x0"], float), STATIC_LO, STATIC_HI)
    else:
        raise ValueError(kind)
    rec.update({k: task[k] for k in task if k != "x0"})
    return rec


def run_tasks(tasks, n_proc):
    if not tasks:
        return []
    if n_proc <= 1:
        return [_worker(t) for t in tasks]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_proc) as pool:
        return pool.map(_worker, tasks, chunksize=1)


# --------------------------------------------------------------------------------------
# Selection, flags, anchors
# --------------------------------------------------------------------------------------
def select_records(records):
    losses = [r["loss"] for r in records]
    L = min(losses)
    sel = min(i for i, l in enumerate(losses) if l <= L + 1e-10)
    srt = sorted(losses)
    return sel, {"n_starts_agree": int(sum(l <= L + 1e-4 for l in losses)),
                 "restart_sensitive": bool(srt[1] - srt[0] > 1e-3) if len(srt) > 1 else False}


def at_bound_flags(x, names, lo, hi):
    return {n: bool(abs(xi - l) < 1e-3 or abs(xi - h) < 1e-3) for n, xi, l, h in zip(names, x, lo, hi)}


def decide(L, U):
    if L > 0:
        return "improves"
    if U < 0:
        return "harms (within margin)" if L > -NON_INFERIORITY_MARGIN else "harms"
    if L > -NON_INFERIORITY_MARGIN:
        return "non-inferior"
    return "inconclusive"


def inside_box_ab(a, b):
    if not (a > 0) or abs(b) >= 4:
        return False
    return abs(np.log(a)) <= 3 and abs(np.arctanh(b / 4)) <= 3


def anchor_table(prep, stored):
    """Anchor thresholds of Sec. 6 (A_temp threshold filled after geo is fitted)."""
    y = prep.y
    lr = LogisticRegression(C=1e6, max_iter=2000, random_state=SEED).fit(prep.mind.reshape(-1, 1), y)
    lr_a, lr_b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    ll_lr = ll(y, expit(lr_a * prep.mind + lr_b))
    ap, bp = stored["parameters"]["platt"]["a"], stored["parameters"]["platt"]["b"]
    ll_mp = ll(y, np.array([expit(ap * r["gaps"] + bp).mean() for r in prep.rows]))
    return {"A_min": {"ll_anchor": ll_lr, "threshold": ll_lr + 5e-4, "lr_a": lr_a, "lr_b": lr_b,
                      "enforced": bool(lr_a > 0 and inside_box_ab(lr_a, lr_b)), "applies_to": ["min"]},
            "A_bkt": {"ll_anchor": ll_mp, "threshold": ll_mp + 5e-4, "a_platt": ap, "b_platt": bp,
                      "enforced": bool(ap > 0 and inside_box_ab(ap, bp)),
                      "applies_to": ["mean_bkt_none", "mean_bkt_model", "mean_bkt_sava", "mean_bkt_gold"]},
            "A_temp": {"ll_anchor": None, "threshold": None, "enforced": True, "applies_to": ["tempered_and"]}}


def anchor_status(anchors, variant, ll_val):
    for name, a in anchors.items():
        if variant in a["applies_to"]:
            thr = a["threshold"]
            holds = None if thr is None else bool(ll_val <= thr)
            return {"anchor": name, "threshold": thr, "holds": holds, "enforced": a["enforced"],
                    "violation": bool(a["enforced"] and holds is False)}
    return {"anchor": None, "threshold": None, "holds": None, "enforced": False, "violation": False}


# --------------------------------------------------------------------------------------
# Validation checks C1, C2, C3, C5, C6, C7
# --------------------------------------------------------------------------------------
def check_c1(rows, verdicts, n_rows, n_dialogues):
    assert len(rows) == n_rows, (len(rows), n_rows)
    ids = [r["dialogue"] for r in rows]
    assert len(set(ids)) == n_dialogues, (len(set(ids)), n_dialogues)
    blocks = 1 + sum(ids[i] != ids[i - 1] for i in range(1, len(ids)))
    assert blocks == n_dialogues, "dialogue blocks are not contiguous"
    assert all(np.all(np.isfinite(r["gaps"])) for r in rows), "non-finite gaps"
    assert all(len(r["gaps"]) == len(r["kcs"]) >= 1 for r in rows)
    assert all(len(set(r["kcs"])) == len(r["kcs"]) for r in rows), "duplicate KC within a turn"
    assert len(verdicts) == n_rows
    counts = {"correct": sum(v == 1 for v in verdicts), "incorrect": sum(v == 0 for v in verdicts),
              "undetermined": sum(v is None for v in verdicts)}
    return {"n_rows": len(rows), "n_dialogues": len(set(ids)), "verdict_counts": counts, "passed": True}


def check_c2(prep, stored):
    out, ok = {}, True
    for mode, name in CCMF_MODE_NAME.items():
        raw = np.array(stored["parameters"][name]["raw"], float)
        got = ll(prep.y, predict_ccmf(prep.rows, raw, mode, prep.verdicts))
        exp = min(f["loss"] for f in stored["fits"][name])
        out[name] = {"recomputed": got, "stored_min_loss": exp, "abs_diff": abs(got - exp), "ok": abs(got - exp) <= 1e-8}
        ok &= out[name]["ok"]
    out["passed"] = bool(ok)
    return out


def check_c3(prep, stored):
    a, b = fit_platt(prep.rows)
    s, g = fit_noisy_and(prep.rows, a, b)
    sa, sb = stored["parameters"]["platt"]["a"], stored["parameters"]["platt"]["b"]
    ss, sg = stored["parameters"]["noisy_and"]["s"], stored["parameters"]["noisy_and"]["g"]
    diffs = {"platt_a": abs(a - sa), "platt_b": abs(b - sb), "s": abs(s - ss), "g": abs(g - sg)}
    return {"refit": {"a": a, "b": b, "s": s, "g": g}, "stored": {"a": sa, "b": sb, "s": ss, "g": sg},
            "abs_diff": diffs, "library_version_warning": bool(max(diffs.values()) > 1e-6)}


def check_c5(prep, stored):
    rows, verdicts, y = prep.rows, prep.verdicts, prep.y
    raws = [np.array(stored["parameters"][n]["raw"], float) for n in CCMF_MODE_NAME.values()] + BKT_STARTS
    max_diff = {"predict_agg_prod_vs_predict_ccmf": 0.0, "engine_mean_vs_predict_agg_mean": 0.0,
                "engine_prod_vs_predict_agg_prod": 0.0, "static_vec_vs_naive": 0.0,
                "tempered_gamma1_vs_geo": 0.0, "tempered_gamma0_vs_noisy_and_platt": 0.0,
                "bkt_lambda1_vs_mean_sigmoid": 0.0}
    for raw in raws:
        for mode in ["none", "model", "sava", "gold"]:
            ref = predict_ccmf(rows, raw, mode, verdicts)
            p1 = predict_agg(rows, raw, mode, verdicts, np.prod)
            max_diff["predict_agg_prod_vs_predict_ccmf"] = max(max_diff["predict_agg_prod_vs_predict_ccmf"],
                                                               float(np.max(np.abs(ref - p1))))
            params = unpack_ccmf(raw)
            pm_loop = predict_agg(rows, raw, mode, verdicts, np.mean)
            pm_vec = bkt_engine(prep, params, mode, "mean")
            max_diff["engine_mean_vs_predict_agg_mean"] = max(max_diff["engine_mean_vs_predict_agg_mean"],
                                                              float(np.max(np.abs(pm_loop - pm_vec))))
            pp_vec = bkt_engine(prep, params, mode, "prod")
            max_diff["engine_prod_vs_predict_agg_prod"] = max(max_diff["engine_prod_vs_predict_agg_prod"],
                                                              float(np.max(np.abs(p1 - pp_vec))))
    for kind, starts in [("geo", STATIC_STARTS), ("min", STATIC_STARTS), ("tempered_and", TEMP_STARTS)]:
        for x in starts:
            raw8, gamma = to_raw8(kind, "none", x)
            a, b, _, _, _, _, s, g = unpack_ccmf(raw8)
            d = np.max(np.abs(static_predict(prep, kind, a, b, s, g, gamma) - static_naive(rows, kind, a, b, s, g, gamma)))
            max_diff["static_vec_vs_naive"] = max(max_diff["static_vec_vs_naive"], float(d))
    for x in STATIC_STARTS:
        raw8, _ = to_raw8("geo", "none", x)
        a, b, _, _, _, _, s, g = unpack_ccmf(raw8)
        d = np.max(np.abs(static_predict(prep, "tempered_and", a, b, s, g, 1.0) - static_predict(prep, "geo", a, b, s, g)))
        max_diff["tempered_gamma1_vs_geo"] = max(max_diff["tempered_gamma1_vs_geo"], float(d))
    ap, bp = stored["parameters"]["platt"]["a"], stored["parameters"]["platt"]["b"]
    sp, gp = stored["parameters"]["noisy_and"]["s"], stored["parameters"]["noisy_and"]["g"]
    products = np.array([np.prod(expit(ap * r["gaps"] + bp)) for r in rows])
    nap = np.clip((1 - sp) * products + gp * (1 - products), CLIP[0], CLIP[1])
    t0 = static_predict(prep, "tempered_and", ap, bp, sp, gp, 0.0)
    max_diff["tempered_gamma0_vs_noisy_and_platt"] = float(np.max(np.abs(t0 - nap)))
    ll_nap = ll(y, nap)
    for x in STATIC_STARTS:
        raw8, _ = to_raw8("geo", "none", x)
        a, b, _, _, _, _, s, g = unpack_ccmf(raw8)
        params = (a, b, 1.0, 0.0, 0.3, 0.4, s, g)
        pb = predict_agg(rows, None, "none", verdicts, np.mean, params=params)
        pv = bkt_engine(prep, params, "none", "mean")
        ms = np.array([expit(a * r["gaps"] + b).mean() for r in rows])
        ms = np.clip((1 - s) * ms + g * (1 - ms), CLIP[0], CLIP[1])
        max_diff["bkt_lambda1_vs_mean_sigmoid"] = max(max_diff["bkt_lambda1_vs_mean_sigmoid"],
                                                      float(np.max(np.abs(pb - ms))), float(np.max(np.abs(pv - ms))))
    passed = all(v <= 1e-12 for v in max_diff.values())
    return {"max_abs_diff": max_diff, "ll_val_noisy_and_platt_stored": ll_nap, "passed": bool(passed)}


def check_c6(prep):
    out = {"transform_box": [], "min_auc_invariance_max_diff": 0.0, "sg_invariance_max_diff": 0.0}
    for r in [-1e6, -8.0, 0.0, 8.0, 1e6]:
        a, b, lam, mu, pL, m0, s, g = unpack_ccmf(np.full(8, r))
        ok = (np.exp(-3) <= a <= np.exp(3)) and abs(b) <= 4 and all(0 <= v <= 1 for v in (lam, mu, pL, m0)) and \
             0 <= s <= 0.49 and 0 <= g <= 0.49
        out["transform_box"].append({"r": r, "a": a, "b": b, "lambda": lam, "s": s, "g": g, "ok": bool(ok)})
    a, b, lam, mu, pL, m0, s, g = unpack_ccmf(np.array([0, 0, 0, 0, 0, 0, 8, 8.]))
    out["min_1_minus_s_minus_g_in_box"] = 1 - s - g
    assert 1 - s - g >= 0.02
    y = prep.y
    auc_mind = roc_auc_score(y, prep.mind)
    n_distinct = len(np.unique(prep.mind))
    test_vecs = STATIC_STARTS + [np.array([3, 3, 8, 8.]), np.array([-3, -3, 8, -8.]), np.array([3, -3, -8, 8.])]
    out["min_auc_saturation"], out["min_auc_starts_nonsaturating"] = [], True
    for x in test_vecs:
        raw8, _ = to_raw8("min", "none", x)
        a, b, _, _, _, _, s, g = unpack_ccmf(raw8)
        p = static_predict(prep, "min", a, b, s, g)
        d = float(abs(roc_auc_score(y, p) - auc_mind))
        new_ties = n_distinct - len(np.unique(p))
        # Logged deviation: yhat is strictly increasing in min_k d, but at extreme parameters float64 expit and the
        # slip/guess map merge distinct predictions; the invariance gate applies to vectors that add no new ties
        # (all fixed starts must be among them) and saturating vectors are recorded.
        if new_ties == 0:
            out["min_auc_invariance_max_diff"] = max(out["min_auc_invariance_max_diff"], d)
        else:
            out["min_auc_saturation"].append({"raw": x.tolist(), "a": a, "b": b, "s": s, "g": g,
                                              "new_ties": int(new_ties), "auc_abs_diff": d})
            if any(np.array_equal(x, s0) for s0 in STATIC_STARTS):
                out["min_auc_starts_nonsaturating"] = False
    for kind, starts in [("geo", STATIC_STARTS), ("tempered_and", TEMP_STARTS)]:
        for x in starts:
            raw8, gamma = to_raw8(kind, "none", x)
            a, b, _, _, _, _, s, g = unpack_ccmf(raw8)
            d = abs(roc_auc_score(y, static_predict(prep, kind, a, b, s, g, gamma)) -
                    roc_auc_score(y, static_predict(prep, kind, a, b, 0.1, 0.1, gamma)))
            out["sg_invariance_max_diff"] = max(out["sg_invariance_max_diff"], float(d))
    for x in BKT_STARTS:
        p = list(unpack_ccmf(x))
        p2 = p[:6] + [0.1, 0.1]
        d = abs(roc_auc_score(y, bkt_engine(prep, tuple(p), "none")) - roc_auc_score(y, bkt_engine(prep, tuple(p2), "none")))
        out["sg_invariance_max_diff"] = max(out["sg_invariance_max_diff"], float(d))
    out["passed"] = bool(all(t["ok"] for t in out["transform_box"]) and out["min_auc_invariance_max_diff"] <= 1e-12
                         and out["min_auc_starts_nonsaturating"] and out["sg_invariance_max_diff"] <= 1e-12)
    return out


def check_c7(prep, stored, m0_ref):
    rows, verdicts = prep.rows, prep.verdicts
    first20 = []
    for d in prep.dialogues:
        if d not in first20:
            first20.append(d)
        if len(first20) == 20:
            break
    mask = np.isin(prep.dialogues, first20)
    sub_rows = [r for r, m in zip(rows, mask) if m]
    sub_ver = [v for v, m in zip(verdicts, mask) if m]
    sub_prep = Prep(sub_rows, sub_ver)
    iso = 0.0
    vecs = [np.array(stored["parameters"][n]["raw"], float) for n in CCMF_MODE_NAME.values()] + BKT_STARTS[:2]
    for raw in vecs:
        params = unpack_ccmf(raw)
        for mode in ["none", "model", "sava", "gold"]:
            full = bkt_engine(prep, params, mode)[mask]
            sub = bkt_engine(sub_prep, params, mode)
            iso = max(iso, float(np.max(np.abs(full - sub))))
            full = predict_agg(rows, raw, mode, verdicts, np.mean)[mask]
            sub = predict_agg(sub_rows, raw, mode, sub_ver, np.mean)
            iso = max(iso, float(np.max(np.abs(full - sub))))
    for kind, x in [("geo", STATIC_STARTS[0]), ("min", STATIC_STARTS[1]), ("tempered_and", TEMP_STARTS[6])]:
        raw8, gamma = to_raw8(kind, "none", x)
        a, b, _, _, _, _, s, g = unpack_ccmf(raw8)
        iso = max(iso, float(np.max(np.abs(static_predict(prep, kind, a, b, s, g, gamma)[mask] -
                                           static_predict(sub_prep, kind, a, b, s, g, gamma)))))
    for mode in ["gold", "sava"]:
        iso = max(iso, float(np.max(np.abs(predict_label_hist(rows, mode, verdicts, m0_ref)[mask] -
                                           predict_label_hist(sub_rows, mode, sub_ver, m0_ref)))))
    # causality: within each of the first 20 dialogues, roll labels/verdicts of turns >= j by one
    caus, n_perm = 0.0, 0
    raw_c = BKT_STARTS[1]
    params_c = unpack_ccmf(raw_c)
    base = {"gold": predict_agg(sub_rows, raw_c, "gold", sub_ver, np.mean),
            "sava": predict_agg(sub_rows, raw_c, "sava", sub_ver, np.mean),
            "lh_gold": predict_label_hist(sub_rows, "gold", sub_ver, m0_ref),
            "lh_sava": predict_label_hist(sub_rows, "sava", sub_ver, m0_ref)}
    sub_dial = np.array([r["dialogue"] for r in sub_rows])
    for d in first20:
        idx = np.flatnonzero(sub_dial == d)
        for j in range(1, len(idx)):
            tail = idx[j:]
            rows_p = [dict(r) for r in sub_rows]
            ver_p = list(sub_ver)
            lab = [sub_rows[i]["label"] for i in tail]
            lab = lab[1:] + lab[:1]
            vv = [sub_ver[i] for i in tail]
            vv = vv[1:] + vv[:1]
            for i, l, v in zip(tail, lab, vv):
                rows_p[i]["label"] = l
                ver_p[i] = v
            head = idx[:j]
            for name, pred in [("gold", predict_agg(rows_p, raw_c, "gold", ver_p, np.mean)),
                               ("sava", predict_agg(rows_p, raw_c, "sava", ver_p, np.mean)),
                               ("lh_gold", predict_label_hist(rows_p, "gold", ver_p, m0_ref)),
                               ("lh_sava", predict_label_hist(rows_p, "sava", ver_p, m0_ref))]:
                caus = max(caus, float(np.max(np.abs(pred[head] - base[name][head]))))
            n_perm += 1
    return {"state_isolation_max_diff": iso, "causality_max_diff": caus, "n_causality_permutations": n_perm,
            "passed": bool(iso <= 1e-12 and caus <= 1e-12)}


def run_val_checks(tag, prep, stored, m0_ref, verbose=True):
    out = {}
    out["C1"] = check_c1(prep.rows, prep.verdicts, N_VAL, D_VAL)
    out["C2"] = check_c2(prep, stored)
    out["C3"] = check_c3(prep, stored)
    out["C5"] = check_c5(prep, stored)
    out["C6"] = check_c6(prep)
    out["C7"] = check_c7(prep, stored, m0_ref)
    out["prior_occurrence_share"] = float(prep.seen_before.mean())
    if verbose:
        for k in ["C1", "C2", "C5", "C6", "C7"]:
            print(f"[{tag}] {k} passed={out[k]['passed']}", flush=True)
        print(f"[{tag}] C2 detail:", {k: v["abs_diff"] for k, v in out["C2"].items() if k != "passed"}, flush=True)
        print(f"[{tag}] C3 abs diffs:", out["C3"]["abs_diff"], "warning:", out["C3"]["library_version_warning"], flush=True)
        print(f"[{tag}] C5 max diffs:", out["C5"]["max_abs_diff"], "LL noisy_and_platt:", out["C5"]["ll_val_noisy_and_platt_stored"], flush=True)
        print(f"[{tag}] C6:", out["C6"]["min_auc_invariance_max_diff"], out["C6"]["sg_invariance_max_diff"],
              "saturating min vectors (new ties, |dAUC|):",
              [(v["new_ties"], round(v["auc_abs_diff"], 6)) for v in out["C6"]["min_auc_saturation"]], flush=True)
        print(f"[{tag}] C7:", out["C7"]["state_isolation_max_diff"], out["C7"]["causality_max_diff"], flush=True)
        print(f"[{tag}] prior-occurrence share of KC slots: {out['prior_occurrence_share']:.4f}", flush=True)
    for k in ["C1", "C2", "C5", "C6", "C7"]:
        if not out[k]["passed"]:
            raise SystemExit(f"[{tag}] regression check {k} FAILED; stopping before any further step")
    return out


def environment():
    return {"python": sys.version, "python_executable": sys.executable, "python_path_expected": PYTHON_PATH,
            "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "pandas": pd.__version__,
            "script_sha256": sha256_file(SCRIPT_PATH), "cell26_sha256": sha256_file(CELL26),
            "preregistration_sha256": sha256_text(PREREG_TEXT)}


# --------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------
def cmd_check(args):
    install_guard()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": now_iso(), "environment": environment()}
    for model, seed, tag in BACKBONES:
        rows = load_rows(model, seed, "val", "check")
        verdicts = load_verdicts("val", rows, "check")
        prep = Prep(rows, verdicts)
        stored = load_stored(tag, "check")
        m0_ref = float(prep.y.mean())
        report[tag] = run_val_checks(tag, prep, stored, m0_ref)
        report[tag]["m0_ref"] = m0_ref
    report["opened_paths"] = sorted(set(OPENED_PATHS))
    assert not any(_is_test_path(p) for p in OPENED_PATHS)
    write_json(OUT / "check_report.json", report)
    remove_guard()
    print("check: all validation gates passed ->", OUT / "check_report.json")


# --------------------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------------------
def ckpt_path(tag, key):
    return CKPT / f"{tag}__{key}.json"


def load_ckpt(tag, key):
    p = ckpt_path(tag, key)
    return json.loads(p.read_text()) if p.exists() else {}


def save_ckpt(tag, key, obj):
    write_json(ckpt_path(tag, key), obj)


def fallback_draws():
    """All fallback starts pre-drawn in the fixed order (backbone outer, variant inner)."""
    rng = np.random.default_rng(SEED)
    draws = {}
    for _, _, tag in BACKBONES:
        for name in FIT_ORDER:
            v = VARIANTS[name]
            draws[(tag, name)] = [rng.uniform(v["lo"], v["hi"]).tolist() for _ in range(N_FALLBACK)]
    return draws


def cmd_fit(args):
    t_fit0 = time.perf_counter()
    install_guard()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    (OUT / "PREREGISTRATION.md").write_text(PREREG_TEXT)
    env = environment()
    deviations = [
        "hit_maxiter is also raised when a Nelder-Mead run exhausts maxfev=8000 before nit reaches 4000 "
        "(budget exhaustion); such runs are continued exactly like nit >= 4000 runs.",
        "Fallback starts are pre-drawn for all 14 fits from one np.random.default_rng(221) in the fixed order "
        "(backbone qlora then zero_shot; within a backbone geo, min, tempered_and, mean_bkt none/model/sava/gold) "
        "and a fit uses its own block only when triggered, so the draws are identical whichever fits trigger.",
        "Checkpoints are one JSON per (backbone, variant) start batch under 03_ccmf/checkpoints/ "
        "(fits_<tag>.json is assembled from them at the end); the determinism duplicate is never checkpointed.",
        "Starts of different fits/backbones are run in parallel worker processes (fork); results are deterministic "
        "and independent of scheduling; wall-clock is reported per start and per fit (sum of start times).",
        "C6 (min AUC invariance): the first `check` run failed with max |dAUC| = 0.0030 because at the two extreme "
        "test vectors with a = e^3 float64 expit and the slip/guess map merge hundreds of distinct predictions (new "
        "ties); all six fixed starts gave |dAUC| = 0 with no new ties. The gate now applies to vectors that add no "
        "new ties (the six starts must be among them) and saturating vectors are recorded. Found before any fit and "
        "before any test access.",
        "The min AUC invariance of the fitted parameters is recorded on validation and test (number of new ties and "
        "a parameter-free flag) instead of asserted, so that a saturating fit is reported rather than aborting the "
        "single test evaluation; test-data integrity checks run before any test prediction is computed.",
        "The determinism duplicate of geo/QLoRA runs in worker processes rather than in the parent process.",
        "Workers are forked with VECLIB_MAXIMUM_THREADS=1, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and "
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES (macOS Accelerate fork safety); a smoke test gave bit-identical "
        "parallel and sequential fits.",
        "The embedded pre-registration text carried a trailing newline absent from the frozen spec; it is stripped "
        "so PREREGISTRATION.md is byte-identical to the frozen pre-registration.",
    ]
    manifest = {"stage": "fit", "started": now_iso(), "environment": env, "deviations": deviations,
                "preregistration_sha256": env["preregistration_sha256"]}
    write_json(OUT / "manifest.json", manifest)
    n_proc = args.procs or max(1, os.cpu_count() or 1)
    print(f"fit: {n_proc} worker processes; python {sys.executable}", flush=True)

    # ---- load validation data, run gates -------------------------------------------------
    checks, stored_all, anchors_all = {}, {}, {}
    for model, seed, tag in BACKBONES:
        rows = load_rows(model, seed, "val", "fit")
        verdicts = load_verdicts("val", rows, "fit")
        prep = Prep(rows, verdicts)
        stored = load_stored(tag, "fit")
        m0_ref = float(prep.y.mean())
        DATA[tag] = {"prep": prep, "stored": stored, "m0_ref": m0_ref}
        stored_all[tag] = stored
        checks[tag] = run_val_checks(tag, prep, stored, m0_ref)
        checks[tag]["m0_ref"] = m0_ref
        anchors_all[tag] = anchor_table(prep, stored)
    global USE_VEC_BKT
    USE_VEC_BKT = all(checks[t]["C5"]["passed"] for t in TAGS)
    print("vectorised mean_bkt engine enabled:", USE_VEC_BKT, flush=True)
    write_json(OUT / "regression_checks_val.json", checks)

    # ---- batch 1: fixed starts, gamma profile, determinism duplicate ----------------------
    draws = fallback_draws()
    records = {}   # (tag, variant) -> {"fixed": {idx: rec}, "fallback": {idx: rec}}
    gamma_recs = {}  # (tag, gamma) -> {idx: rec}
    tasks = []
    for _, _, tag in BACKBONES:
        for name in FIT_ORDER:
            ck = load_ckpt(tag, name)
            records[(tag, name)] = {"fixed": {int(k): v for k, v in ck.get("fixed", {}).items()},
                                   "fallback": {int(k): v for k, v in ck.get("fallback", {}).items()}}
            for i, x0 in enumerate(VARIANTS[name]["starts"]):
                if i not in records[(tag, name)]["fixed"]:
                    tasks.append({"kind": "fit", "tag": tag, "variant": name, "start": i, "x0": x0.tolist()})
        for gm in GAMMA_PROFILE:
            ck = load_ckpt(tag, f"gamma_{gm}")
            gamma_recs[(tag, gm)] = {int(k): v for k, v in ck.get("fixed", {}).items()}
            for i, x0 in enumerate(STATIC_STARTS):
                if i not in gamma_recs[(tag, gm)]:
                    tasks.append({"kind": "gamma", "tag": tag, "gamma": gm, "start": i, "x0": x0.tolist()})
    for i, x0 in enumerate(VARIANTS["geo"]["starts"]):
        tasks.append({"kind": "dup", "tag": "qlora_seed221", "variant": "geo", "start": i, "x0": x0.tolist()})
    print(f"batch 1: {len(tasks)} start tasks", flush=True)
    t0 = time.perf_counter()
    results = run_tasks(tasks, n_proc)
    print(f"batch 1 wall-clock: {time.perf_counter() - t0:.1f}s", flush=True)
    dup_recs = {}
    for rec in results:
        if rec["kind"] == "fit":
            records[(rec["tag"], rec["variant"])]["fixed"][rec["start"]] = rec
        elif rec["kind"] == "gamma":
            gamma_recs[(rec["tag"], rec["gamma"])][rec["start"]] = rec
        else:
            dup_recs[rec["start"]] = rec
    for (tag, name), r in records.items():
        save_ckpt(tag, name, r)
    for (tag, gm), r in gamma_recs.items():
        save_ckpt(tag, f"gamma_{gm}", {"fixed": r})

    # determinism check (geo on QLoRA fitted twice in the same process)
    geo_q = records[("qlora_seed221", "geo")]["fixed"]
    det_ok = all(geo_q[i]["x"] == dup_recs[i]["x"] and geo_q[i]["loss"] == dup_recs[i]["loss"] for i in dup_recs)
    print("determinism check (geo/QLoRA fitted twice, bit-identical):", det_ok, flush=True)
    assert det_ok, "determinism check failed"

    # ---- selection, anchors, fallback ----------------------------------------------------
    def assemble(tag, name):
        v = VARIANTS[name]
        recs = records[(tag, name)]
        ordered = [recs["fixed"][i] for i in range(len(v["starts"]))] + \
                  [recs["fallback"][i] for i in sorted(recs["fallback"])]
        sel, agree = select_records(ordered)
        best = ordered[sel]
        x = np.array(best["x"], float)
        prep = DATA[tag]["prep"]
        pred = predict_variant(prep, v["family"], v["mode"], x)
        ll_val = ll(prep.y, pred)
        assert abs(ll_val - best["loss"]) < 1e-12, (ll_val, best["loss"])
        auc_val = safe_auc(prep.y, pred)
        return {"family": v["family"], "mode": v["mode"], "k_free": v["k_free"], "free_names": v["names"],
                "raw_free": x.tolist(), "raw8": to_raw8(v["family"], v["mode"], x)[0].tolist(),
                "natural": natural_params(v["family"], v["mode"], x),
                "starts": ordered, "selected_start": int(sel), "n_starts": len(ordered),
                "ll_val": ll_val, "auc_val": auc_val,
                "flags": {"converged": bool(best["success"] and not best["hit_maxiter"]),
                          "hit_maxiter": bool(best["hit_maxiter"]), "success": bool(best["success"]),
                          "n_polish_accepted": best["n_polish_accepted"], **agree,
                          "at_bound": at_bound_flags(x, v["names"], v["lo"], v["hi"]),
                          "fallback_used": bool(recs["fallback"]),
                          "n_starts_converged": int(sum(r["success"] and not r["hit_maxiter"] for r in ordered))},
                "wall_clock_s": float(sum(r["elapsed_s"] for r in ordered))}

    fits = {tag: {} for tag in TAGS}
    for tag in TAGS:
        for name in FIT_ORDER:
            fits[tag][name] = assemble(tag, name)
        anchors_all[tag]["A_temp"]["ll_anchor"] = fits[tag]["geo"]["ll_val"]
        anchors_all[tag]["A_temp"]["threshold"] = fits[tag]["geo"]["ll_val"] + 5e-4
        for name in FIT_ORDER:
            fits[tag][name]["anchor"] = anchor_status(anchors_all[tag], name, fits[tag][name]["ll_val"])

    triggered = [(tag, name) for tag in TAGS for name in FIT_ORDER
                 if (not fits[tag][name]["flags"]["converged"] or fits[tag][name]["anchor"]["violation"])
                 and not records[(tag, name)]["fallback"]]
    print("fallback triggered for:", triggered, flush=True)
    tasks = [{"kind": "fit", "tag": tag, "variant": name, "start": len(VARIANTS[name]["starts"]) + i, "x0": x0}
             for tag, name in triggered for i, x0 in enumerate(draws[(tag, name)])]
    if tasks:
        t0 = time.perf_counter()
        for rec in run_tasks(tasks, n_proc):
            records[(rec["tag"], rec["variant"])]["fallback"][rec["start"]] = rec
        print(f"batch 2 (fallback) wall-clock: {time.perf_counter() - t0:.1f}s", flush=True)
        for tag, name in triggered:
            save_ckpt(tag, name, records[(tag, name)])
            fits[tag][name] = assemble(tag, name)
        for tag in TAGS:
            anchors_all[tag]["A_temp"]["ll_anchor"] = fits[tag]["geo"]["ll_val"]
            anchors_all[tag]["A_temp"]["threshold"] = fits[tag]["geo"]["ll_val"] + 5e-4
            for name in FIT_ORDER:
                fits[tag][name]["anchor"] = anchor_status(anchors_all[tag], name, fits[tag][name]["ll_val"])
    for tag in TAGS:
        for name in FIT_ORDER:
            f = fits[tag][name]
            f["optimiser_failure"] = bool(not f["flags"]["converged"] or f["anchor"]["violation"])

    # ---- per-fit diagnostics --------------------------------------------------------------
    for tag in TAGS:
        prep, stored, m0_ref = DATA[tag]["prep"], DATA[tag]["stored"], DATA[tag]["m0_ref"]
        y = prep.y
        for name in FIT_ORDER:
            f, v = fits[tag][name], VARIANTS[name]
            x = np.array(f["raw_free"], float)
            ni = []
            if v["family"] == "mean_bkt":
                lam = f["natural"]["lambda"]
                f["lambda_saturated"] = bool(lam > 0.999)
                pert = {}
                for i, nm in enumerate(v["names"]):
                    x2 = x.copy()
                    x2[i] = 0.0
                    d = abs(ll(y, predict_variant(prep, v["family"], v["mode"], x2)) - f["ll_val"])
                    pert[nm] = d
                    if d < 1e-4:
                        ni.append(nm)
                f["perturbation_ll_change"] = pert
                if f["lambda_saturated"]:
                    for nm in (["p_L", "m0"] + (["mu"] if v["mode"] != "none" else [])):
                        if nm not in ni:
                            ni.append(nm)
            f["not_identified"] = ni
            if name == "min":
                pred = predict_variant(prep, "min", "none", x)
                d = abs(roc_auc_score(y, pred) - roc_auc_score(y, prep.mind))
                f["min_auc_invariance_abs_diff"] = float(d)
                f["min_new_ties_val"] = int(len(np.unique(prep.mind)) - len(np.unique(pred)))
                f["min_auc_parameter_free_val"] = bool(d <= 1e-12 and f["min_new_ties_val"] == 0)
        # geo sensitivity grid
        grid = []
        for a in [0.5, 1.0, 2.0, 4.0]:
            for b in [-2.0, 0.0, 2.0]:
                grid.append({"a": a, "b": b, "auc_val": safe_auc(y, static_predict(prep, "geo", a, b, 0.1, 0.1))})
        aucs = [g["auc_val"] for g in grid]
        # references on validation
        refs = {}
        ref_preds = reference_predictions(prep, stored, m0_ref)
        for k, p in ref_preds.items():
            refs[k] = {"auc_val": safe_auc(y, p), "ll_val": None if k.startswith("label_hist") else ll(y, p)}
        # gamma profile
        gp_rows = []
        for gm in GAMMA_PROFILE:
            ordered = [gamma_recs[(tag, gm)][i] for i in range(len(STATIC_STARTS))]
            sel, agree = select_records(ordered)
            best = ordered[sel]
            xb = np.array(best["x"], float)
            raw8 = np.array([xb[0], xb[1], 0, 0, 0, 0, xb[2], xb[3]])
            a, b, _, _, _, _, s, g = unpack_ccmf(raw8)
            pred = static_predict(prep, "tempered_and", a, b, s, g, gm)
            gp_rows.append({"tag": tag, "gamma": gm, "ll_val": ll(y, pred), "auc_val": safe_auc(y, pred),
                            "a": a, "b": b, "s": s, "g": g, "selected_start": sel,
                            "converged": bool(best["success"] and not best["hit_maxiter"]), **agree})
        pd.DataFrame(gp_rows).to_csv(OUT / f"gamma_profile_{tag}.csv", index=False)
        # selection of the primary variant
        scores = {n: fits[tag][n]["ll_val"] + VARIANTS[n]["k_free"] / N_VAL for n in FIT_ORDER}
        best_score = min(scores[n] for n in PRIMARY_SET)
        primary = next(n for n in PRIMARY_SET if scores[n] <= best_score + 1e-10)
        for n in FIT_ORDER:
            fits[tag][n]["selection_score"] = scores[n]
            fits[tag][n]["in_primary_set"] = n in PRIMARY_SET
            fits[tag][n]["is_primary"] = n == primary
        fits_json = {"tag": tag, "n_val": int(len(y)), "m0_ref": m0_ref, "primary_variant": primary,
                     "variants": fits[tag], "references": refs, "gamma_profile": gp_rows,
                     "geo_sensitivity_grid": {"grid": grid, "auc_range": float(max(aucs) - min(aucs)),
                                              "effectively_parameter_free": bool(max(aucs) - min(aucs) < 0.002)},
                     "anchors": anchors_all[tag], "prior_occurrence_share": checks[tag]["prior_occurrence_share"],
                     "checks": checks[tag], "use_vectorised_bkt_engine": USE_VEC_BKT, "timestamp": now_iso()}
        write_json(OUT / f"fits_{tag}.json", fits_json)
        write_json(OUT / f"params_{tag}.json", fits_json)
        for n in FIT_ORDER:
            f = fits[tag][n]
            print(f"[{tag}] {n:15s} LL_val={f['ll_val']:.6f} AUC_val={f['auc_val']:.4f} start={f['selected_start']} "
                  f"converged={f['flags']['converged']} agree={f['flags']['n_starts_agree']} "
                  f"fallback={f['flags']['fallback_used']} anchor_viol={f['anchor']['violation']} "
                  f"wall={f['wall_clock_s']:.1f}s", flush=True)
        print(f"[{tag}] primary variant: {primary}", flush=True)

    # ---- validation report, preregistration.json, manifest --------------------------------
    rows_out = []
    for tag in TAGS:
        fj = json.loads((OUT / f"fits_{tag}.json").read_text())
        model, seed = [(m, s) for m, s, t in BACKBONES if t == tag][0]
        for n in FIT_ORDER:
            f = fj["variants"][n]
            rows_out.append({"dataset": "mathdial", "model": model, "seed": seed, "variant": n, "family": f["family"],
                             "mode": f["mode"], "k_free": f["k_free"], "ll_val": f["ll_val"], "auc_val": f["auc_val"],
                             "selection_score": f["selection_score"], "converged": f["flags"]["converged"],
                             "hit_maxiter": f["flags"]["hit_maxiter"], "fallback_used": f["flags"]["fallback_used"],
                             "n_starts_agree": f["flags"]["n_starts_agree"], "n_starts": f["n_starts"],
                             "n_starts_converged": f["flags"]["n_starts_converged"],
                             "restart_sensitive": f["flags"]["restart_sensitive"],
                             "at_bound": json.dumps(f["flags"]["at_bound"]), "not_identified": json.dumps(f["not_identified"]),
                             "anchor": f["anchor"]["anchor"], "anchor_violation": f["anchor"]["violation"],
                             "optimiser_failure": f["optimiser_failure"],
                             "params_natural_json": json.dumps(f["natural"]), "raw_json": json.dumps(f["raw_free"]),
                             "is_primary": f["is_primary"], "is_reference": False, "wall_clock_s": f["wall_clock_s"]})
        for n, r in fj["references"].items():
            rows_out.append({"dataset": "mathdial", "model": model, "seed": seed, "variant": n, "family": "reference",
                             "mode": "none", "k_free": 0, "ll_val": r["ll_val"], "auc_val": r["auc_val"],
                             "is_primary": False, "is_reference": True})
    pd.DataFrame(rows_out).to_csv(OUT / "validation_report.csv", index=False)
    fits_sha = {tag: sha256_file(OUT / f"fits_{tag}.json") for tag in TAGS}
    prereg = {"timestamp": now_iso(), "preregistration_sha256": env["preregistration_sha256"],
              "variants": json.loads(VARIANT_SPEC_JSON),
              "fit_variants": [{"name": n, "family": VARIANTS[n]["family"], "mode": VARIANTS[n]["mode"],
                                "k_free": VARIANTS[n]["k_free"], "free_names": VARIANTS[n]["names"],
                                "box_lo": VARIANTS[n]["lo"].tolist(), "box_hi": VARIANTS[n]["hi"].tolist(),
                                "starts": [s.tolist() for s in VARIANTS[n]["starts"]]} for n in FIT_ORDER],
              "reference_rows": REFERENCE_ROWS, "table5_rows": TABLE5_ROWS,
              "optimiser": {"method": "Nelder-Mead", "options": NM_OPTIONS, "simplex_step": SIMPLEX_STEP,
                            "max_continuations": MAX_CONTINUATIONS, "polish_step": POLISH_STEP, "max_polish": MAX_POLISH,
                            "polish_min_improvement": 1e-7, "tie_tolerance": 1e-10, "n_fallback_starts": N_FALLBACK,
                            "fallback_rng_seed": SEED},
              "gamma_profile": GAMMA_PROFILE,
              "primary_set": PRIMARY_SET, "selection_rule": "argmin LL_val + k_free/N_val over primary_set; ties 1e-10 by order",
              "primary_variant": {tag: json.loads((OUT / f"fits_{tag}.json").read_text())["primary_variant"] for tag in TAGS},
              "decision_rules": {"improves": "delta CI lower bound L > 0", "harms": "delta CI upper bound U < 0 (within margin iff L > -0.010)",
                                 "non-inferior": "L > -0.010 and neither of the above", "inconclusive": "otherwise",
                                 "claim_rule": "improvement may be claimed only if V*_qlora is 'improves'; zero-shot supports no claim",
                                 "bootstrap": {"N_BOOT": 1000, "seed": SEED, "level": "dialogue", "reference": "mean_raw"},
                                 "contrast_references": CONTRAST_REFS, "non_inferiority_margin": NON_INFERIORITY_MARGIN},
              "environment": env, "fits_sha256": fits_sha, "n_val": N_VAL, "n_test_expected": N_TEST}
    write_json(OUT / "preregistration.json", prereg)
    write_json(OUT / "spec.json", {k: v for k, v in prereg.items() if k not in ("fits_sha256",)} |
               {"preregistration_text_file": "PREREGISTRATION.md", "deviations": deviations,
                "output_files": ["fits_<tag>.json", "params_<tag>.json", "gamma_profile_<tag>.csv", "validation_report.csv",
                                 "regression_checks_val.json", "preregistration.json", "manifest.json", "PREREGISTRATION.md"]})
    manifest.update({"finished": now_iso(), "wall_clock_s": time.perf_counter() - t_fit0,
                     "fit_opened_paths": sorted(set(OPENED_PATHS)), "fits_sha256": fits_sha,
                     "n_worker_processes": n_proc, "use_vectorised_bkt_engine": USE_VEC_BKT,
                     "determinism_check_geo_qlora_bit_identical": det_ok,
                     "fallback_triggered": [f"{t}/{n}" for t, n in triggered],
                     "optimiser_failures": [f"{t}/{n}" for t in TAGS for n in FIT_ORDER if fits[t][n]["optimiser_failure"]],
                     "cell26_path": str(CELL26)})
    assert not any(_is_test_path(p) for p in OPENED_PATHS), "C8: a test path was opened during fit"
    write_json(OUT / "manifest.json", manifest)
    remove_guard()
    print(f"fit finished in {manifest['wall_clock_s']:.1f}s; outputs in {OUT}", flush=True)


# --------------------------------------------------------------------------------------
# evaluate (single shot)
# --------------------------------------------------------------------------------------
def cmd_evaluate(args):
    t0 = time.perf_counter()
    sentinel = OUT / "TEST_EVALUATED"
    if sentinel.exists():
        raise SystemExit("TEST_EVALUATED sentinel exists: the test set has already been evaluated (no override flag).")
    prereg = json.loads((OUT / "preregistration.json").read_text())
    fits_sha = {tag: sha256_file(OUT / f"fits_{tag}.json") for tag in TAGS}
    for tag in TAGS:
        assert fits_sha[tag] == prereg["fits_sha256"][tag], f"fits_{tag}.json sha256 mismatch vs preregistration.json"
    fits = {tag: json.loads((OUT / f"fits_{tag}.json").read_text()) for tag in TAGS}
    variant_names = [v["name"] for v in prereg["fit_variants"]]
    assert len(variant_names) == 7
    for tag in TAGS:
        for n in variant_names:
            assert "selected_start" in fits[tag]["variants"][n], f"{tag}/{n} has no selected start"
        assert prereg["primary_variant"][tag] in variant_names
    env = environment()
    table5 = pd.read_csv(KQ / "03_ccmf" / "table5_reference_inputs" / "ccmf_real_logits_ablation.csv")
    table5 = table5[table5.dataset == "mathdial"]

    results, contrasts, regression, pred_frames = [], [], {"timestamp": now_iso()}, {}
    for model, seed, tag in BACKBONES:
        rows = load_rows(model, seed, "test", "evaluate")
        verdicts = load_verdicts("test", rows, "evaluate")
        c1 = check_c1(rows, verdicts, N_TEST, D_TEST)
        # integrity checks run before any prediction so that a failure cannot interrupt a started evaluation
        stored_pred = pd.read_csv(KQ / "03_ccmf" / "table5_reference_inputs" / f"ccmf_test_predictions_mathdial_{tag}.csv",
                                  dtype={"dialogue": str, "turn_id": str})
        assert list(zip(stored_pred.dialogue, stored_pred.turn_id)) == [(r["dialogue"], r["turn_id"]) for r in rows]
        assert set(TABLE5_ROWS) <= set(table5[table5.model == model].variant), "Table 5 rows missing"
        prep = Prep(rows, verdicts)
        stored = json.loads((KQ / "03_ccmf" / "table5_reference_inputs" / f"ccmf_parameters_mathdial_{tag}.json").read_text())
        m0_ref = fits[tag]["m0_ref"]
        y, dialogues = prep.y, prep.dialogues
        # single pass: all candidate + reference predictions
        preds = all_predictions(prep, stored, m0_ref, {n: fits[tag]["variants"][n]["raw_free"] for n in variant_names})
        # min AUC invariance on test
        d_min = abs(roc_auc_score(y, preds["min"]) - roc_auc_score(y, prep.mind))
        min_new_ties_test = int(len(np.unique(prep.mind)) - len(np.unique(preds["min"])))
        # bootstrap: one main call, contrast calls with the same dict
        boot = paired_bootstrap(y, preds, dialogues, reference="mean_raw")
        boot_c = {ref: paired_bootstrap(y, preds, dialogues, reference=ref) for ref in CONTRAST_REFS}
        for ref, b in boot_c.items():
            assert b["mean_raw"]["ci"] == boot["mean_raw"]["ci"], f"mean_raw CI differs in contrast call {ref}"
        # C4 regression vs Table 5
        t5 = table5[table5.model == model].set_index("variant")
        c4 = {}
        for n in TABLE5_ROWS:
            auc = safe_auc(y, preds[n])
            c4[n] = {"auc": auc, "table5_auc": float(t5.loc[n, "auc"]), "auc_abs_diff": abs(auc - float(t5.loc[n, "auc"])),
                     "pred_max_abs_diff": float(np.max(np.abs(preds[n] - np.clip(stored_pred[n].values, CLIP[0], CLIP[1])))),
                     "ci_abs_diff": max(abs(boot[n]["ci"][0] - float(t5.loc[n, "ci_low"])),
                                        abs(boot[n]["ci"][1] - float(t5.loc[n, "ci_high"]))),
                     "delta_ci_abs_diff": max(abs(boot[n]["delta_ci"][0] - float(t5.loc[n, "delta_ci_low"])),
                                              abs(boot[n]["delta_ci"][1] - float(t5.loc[n, "delta_ci_high"])))}
            c4[n]["auc_gate_ok"] = c4[n]["auc_abs_diff"] <= 1e-3
            c4[n]["pred_gate_ok"] = c4[n]["pred_max_abs_diff"] <= 1e-6
            c4[n]["ci_ok_1e-9"] = c4[n]["ci_abs_diff"] <= 1e-9 and c4[n]["delta_ci_abs_diff"] <= 1e-9
            c4[n]["auc_within_1e-9"] = c4[n]["auc_abs_diff"] <= 1e-9
        c4_pass = all(c4[n]["auc_gate_ok"] and c4[n]["pred_gate_ok"] for n in TABLE5_ROWS)
        regression[tag] = {"C1": c1, "C4": c4, "C4_passed": bool(c4_pass),
                           "C4_max_auc_abs_diff": max(c4[n]["auc_abs_diff"] for n in TABLE5_ROWS),
                           "C4_max_ci_abs_diff": max(max(c4[n]["ci_abs_diff"], c4[n]["delta_ci_abs_diff"]) for n in TABLE5_ROWS),
                           "min_auc_invariance_abs_diff": float(d_min), "min_new_ties_test": min_new_ties_test,
                           "min_auc_parameter_free_test": bool(d_min <= 1e-12 and min_new_ties_test == 0),
                           "mean_raw_ci_identical_across_calls": True}
        # stratum AUCs
        strata = {"auc_n1": prep.n == 1, "auc_n2": prep.n == 2, "auc_n3": prep.n == 3, "auc_n4plus": prep.n >= 4}
        auc_mr = safe_auc(y, preds["mean_raw"])
        primary = prereg["primary_variant"][tag]
        for n, p in preds.items():
            is_ref = n in REFERENCE_ROWS
            f = fits[tag]["variants"].get(n)
            ref = fits[tag]["references"].get(n)
            auc = safe_auc(y, p)
            L, U = boot[n]["delta_ci"]
            row = {"dataset": "mathdial", "model": model, "seed": seed, "variant": n,
                   "family": f["family"] if f else "reference", "mode": f["mode"] if f else "none",
                   "k_free": f["k_free"] if f else 0,
                   "ll_val": f["ll_val"] if f else ref["ll_val"], "auc_val": f["auc_val"] if f else ref["auc_val"],
                   "selection_score": f["selection_score"] if f else None,
                   "n": int(len(y)), "auc": auc, "ci_low": boot[n]["ci"][0], "ci_high": boot[n]["ci"][1],
                   "delta_vs_mean_raw": auc - auc_mr, "delta_ci_low": L, "delta_ci_high": U,
                   "ll_test": None if n.startswith("label_hist") else ll(y, p),
                   **{k: safe_auc(y[m], p[m]) for k, m in strata.items()},
                   "converged": f["flags"]["converged"] if f else None, "hit_maxiter": f["flags"]["hit_maxiter"] if f else None,
                   "fallback_used": f["flags"]["fallback_used"] if f else None,
                   "n_starts_agree": f["flags"]["n_starts_agree"] if f else None,
                   "restart_sensitive": f["flags"]["restart_sensitive"] if f else None,
                   "at_bound": json.dumps(f["flags"]["at_bound"]) if f else None,
                   "not_identified": json.dumps(f["not_identified"]) if f else None,
                   "params_natural_json": json.dumps(f["natural"]) if f else None,
                   "raw_json": json.dumps(f["raw_free"]) if f else None,
                   "is_primary": bool(f and n == primary),
                   "decision": decide(L, U) if (f and n == primary) else ("secondary: " + decide(L, U) if f else ""),
                   "is_reference": is_ref,
                   "table5_reproduced": (c4[n]["auc_gate_ok"] and c4[n]["pred_gate_ok"]) if n in c4 else None}
            results.append(row)
        for ref, targets in CONTRAST_REFS.items():
            auc_ref = safe_auc(y, preds[ref])
            for n in targets:
                Lc, Uc = boot_c[ref][n]["delta_ci"]
                contrasts.append({"dataset": "mathdial", "model": model, "seed": seed, "variant": n, "reference": ref,
                                  "auc": safe_auc(y, preds[n]), "auc_reference": auc_ref,
                                  "delta": safe_auc(y, preds[n]) - auc_ref, "delta_ci_low": Lc, "delta_ci_high": Uc,
                                  "label_history_effect_excluded": bool(Lc > 0) if ref.startswith("label_hist") else None})
        pred_frames[tag] = pd.DataFrame({"dialogue": dialogues, "turn_id": [r["turn_id"] for r in rows], "label": y, **preds})
        print(f"[{tag}] test pass done; C4 passed={c4_pass}; max AUC diff={regression[tag]['C4_max_auc_abs_diff']:.2e}", flush=True)

    # ---- log + sentinel (before writing results), then outputs ---------------------------
    line = json.dumps({"timestamp": now_iso(), "script_sha256": env["script_sha256"], "fits_sha256": fits_sha})
    with (OUT / "test_evaluations.log").open("a") as fh:
        fh.write(line + "\n")
    sentinel.write_text(line + "\n")
    pd.DataFrame(results).to_csv(OUT / "test_results.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "test_contrasts.csv", index=False)
    for tag, df in pred_frames.items():
        df.to_csv(OUT / f"test_predictions_{tag}.csv", index=False)
    write_json(OUT / "regression_checks_test.json", regression)
    manifest = json.loads((OUT / "manifest.json").read_text())
    manifest["evaluate"] = {"timestamp": now_iso(), "environment": env, "fits_sha256": fits_sha,
                            "wall_clock_s": time.perf_counter() - t0, "sentinel": str(sentinel),
                            "C4_passed": {tag: regression[tag]["C4_passed"] for tag in TAGS}}
    write_json(OUT / "manifest.json", manifest)
    df = pd.DataFrame(results)
    print(df[["model", "variant", "auc", "delta_vs_mean_raw", "delta_ci_low", "delta_ci_high", "ll_test", "is_primary", "decision"]]
          .to_string(index=False))
    if not all(regression[tag]["C4_passed"] for tag in TAGS):
        raise SystemExit("C4 regression gate FAILED (results written; see regression_checks_test.json)")
    print(f"evaluate finished in {time.perf_counter() - t0:.1f}s", flush=True)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--procs", type=int, default=0, help="worker processes (default: cpu count)")
    sub.add_parser("evaluate")
    args = ap.parse_args()
    {"check": cmd_check, "fit": cmd_fit, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
