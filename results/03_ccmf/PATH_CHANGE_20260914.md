# Path change 2026-09-14: `03_ccmf/redesign/` → `03_ccmf/`

The author flattened the folder on 2026-09-14, after the single test evaluation (TEST_EVALUATED, 2026-09-13T19:36:45Z). No result file content changed.
At the author's request only path strings were edited in the files below. `PREREGISTRATION.md` and the pre-registration text embedded in `ccmf_redesign.py` (sha256 e17bd98c086a3e1b…, re-checked unchanged) still name the original folder, and the append-only `TEST_EVALUATED` and `test_evaluations.log` keep the script sha256 recorded at evaluation time (f8284c19…). `check_report.json` only names the script and was not edited.

| File | sha256 before | sha256 after | replacements |
|---|---|---|---|
| `08_scripts/ccmf_redesign.py` | f8284c191bf6640cac7246ac319395de2cb4891d26e6390bc38f93867633ea70 | 3a8194419251ec6f139a84f279967111f7ee95bd1a706d4fb2643e1855b6fbb9 | 2 |
| `08_scripts/verify_ccmf_redesign.py` | 6c84dbe622b0a10289035a4bc151bb8c55af15a4618eda294ca6a1fbb51efd91 | 822e0a3aef01580669e51b6a2468cb5b21de02aa2f086aa1f294877774953325 | 2 |
| `03_ccmf/manifest.json` | 0b8dde69f63f9136cd9f86ab8a6f58b5ff1a3c36161a58b5f9727d208a35fbcf | ffabbed256b7bf8876b08c4c2e319b53e72ba0e834ed7f0a7d5ca5c41d8662d0 | 38 |
| `03_ccmf/spec.json` | 69eee051db1e6bd4e9f36a652b33c62f59bda39014e4c7922627489f5b5fec65 | ccbd4581d32db163307b0069de4a4f5de9ff0b0ca55ce68da28b732c269f8141 | 1 |

JSON files: after mapping `03_ccmf/redesign` → `03_ccmf` and `under redesign/checkpoints/` → `under 03_ccmf/checkpoints/`, every value is identical (checked programmatically).

## Script diffs

```diff
--- a/08_scripts/ccmf_redesign.py
+++ b/08_scripts/ccmf_redesign.py
@@ -47 +47 @@
-OUT = KQ / "03_ccmf" / "redesign"
+OUT = KQ / "03_ccmf"
@@ -1066 +1066 @@
-        "Checkpoints are one JSON per (backbone, variant) start batch under redesign/checkpoints/ "
+        "Checkpoints are one JSON per (backbone, variant) start batch under 03_ccmf/checkpoints/ "
```

```diff
--- a/08_scripts/verify_ccmf_redesign.py
+++ b/08_scripts/verify_ccmf_redesign.py
@@ -6 +6 @@
-decision rule). Writes 03_ccmf/redesign/independent_verification.json.
+decision rule). Writes 03_ccmf/independent_verification.json.
@@ -20 +20 @@
-OUT = KQ / "03_ccmf" / "redesign"
+OUT = KQ / "03_ccmf"
```


## Second change (same day): Table 5 reference inputs

The folder clean-up had moved the stored Table 5 inputs of the original product-head CCMF out of `03_ccmf/`. Both scripts read them (regression gates C2/C4 and the verification's Table 5 reproduction), so they were restored unchanged into `03_ccmf/table5_reference_inputs/` and the read paths were updated. `PREREGISTRATION.md` and the embedded pre-registration text still cite `03_ccmf/ccmf_parameters_mathdial_<tag>.json` (sha256 re-checked unchanged).

| Restored input | sha256 |
|---|---|
| `03_ccmf/table5_reference_inputs/ccmf_parameters_mathdial_qlora_seed221.json` | ac4bf99f3d1e7055655506a0dccebbe680045fd7fe589a57e15f3b72b0cfb891 |
| `03_ccmf/table5_reference_inputs/ccmf_parameters_mathdial_zero_shot_seed-na.json` | 6e7a5d1b51659c2d408cb2e760b7db04173f67ce7ec7c40fb8c31df0ea06eaa6 |
| `03_ccmf/table5_reference_inputs/ccmf_real_logits_ablation.csv` | d9de1ef5207c3b3bc494bc0809789cde93e29225d6fec2bb21f12b51cb651879 |
| `03_ccmf/table5_reference_inputs/ccmf_test_predictions_mathdial_qlora_seed221.csv` | 4e6a98195184b0527d73d30e972d717bbbc5e034558d73836186f3287a986800 |
| `03_ccmf/table5_reference_inputs/ccmf_test_predictions_mathdial_zero_shot_seed-na.csv` | 9ba8d41de5a443d2fd9dfdcbf91d8f7a03f6518948fb2fece450d221245a0c46 |

| Script | sha256 before | sha256 after |
|---|---|---|
| `08_scripts/ccmf_redesign.py` | 3a8194419251ec6f139a84f279967111f7ee95bd1a706d4fb2643e1855b6fbb9 | 6b58066345281e9cd1f1f726c3350f43b761fe58d96d2bdccd9023a1f68891fd |
| `08_scripts/verify_ccmf_redesign.py` | 822e0a3aef01580669e51b6a2468cb5b21de02aa2f086aa1f294877774953325 | bdd9cb1b888056a039ebd849572b8ad9458305f7e8745fd6d5ee427b055f107d |

```diff
--- a/08_scripts/ccmf_redesign.py
+++ b/08_scripts/ccmf_redesign.py
@@ -407 +407 @@
-    return read_json(KQ / "03_ccmf" / f"ccmf_parameters_mathdial_{tag}.json", stage)
+    return read_json(KQ / "03_ccmf" / "table5_reference_inputs" / f"ccmf_parameters_mathdial_{tag}.json", stage)
@@ -1376 +1376 @@
-    table5 = pd.read_csv(KQ / "03_ccmf" / "ccmf_real_logits_ablation.csv")
+    table5 = pd.read_csv(KQ / "03_ccmf" / "table5_reference_inputs" / "ccmf_real_logits_ablation.csv")
@@ -1385 +1385 @@
-        stored_pred = pd.read_csv(KQ / "03_ccmf" / f"ccmf_test_predictions_mathdial_{tag}.csv",
+        stored_pred = pd.read_csv(KQ / "03_ccmf" / "table5_reference_inputs" / f"ccmf_test_predictions_mathdial_{tag}.csv",
@@ -1390 +1390 @@
-        stored = json.loads((KQ / "03_ccmf" / f"ccmf_parameters_mathdial_{tag}.json").read_text())
+        stored = json.loads((KQ / "03_ccmf" / "table5_reference_inputs" / f"ccmf_parameters_mathdial_{tag}.json").read_text())
```

```diff
--- a/08_scripts/verify_ccmf_redesign.py
+++ b/08_scripts/verify_ccmf_redesign.py
@@ -113 +113 @@
-    table5 = pd.read_csv(KQ / "03_ccmf" / "ccmf_real_logits_ablation.csv")
+    table5 = pd.read_csv(KQ / "03_ccmf" / "table5_reference_inputs" / "ccmf_real_logits_ablation.csv")
```


## Third change (same day): `_archive/` removed by the author

The author deleted `_archive/`. Its two remaining dependencies were moved into the package: `cell26.txt` (unchanged, sha256 97a58c17f42e591375cfe50f36371a7b230a8a3e0f0ccf91ca6c0159d9561d8e, equal to `cell26_sha256` in manifest.json) into `03_ccmf/table5_reference_inputs/`, and the SAVA verdict columns of the product-head loop (dialogue, turn_id, label, final_turn, sava_verdict, sava_stage, diagnosis_gate, sava_extracted) into `04_integrated_loop/sava_verdicts_test.csv` (identical to the same columns of the current loop). The embedded pre-registration text still cites `_archive/results_sava_conclusion/cell26.txt` (sha256 re-checked unchanged).

| File | sha256 before | sha256 after |
|---|---|---|
| `08_scripts/ccmf_redesign.py` | 6b58066345281e9cd1f1f726c3350f43b761fe58d96d2bdccd9023a1f68891fd | 6316cac39106ef39ca76a9c53be22157803f404792884453e9caf5f16a60017d |
| `08_scripts/rerun_integrated_loop_ccmf.py` | d5bae17782ea971a1c8910db10bf6158208a8695ebe9b18626e4ab83add35b17 | dac0c753339912473a961c1c71d1c337f9358700f153d0812eadf9ae5e848932 |
| `03_ccmf/manifest.json` | ffabbed256b7bf8876b08c4c2e319b53e72ba0e834ed7f0a7d5ca5c41d8662d0 | 8e86b913faff2fbda4d054afb3270f3bc7643522871a2b775722c5ac311e19ba |
| `03_ccmf/check_report.json` | 4ac0c40007477e0d6359831136afaed0926b89908a227b7a3648c4f0b975254d | 78bd7dcccb220b0c32fadc8b1cbe1a1aad56912a90e38c665af349c6ade5a47a |

```diff
--- a/08_scripts/ccmf_redesign.py
+++ b/08_scripts/ccmf_redesign.py
@@ -49 +49 @@
-CELL26 = ROOT / "_archive" / "results_sava_conclusion" / "cell26.txt"
+CELL26 = KQ / "03_ccmf" / "table5_reference_inputs" / "cell26.txt"
```

```diff
--- a/08_scripts/rerun_integrated_loop_ccmf.py
+++ b/08_scripts/rerun_integrated_loop_ccmf.py
@@ -16 +16 @@
-OLD = PAPER / "_archive/04_integrated_loop_product_head/integrated_run_mathdial_qlora_seed221.csv"
+OLD = PKG / "04_integrated_loop/sava_verdicts_test.csv"
```


## Fourth change (same day): docstring

| `08_scripts/ccmf_redesign.py` | 6316cac39106ef39ca76a9c53be22157803f404792884453e9caf5f16a60017d | 05150f953f180eda861f7153e66d9ad044a47b1809678f1e7c7b8f99fe1f689c |

```diff
--- a/08_scripts/ccmf_redesign.py
+++ b/08_scripts/ccmf_redesign.py
@@ -10 +10 @@
-Reused code: the notebook cell `_archive/results_sava_conclusion/cell26.txt` is exec'd
+Reused code: the notebook cell `ket_qua_bai_bao/03_ccmf/table5_reference_inputs/cell26.txt` is exec'd
```

