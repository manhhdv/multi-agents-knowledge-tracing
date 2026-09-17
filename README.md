# MathKT-Agent: From LLM-Based Dialogue Knowledge Tracing to Tutoring Decisions without Offline Correctness Labels

Van-Manh Dang<sup>1</sup>, Duy-Hung Dao<sup>1</sup>, Thu-Hien Nguyen<sup>2</sup>, Van-Doc Vu<sup>2</sup>, Van-Tan Bui<sup>2</sup>, Hong-Viet Tran<sup>1,*</sup>

<sup>1</sup> University of Engineering and Technology, Vietnam National University, Hanoi, Vietnam
<sup>2</sup> University of Economics - Technology for Industries, Hanoi, Vietnam
<sup>*</sup> Corresponding author: thviet@vnu.edu.vn

This repository contains the code, notebooks, released run outputs and human annotation materials for the MathKT-Agent paper, submitted to SoICT 2026. The CPU analyses can be rerun directly after cloning (see [Quick Start](#quick-start)).

---

## Abstract

LLM-based dialogue knowledge tracing (KT), as in LLMKT, estimates the mastery of each knowledge component (KC) from tutor–student dialogues. It depends on offline GPT-4o correctness labels and stops at estimation. **MathKT-Agent** is a multi-agent extension that keeps LLMKT's estimator (retrained with 4-bit QLoRA; AUC 0.745 on MathDial) and adds agents that each replace one decision step. Each agent is evaluated against the baseline it replaces.

- **SAVA** (Symbolic Answer Verification Agent) replaces offline labels at inference. It grades answers against a reference with computer algebra and abstains when the answer alone is insufficient.
- **CCMF** (Calibrated Cumulative Mastery Filtering) matches mean pooling in AUC while calibrating its probabilities.
- The **Error Analyzer** classifies student errors into five classes.
- The **Feedback Generator** conditions on mastery and diagnosis.
- The **MTA** (Maintain-Then-Advance) planner selects the next KC from mastery and prerequisites.

## Key Results

| Component | Evaluation | Result |
|---|---|---|
| Backbone (LLMKT estimator, 4-bit QLoRA) | MathDial test, n = 1,985 | Acc 68.11, AUC 74.53, F1 62.61 |
| SAVA | Constructed benchmark, n = 5,448 | Accuracy 1.000 (numeric regex 0.917) |
| SAVA | 139 human-verified real turns | 84.6% accurate on the 78.7% of turns it judges, on par with regex and an 8B LLM judge |
| CCMF | MathDial test, fine-tuned backbone | AUC 0.745 (Δ +0.000 vs. mean pooling); log-loss 0.611 → 0.595 |
| CCMF ablation | Product (noisy-AND) aggregation | −0.049 AUC |
| Error Analyzer | 46 CoMTA turns, adjudicated human labels | Cohen's κ = 0.16 |
| Feedback Generator | 30 paired cases vs. LLMKT-only baseline | Correctness +0.45 / 5 (Wilcoxon p = 0.013); relevance and clarity not improved |
| MTA planner | 500 simulated learners | 61.3% of KCs mastered, on par with a hand-ordered curriculum; 2.6× prerequisite-aware lowest-first |
| MTA planner | Logged MathDial dialogues | Prerequisite rule selects the next KC on 3.9% of turns |

The paper also reports negative results and limitations. See Sections 4–5 of the manuscript.

## Architecture

```
                 OFFLINE (uses correctness labels)
   ┌───────────────────────────────────────────────────────────┐
   │  LLMKT estimator training (QLoRA)   CCMF head fit on val  │
   └───────────────────────────────────────────────────────────┘
                 INFERENCE (no correctness labels)
   dialogue ──► LLMKT estimator ──► per-KC logit gaps ──► CCMF ──► learner memory
      │                                                   ▲            │
      └──► SAVA (correct / incorrect / undetermined) ─────┘            │
               │ incorrect                                              ▼
               └──► Error Analyzer ──► Feedback Generator ◄── mastery ─┤
                                                                        ▼
                                                                  MTA planner ──► next KC
```

SAVA's verdict updates mastery only for later turns and gates the Error Analyzer.

## Quick Start

Everything needed to rerun the CPU analyses (code, released run outputs and human labels) is in this repository. No GPU, API key or dataset download is required.

```bash
git clone https://github.com/manhhdv/multi-agents-knowledge-tracing.git
cd multi-agents-knowledge-tracing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` reruns the 9 scripts in `cpu/` and the 5 in `supplementary/scripts/`, which takes about 2 minutes on a laptop. It then checks with `git diff` that every regenerated file in `cpu/` and `out/` is byte-identical to the released version, and ends with:

```
OK: all regenerated outputs in cpu/ and out/ match the released files.
```

## Repository Structure

```
.
├── README.md
├── requirements.txt                Python dependencies for the CPU analyses
├── reproduce.sh                    Reruns all CPU analyses and checks the outputs
├── MANIFEST.sha256                 SHA-256 checksums of every released file
├── cpu/                            Analysis scripts added in revision (CPU only)
│   ├── 01_judge_threeway.py            LLM judge with an abstain option
│   ├── 02_calibration_downstream.py    Effect of calibration on MTA decisions
│   ├── 03_table1_cis.py                Dialogue-level bootstrap CIs for Table 1
│   ├── 04_ea_confusion_table.py        Error Analyzer confusion matrix (Table 5)
│   ├── 05_regen_automatic_checks.py    Recomputes automatic feedback checks from row-level files
│   ├── 06_score_human_verification.py  Scores the 139-turn human verification
│   ├── 07_score_feedback_baseline.py   Paired comparison with the LLMKT-only feedback baseline
│   ├── 08_endtoend_mta.py              End-to-end SAVA → CCMF → MTA replay on MathDial
│   ├── 09_build_atc_graph.py           Prerequisite graph from the Achieve the Core Coherence Map
│   ├── strata_weights.csv              Stratum weights for the human verification sample
│   └── out_*.{csv,json,tex}            Released outputs of the scripts above
├── supplementary/
│   ├── scripts/                        Further analyses (SAVA cost and benchmark overlap,
│   │                                   matched judge comparison, CCMF bounds, planner under real noise)
│   ├── data/                           Outputs of those analyses
│   └── human_verification/             Sampling key (system verdicts per item) for the 139 turns
├── results/                        Released results of the main experiments
│   ├── 00_run_info/                    Environment, commit, split counts
│   ├── 01_backbone/                    LLMKT estimator zero-shot and 4-bit QLoRA (Table 1)
│   ├── 02_sava/                        Constructed benchmark and real-turn verdicts (Table 2, Sec. 4.3)
│   ├── 03_ccmf/                        Pre-registration, fitted parameters, test predictions (Table 3)
│   ├── 04_integrated_loop/             Full SAVA → CCMF → agents run on MathDial test
│   ├── 05_error_analyzer/              Predictions, adjudicated gold labels, metrics (Tables 4–5)
│   ├── 06_feedback/                    Feedback inputs, generations and human ratings (Sec. 4.6)
│   ├── 07_planner/                     MTA simulation, ablation and sensitivity grid (Sec. 4.7)
│   └── 08_scripts/                     Scripts used to build the result files
├── raw_runs/                       Per-turn True/False logit gaps and LLMKT evaluation logs
│   ├── mathdial/{zero_shot,qlora}/     val and test splits
│   └── comta/zero_shot/                val and test splits (student text removed, see Data)
├── replication/
│   ├── colab_full_framework_replication.ipynb   End-to-end pipeline: training, SAVA, CCMF, agents, planner
│   └── fetch_mathdial.py                        Downloads MathDial from its authors
├── colab/                          GPU notebooks added in revision
│   ├── A_prereq_graph_mathdial.ipynb   Builds the KC prerequisite graph (wraps cpu/09)
│   ├── B_feedback_baseline.ipynb       Generates LLMKT-only baseline feedback and blind rating sheets
│   └── C_ea_abstention.ipynb           Error Analyzer rerun with an "unclear" class
├── colab_out/                      Outputs of notebooks B and C, including the rating sheets
├── out/                            Prerequisite graph, KC-to-standard map, Achieve the Core source
└── labelling/                      Human verification of turn correctness (139 MathDial turns)
    ├── HUONG_DAN_XAC_MINH.md           Annotation guideline (Vietnamese)
    ├── human_verification_sheet.csv    Annotation sheet
    ├── xac_minh_A.xlsx, xac_minh_B.xlsx  Independent labels of annotators A and B
    └── adjudication_xac_minh.xlsx      Merged labels and adjudicated final labels
```

## Reproducing the Analyses

### CPU analyses

Scripts locate the repository root (the folder containing `results/` and `raw_runs/`) automatically. To point them elsewhere, set `MATHKT_ROOT`. Most scripts first reproduce a published number, such as the judge agreement or the CCMF test AUC, and stop if it does not match.

To run the scripts one at a time:

```bash
cd cpu
python3 01_judge_threeway.py
python3 02_calibration_downstream.py
python3 03_table1_cis.py
python3 04_ea_confusion_table.py
python3 05_regen_automatic_checks.py
python3 09_build_atc_graph.py             # writes ../out/prereq_graph_mathdial_atc.json
python3 08_endtoend_mta.py                # run after 09; optional --graph PATH --tau 0.8
python3 06_score_human_verification.py    # scores the adjudicated labels in ../labelling/
python3 07_score_feedback_baseline.py     # scores the rating sheets in ../colab_out/
```

`06_score_human_verification.py --merge` rebuilds the adjudication sheet from the two annotator files. It writes `adjudication_xac_minh_MERGED.xlsx` for a new annotation round.

### GPU notebooks

The notebooks need a Colab GPU runtime (L4 or A100) and a Hugging Face token with access to [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct). The first cell mounts Google Drive and clones this repository into `MyDrive/multi-agents-knowledge-tracing` if it is not already there. Outputs are written to `colab_out/` in that clone.

- `colab/A_prereq_graph_mathdial.ipynb` also runs on CPU.
- `colab/B_feedback_baseline.ipynb` runs from the repository alone.
- `colab/C_ea_abstention.ipynb` additionally needs `out/comta_context.json`, which contains CoMTA dialogues and is not distributed (see Data).

The full pipeline (`replication/colab_full_framework_replication.ipynb`) retrains the estimator with 4-bit QLoRA. This took 4.6 h on one A100, including evaluation. It needs MathDial (`python3 replication/fetch_mathdial.py`) and, for CoMTA evaluation, the CoMTA release.

### Note on released numbers

`cpu/out_05_automatic_checks_FIXED.csv` reports a 0.990 valid-JSON rate for the Error Analyzer on MathDial (198/200). It is computed from `results/05_error_analyzer/error_analyzer_predictions.csv`, the run before CCMF. The paper reports 99.5% (199/200) from the CCMF rerun, `results/06_feedback/02_generation/error_analyzer_predictions_mathdial_ccmf.csv`, which also produced the 30 feedback cases.

## Human Evaluation Protocol

Two co-authors independently labelled 139 MathDial test turns as *correct*, *incorrect* or *unclear*, without AI tools. They were blind to every system verdict and to the GPT-4o label. The sample is stratified over the six SAVA/judge agreement patterns, with at least 15 turns per stratum. Estimates are reweighted to the 1,985 test turns with `strata_weights.csv`.

- Raw agreement: 84.9%, Cohen's κ = 0.71.
- 21 disagreements were adjudicated. 4 turns left unclear were excluded.
- Accuracy against human labels: GPT-4o labels 89.9% [86.2, 94.0]; SAVA 84.6% [79.8, 89.2] on the turns it judges.

The guideline (`labelling/HUONG_DAN_XAC_MINH.md`) and the sheets are in Vietnamese. Column meanings:

| Column | Meaning |
|---|---|
| `item_id`, `dialogue`, `turn_id` | Sample and MathDial identifiers |
| `de_bai` | Problem statement |
| `ngu_canh` | Dialogue context before the turn |
| `luot_hoc_sinh` | Student turn |
| `dap_an` | Reference final answer |
| `dung_sai` (`_A`, `_B`) | Label: `dung` (correct), `sai` (incorrect), `chua_ro` (unclear) |
| `ghi_chu` (`_A`, `_B`) | Annotator note |
| `trang_thai`, `final`, `final_ghi_chu` | Agreement status, adjudicated label and note |

## Data

| Dataset | Use | Source and licence |
|---|---|---|
| MathDial | Training, validation, testing | Macina et al. (2023), [eth-nlped/mathdial](https://github.com/eth-nlped/mathdial); KC tags from the LLMKT release |
| CoMTA | Evaluation only (no fitting) | LLMKT release; its licence prohibits training |
| Achieve the Core Coherence Map | Prerequisite graph | [allenai/achieve-the-core](https://huggingface.co/datasets/allenai/achieve-the-core) as released in MathFish, ODC-BY 1.0 |

This repository redistributes derived run outputs, not the datasets themselves:

- **MathDial.** Result files contain the student turns and reference answers needed to rerun the analyses. The full dataset is available from its authors (`replication/fetch_mathdial.py`).
- **CoMTA.** The licence permits internal evaluation only, so no CoMTA dialogue text is included. The `student_text` field in `raw_runs/comta/**/raw_logit_gaps.json` is set to `null`, and the per-turn LLMKT dumps, the Error Analyzer annotation sheets and `out/comta_context.json` are omitted. Released labels, logits, predictions and adjudicated gold classes are kept, so every CPU analysis still runs.
- **Achieve the Core.** `out/external/achieve-the-core_standards.jsonl` is redistributed under ODC-BY 1.0 (see `out/external/NGUON.md`).

## Integrity Check

`MANIFEST.sha256` lists a checksum for every released file:

```bash
shasum -a 256 -c MANIFEST.sha256
```

## Limitations

- The estimator is inherited from LLMKT and not improved. Results come from a single seed and fold.
- SAVA handles only single-valued answers and needs a reference answer. On CoMTA it abstains on every turn.
- Labels are removed only at inference. The estimator and CCMF are fitted on labelled turns.
- Human evaluations are small and conducted by co-authors.
- The planner is evaluated only in simulation and on logged dialogues, not with real learners.

## Citation

```bibtex
@unpublished{dang2026mathktagent,
  title  = {{MathKT-Agent}: From {LLM}-Based Dialogue Knowledge Tracing to Tutoring Decisions without Offline Correctness Labels},
  author = {Dang, Van-Manh and Dao, Duy-Hung and Nguyen, Thu-Hien and Vu, Van-Doc and Bui, Van-Tan and Tran, Hong-Viet},
  note   = {Manuscript submitted to SoICT 2026},
  year   = {2026}
}
```

## Acknowledgements

This work builds on [LLMKT](https://github.com/umass-ml4ed/dialogue-kt) (Scarlatos et al.), whose prompt, logit readout and training code we reuse, and on the MathDial, CoMTA and Achieve the Core resources.

## Contact

For questions, open an issue or contact Hong-Viet Tran (thviet@vnu.edu.vn) or Van-Manh Dang (manhdvvnu@gmail.com).
