#!/usr/bin/env bash
# Rerun every CPU analysis and check that the outputs match the released files.
# Usage: bash reproduce.sh            (from the repository root)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

run() { echo "==> $*"; (cd "$(dirname "$1")" && "$PY" "$(basename "$1")" "${@:2}" > /dev/null); }

run cpu/01_judge_threeway.py
run cpu/02_calibration_downstream.py
run cpu/03_table1_cis.py
run cpu/04_ea_confusion_table.py
run cpu/05_regen_automatic_checks.py
run cpu/09_build_atc_graph.py
run cpu/08_endtoend_mta.py
run cpu/06_score_human_verification.py
run cpu/07_score_feedback_baseline.py
run supplementary/scripts/analysis_sava_overlap.py
run supplementary/scripts/analysis_judge_matched.py
run supplementary/scripts/analysis_ccmf_bounds.py
run supplementary/scripts/analysis_sava_cost_sample.py
run supplementary/scripts/analysis_planner_realnoise.py

if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  if git diff --quiet -- cpu out; then
    echo "OK: all regenerated outputs in cpu/ and out/ match the released files."
  else
    echo "WARNING: regenerated outputs differ from the released files:"; git diff --stat -- cpu out; exit 1
  fi
fi
