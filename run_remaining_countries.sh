#!/usr/bin/env bash
# Runs fleet_evaluation_lca_algebraic.py (no baseline needed) across a list of countries,
# one subprocess per country so Brightway/lca_algebraic state never leaks between them.
# Continues past a per-country failure rather than aborting the whole batch — see
# NEXT_STEPS_lca_algebraic.md item 4 / PLAN_lca_algebraic.md "Completed 21 Jul 2026".
set -u
cd "$(dirname "$0")"
source /home/elie/Desktop/masters/EVERGI/rewind_env/bin/activate

COUNTRIES=(ES FR SE IT NL PT PL IE AT GR FI RO HR UA BG CZ RS LT EE HU CY LU BA LV ME CH FO MK BY XK SK IS SI)

mkdir -p logs/algebraic_batch
FAILED=()
i=0
total=${#COUNTRIES[@]}
for c in "${COUNTRIES[@]}"; do
  i=$((i+1))
  echo "=== [$i/$total] START $c ==="
  if python fleet_evaluation_lca_algebraic.py "$c" > "logs/algebraic_batch/${c}.log" 2>&1; then
    echo "=== [$i/$total] DONE $c ==="
  else
    echo "=== [$i/$total] FAILED $c (see logs/algebraic_batch/${c}.log) ==="
    FAILED+=("$c")
  fi
done

echo "BATCH COMPLETE. ${#FAILED[@]} failures: ${FAILED[*]:-none}"
