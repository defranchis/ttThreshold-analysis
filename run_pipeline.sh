#!/bin/bash


source setup.sh

echo "=== Step 1: treemaker ==="
fccanalysis run treemaker_lnuqq_step1.py --ncores 12

echo "=== Fits: resolution parametrization ==="
python3 fit_resolutions.py

echo "=== Build log Z table for kinfit BW normalization ==="
tools/ensure_logz_table.sh

echo "=== Step 2: treemaker + kinfit (3 ECMs in parallel) ==="
mkdir -p logs
pids=()
for ecm in 157 160 163; do
    WW_ECM=$ecm fccanalysis run treemaker_lnuqq_step2.py --ncores 4 \
        > logs/step2_ecm${ecm}.log 2>&1 &
    pids+=($!)
    echo "  launched ecm${ecm} (PID=${pids[-1]})"
done
fail=0
for pid in "${pids[@]}"; do
    wait $pid || fail=1
done
if [ $fail -ne 0 ]; then
    echo "step2 failed for at least one ECM — see logs/step2_ecm*.log"
    exit 1
fi

echo "=== Diagnostics: kinfit summary ==="
python3 kinfit_diagnostics.py

echo "=== Plot: kinfit results (mW overlays + kinfit vars) ==="
python3 plot_kinfit_results.py

echo "=== Done ==="
