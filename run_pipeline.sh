#!/bin/bash


source setup.sh

echo "=== Step 1: treemaker ==="
fccanalysis run treemaker_lnuqq_step1.py --ncores 12

echo "=== Fits: resolution parametrization ==="
python3 fit_resolutions.py

echo "=== Step 2: treemaker + kinfit ==="
fccanalysis run treemaker_lnuqq_step2.py --ncores 12

echo "=== Plot: mW overlay ==="
python3 plot_mW_overlay.py

echo "=== Plot: kinfit variables ==="
python3 plot_kinfit_vars.py

echo "=== Done ==="
