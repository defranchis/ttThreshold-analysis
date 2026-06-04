#!/bin/bash
# Run the full standalone BW-pairing chain: WW signal, ZZ control, then analysis.
# Must be run from the repository root in a key4hep / FCCAnalyses environment.
#   source <your FCCAnalyses setup>   # provides `fccanalysis`
#   bash bw_pairing_standalone/run.sh
set -e
HERE="bw_pairing_standalone"

echo ">>> WW -> 4q (with gen-truth pairing)"
BW_BOSON=W BW_SAMPLE=p8_ee_WW_ecm160 \
    fccanalysis run ${HERE}/treemaker_bw_pairing.py

echo ">>> ZZ -> 4q (WW hypothesis, control)"
BW_BOSON=Z BW_SAMPLE=p8_ee_ZZ_ecm160 \
    fccanalysis run ${HERE}/treemaker_bw_pairing.py

echo ">>> analysis (pairing efficiency + WW vs ZZ gof)"
python3 ${HERE}/analyze_bw_pairing.py \
    outputs/bw_pairing/W/p8_ee_WW_ecm160.root \
    outputs/bw_pairing/Z/p8_ee_ZZ_ecm160.root \
    bw_pairing_plots

echo ">>> done. plots in ./bw_pairing_plots/"
