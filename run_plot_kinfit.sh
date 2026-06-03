#!/bin/bash
cd /afs/cern.ch/work/m/mdefranc/private/WW/WW_reco
source setup.sh
exec python3 plot_kinfit_results.py
