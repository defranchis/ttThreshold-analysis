#!/bin/bash
# Ensure kinfit_inputs/logz_table.bin is present and up-to-date.
#
# Idempotent helper called by run_pipeline.sh and run_step2.sh:
#   1. (Re)compile tools/build_logz_table if missing or source is newer.
#   2. (Re)generate kinfit_inputs/logz_table.bin if missing or source is newer.
#
# Run directly (project root cwd) or via the launchers — both work.
set -e

cd "$(dirname "$0")/.."   # project root

SRC=tools/build_logz_table.cxx
BLD=tools/build_logz_table
TBL=kinfit_inputs/logz_table.bin

mkdir -p kinfit_inputs

if [ ! -x $BLD ] || [ $SRC -nt $BLD ]; then
    echo "[ensure_logz_table] (re)compiling builder..."
    g++ -O2 -std=c++17 -march=native $SRC -o $BLD
fi

if [ ! -f $TBL ] || [ $SRC -nt $TBL ]; then
    echo "[ensure_logz_table] (re)generating table..."
    $BLD $TBL
fi

echo "[ensure_logz_table] OK: $TBL ($(du -h $TBL | cut -f1))"
