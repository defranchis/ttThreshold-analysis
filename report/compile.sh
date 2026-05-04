#!/bin/bash
# Build the report and publish to EOS.
# Runs pdflatex twice so cross-references / hyperref bookmarks resolve.
set -e
cd "$(dirname "$0")"

EOS_DEST=/eos/user/m/mdefranc/www/mW/report.pdf

run_pdflatex() {
    pdflatex -interaction=nonstopmode -halt-on-error report.tex >/tmp/pdflatex.log 2>&1 \
        || { tail -50 /tmp/pdflatex.log; exit 1; }
}

# Two passes: first builds the .aux, second consumes it for cross-refs.
run_pdflatex
run_pdflatex

# Sanity check: scan for "??" (literal) reference markers in the PDF.
if pdftotext report.pdf - 2>/dev/null | grep -qF "??"; then
    echo "WARNING: unresolved references still present after 2 passes:"
    pdftotext report.pdf - | grep -nF "??" | head -5
    echo "(running a third pass...)"
    run_pdflatex
fi

cp -f report.pdf "$EOS_DEST"
echo "OK -> $(pwd)/report.pdf"
echo "     -> $EOS_DEST"
