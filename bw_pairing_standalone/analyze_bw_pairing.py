#!/usr/bin/env python3
# =============================================================================
#  Standalone analysis for the BW jet -> W pairing.
#    - computes the pairing efficiency on WW (all / matched / unmatched)
#    - makes the WW-vs-ZZ "gof" discrimination plot, split by matched/unmatched
#
#  "matched" = all 4 reco jets are within dR < MATCH_DR of their gen quark
#  (the events where the gen-truth pairing is reliable).
#
#  Usage:
#    python3 analyze_bw_pairing.py [WW.root] [ZZ.root] [outdir]
#  Defaults point at the treemaker output dirs and an ./bw_pairing_plots folder.
# =============================================================================
import os
import sys
import numpy as np
import ROOT

MATCH_DR = 0.1   # jet<->quark matching criterion (per jet)

import glob

WW_FN  = sys.argv[1] if len(sys.argv) > 1 else "outputs/bw_pairing/W"
ZZ_FN  = sys.argv[2] if len(sys.argv) > 2 else "outputs/bw_pairing/Z"
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "bw_pairing_plots"


def _files(path):
    """Accept a single .root file, a directory (all *.root inside), or a glob."""
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.root")))
    if any(c in path for c in "*?["):
        return sorted(glob.glob(path))
    return [path]


def load(path, cols):
    files = _files(path)
    if not files:
        raise FileNotFoundError(f"no ROOT files found at {path}")
    rdf = ROOT.RDataFrame("events", files)   # reads all chunks together
    a = rdf.AsNumpy(cols)
    return {c: np.asarray(a[c]) for c in cols}


def pairing_efficiency(ww_fn):
    """Read a WW treemaker output and return the per-pairing choice + efficiency.

    Returns dict with:
      pairing   : np.array, the chosen pairing (0/1/2) per event   <- "the pair"
      eff_all / eff_matched / eff_unmatched : correct-pairing fractions
      matched   : boolean mask (all 4 jets matched to quarks within MATCH_DR)
    """
    cols = ["bwpair_pairing", "bwpair_correct", "gen_pairing_true"] + \
           [f"jet{i}_matched_q_dR" for i in (1, 2, 3, 4)]
    C = load(ww_fn, cols)
    pairing = C["bwpair_pairing"].astype(int)        # the BW tool's choice, per event
    correct = C["bwpair_correct"].astype(float)      # 1 if it equals the true pairing, else 0
    # an event is "matched" if ALL 4 jets sit within MATCH_DR of their quark, i.e.
    # the reco faithfully represents the quarks and the true pairing is reliable.
    dRmax   = np.stack([C[f"jet{i}_matched_q_dR"] for i in (1, 2, 3, 4)], 1).max(1)
    matched = dRmax < MATCH_DR
    return {
        "pairing":       pairing,
        "matched":       matched,
        "eff_all":       float(correct.mean()),
        "eff_matched":   float(correct[matched].mean()),
        "eff_unmatched": float(correct[~matched].mean()),
        "n":             len(pairing),
        "n_matched":     int(matched.sum()),
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- pairing efficiency (WW) ----
    r = pairing_efficiency(WW_FN)
    print("=" * 60)
    print(f"BW PAIRING EFFICIENCY (WW->4q, n={r['n']})   [chance = 0.333]")
    print(f"   all events           : {r['eff_all']:.3f}")
    print(f"   matched (dR<{MATCH_DR})     : {r['eff_matched']:.3f}   "
          f"(n={r['n_matched']}, {100*r['n_matched']/r['n']:.0f}% of events)")
    print(f"   unmatched            : {r['eff_unmatched']:.3f}")
    print("=" * 60)

    # ---- WW vs ZZ gof, split matched / unmatched ----
    # The winner's gof is also a WW-vs-ZZ discriminant: ZZ di-jet masses prefer the
    # Z (~91 GeV), far from the W pole, so ZZ sits at higher gof. Overlay the three.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    WW = load(WW_FN, ["bwpair_gof_best"] + [f"jet{i}_matched_q_dR" for i in (1, 2, 3, 4)])
    ZZ = load(ZZ_FN, ["bwpair_gof_best"])
    dRmax = np.stack([WW[f"jet{i}_matched_q_dR"] for i in (1, 2, 3, 4)], 1).max(1)
    matched = dRmax < MATCH_DR
    gW, gZ = WW["bwpair_gof_best"], ZZ["bwpair_gof_best"]

    bins = np.linspace(0, 40, 61)
    plt.figure(figsize=(7.5, 5))
    for arr, lab in [(gW[matched], "WW matched"),
                     (gW[~matched], "WW unmatched"),
                     (gZ, "ZZ")]:
        plt.hist(arr, bins=bins, density=True, histtype="step", lw=2,
                 label=f"{lab} (n={len(arr)}, med={np.median(arr):.1f})")
    plt.xlabel(r"BW pairing gof  $-2\log[\mathrm{BW}_a\,\mathrm{BW}_b]$ (pole-ref)")
    plt.ylabel("normalized")
    plt.title("WW vs ZZ : BW pairing gof (W hypothesis)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUTDIR, "gof_WW_vs_ZZ_matched_unmatched.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print("wrote:", out)


if __name__ == "__main__":
    main()
