"""Lepton resolution dependence on kinematic variables (theta, pT).

For each binning variable in (theta, pT) we produce one plot:
  across-bin overlay of lep resolutions (3 resolutions × all bins).

Same data source and dR<0.1 mask as the jet-resolution study.
"""
import os
import numpy as np
import uproot
import matplotlib.pyplot as plt

WD       = "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco"
ECMS     = (157, 160, 163)
SRC_TMPL = f"{WD}/outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm{{ecm}}.root"
OUT_BASE = f"{WD}/outputs/plots/lep_resol_vs_kin"

RESOLS = [
    ("p_resp",      "p response (reco_p / gen_p)",            (0.85, 1.10), 100),
    ("theta_resol", "theta resolution (reco - gen) [rad]",   (-0.0005, 0.0005), 100),
    ("phi_resol",   "phi resolution (reco - gen) [rad]",     (-0.0005, 0.0005), 100),
]

BINNINGS = [
    {"name": "abscostheta", "label": "|cos(theta)|", "var": "reco_lep_costheta",
     "transform": lambda v: np.abs(v),
     "n_bins": 5,
     "fmt": lambda lo, hi: f"|cos(theta)| in [{lo:.2f}, {hi:.2f}]"},
    {"name": "pt", "label": "pT [GeV]", "var": "reco_lep_pt",
     "n_bins": 4,
     "fmt": lambda lo, hi: f"pT in [{lo:.1f}, {hi:.1f}]"},
    {"name": "p", "label": "p [GeV]", "var": "reco_lep_p",
     "n_bins": 4,
     "fmt": lambda lo, hi: f"p in [{lo:.1f}, {hi:.1f}]"},
    {"name": "phi", "label": "phi [rad]", "var": "reco_lep_phi",
     "n_bins": 4,
     "fmt": lambda lo, hi: f"phi in [{lo:+.2f}, {hi:+.2f}]"},
]

need = ["jet1_matched_q_dR", "jet2_matched_q_dR"]
for r in RESOLS:
    need.append(f"lep_{r[0]}")
for b in BINNINGS:
    need.append(b["var"])

def _load(ECM):
    SRC = SRC_TMPL.format(ecm=ECM)
    arr = uproot.open(SRC)["events"].arrays(need, library="np")
    mask = (arr["jet1_matched_q_dR"] < 0.1) & (arr["jet2_matched_q_dR"] < 0.1)
    OUTDIR = f"{OUT_BASE}/ecm{ECM}"
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"\n=== ecm{ECM} → {OUTDIR}, events={mask.sum()} (dR<0.1 on both jets) ===")
    return arr, mask, OUTDIR

for ECM in ECMS:
  arr, mask, OUTDIR = _load(ECM)
  for B in BINNINGS:
    n_bins = B["n_bins"]
    tfm = B.get("transform", lambda v: v)
    v = tfm(arr[B["var"]])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(v[mask], qs)
    edges[0]  -= 1e-9
    edges[-1] += 1e-9
    bin_labels = [B["fmt"](edges[i], edges[i+1]) for i in range(n_bins)]
    bi = np.full(len(v), -1, dtype=np.int32)
    for i in range(n_bins):
        bi[(v >= edges[i]) & (v < edges[i+1])] = i

    fig, axes = plt.subplots(1, len(RESOLS), figsize=(15, 4.5))
    cmap = plt.cm.viridis(np.linspace(0, 1, n_bins))
    for j, (r, lbl, rng, nb) in enumerate(RESOLS):
        ax = axes[j]
        for i in range(n_bins):
            d = arr[f"lep_{r}"][mask & (bi == i)]
            if len(d) < 10:
                continue
            ax.hist(d, bins=nb, range=rng, histtype="step", lw=1.6,
                    density=True, color=cmap[i],
                    label=f"{bin_labels[i]} (N={len(d)}, σ={d.std():.3g})")
        ax.set_xlabel(lbl)
        if j == 0:
            ax.set_ylabel("normalised")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle(f"ecm{ECM}: lepton resolutions across {B['name']} bins")
    plt.tight_layout()
    out = f"{OUTDIR}/lep_across_{B['name']}_bins.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  saved {out}")

import sys
sys.path.insert(0, WD)
from eos_publish import publish
publish(OUT_BASE, "resol_vs_kin/lep")
print("done")
