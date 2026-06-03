"""Jet resolution dependence on kinematic variables (theta, pT).

For each binning variable in (theta, pT) we produce 3 plots:
  A) per-bin overlay of jet1 vs jet2 resolutions (3 resolutions × N bins).
  B) across-bin overlay for jet1 only (3 resolutions × all bins).
  C) across-bin overlay for jet2 only (3 resolutions × all bins).

Uses step2 ROOT files (any config — per-event resolution branches identical
across configs since step2 keeps all events; the dR<0.1 cut here is applied
in-plot to drop matching-failure tails).
"""
import os
import numpy as np
import uproot
import matplotlib.pyplot as plt

WD       = "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco"
ECMS     = (157, 160, 163)
SRC_TMPL = f"{WD}/outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm{{ecm}}.root"
OUT_BASE = f"{WD}/outputs/plots/jet_resol_vs_kin"

RESOLS = [
    ("p_resp",      "p response (reco_p / gen_p)",       (0.7, 1.3),  100),
    ("theta_resol", "theta resolution (reco - gen) [rad]", (-0.05, 0.05), 100),
    ("phi_resol",   "phi resolution (reco - gen) [rad]",   (-0.06, 0.06), 100),
]

# binning vars: branch on jet1, branch on jet2, edges, label, prettyfn
BINNINGS = [
    {"name": "abscostheta", "label": "|cos(theta)|",
     "j1": "reco_jet1_costheta", "j2": "reco_jet2_costheta",
     "transform": lambda v: np.abs(v),
     "n_bins": 5,
     "fmt": lambda lo, hi: f"|cos(theta)| in [{lo:.2f}, {hi:.2f}]"},
    {"name": "pt", "label": "pT [GeV]",
     "j1": "reco_jet1_pt", "j2": "reco_jet2_pt",
     "n_bins": 4,
     "fmt": lambda lo, hi: f"pT in [{lo:.1f}, {hi:.1f}]"},
    {"name": "p", "label": "p [GeV]",
     "j1": "reco_jet1_p", "j2": "reco_jet2_p",
     "n_bins": 4,
     "fmt": lambda lo, hi: f"p in [{lo:.1f}, {hi:.1f}]"},
    {"name": "phi", "label": "phi [rad]",
     "j1": "reco_jet1_phi", "j2": "reco_jet2_phi",
     "n_bins": 4,
     "fmt": lambda lo, hi: f"phi in [{lo:+.2f}, {hi:+.2f}]"},
]

need = ["jet1_matched_q_dR", "jet2_matched_q_dR"]
for r in RESOLS:
    need += [f"jet1_{r[0]}", f"jet2_{r[0]}"]
for b in BINNINGS:
    need += [b["j1"], b["j2"]]

def run_for_ecm(ECM):
    SRC = SRC_TMPL.format(ecm=ECM)
    OUTDIR = f"{OUT_BASE}/ecm{ECM}"
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"\n=== ecm{ECM} → {OUTDIR} ===")
    arr = uproot.open(SRC)["events"].arrays(need, library="np")
    mask = (arr["jet1_matched_q_dR"] < 0.1) & (arr["jet2_matched_q_dR"] < 0.1)
    print(f"  events: {mask.sum()} (dR<0.1 on both jets)")
    return arr, mask, OUTDIR, ECM

for ECM in ECMS:
  arr, mask, OUTDIR, _ = run_for_ecm(ECM)
  for B in BINNINGS:
    n_bins = B["n_bins"]
    tfm = B.get("transform", lambda v: v)
    v_j1 = tfm(arr[B["j1"]])
    v_j2 = tfm(arr[B["j2"]])
    combined = np.concatenate([v_j1[mask], v_j2[mask]])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(combined, qs)
    edges[0]  -= 1e-9
    edges[-1] += 1e-9
    bin_labels = [B["fmt"](edges[i], edges[i+1]) for i in range(n_bins)]
    def bin_idx(values):
        idx = np.full(len(values), -1, dtype=np.int32)
        for i in range(n_bins):
            idx[(values >= edges[i]) & (values < edges[i+1])] = i
        idx[values >= edges[-1]] = -1
        return idx
    bi1 = bin_idx(v_j1)
    bi2 = bin_idx(v_j2)

    # Plot A: per-bin overlay jet1 vs jet2
    fig, axes = plt.subplots(n_bins, len(RESOLS), figsize=(15, 3.0*n_bins),
                             squeeze=False)
    for i in range(n_bins):
        for j, (r, lbl, rng, nb) in enumerate(RESOLS):
            ax = axes[i, j]
            d1 = arr[f"jet1_{r}"][mask & (bi1 == i)]
            d2 = arr[f"jet2_{r}"][mask & (bi2 == i)]
            ax.hist(d1, bins=nb, range=rng, histtype="step", lw=1.6,
                    color="C0", density=True,
                    label=f"jet1 (N={len(d1)}, σ={d1.std():.4g})")
            ax.hist(d2, bins=nb, range=rng, histtype="step", lw=1.6,
                    color="C3", density=True,
                    label=f"jet2 (N={len(d2)}, σ={d2.std():.4g})")
            ax.set_xlabel(lbl)
            if j == 0:
                ax.set_ylabel(f"{bin_labels[i]}\n\nnormalised")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.3)
    fig.suptitle(f"ecm{ECM}: jet resolutions vs {B['name']} — jet1 (blue) vs jet2 (red), per bin")
    plt.tight_layout()
    out = f"{OUTDIR}/A_per_bin_jet1_vs_jet2_by_{B['name']}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  saved {out}")

    # Plot B/C: across-bins, per-jet (equal-occupancy edges for THIS jet only)
    for jet in (1, 2):
        var_jet = tfm(arr[B["j1"] if jet == 1 else B["j2"]])
        edges_jet = np.quantile(var_jet[mask], qs)
        edges_jet[0]  -= 1e-9
        edges_jet[-1] += 1e-9
        bi_jet = np.full(len(var_jet), -1, dtype=np.int32)
        for i in range(n_bins):
            bi_jet[(var_jet >= edges_jet[i]) & (var_jet < edges_jet[i+1])] = i
        labels_jet = [B["fmt"](edges_jet[i], edges_jet[i+1]) for i in range(n_bins)]

        fig, axes = plt.subplots(1, len(RESOLS), figsize=(15, 4.5))
        cmap = plt.cm.viridis(np.linspace(0, 1, n_bins))
        for j, (r, lbl, rng, nb) in enumerate(RESOLS):
            ax = axes[j]
            for i in range(n_bins):
                d = arr[f"jet{jet}_{r}"][mask & (bi_jet == i)]
                ax.hist(d, bins=nb, range=rng, histtype="step", lw=1.6,
                        density=True, color=cmap[i],
                        label=f"{labels_jet[i]} (N={len(d)}, σ={d.std():.4g})")
            ax.set_xlabel(lbl)
            if j == 0:
                ax.set_ylabel("normalised")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.3)
        fig.suptitle(f"ecm{ECM}: jet{jet} resolutions across {B['name']} bins (equal-occupancy per jet)")
        plt.tight_layout()
        out = f"{OUTDIR}/{'B' if jet==1 else 'C'}_jet{jet}_across_{B['name']}_bins.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"  saved {out}")

# Publish (OUT_BASE contains ecm{N}/ subdirs)
import sys
sys.path.insert(0, WD)
from eos_publish import publish
publish(OUT_BASE, "resol_vs_kin/jet")
print("done")
