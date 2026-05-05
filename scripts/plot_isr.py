#!/usr/bin/env python3
"""Plot ISR 4-momentum components from step1 — gen_isr_{px,py,pz,E} are the
(depth=1 e+e-) - (depth=2 e+e-) difference, i.e. the ISR carried out before
the hard process. Reads outputs/treemaker/lnuqq/step1, writes to
outputs/plots/isr/."""
import os
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WD         = "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco"
INDIR      = f"{WD}/outputs/treemaker/lnuqq/step1/semihad"
OUTDIR     = f"{WD}/outputs/plots/isr"
ECM_LIST   = [157, 160, 163]
ECM_COLORS = {157: "tab:purple", 160: "tab:orange", 163: "tab:cyan"}

os.makedirs(OUTDIR, exist_ok=True)

data = {ecm: {} for ecm in ECM_LIST}
for ecm in ECM_LIST:
    with uproot.open(f"{INDIR}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root") as f:
        t = f["events"]
        for k in ("gen_isr_px", "gen_isr_py", "gen_isr_pz", "gen_isr_E"):
            data[ecm][k] = t[k].array(library="np")


def overlay(branch, xlim, xlabel, fname, log=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(*xlim, 121)
    for ecm in ECM_LIST:
        a = data[ecm][branch]
        a_clip = a[(a >= xlim[0]) & (a <= xlim[1])]
        label = (f"ecm{ecm}: μ={a.mean():+.3f} GeV  σ={a.std():.3f} GeV  "
                 f"(N={len(a):,})")
        ax.hist(a_clip, bins=bins, histtype="step", linewidth=1.6,
                color=ECM_COLORS[ecm], label=label, density=True)
    ax.axvline(0, color="grey", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density [1/GeV]")
    ax.set_title("ISR 4-momentum (depth=1 e+e- − depth=2 e+e-)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    if log:
        ax.set_yscale("log")
    fig.tight_layout()
    out = f"{OUTDIR}/{fname}"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    plt.close(fig)


# Transverse: tightly peaked near 0 (ISR mostly along beam), plot ±5 GeV log scale.
overlay("gen_isr_px", (-5, 5),  r"ISR $p_x$  [GeV]", "isr_px.png", log=True)
overlay("gen_isr_py", (-5, 5),  r"ISR $p_y$  [GeV]", "isr_py.png", log=True)
# Longitudinal: bimodal (ISR from either beam), spike at 0 + wide tails.
overlay("gen_isr_pz", (-30, 30), r"ISR $p_z$  [GeV]", "isr_pz.png", log=True)
# Total ISR energy (positive by construction).
overlay("gen_isr_E",  (0, 30),   r"ISR $E$  [GeV]",  "isr_E.png",  log=True)
