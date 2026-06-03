#!/usr/bin/env python3
"""Plot the beam-energy-spread proxy m(e+e-) - ECM measured from depth=1
e+e- (post-BES, pre-ISR), per ECM. Reads gen_ee_m_minus_ecm from step1
outputs and writes to outputs/plots/bes/."""
import os
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WD         = "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco"
INDIR      = f"{WD}/outputs/treemaker/lnuqq/step1/semihad"
OUTDIR     = f"{WD}/outputs/plots/bes"
ECM_LIST   = [157, 160, 163]
ECM_COLORS = {157: "tab:purple", 160: "tab:orange", 163: "tab:cyan"}

os.makedirs(OUTDIR, exist_ok=True)

data = {}
for ecm in ECM_LIST:
    f = uproot.open(f"{INDIR}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root")
    arr = f["events"]["gen_ee_m_minus_ecm"].array(library="np") * 1000.0  # GeV → MeV
    data[ecm] = arr

# Overlay m(ee) - ECM for the 3 ECMs
fig, ax = plt.subplots(figsize=(7, 5))
bins = np.linspace(-500, 500, 101)
for ecm in ECM_LIST:
    a = data[ecm]
    label = (f"ecm{ecm}: μ={a.mean():+.1f}  σ={a.std():.1f} MeV  "
             f"(N={len(a):,})")
    ax.hist(a, bins=bins, histtype="step", linewidth=1.6,
            color=ECM_COLORS[ecm], label=label, density=True)
ax.axvline(0, color="grey", linewidth=0.8, alpha=0.6)
ax.set_xlabel(r"$m(e^+e^-) - E_\mathrm{cm}$  [MeV]")
ax.set_ylabel("density [1/MeV]")
ax.set_title("Beam-energy-spread proxy (depth=1 e±, post-BES pre-ISR)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
out = f"{OUTDIR}/m_ee_minus_ecm.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")

# Per-beam σ vs ECM (σ_per_beam = σ_total / √2 since the two beams add in quadrature).
fig2, ax2 = plt.subplots(figsize=(5, 4))
ecms = np.array(ECM_LIST)
sigmas_total = np.array([data[e].std() for e in ECM_LIST])
sigmas_perbeam = sigmas_total / np.sqrt(2)
ax2.plot(ecms, sigmas_total, "o-", label=r"$\sigma(m_{ee})$", color="black")
ax2.plot(ecms, sigmas_perbeam, "s--", label=r"$\sigma/\sqrt{2}$ (per beam)",
         color="tab:red")
for e, s, sb in zip(ecms, sigmas_total, sigmas_perbeam):
    ax2.annotate(f"{s:.1f}", (e, s), textcoords="offset points", xytext=(4, 4))
    ax2.annotate(f"{sb:.1f}", (e, sb), textcoords="offset points", xytext=(4, 4))
ax2.set_xlabel(r"$E_\mathrm{cm}$  [GeV]")
ax2.set_ylabel(r"$\sigma$  [MeV]")
ax2.set_title("BES width vs ECM")
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
out2 = f"{OUTDIR}/sigma_vs_ecm.png"
fig2.savefig(out2, dpi=140)
print(f"wrote {out2}")
