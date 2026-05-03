"""kinfit valid_frac vs d_32 cut — for the dr01 SWAP binned config (best so far).

For each ECM, plot:
  - d_32 distribution split by kinfit_valid (do failures concentrate at high d_32?)
  - cumulative valid_frac as a function of d_32 upper cut
  - also report event-fraction kept at each cut threshold
"""
import os
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WD = "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco"
OUT = f"{WD}/outputs/plots/d32_vs_kinfit"
os.makedirs(OUT, exist_ok=True)
ECMS = (157, 160, 163)
SRC_TMPL = f"{WD}/outputs/treemaker/lnuqq/step2_dr01_swap/semihad/wzp6_ee_munumuqq_noCut_ecm{{ecm}}.root"

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
print(f"{'ECM':>4}  {'cut':>8}  {'kept':>8}  {'valid_frac':>10}")
for ic, ecm in enumerate(ECMS):
    arr = uproot.open(SRC_TMPL.format(ecm=ecm))["events"].arrays(["d_32","kinfit_valid"], library="np")
    d32 = arr["d_32"].astype(float).ravel() if arr["d_32"].dtype != float else arr["d_32"]
    valid = arr["kinfit_valid"].astype(int).ravel()
    # d_32 has units of GeV (sqrt of squared kt scale) — typical range 0..40 GeV
    d32 = np.sqrt(np.maximum(d32, 0.0))   # convert from GeV^2 → GeV (d_ij is scale^2 by convention)

    ax = axes[0, ic]
    rng = (0, np.percentile(d32, 99.5))
    ax.hist(d32[valid==1], bins=60, range=rng, histtype="step", lw=1.6, color="steelblue",
            density=True, label=f"valid (N={(valid==1).sum()})")
    ax.hist(d32[valid==0], bins=60, range=rng, histtype="step", lw=1.6, color="crimson",
            density=True, label=f"invalid (N={(valid==0).sum()})")
    ax.set_xlabel("$\\sqrt{d_{32}}$ [GeV]")
    ax.set_ylabel("density")
    ax.set_title(f"ecm{ecm}: d32 split by kinfit_valid")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # cumulative kept-fraction and valid_frac as a function of upper cut
    cuts = np.linspace(0.5, rng[1], 60)
    keep_frac, vfrac = [], []
    for c in cuts:
        m = d32 < c
        keep_frac.append(m.mean())
        vfrac.append(valid[m].mean() if m.sum() else np.nan)
    ax = axes[1, ic]
    ax.plot(cuts, vfrac, color="tab:blue", lw=2, label="valid_frac after cut")
    ax.plot(cuts, keep_frac, color="tab:orange", lw=1.6, ls="--", label="event keep frac")
    ax.set_xlabel("$\\sqrt{d_{32}}$ < cut [GeV]")
    ax.set_ylabel("fraction")
    ax.set_title(f"ecm{ecm}: cuts on d32")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    # report a few cuts of interest
    for c in (np.percentile(d32, 50), np.percentile(d32, 75), np.percentile(d32, 90)):
        m = d32 < c
        print(f"{ecm:>4}  d32<{c:>5.2f}  {m.mean()*100:>6.1f}%  {valid[m].mean()*100:>9.2f}%")

fig.suptitle("dr01 SWAP binned — d_32 vs kinfit convergence")
plt.tight_layout()
out = f"{OUT}/d32_vs_kinfit.png"
fig.savefig(out, dpi=120); plt.close(fig)
print(f"saved {out}")

import sys
sys.path.insert(0, WD)
from eos_publish import publish
publish(OUT, "d32_vs_kinfit")
print("done")
