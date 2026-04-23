import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

infile = "outputs/histmaker/lnuqq/wzp6_ee_munumuqq_noCut_ecm160.root"
outdir = "outputs/plots/lnuqq/allbranches/"
os.makedirs(outdir, exist_ok=True)

hists_cfg = [
    ("no_cut_Wlep_reco_mass", "Wlep reco (pre-fit)",  "tab:blue",  ":"),
    ("no_cut_Whad_reco_mass", "Whad reco (pre-fit)",  "tab:green", ":"),
    ("no_cut_kinfit_mWlep",   "Wlep (kinfit)",         "tab:blue",  "--"),
    ("no_cut_kinfit_mWhad",   "Whad (kinfit)",          "tab:green", "--"),
    ("no_cut_kinfit_mW",      "W (kinfit combined)",    "tab:red",   "-"),
]

f = uproot.open(infile)

fig, ax = plt.subplots(figsize=(8, 6))

for hname, label, color, ls in hists_cfg:
    if hname not in f:
        print(f"WARNING: {hname} not found in {infile}")
        continue
    counts, edges = f[hname].to_numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= 50) & (centers <= 100)
    c = counts[mask]
    x = centers[mask]
    norm = c.sum()
    if norm > 0:
        c = c / norm
    ax.step(x, c, where="mid", color=color, linestyle=ls, linewidth=2, label=label)

mW = 80.419
ax.axvline(mW, color="grey", linestyle="-", linewidth=1.5, label=f"$m_W$ = {mW:.3f} GeV")

ax.set_xlabel(r"$m_W$ [GeV]", fontsize=13)
ax.set_ylabel("A.U.", fontsize=13)
ax.set_xlim(50, 100)
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=11, loc="upper left")
fig.tight_layout()

for fmt in ("png", "pdf"):
    fig.savefig(f"{outdir}/mW_overlay.{fmt}", dpi=150)

print(f"Saved {outdir}/mW_overlay.[png|pdf]")
