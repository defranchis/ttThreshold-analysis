import os
import uproot
import numpy as np
import matplotlib.pyplot as plt

indir = "outputs/histmaker/lnuqq/semihad"
outdir = "outputs/plots/lnuqq/kinfit_ecm_comparison"
os.makedirs(outdir, exist_ok=True)

ecm_files = [
    (157, f"{indir}/wzp6_ee_munumuqq_noCut_ecm157.root"),
    (160, f"{indir}/wzp6_ee_munumuqq_noCut_ecm160.root"),
    (163, f"{indir}/wzp6_ee_munumuqq_noCut_ecm163.root"),
]

colors = {157: "tab:blue", 160: "tab:orange", 163: "tab:green"}

kinfit_hists = [
    ("no_cut_kinfit_mW",   r"$m_W$ kinfit combined",  "kinfit_mW"),
    ("no_cut_kinfit_mWlep", r"$m_{W}^{\ell\nu}$ kinfit", "kinfit_mWlep"),
    ("no_cut_kinfit_mWhad", r"$m_{W}^{qq}$ kinfit",     "kinfit_mWhad"),
]

mW = 80.419

for hname, title, fname in kinfit_hists:
    fig, ax = plt.subplots(figsize=(8, 6))

    for ecm, fpath in ecm_files:
        with uproot.open(fpath) as f:
            if hname not in f:
                print(f"WARNING: {hname} not found in {fpath}")
                continue
            counts, edges = f[hname].to_numpy()
        centers = 0.5 * (edges[:-1] + edges[1:])
        mask = (centers >= 50) & (centers <= 100)
        c = counts[mask]
        x = centers[mask]
        norm = c.sum()
        if norm > 0:
            c = c / norm
        ax.step(x, c, where="mid", color=colors[ecm], linewidth=2,
                label=rf"$\sqrt{{s}}$ = {ecm} GeV")

    ax.axvline(mW, color="grey", linestyle="--", linewidth=1.5,
               label=rf"$m_W$ = {mW:.3f} GeV")

    ax.set_xlabel(r"$m_W$ [GeV]", fontsize=13)
    ax.set_ylabel("A.U.", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(50, 100)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(f"{outdir}/{fname}.{fmt}", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir}/{fname}.[png|pdf]")
