import os
import uproot
import numpy as np
import matplotlib.pyplot as plt

ECM_LIST  = [157, 160, 163]
INDIR     = "outputs/histmaker/lnuqq/step2/semihad"
OUTDIR    = "outputs/plots/lnuqq/allbranches"
os.makedirs(OUTDIR, exist_ok=True)

HISTS_CFG = [
    ("no_cut_Wlep_reco_mass", "Wlep reco (pre-fit)",  "tab:blue",  ":"),
    ("no_cut_Whad_reco_mass", "Whad reco (pre-fit)",  "tab:green", ":"),
    ("no_cut_kinfit_mWlep",   "Wlep (kinfit)",         "tab:blue",  "--"),
    ("no_cut_kinfit_mWhad",   "Whad (kinfit)",          "tab:green", "--"),
    ("no_cut_kinfit_mW",      "W (kinfit combined)",    "tab:red",   "-"),
]

MW = 80.419
XLIM = (50, 100)

ECM_COLORS = {157: "tab:purple", 160: "tab:orange", 163: "tab:cyan"}


def _load(f, hname):
    if hname not in f:
        return None, None
    counts, edges = f[hname].to_numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= XLIM[0]) & (centers <= XLIM[1])
    c = counts[mask]
    x = centers[mask]
    norm = c.sum()
    if norm > 0:
        c = c / norm
    return x, c


def plot_overlay_per_ecm(ecm, f):
    """One plot showing all mW histogram variants for a single ECM."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for hname, label, color, ls in HISTS_CFG:
        x, c = _load(f, hname)
        if x is None:
            print(f"  WARNING [{ecm}]: {hname} not found")
            continue
        ax.step(x, c, where="mid", color=color, linestyle=ls, linewidth=2, label=label)

    ax.axvline(MW, color="grey", linestyle="-", linewidth=1.5, label=f"$m_W$ = {MW:.3f} GeV")
    ax.set_title(rf"$\sqrt{{s}}$ = {ecm} GeV", fontsize=13)
    ax.set_xlabel(r"$m_W$ [GeV]", fontsize=13)
    ax.set_ylabel("A.U.", fontsize=13)
    ax.set_xlim(XLIM)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(f"{OUTDIR}/mW_overlay_ecm{ecm}.{fmt}", dpi=150)
    plt.close(fig)


def plot_ecm_comparison(files):
    """One plot per histogram showing all ECMs overlaid."""
    for hname, label, _, _ in HISTS_CFG:
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted = False
        for ecm, f in files.items():
            if f is None:
                continue
            x, c = _load(f, hname)
            if x is None:
                print(f"  WARNING [{ecm}]: {hname} not found")
                continue
            ax.step(x, c, where="mid", color=ECM_COLORS[ecm], linewidth=2,
                    label=rf"$\sqrt{{s}}$ = {ecm} GeV")
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.axvline(MW, color="grey", linestyle="-", linewidth=1.5,
                   label=f"$m_W$ = {MW:.3f} GeV")
        ax.set_title(label, fontsize=13)
        ax.set_xlabel(r"$m_W$ [GeV]", fontsize=13)
        ax.set_ylabel("A.U.", fontsize=13)
        ax.set_xlim(XLIM)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=11, loc="upper left")
        fig.tight_layout()
        safe = hname.replace("no_cut_", "")
        for fmt in ("png", "pdf"):
            fig.savefig(f"{OUTDIR}/ecm_comparison_{safe}.{fmt}", dpi=150)
        plt.close(fig)


files = {}
for ecm in ECM_LIST:
    path = f"{INDIR}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping ecm{ecm}")
        files[ecm] = None
        continue
    files[ecm] = uproot.open(path)

for ecm, f in files.items():
    if f is not None:
        plot_overlay_per_ecm(ecm, f)
        print(f"Saved mW_overlay_ecm{ecm}.[png|pdf]")

plot_ecm_comparison(files)
print(f"Saved ecm_comparison_*.[png|pdf]  →  {OUTDIR}/")
