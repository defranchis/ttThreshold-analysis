import os
import uproot
import numpy as np
import matplotlib.pyplot as plt

ECM_LIST  = [157, 160, 163]
INDIR     = "outputs/treemaker/lnuqq/step2/semihad"
TREE_NAME = "events"
OUTDIR    = "outputs/plots/lnuqq/allbranches"
os.makedirs(OUTDIR, exist_ok=True)

HISTS_CFG = [
    ("Wlep_reco_mass", "Wlep reco (pre-fit)",  "tab:blue",  ":",  False),
    ("Whad_reco_mass", "Whad reco (pre-fit)",  "tab:green", ":",  False),
    ("kinfit_mWlep",   "Wlep (kinfit)",         "tab:blue",  "--", False),
    ("kinfit_mWhad",   "Whad (kinfit)",         "tab:green", "--", False),
    ("kinfit_mW",      "W (kinfit combined)",   "tab:red",   "-",  False),
]

MW = 80.419
XLIM = (50, 100)
NBINS = 100

ECM_COLORS = {157: "tab:purple", 160: "tab:orange", 163: "tab:cyan"}


def _load(t, branch, require_valid):
    if branch not in t.keys():
        return None, None
    if require_valid:
        arrs = t.arrays([branch, "kinfit_valid"], library="np")
        vals = arrs[branch][arrs["kinfit_valid"].astype(bool)]
    else:
        vals = t[branch].array(library="np")
    counts, edges = np.histogram(vals, bins=NBINS, range=XLIM)
    centers = 0.5 * (edges[:-1] + edges[1:])
    norm = counts.sum()
    c = counts.astype(float)
    if norm > 0:
        c = c / norm
    return centers, c


def plot_overlay_per_ecm(ecm, t):
    """One plot showing all mW histogram variants for a single ECM."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for branch, label, color, ls, require_valid in HISTS_CFG:
        x, c = _load(t, branch, require_valid)
        if x is None:
            print(f"  WARNING [{ecm}]: {branch} not found")
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


def plot_ecm_comparison(trees):
    """One plot per histogram showing all ECMs overlaid."""
    for branch, label, _, _, require_valid in HISTS_CFG:
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted = False
        for ecm, t in trees.items():
            if t is None:
                continue
            x, c = _load(t, branch, require_valid)
            if x is None:
                print(f"  WARNING [{ecm}]: {branch} not found")
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
        for fmt in ("png", "pdf"):
            fig.savefig(f"{OUTDIR}/ecm_comparison_{branch}.{fmt}", dpi=150)
        plt.close(fig)


trees = {}
for ecm in ECM_LIST:
    path = f"{INDIR}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping ecm{ecm}")
        trees[ecm] = None
        continue
    trees[ecm] = uproot.open(path)[TREE_NAME]

for ecm, t in trees.items():
    if t is not None:
        plot_overlay_per_ecm(ecm, t)
        print(f"Saved mW_overlay_ecm{ecm}.[png|pdf]")

plot_ecm_comparison(trees)
print(f"Saved ecm_comparison_*.[png|pdf]  →  {OUTDIR}/")
