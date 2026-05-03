"""Plot kinfit valid_frac across dR cut configs × jet-prior modes × ECMs.

Top panels: absolute valid_frac per mode, grouped by ECM, bars per dR cut.
Bottom panels: difference (dr01 - dr02) and (dr01 - nocut if available)
showing the gain from tightening the dR cut.
"""
import os, glob
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WD = "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco"
OUT = f"{WD}/outputs/plots/dr_compare"
os.makedirs(OUT, exist_ok=True)
ECMS = (157, 160, 163)
MODES = ("pool", "swap", "fixed")
DR_TAGS = ("dr01", "dr02", "nocut")  # tags present today; missing ones are skipped
DR_LABEL = {"dr01": "dR<0.1", "dr02": "dR<0.2", "nocut": "no cut"}
DR_COLOR = {"dr01": "tab:blue", "dr02": "tab:orange", "nocut": "tab:green"}


def vfrac(tag, mode, ecm):
    f = f"{WD}/outputs/treemaker/lnuqq/step2_{tag}_{mode}/semihad/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
    if not os.path.exists(f): return None
    try:
        v = uproot.open(f)["events"]["kinfit_valid"].array(library="np")
        return 100.0 * v.sum() / len(v)
    except Exception:
        return None  # file still being written / no events tree yet


# Collect: data[tag][mode][ecm] = valid_frac (or None)
data = {tag: {m: {e: vfrac(tag, m, e) for e in ECMS} for m in MODES} for tag in DR_TAGS}
present = [t for t in DR_TAGS if any(data[t][m][e] is not None for m in MODES for e in ECMS)]

n_panels = len(MODES)
fig, axes = plt.subplots(2, n_panels, figsize=(4.5*n_panels, 8), sharey="row")
x = np.arange(len(ECMS))
bar_w = 0.8 / max(len(present), 1)
for ic, mode in enumerate(MODES):
    ax = axes[0, ic]
    for it, tag in enumerate(present):
        ys = [data[tag][mode][e] for e in ECMS]
        ys_plot = [y if y is not None else 0 for y in ys]
        ax.bar(x + (it - (len(present)-1)/2)*bar_w, ys_plot, bar_w,
               color=DR_COLOR[tag], label=DR_LABEL[tag])
        for xi, y in zip(x + (it - (len(present)-1)/2)*bar_w, ys):
            if y is not None:
                ax.text(xi, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([f"ecm{e}" for e in ECMS])
    ax.set_ylim(0, 105)
    ax.set_title(f"{mode.upper()}", fontweight="bold")
    if ic == 0: ax.set_ylabel("valid_frac [%]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, ic]
    if "dr01" in present and "dr02" in present:
        diff = [(data["dr01"][mode][e] - data["dr02"][mode][e])
                if data["dr01"][mode][e] is not None and data["dr02"][mode][e] is not None
                else 0 for e in ECMS]
        ax.bar(x - 0.18, diff, 0.36, color="tab:purple", label="dr01 − dr02")
        for xi, y in zip(x - 0.18, diff):
            ax.text(xi, y + np.sign(y)*0.5, f"{y:+.1f}", ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=8)
    if "dr01" in present and "nocut" in present:
        diff2 = [(data["dr01"][mode][e] - data["nocut"][mode][e])
                 if data["dr01"][mode][e] is not None and data["nocut"][mode][e] is not None
                 else 0 for e in ECMS]
        ax.bar(x + 0.18, diff2, 0.36, color="tab:olive", label="dr01 − nocut")
        for xi, y in zip(x + 0.18, diff2):
            ax.text(xi, y + np.sign(y)*0.5, f"{y:+.1f}", ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"ecm{e}" for e in ECMS])
    if ic == 0: ax.set_ylabel("Δ valid_frac [pp]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Binned-prior step2 — valid_frac vs dR cut, by jet-prior mode")
plt.tight_layout()
out = f"{OUT}/dr_compare.png"
fig.savefig(out, dpi=120); plt.close(fig)
print(f"saved {out}")

import sys
sys.path.insert(0, WD)
from eos_publish import publish
publish(OUT, "dr_compare")
print("done")
