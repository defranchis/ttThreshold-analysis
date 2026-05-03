"""Per-axis comparison: reco MET vs −(reco jet+jet+lep) vs gen_nu vs −(gen quark+quark+lep).

Reco MET ≡ −sum(reco visible) by construction, so the first two overlap exactly.
The interesting differences are:
  - gen_nu vs −(gen quark1+quark2+lep): the ISR contribution at gen level
  - reco MET vs gen_nu: MET measurement resolution (reco − gen)
3×3 grid: rows = px/py/pz, cols = ECM.
"""
import os, numpy as np, uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WD = "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco"
SRC = f"{WD}/outputs/treemaker/lnuqq/step2_dr01_swap/semihad/wzp6_ee_munumuqq_noCut_ecm{{ecm}}.root"
OUT = f"{WD}/outputs/plots/met_vs_visible"
os.makedirs(OUT, exist_ok=True)
ECMS = (157, 160, 163)

def _xyz(p, theta, phi):
    s = np.sin(theta); return p*s*np.cos(phi), p*s*np.sin(phi), p*np.cos(theta)

need_reco = ("reco_met_p","reco_met_theta","reco_met_phi",
             "reco_jet1_p","reco_jet1_theta","reco_jet1_phi",
             "reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
             "reco_lep_p","reco_lep_theta","reco_lep_phi")
# gen px/py/pz aren't stored as flat branches; compute from (p,theta,phi).
need_gen  = ("gen_nu_p","gen_nu_theta","gen_nu_phi",
             "gen_quark1_p","gen_quark1_theta","gen_quark1_phi",
             "gen_quark2_p","gen_quark2_theta","gen_quark2_phi",
             "gen_lep_p","gen_lep_theta","gen_lep_phi")

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
print(f"{'ECM':>4}  {'axis':>3}  {'σ(reco MET − gen_nu)':>22}  {'σ(gen_nu + visible_gen)':>26}  [GeV]")
for ic, ecm in enumerate(ECMS):
    a = uproot.open(SRC.format(ecm=ecm))["events"].arrays(list(need_reco)+list(need_gen), library="np")
    # reco MET (in xyz)
    mx,my,mz = _xyz(a["reco_met_p"],   a["reco_met_theta"],   a["reco_met_phi"])
    # −(reco visible)
    j1x,j1y,j1z = _xyz(a["reco_jet1_p"], a["reco_jet1_theta"], a["reco_jet1_phi"])
    j2x,j2y,j2z = _xyz(a["reco_jet2_p"], a["reco_jet2_theta"], a["reco_jet2_phi"])
    lx,ly,lz    = _xyz(a["reco_lep_p"],  a["reco_lep_theta"],  a["reco_lep_phi"])
    neg_vis_x = -(j1x+j2x+lx); neg_vis_y = -(j1y+j2y+ly); neg_vis_z = -(j1z+j2z+lz)
    # gen
    nx, ny, nz = _xyz(a["gen_nu_p"], a["gen_nu_theta"], a["gen_nu_phi"])
    q1x, q1y, q1z = _xyz(a["gen_quark1_p"], a["gen_quark1_theta"], a["gen_quark1_phi"])
    q2x, q2y, q2z = _xyz(a["gen_quark2_p"], a["gen_quark2_theta"], a["gen_quark2_phi"])
    glx, gly, glz = _xyz(a["gen_lep_p"],   a["gen_lep_theta"],    a["gen_lep_phi"])
    neg_genvis_x = -(q1x+q2x+glx); neg_genvis_y = -(q1y+q2y+gly); neg_genvis_z = -(q1z+q2z+glz)

    for ia, (axis_name, met, vis, gen_n, gen_v) in enumerate(zip(
            ("px","py","pz"),
            (mx, my, mz), (neg_vis_x, neg_vis_y, neg_vis_z),
            (nx, ny, nz), (neg_genvis_x, neg_genvis_y, neg_genvis_z))):
        ax = axes[ia, ic]
        rng = (np.percentile(np.r_[met, gen_n], 1), np.percentile(np.r_[met, gen_n], 99))
        ax.hist(met,    bins=80, range=rng, histtype="step", lw=1.6, color="tab:blue",
                label=f"reco MET (= −Σreco vis)  σ={met.std():.2f}")
        ax.hist(gen_n,  bins=80, range=rng, histtype="step", lw=1.6, color="tab:orange",
                label=f"gen ν  σ={gen_n.std():.2f}")
        ax.hist(gen_v,  bins=80, range=rng, histtype="step", lw=1.6, color="tab:green", ls="--",
                label=f"−Σ gen vis  σ={gen_v.std():.2f}")
        ax.set_xlabel(f"{axis_name} [GeV]")
        ax.set_ylabel(f"events  (ecm{ecm})" if ic == 0 else "")
        if ia == 0: ax.set_title(f"ecm{ecm}", fontweight="bold")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
        d_reco_gen = (met - gen_n).std()
        d_isr      = (gen_n - gen_v).std()
        print(f"{ecm:>4}  {axis_name:>3}  {d_reco_gen:>22.3f}  {d_isr:>26.3f}")
fig.suptitle("MET vs −(visible) — reco vs gen, per axis")
plt.tight_layout()
out = f"{OUT}/met_vs_visible.png"
fig.savefig(out, dpi=120); plt.close(fig)
print(f"\nsaved {out}")

import sys
sys.path.insert(0, WD)
from eos_publish import publish
publish(OUT, "met_vs_visible")
print("done")
