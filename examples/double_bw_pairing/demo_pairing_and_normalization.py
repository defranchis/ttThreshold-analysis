"""
Demo: correlated double-BW jet-pairing pick + why normalization matters,
run on real WW->4q reconstructed jets.

Usage:
    source /cvmfs/sw.hsf.org/key4hep/setup.sh   # provides uproot, numpy, matplotlib
    python3 demo_pairing_and_normalization.py

Env vars:
    ROOT_FILE_PAIRING  step2 WW->4q ntuple for the pairing demo (part 1).
                       Default: an ecm240 sample -- well above the 2*mW
                       threshold, so the phase-space triangle isn't tight.
    ROOT_FILE_NORM     step2 WW->4q ntuple for the normalization demo (part 2).
                       Default: an ecm160 sample -- right at the 2*mW
                       threshold, where log_Z(mW) varies fastest with mW and
                       the un-normalized bias is most dramatic (this repo's
                       own use case is a WW threshold scan).
    OUTDIR             where to write the 3 PNGs (default: an EOS www dir)
    NTOY               #events for the (slower) mW-scan toy in part 2 (default 3000)

What it shows
--------------
1. Jet-pairing selection: for each event, the double-BW*phase-space term
   (double_bw_term.neg2ll_bw_phasespace) is evaluated for the 3 ways to split
   4 jets into 2 dijets, and the minimum is picked -- compared against the
   naive "closest to mW in each dijet separately" pick and against MC truth.
2. Normalization: log_Z(mW, ...) is not flat in mW, so an UN-normalized
   -2lnL(mW) profile (built with double_bw_term.neg2ll_bw_phasespace) is
   biased away from the true mW, while the NORMALIZED profile
   (neg2ll_bw_phasespace_normalized) is not. This is the reason to add the
   log_Z term even in a plain chi2 kinfit that floats mW, not only in an
   explicit likelihood fit. The two curves both sit well below the generator
   mW in absolute terms -- that offset is the (unrelated, uncalibrated) jet
   energy scale of these raw reco jets; what matters here is the SHIFT
   between the normalized and un-normalized minima.
"""
import os

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from double_bw_term import (
    MW_PDG, GW_PDG,
    neg2ll_bw_phasespace, neg2ll_bw_phasespace_normalized,
    log_z_bw_phasespace, choose_pairing,
)

REPO = "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco"
ROOT_FILE_PAIRING = os.environ.get(
    "ROOT_FILE_PAIRING", f"{REPO}/outputs/treemaker/4q/step2_ff/had_ff_ecm240/p8_ee_WW_ecm240.root")
ROOT_FILE_NORM = os.environ.get(
    "ROOT_FILE_NORM", f"{REPO}/outputs/treemaker/4q/step2/had_scap_ctrl_s1/p8_ee_WW_ecm160.root")
OUTDIR = os.environ.get("OUTDIR", "/eos/user/m/mdefranc/www/mW/example_double_bw")
NTOY = int(os.environ.get("NTOY", "3000"))
os.makedirs(OUTDIR, exist_ok=True)


def load_events(root_file):
    """Read reco jets + MC-truth pairing from a step2 WW->4q ntuple.

    Returns (mA, mB, m_WW_reco, gen_pairing_true): mA/mB are (nEvt, 3) dijet
    masses for the 3 ways to split 4 jets into 2 dijets; m_WW_reco is the
    (pairing-invariant) invariant mass of the sum of all 4 jets.
    """
    t = uproot.open(root_file)["events"]
    need = (["gen_pairing_true"]
            + [f"reco_jet{i}_{c}" for i in (1, 2, 3, 4) for c in ("p", "theta", "phi")])
    a = t.arrays(need, library="np")

    def jet_vec(i):
        """Massless reco-jet 4-vector (px, py, pz, E) from (p, theta, phi)."""
        p, th, ph = a[f"reco_jet{i}_p"], a[f"reco_jet{i}_theta"], a[f"reco_jet{i}_phi"]
        st = np.sin(th)
        return np.stack([p * st * np.cos(ph), p * st * np.sin(ph), p * np.cos(th), p], axis=1)

    J = {i: jet_vec(i) for i in (1, 2, 3, 4)}

    def dimass(u, v):
        s = u + v
        m2 = s[:, 3] ** 2 - s[:, 0] ** 2 - s[:, 1] ** 2 - s[:, 2] ** 2
        return np.sqrt(np.maximum(m2, 0.0))

    # the 3 ways to split 4 jets into two dijets ("jet pairings")
    partitions = [((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))]
    mA = np.stack([dimass(J[p[0][0]], J[p[0][1]]) for p in partitions], axis=1)  # (n, 3)
    mB = np.stack([dimass(J[p[1][0]], J[p[1][1]]) for p in partitions], axis=1)  # (n, 3)

    # parent invariant mass is pairing-invariant: sum all 4 jets once
    Jsum = J[1] + J[2] + J[3] + J[4]
    m_WW_reco = np.sqrt(np.maximum(
        Jsum[:, 3] ** 2 - Jsum[:, 0] ** 2 - Jsum[:, 1] ** 2 - Jsum[:, 2] ** 2, 0.0))

    gen_pairing_true = a["gen_pairing_true"].astype(int)
    ok = (gen_pairing_true >= 0) & (gen_pairing_true <= 2) & np.isfinite(m_WW_reco)
    return mA[ok], mB[ok], m_WW_reco[ok], gen_pairing_true[ok]


# ══ 1. jet-pairing selection (ecm240, away from the 2*mW threshold) ═════════
mA, mB, m_WW_reco, gen_pairing_true = load_events(ROOT_FILE_PAIRING)
n = len(gen_pairing_true)
print(f"[demo] pairing: {n} WW->4q events read from {ROOT_FILE_PAIRING}")

picked_bw = choose_pairing(mA, mB, m_WW_reco, mW=MW_PDG, gW=GW_PDG)
eff_bw = float(np.mean(picked_bw == gen_pairing_true))

# naive baseline: minimize |mA-mW| + |mB-mW| per pairing -- a Gaussian-like
# pick with no lineshape/phase-space structure, for contrast
picked_naive = np.argmin(np.abs(mA - MW_PDG) + np.abs(mB - MW_PDG), axis=1)
eff_naive = float(np.mean(picked_naive == gen_pairing_true))

print(f"[demo] pairing efficiency: correlated double-BW = {eff_bw:.1%}"
      f"   naive |m-mW| pick = {eff_naive:.1%}")

rows = np.arange(n)
m_hi_true = np.maximum(mA[rows, gen_pairing_true], mB[rows, gen_pairing_true])
m_lo_true = np.minimum(mA[rows, gen_pairing_true], mB[rows, gen_pairing_true])
m_hi_bw = np.maximum(mA[rows, picked_bw], mB[rows, picked_bw])
m_lo_bw = np.minimum(mA[rows, picked_bw], mB[rows, picked_bw])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
bins = np.linspace(40, 120, 81)
for ax, (mh, ml, title) in zip(
        axes,
        [(m_hi_true, m_lo_true, "MC-truth pairing"),
         (m_hi_bw, m_lo_bw, f"double-BW pick (efficiency = {eff_bw:.1%})")]):
    ax.hist(mh, bins=bins, histtype="step", lw=1.6, label=r"$m_{hi}$")
    ax.hist(ml, bins=bins, histtype="step", lw=1.6, label=r"$m_{lo}$")
    ax.axvline(MW_PDG, color="k", ls=":", lw=1)
    ax.set_xlabel("dijet mass [GeV]")
    ax.set_title(title)
    ax.legend()
fig.suptitle(r"WW$\to$4q dijet masses: truth vs. double-BW$\times$phase-space pairing pick")
fig.tight_layout()
p1 = os.path.join(OUTDIR, "pairing_dijet_masses.png")
fig.savefig(p1, dpi=140)
plt.close(fig)


# ══ 2. why normalization matters (ecm160, right at the 2*mW threshold) ══════
# log_Z(mW, gW, m_WW) is NOT constant in mW, so an un-normalized -2lnL(mW)
# profile is not a genuine likelihood and its minimum is biased. The effect
# is largest right at threshold, where the (m_h,m_l) triangle area is most
# sensitive to the mW hypothesis -- exactly this repo's WW threshold-scan use
# case -- so this section reloads the ecm160 sample rather than reusing part 1.
mA_n, mB_n, m_WW_reco_n, gen_pairing_true_n = load_events(ROOT_FILE_NORM)
n_n = len(gen_pairing_true_n)
print(f"[demo] normalization: {n_n} WW->4q events read from {ROOT_FILE_NORM}")
rows_n = np.arange(n_n)
m_hi_true = np.maximum(mA_n[rows_n, gen_pairing_true_n], mB_n[rows_n, gen_pairing_true_n])
m_lo_true = np.minimum(mA_n[rows_n, gen_pairing_true_n], mB_n[rows_n, gen_pairing_true_n])

mw_grid = np.linspace(75.0, 86.0, 45)
m_WW_ref = float(np.median(m_WW_reco_n))
logz_curve = np.array([log_z_bw_phasespace(np.array([m_WW_ref]), mw, GW_PDG)[0] for mw in mw_grid])

fig, ax = plt.subplots(figsize=(5.2, 4))
ax.plot(mw_grid, 2 * logz_curve)
ax.set_xlabel(r"$m_W$ hypothesis [GeV]")
ax.set_ylabel(r"$2\log Z(m_W,\Gamma_W,m_{WW}$" + f"={m_WW_ref:.0f} GeV)")
ax.set_title(r"Normalization is not flat in $m_W$ -- add it if $m_W$ floats")
fig.tight_layout()
p2 = os.path.join(OUTDIR, "logz_vs_mw.png")
fig.savefig(p2, dpi=140)
plt.close(fig)

# toy mW scan using the MC-truth pairing (no combinatorial confusion), with
# vs without the log_Z term. Subsampled (NTOY) to keep the 2-D quadrature
# in the normalized curve fast.
rng = np.random.default_rng(0)
sel = rng.choice(n_n, size=min(NTOY, n_n), replace=False)
mh_t, ml_t, mww_t = m_hi_true[sel], m_lo_true[sel], m_WW_reco_n[sel]

raw_curve = np.array([np.sum(neg2ll_bw_phasespace(mh_t, ml_t, mww_t, mw, GW_PDG)) for mw in mw_grid])
norm_curve = np.array(
    [np.sum(neg2ll_bw_phasespace_normalized(mh_t, ml_t, mww_t, mw, GW_PDG)) for mw in mw_grid])


def parab_min(xs, ys):
    i = int(np.clip(np.argmin(ys), 2, len(ys) - 3))
    c = np.polyfit(xs[i - 2:i + 3], ys[i - 2:i + 3], 2)
    return -c[1] / (2 * c[0])


mw_raw = parab_min(mw_grid, raw_curve)
mw_norm = parab_min(mw_grid, norm_curve)
print(f"[demo] toy -2lnL(mW) minimum over {len(sel)} truth-paired events:"
      f" un-normalized = {mw_raw:.2f} GeV, normalized = {mw_norm:.2f} GeV"
      f" (generator mW = {MW_PDG:.3f} GeV)")

fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.plot(mw_grid, raw_curve - raw_curve.min(), label=f"un-normalized (min @ {mw_raw:.2f})")
ax.plot(mw_grid, norm_curve - norm_curve.min(), label=f"normalized (min @ {mw_norm:.2f})")
ax.axvline(MW_PDG, color="k", ls=":", lw=1, label=f"generator $m_W$={MW_PDG:.3f}")
ax.set_xlabel(r"$m_W$ hypothesis [GeV]")
ax.set_ylabel(r"$-2\ln L$ $-$ min")
ax.set_title("Un-normalized double-BW term biases a floating-$m_W$ fit")
ax.legend(fontsize=8)
fig.tight_layout()
p3 = os.path.join(OUTDIR, "mw_scan_norm_vs_unnorm.png")
fig.savefig(p3, dpi=140)
plt.close(fig)

print("[demo] plots written to:", OUTDIR)
for p in (p1, p2, p3):
    print("  -", p)
