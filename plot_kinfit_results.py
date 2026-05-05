#!/usr/bin/env python3
"""
Plot kinfit results from step2 treemaker output.

Two stages, run in order:
  1. mW overlay      — per-ECM and ECM-comparison plots of W-mass branches
                       (reco / kinfit pre/post / kinfit combined)
                       → outputs/plots/lnuqq/allbranches/  (publish: "mW_overlay")
  2. kinfit variables — per-ECM and ECM-comparison plots for every kinfit branch,
                       with input-PDF overlays from fit_resolutions.py JSON,
                       reco/gen comparisons, and equal-N NLL slices.
                       → outputs/plots/kinfit_vars/         (publish: "kinfit_vars")
"""

import os, json, math
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from eos_publish import publish

# ── Shared constants ─────────────────────────────────────────────────────────

ECM_LIST    = [157, 160, 163]
INDIR       = os.environ.get("STEP2_INDIR", "outputs/treemaker/lnuqq/step2/semihad")
INFILE_TMPL = INDIR + "/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
JSON_TMPL   = "outputs/response/functions/dcb_results_ecm{ecm}.json"
TREE_NAME   = "events"
ECM_COLORS  = {157: "tab:purple", 160: "tab:orange", 163: "tab:cyan"}


# =============================================================================
# Stage 1 — mW overlay
# =============================================================================

MW_OUTDIR = "outputs/plots/lnuqq/allbranches"

# Last column is_kinfit: branches that respect the kinfit_valid mask in the
# `_valid` variant. Pre-fit reco branches are unfiltered in both variants.
MW_HISTS_CFG = [
    ("reco_Wlep_m",   "Wlep reco (pre-fit)", "tab:blue",  ":",  False),
    ("reco_Whad_m",   "Whad reco (pre-fit)", "tab:green", ":",  False),
    ("kinfit_Wlep_m", "Wlep (kinfit)",       "tab:blue",  "--", True),
    ("kinfit_Whad_m", "Whad (kinfit)",       "tab:green", "--", True),
    ("kinfit_mW",     "W (kinfit combined)", "tab:red",   "-",  True),
]

MW_REF   = 80.419
MW_XLIM  = (50, 100)
MW_NBINS = 100


def _mw_load(t, branch, valid_only, is_kinfit):
    """Load mW histogram. valid_only masks kinfit branches by kinfit_valid;
    pre-fit reco branches are never masked (no validity concept)."""
    if branch not in t.keys():
        return None, None, 0
    if valid_only and is_kinfit:
        arrs = t.arrays([branch, "kinfit_valid"], library="np")
        vals = arrs[branch][arrs["kinfit_valid"].astype(bool)]
    else:
        vals = t[branch].array(library="np")
    counts, edges = np.histogram(vals, bins=MW_NBINS, range=MW_XLIM)
    centers = 0.5 * (edges[:-1] + edges[1:])
    norm = int(counts.sum())  # N entries in histogram range
    c = counts.astype(float)
    if norm > 0:
        c = c / norm
    return centers, c, norm


def plot_mW_overlay_per_ecm(ecm, t, variant):
    """One plot showing all mW histogram variants for a single ECM.
    variant is 'valid' or 'all' — controls kinfit_valid filtering."""
    valid_only = (variant == "valid")
    fig, ax = plt.subplots(figsize=(8, 6))
    for branch, label, color, ls, is_kinfit in MW_HISTS_CFG:
        x, c, n = _mw_load(t, branch, valid_only, is_kinfit)
        if x is None:
            print(f"  WARNING [{ecm}]: {branch} not found")
            continue
        ax.step(x, c, where="mid", color=color, linestyle=ls, linewidth=2,
                label=f"{label}  (N={n})")

    ax.axvline(MW_REF, color="grey", linestyle="-", linewidth=1.5,
               label=f"$m_W$ = {MW_REF:.3f} GeV")
    ax.set_title(rf"$\sqrt{{s}}$ = {ecm} GeV  —  kinfit {variant}", fontsize=13)
    ax.set_xlabel(r"$m_W$ [GeV]", fontsize=13)
    ax.set_ylabel("A.U.", fontsize=13)
    ax.set_xlim(MW_XLIM)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(f"{MW_OUTDIR}/mW_overlay_ecm{ecm}_{variant}.{fmt}", dpi=150)
    plt.close(fig)


def plot_mW_ecm_comparison(trees, variant):
    """One plot per histogram showing all ECMs overlaid.
    variant is 'valid' or 'all'."""
    valid_only = (variant == "valid")
    for branch, label, _, _, is_kinfit in MW_HISTS_CFG:
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted = False
        for ecm, t in trees.items():
            if t is None:
                continue
            x, c, n = _mw_load(t, branch, valid_only, is_kinfit)
            if x is None:
                print(f"  WARNING [{ecm}]: {branch} not found")
                continue
            ax.step(x, c, where="mid", color=ECM_COLORS[ecm], linewidth=2,
                    label=rf"$\sqrt{{s}}$ = {ecm} GeV  (N={n})")
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.axvline(MW_REF, color="grey", linestyle="-", linewidth=1.5,
                   label=f"$m_W$ = {MW_REF:.3f} GeV")
        ax.set_title(f"{label}  —  kinfit {variant}", fontsize=13)
        ax.set_xlabel(r"$m_W$ [GeV]", fontsize=13)
        ax.set_ylabel("A.U.", fontsize=13)
        ax.set_xlim(MW_XLIM)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=11, loc="upper left")
        fig.tight_layout()
        for fmt in ("png", "pdf"):
            fig.savefig(f"{MW_OUTDIR}/ecm_comparison_{branch}_{variant}.{fmt}", dpi=150)
        plt.close(fig)


def run_mW_overlay():
    os.makedirs(MW_OUTDIR, exist_ok=True)
    trees = {}
    for ecm in ECM_LIST:
        path = INFILE_TMPL.format(ecm=ecm)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — skipping ecm{ecm}")
            trees[ecm] = None
            continue
        trees[ecm] = uproot.open(path)[TREE_NAME]

    for variant in ("valid", "all"):
        for ecm, t in trees.items():
            if t is not None:
                plot_mW_overlay_per_ecm(ecm, t, variant)
                print(f"Saved mW_overlay_ecm{ecm}_{variant}.[png|pdf]")
        plot_mW_ecm_comparison(trees, variant)
        print(f"Saved ecm_comparison_*_{variant}.[png|pdf]  →  {MW_OUTDIR}/")

    publish(MW_OUTDIR, os.environ.get("MW_PUBSUB", "mW_overlay"))


# =============================================================================
# Stage 2 — kinfit-variable plots with PDF overlays
# =============================================================================

KINFIT_OUTDIR = "outputs/plots/kinfit_vars"

_SQRT2   = math.sqrt(2.0)
_LOG_MAX = math.log(np.finfo(np.float64).max)
_LOG_MIN = -_LOG_MAX

# Mapping from kinfit output branch → fitted resolution branch in JSON
KINFIT_TO_RESOL = {
    "kinfit_s1":  "jet1_p_resp",
    "kinfit_s2":  "jet2_p_resp",
    "kinfit_sl":  "lep_p_resp",
    "kinfit_sn":  "met_p_resp",
    "kinfit_t1":  "jet1_theta_resol",
    "kinfit_t2":  "jet2_theta_resol",
    "kinfit_tl":  "lep_theta_resol",
    "kinfit_tn":  "met_theta_resol",
    "kinfit_p1":  "jet1_phi_resol",
    "kinfit_p2":  "jet2_phi_resol",
    "kinfit_pl":  "lep_phi_resol",
    "kinfit_pn":  "met_phi_resol",
    # BES nuisances (Gaussian priors)
    "kinfit_bes_m_minus_ecm": "gen_ee_m_minus_ecm",
    "kinfit_bes_pz":          "gen_ee_pz",
}

# Mapping: kinfit post-fit branch → reco / gen counterpart, derived from the
# convention <level>_<object>_<quantity>. Objects with all three levels:
#   constituents (jet1, jet2, lep, nu): p, pt, theta, phi
#   Ws (Wlep, Whad):                    m, p, pt, px, py, pz
#   WW system:                          m, m_minus_ecm, px, py, pz, p_imbalance_tot
def _build_kinfit_maps():
    reco, gen = {}, {}
    # reco-level neutrino is the detector MET (no real nu reco); gen-level uses
    # "quark" naming for the matched parton (jets at gen level would be misleading).
    _reco_obj = {"jet1": "jet1", "jet2": "jet2", "lep": "lep", "nu": "met"}
    _gen_obj  = {"jet1": "quark1", "jet2": "quark2", "lep": "lep", "nu": "nu"}
    for obj in ("jet1", "jet2", "lep", "nu"):
        for q in ("p", "pt", "theta", "phi"):
            kf = f"kinfit_{obj}_{q}"
            reco[kf] = f"reco_{_reco_obj[obj]}_{q}"
            gen[kf]  = f"gen_{_gen_obj[obj]}_{q}"
    for obj in ("Wlep", "Whad"):
        for q in ("m", "p", "pt", "px", "py", "pz"):
            kf = f"kinfit_{obj}_{q}"
            reco[kf] = f"reco_{obj}_{q}"
            gen[kf]  = f"gen_{obj}_{q}"
    for q in ("m", "m_minus_ecm", "px", "py", "pz", "p_imbalance_tot"):
        kf = f"kinfit_WW_{q}"
        reco[kf] = f"reco_WW_{q}"
        gen[kf]  = f"gen_WW_{q}"
    return reco, gen

KINFIT_TO_RECO, KINFIT_TO_GEN = _build_kinfit_maps()
EXTRA_BRANCHES = sorted(set(KINFIT_TO_RECO.values()) | set(KINFIT_TO_GEN.values()))

# Subdirectory assignment for per-ECM plots
_PULL_BRANCHES = {
    "kinfit_s1","kinfit_s2","kinfit_sl","kinfit_sn",
    "kinfit_t1","kinfit_t2","kinfit_tn","kinfit_tl",
    "kinfit_p1","kinfit_p2","kinfit_pn","kinfit_pl",
    "kinfit_bes_m_minus_ecm","kinfit_bes_pz",
}

def _subdir(bname):
    if bname in _PULL_BRANCHES:
        return "pulls"
    if bname in KINFIT_TO_RECO and bname in KINFIT_TO_GEN:
        return "fit_vs_reco_gen"
    if bname in KINFIT_TO_GEN:
        return "fit_vs_gen"
    return "misc"

# All kinfit branches from treemaker_lnuqq_step2.py
KINFIT_BRANCHES = [
    "kinfit_mW", "kinfit_gW",
    "kinfit_s1", "kinfit_s2", "kinfit_sl", "kinfit_sn",
    "kinfit_t1", "kinfit_t2", "kinfit_tn", "kinfit_tl",
    "kinfit_p1", "kinfit_p2", "kinfit_pn", "kinfit_pl",
    "kinfit_bes_m_minus_ecm", "kinfit_bes_pz",
    "kinfit_chi2", "kinfit_chi2_ndof", "kinfit_valid", "kinfit_status",
    # constituent kinematics
    "kinfit_jet1_p",  "kinfit_jet2_p",  "kinfit_lep_p",  "kinfit_nu_p",
    "kinfit_jet1_pt", "kinfit_jet2_pt", "kinfit_lep_pt", "kinfit_nu_pt",
    "kinfit_jet1_theta", "kinfit_jet2_theta", "kinfit_lep_theta", "kinfit_nu_theta",
    "kinfit_jet1_phi",   "kinfit_jet2_phi",   "kinfit_lep_phi",   "kinfit_nu_phi",
    # W bosons
    "kinfit_Wlep_m", "kinfit_Wlep_p", "kinfit_Wlep_pt",
    "kinfit_Wlep_px", "kinfit_Wlep_py", "kinfit_Wlep_pz",
    "kinfit_Whad_m", "kinfit_Whad_p", "kinfit_Whad_pt",
    "kinfit_Whad_px", "kinfit_Whad_py", "kinfit_Whad_pz",
    # WW system (post-fit derived; no direct prior — overlaid via ISR balance)
    "kinfit_WW_px", "kinfit_WW_py", "kinfit_WW_pz",
    "kinfit_WW_m", "kinfit_WW_p_imbalance_tot",
]


# ── Model functions (copied from fit_resolutions.py — pure math) ─────────────

def _dcb_core(t, aL, nL, aR, nR):
    aL, nL, aR, nR = abs(aL), abs(nL), abs(aR), abs(nR)
    BL = nL / aL - aL
    BR = nR / aR - aR
    log_AL = nL * np.log(nL / aL) - 0.5 * aL * aL
    log_AR = nR * np.log(nR / aR) - 0.5 * aR * aR
    return np.where(
        t < -aL,
        np.exp(np.minimum(np.maximum(log_AL - nL * np.log(np.maximum(BL - t, 1e-10)), _LOG_MIN), _LOG_MAX)),
        np.where(
            t > aR,
            np.exp(np.minimum(np.maximum(log_AR - nR * np.log(np.maximum(BR + t, 1e-10)), _LOG_MIN), _LOG_MAX)),
            np.exp(-0.5 * t * t)
        )
    )

def dcb(x, N, mu, sigma, aL, nL, aR, nR):
    return N * _dcb_core((x - mu) / sigma, aL, nL, aR, nR)

def dcb_gauss(x, N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, mu_w, sigma_w):
    core = _dcb_core((x - mu_c) / sigma_c, aL, nL, aR, nR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)

def _dcb_expleft_core(t, aL, kL, aR, nR):
    kL, aL, aR, nR = abs(kL), abs(aL), abs(aR), abs(nR)
    BR = nR / aR - aR
    log_AR = nR * np.log(nR / aR) - 0.5 * aR * aR
    return np.where(
        t < -aL,
        np.exp(-0.5 * aL * aL + np.minimum(kL * (aL + t), 0.0)),
        np.where(
            t > aR,
            np.exp(np.minimum(np.maximum(log_AR - nR * np.log(np.maximum(BR + t, 1e-10)), _LOG_MIN), _LOG_MAX)),
            np.exp(-0.5 * t * t)
        )
    )

def dcb_expleft_gauss(x, N, mu_c, sigma_c, aL, kL, aR, nR, f_wide, mu_w, sigma_w):
    core = _dcb_expleft_core((x - mu_c) / sigma_c, aL, kL, aR, nR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)

def _dcb_expright_core(t, aL, nL, aR, kR):
    aL, nL, aR, kR = abs(aL), abs(nL), abs(aR), abs(kR)
    BL = nL / aL - aL
    log_AL = nL * np.log(nL / aL) - 0.5 * aL * aL
    return np.where(
        t < -aL,
        np.exp(np.minimum(np.maximum(log_AL - nL * np.log(np.maximum(BL - t, 1e-10)), _LOG_MIN), _LOG_MAX)),
        np.where(
            t > aR,
            np.exp(np.minimum(-0.5 * aR * aR - kR * (t - aR), _LOG_MAX)),
            np.exp(-0.5 * t * t)
        )
    )

def dcb_expright_gauss(x, N, mu_c, sigma_c, aL, nL, aR, kR, f_wide, mu_w, sigma_w):
    core = _dcb_expright_core((x - mu_c) / sigma_c, aL, nL, aR, kR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)

def dcb_gaussbox(x, N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, p_max, sigma_box):
    from scipy.special import erf as _sp_erf
    core = _dcb_core((x - mu_c) / sigma_c, aL, nL, aR, nR)
    p_max = abs(p_max)
    sb = max(abs(sigma_box), 1e-10)
    sq2 = _SQRT2 * sb
    wide = 0.5 * (_sp_erf((x + p_max) / sq2) - _sp_erf((x - p_max) / sq2))
    wide_peak = max(float(_sp_erf(p_max / sq2)), 1e-10)
    return N * ((1.0 - f_wide) * core + f_wide * wide / wide_peak)


def _make_pdf(p):
    """Return a callable f(x) → normalized PDF value, from JSON param dict."""
    model = p["model"]

    if model == "gauss":
        mu, sg = p["mu"], p["sigma"]
        inv_sg = 1.0 / abs(sg)
        norm_g = inv_sg / math.sqrt(2.0 * math.pi)
        def fn(x): return norm_g * np.exp(-0.5 * ((x - mu) * inv_sg) ** 2)
        return fn

    norm = p["norm"]
    if model == "dcb":
        mu, sg = p["mu"], p["sigma"]
        aL, nL, aR, nR = p["aL"], p["nL"], p["aR"], p["nR"]
        def fn(x): return norm * dcb(x, 1.0, mu, sg, aL, nL, aR, nR)

    elif model == "dcb2g":
        mu, sg = p["mu"], p["sigma"]
        aL, nL, aR, nR = p["aL"], p["nL"], p["aR"], p["nR"]
        fw, mw, sw = p["f_wide"], p["mu_wide"], p["sigma_wide"]
        def fn(x): return norm * dcb_gauss(x, 1.0, mu, sg, aL, nL, aR, nR, fw, mw, sw)

    elif model == "expleft2g":
        mu, sg = p["mu"], p["sigma"]
        aL, kL, aR, nR = p["aL"], p["kL"], p["aR"], p["nR"]
        fw, mw, sw = p["f_wide"], p["mu_wide"], p["sigma_wide"]
        def fn(x): return norm * dcb_expleft_gauss(x, 1.0, mu, sg, aL, kL, aR, nR, fw, mw, sw)

    elif model == "dcber2g":
        mu, sg = p["mu"], p["sigma"]
        aL, nL, aR, kR = p["aL"], p["nL"], p["aR"], p["kR"]
        fw, mw, sw = p["f_wide"], p["mu_wide"], p["sigma_wide"]
        def fn(x): return norm * dcb_expright_gauss(x, 1.0, mu, sg, aL, nL, aR, kR, fw, mw, sw)

    elif model == "dcbgb":
        mu, sg = p["mu"], p["sigma"]
        aL, nL, aR, nR = p["aL"], p["nL"], p["aR"], p["nR"]
        fw, pm, sb = p["f_wide"], p["p_max"], p["sigma_box"]
        def fn(x): return norm * dcb_gaussbox(x, 1.0, mu, sg, aL, nL, aR, nR, fw, pm, sb)

    else:
        return None

    return fn


# ── Axis range heuristics ─────────────────────────────────────────────────────

# Branches with a known fixed range; everything else is data-driven.
# Ranges chosen to span the post-fit (green) distribution; the prior PDF
# (red) often has heavier tails which would leave the fitted peak crammed.
_FIXED_RANGE = {
    "kinfit_mW":    (100, 50,   100),
    "kinfit_Wlep_m": (100, 50,   100),
    "kinfit_Whad_m": (100, 50,   100),
    "kinfit_gW":    (100,  1.9,  2.2),
    "kinfit_valid": (  3, -0.5,  2.5),
    "kinfit_chi2_ndof": (100, 0, 50),
    "kinfit_jet1_p":  (100,  0,   80),
    "kinfit_jet2_p":  (100,  0,   80),
    "kinfit_lep_p": (100,  0,   80),
    "kinfit_nu_p":  (100,  0,   80),
    "kinfit_Wlep_px": (100, -50, 50),
    "kinfit_Wlep_py": (100, -50, 50),
    "kinfit_Wlep_pz": (100, -50, 50),
    "kinfit_Whad_px": (100, -50, 50),
    "kinfit_Whad_py": (100, -50, 50),
    "kinfit_Whad_pz": (100, -50, 50),
    # Total WW system momentum: constrained near 0 by px/py/gen_WW_pz PDF
    # (gen has a sub-bin spike at 0 from collinear ISR; fit follows it tightly)
    "kinfit_WW_px": (100, -0.05, 0.05),
    "kinfit_WW_py": (100, -0.05, 0.05),
    "kinfit_WW_pz": (100, -0.3, 0.3),
    # BES nuisances: Gaussian priors with σ ≈ 119 MeV → ±400 MeV is ~3.5σ.
    "kinfit_bes_m_minus_ecm": (100, -0.4, 0.4),
    "kinfit_bes_pz":          (100, -0.4, 0.4),
    "kinfit_jet1_theta":  (100, 0,   3.2),
    "kinfit_jet2_theta":  (100, 0,   3.2),
    "kinfit_nu_theta":  (100, 0,   3.2),
    "kinfit_jet1_phi":    (100, -3.2, 3.2),
    "kinfit_jet2_phi":    (100, -3.2, 3.2),
    "kinfit_nu_phi":    (100, -3.2, 3.2),
    # WW system post-fit: mass near ECM, mass-minus-ECM near 0 (slightly below, ISR)
    "kinfit_WW_m":           (100, 150, 170),
    # Pull/scale parameters: span post-fit data, not the wider prior tails.
    "kinfit_s1":  (100, 0.85, 1.15),
    "kinfit_s2":  (100, 0.85, 1.15),
    "kinfit_sl":  (100, 0.85, 1.15),
    "kinfit_sn":  (100, 0.92, 1.05),
    "kinfit_t1":  (100, -0.15, 0.15),
    "kinfit_t2":  (100, -0.15, 0.15),
    "kinfit_tn":  (100, -0.03, 0.03),
    "kinfit_tl":  (100, -5e-5, 5e-5),
    "kinfit_p1":  (100, -0.15, 0.15),
    "kinfit_p2":  (100, -0.15, 0.15),
    "kinfit_pn":  (100, -0.005, 0.005),
    "kinfit_pl":  (100, -3e-4, 3e-4),
    "kinfit_WW_p_imbalance_tot": (100, 0.0, 0.1),
}

def _auto_range(vals):
    lo, hi = np.percentile(vals[np.isfinite(vals)], [0.5, 99.5])
    margin = max(abs(hi - lo) * 0.1, abs(lo) * 1e-3, 1e-12)
    return lo - margin, hi + margin

def _pdf_natural_range(p, nsigma=5):
    sg = p.get("sigma", 1.0)
    aL = abs(p.get("aL", 2.0));  aR = abs(p.get("aR", 2.0))
    half = nsigma * max(aL, aR, 1.0) * sg
    if "mu" in p:
        mu = p["mu"]
        if "mu_wide" in p and "sigma_wide" in p:
            half = max(half, abs(p["mu_wide"] - mu) + nsigma * abs(p["sigma_wide"]))
        return mu - half, mu + half
    if "x_cut" in p:
        xc = p["x_cut"];  sw = p.get("sigma_wide", sg)
        return xc - nsigma * max(sw, sg), xc + 0.5
    return -half, half

def _binning(var, vals=None, pdf_params=None):
    if var in _FIXED_RANGE:
        return _FIXED_RANGE[var]

    data_lo = data_hi = None
    if vals is not None and len(vals) > 0:
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            data_lo, data_hi = _auto_range(finite)

    if pdf_params is not None:
        pdf_lo, pdf_hi = _pdf_natural_range(pdf_params)
        if data_lo is None:
            return (100, pdf_lo, pdf_hi)
        data_w = data_hi - data_lo
        pdf_w  = pdf_hi  - pdf_lo
        if data_w < 0.2 * pdf_w:
            return (100, pdf_lo, pdf_hi)
        return (100, data_lo, data_hi)

    if data_lo is not None:
        return (100, data_lo, data_hi)
    return (100, -5, 5)


# ── Per-ECM single-branch plot ────────────────────────────────────────────────

def _plot_branch(ax, bname, vals, ecm, pdf_fn=None, pdf_params=None,
                 reco_vals=None, gen_vals=None, color="steelblue", xlim=None):
    all_combined = np.concatenate([v for v in [vals, reco_vals, gen_vals]
                                   if v is not None and len(v)])
    nbins, xlo, xhi = _binning(bname, all_combined, pdf_params=pdf_params)
    if xlim is not None:
        xlo, xhi = xlim

    def _draw_hist(v, label, clr, as_bar=False):
        v_c = v[(v >= xlo) & (v <= xhi)]
        counts, edges = np.histogram(v_c, bins=nbins, range=(xlo, xhi))
        bw = np.diff(edges)[0]
        density = counts / max(counts.sum() * bw, 1e-300)
        centers = 0.5 * (edges[:-1] + edges[1:])
        if as_bar:
            ax.bar(centers, density, width=bw, color=clr, alpha=0.55, label=label)
        else:
            ax.step(centers, density, where="mid", color=clr, lw=2, label=label)

    if reco_vals is not None and len(reco_vals):
        _draw_hist(reco_vals, "reco", "lightskyblue", as_bar=True)
    if gen_vals is not None and len(gen_vals):
        _draw_hist(gen_vals, "gen", "tab:orange")
    _draw_hist(vals, "fitted", "tab:green")

    if pdf_fn is not None and pdf_params is not None:
        xfine = np.linspace(xlo, xhi, 600)
        yfine = pdf_fn(xfine)
        integ_plot = float(np.trapz(yfine, xfine))
        yfine = yfine / max(integ_plot, 1e-300)
        ax.plot(xfine, yfine, color="crimson", lw=2, label="input PDF")

    ax.set_xlabel(bname, fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False)


def plot_per_ecm(ecm, branches_data, json_results):
    base = f"{KINFIT_OUTDIR}/ecm{ecm}"

    for bname in [b for b in KINFIT_BRANCHES if b in branches_data]:
        vals = branches_data[bname]
        resol_name = KINFIT_TO_RESOL.get(bname)
        pdf_fn = None
        pdf_params = None
        if resol_name and resol_name in json_results:
            pdf_params = json_results[resol_name]
            pdf_fn = _make_pdf(pdf_params)

        reco_bname = KINFIT_TO_RECO.get(bname)
        gen_bname  = KINFIT_TO_GEN.get(bname)
        reco_vals  = branches_data.get(reco_bname) if reco_bname else None
        gen_vals   = branches_data.get(gen_bname)  if gen_bname  else None

        out_dir = f"{base}/{_subdir(bname)}"
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        _plot_branch(ax, bname, vals, ecm, pdf_fn=pdf_fn, pdf_params=pdf_params,
                     reco_vals=reco_vals, gen_vals=gen_vals)
        title = f"{bname}  [ecm{ecm}]"
        if resol_name:
            title += f"\ninput PDF: {resol_name}"
            if resol_name in json_results:
                title += f"  ({json_results[resol_name]['model']}, χ²/ndf={json_results[resol_name]['chi2_ndof']:.2f})"
        ax.set_title(title, fontsize=10)

        for fmt in ("png", "pdf"):
            fig.savefig(f"{out_dir}/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    print(f"  [ecm{ecm}]  per-ECM plots → {base}/")


# ── Chi2-slice comparison plots ───────────────────────────────────────────────

def plot_chi2_slices(ecm, raw_arrays, json_results):
    """Per fit_vs_reco_gen branch: 3 panels (reco | gen | fit), each showing
    3 overlaid equal-N NLL slices (NLL = chi2/2)."""
    chi2  = raw_arrays.get("kinfit_chi2")
    if chi2 is None:
        return

    out_dir = f"{KINFIT_OUTDIR}/ecm{ecm}/nll_slices"
    os.makedirs(out_dir, exist_ok=True)

    SLICE_COLORS = ["tab:blue", "tab:orange", "tab:red"]

    for bname in KINFIT_BRANCHES:
        reco_bname = KINFIT_TO_RECO.get(bname)
        gen_bname  = KINFIT_TO_GEN.get(bname)
        if not (reco_bname and gen_bname):
            continue
        if bname not in raw_arrays or reco_bname not in raw_arrays or gen_bname not in raw_arrays:
            continue

        fit_arr  = np.asarray(raw_arrays[bname],      dtype=float)
        reco_arr = np.asarray(raw_arrays[reco_bname], dtype=float)
        gen_arr  = np.asarray(raw_arrays[gen_bname],  dtype=float)
        nll_arr  = np.asarray(chi2,                   dtype=float)

        mask = (np.isfinite(fit_arr) & np.isfinite(reco_arr) &
                np.isfinite(gen_arr) & np.isfinite(nll_arr))

        fit_arr  = fit_arr[mask]
        reco_arr = reco_arr[mask]
        gen_arr  = gen_arr[mask]
        nll_arr  = nll_arr[mask]

        if len(nll_arr) < 30:
            continue

        q33, q67 = np.percentile(nll_arr, [100.0 / 3, 200.0 / 3])
        slice_masks = [
            nll_arr < q33,
            (nll_arr >= q33) & (nll_arr < q67),
            nll_arr >= q67,
        ]
        slice_labels = [
            f"NLL < {q33:.1f}",
            f"{q33:.1f} ≤ NLL < {q67:.1f}",
            f"NLL ≥ {q67:.1f}",
        ]

        nbins, xlo, xhi = _binning(bname, np.concatenate([fit_arr, reco_arr, gen_arr]))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
        fig.suptitle(f"{bname}  [ecm{ecm}]  — equal-N NLL slices", fontsize=11)

        for ax, (arr, panel_label) in zip(axes, [
            (reco_arr, "reco"),
            (gen_arr,  "gen"),
            (fit_arr,  "fitted"),
        ]):
            for sl_mask, sl_label, clr in zip(slice_masks, slice_labels, SLICE_COLORS):
                v_c = arr[sl_mask]
                v_c = v_c[(v_c >= xlo) & (v_c <= xhi)]
                counts, edges = np.histogram(v_c, bins=nbins, range=(xlo, xhi))
                bw = np.diff(edges)[0]
                density = counts / max(counts.sum() * bw, 1e-300)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centers, density, where="mid", color=clr, lw=2,
                        label=f"{sl_label}  (N={int(sl_mask.sum())})")
            ax.set_title(panel_label, fontsize=11)
            ax.set_xlabel(bname, fontsize=10)
            ax.set_ylabel("Probability density", fontsize=10)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8, frameon=False)

        for fmt in ("png", "pdf"):
            fig.savefig(f"{out_dir}/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    print(f"  [ecm{ecm}]  NLL slice plots → {out_dir}/")


# ── Per-status comparison plots ──────────────────────────────────────────────
# Migrad status codes (see WWKinReco.h: KinFitResult::status):
#   0 = OK, 1 = PD-forced cov, 2 = Hesse failed, 3 = EDM>tol,
#   4 = max calls, 5 = other, −1 = early-return (invalid input).
# Two buckets shown: converged (0|1) vs EDM>tol (3). Codes 4/5 sit below 0.05 %
# at all ECMs and are dropped. Pull branches and other branches go to separate
# subdirectories (by_status/pulls/ and by_status/other/).

_STATUS_BUCKETS = [
    (lambda s: (s == 0) | (s == 1), "status=0|1 (converged)", "tab:blue"),
    (lambda s: s == 3,              "status=3 (EDM>tol)",     "tab:red"),
]


def plot_by_status(ecm, raw_arrays):
    status = raw_arrays.get("kinfit_status")
    if status is None:
        print(f"  [ecm{ecm}]  kinfit_status missing — skipping by-status plots")
        return

    base = f"{KINFIT_OUTDIR}/ecm{ecm}/by_status"
    os.makedirs(f"{base}/pulls", exist_ok=True)
    os.makedirs(f"{base}/other", exist_ok=True)

    status = np.asarray(status, dtype=int)

    for bname in KINFIT_BRANCHES:
        if bname in ("kinfit_status", "kinfit_valid"):
            continue
        if bname not in raw_arrays:
            continue
        vals = np.asarray(raw_arrays[bname], dtype=float)
        finite = np.isfinite(vals)
        # Only count status≥0 events (status=−1 are early-returns with placeholder values).
        active = finite & (status >= 0)
        if active.sum() < 30:
            continue

        nbins, xlo, xhi = _binning(bname, vals[active])

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for pred, lbl, clr in _STATUS_BUCKETS:
            m = pred(status) & active
            n = int(m.sum())
            if n < 1:
                continue
            v = vals[m]
            v = v[(v >= xlo) & (v <= xhi)]
            counts, edges = np.histogram(v, bins=nbins, range=(xlo, xhi))
            bw = np.diff(edges)[0]
            density = counts / max(counts.sum() * bw, 1e-300)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centers, density, where="mid", color=clr, lw=2,
                    label=f"{lbl}  (N={n})")

        ax.set_xlabel(bname, fontsize=11)
        ax.set_ylabel("Probability density", fontsize=11)
        ax.set_title(f"{bname}  [ecm{ecm}]  — split by Migrad status", fontsize=10)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, frameon=False)

        sub = "pulls" if bname in _PULL_BRANCHES else "other"
        for fmt in ("png", "pdf"):
            fig.savefig(f"{base}/{sub}/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    print(f"  [ecm{ecm}]  by-status plots → {base}/{{pulls,other}}/")


# ── ECM comparison plots ──────────────────────────────────────────────────────

def plot_kinfit_ecm_comparison(all_data, all_json):
    branch_names = [b for b in KINFIT_BRANCHES
                    if any(b in all_data[e] for e in ECM_LIST if e in all_data)]

    for bname in branch_names:
        out_dir = f"{KINFIT_OUTDIR}/ecm_comparison/{_subdir(bname)}"
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

        resol_name = KINFIT_TO_RESOL.get(bname)

        pdf_params_ref = None
        if resol_name:
            for _e in ECM_LIST:
                if _e in all_json and resol_name in all_json[_e]:
                    pdf_params_ref = all_json[_e][resol_name]
                    break

        all_vals_combined = np.concatenate([
            all_data[ecm][bname] for ecm in ECM_LIST
            if ecm in all_data and bname in all_data[ecm]
        ]) if any(ecm in all_data and bname in all_data[ecm] for ecm in ECM_LIST) else np.array([])
        nbins, xlo, xhi = _binning(bname,
                                    all_vals_combined if len(all_vals_combined) else None,
                                    pdf_params=pdf_params_ref)

        for ecm in ECM_LIST:
            if ecm not in all_data or bname not in all_data[ecm]:
                continue
            vals = all_data[ecm][bname]
            vals_c = vals[(vals >= xlo) & (vals <= xhi)]
            counts, edges = np.histogram(vals_c, bins=nbins, range=(xlo, xhi))
            centers = 0.5 * (edges[:-1] + edges[1:])
            bw = np.diff(edges)[0]
            density = counts / max(counts.sum() * bw, 1e-300)
            ax.step(centers, density, where="mid",
                    color=ECM_COLORS[ecm], lw=2, label=rf"$\sqrt{{s}}$ = {ecm} GeV")

            if resol_name and ecm in all_json and resol_name in all_json[ecm]:
                p = all_json[ecm][resol_name]
                pdf_fn = _make_pdf(p)
                if pdf_fn is not None:
                    xfine = np.linspace(xlo, xhi, 600)
                    yfine = pdf_fn(xfine)
                    integ_plot = float(np.trapz(yfine, xfine))
                    yfine = yfine / max(integ_plot, 1e-300)
                    ax.plot(xfine, yfine,
                            color=ECM_COLORS[ecm], lw=1.5, ls="--",
                            label=f"PDF ecm{ecm}")

        ax.set_xlabel(bname, fontsize=11)
        ax.set_ylabel("Probability density", fontsize=11)
        title = bname
        if resol_name:
            title += f"  (input PDF: {resol_name})"
        ax.set_title(title, fontsize=11)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, frameon=False)

        for fmt in ("png", "pdf"):
            fig.savefig(f"{out_dir}/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    print(f"  ECM comparison plots → {KINFIT_OUTDIR}/ecm_comparison/")


def _process_ecm(ecm):
    """Worker: load, slice, write per-ECM and NLL-slice plots."""
    log = []
    infile = INFILE_TMPL.format(ecm=ecm)
    if not os.path.exists(infile):
        log.append(f"WARNING: {infile} not found — skipping ecm{ecm}")
        return ecm, None, None, log

    log.append(f"\n{'='*55}\nECM {ecm} GeV  —  {infile}\n{'='*55}")

    with uproot.open(infile) as f:
        tree = f[TREE_NAME]
        available = set(tree.keys())
        to_load    = [b for b in KINFIT_BRANCHES if b in available]
        missing    = [b for b in KINFIT_BRANCHES if b not in available]
        extra_load = [b for b in EXTRA_BRANCHES  if b in available and b not in set(to_load)]
        extra_miss = [b for b in EXTRA_BRANCHES  if b not in available]
        if missing:
            log.append(f"  WARNING: kinfit branches not in tree: {missing}")
        if extra_miss:
            log.append(f"  WARNING: comparison branches not in tree (re-run treemaker?): {extra_miss}")
        raw = tree.arrays(to_load + extra_load, library="np")

    raw_data = {}
    branches_data = {}
    for bname in to_load + extra_load:
        arr = np.asarray(raw[bname], dtype=float).ravel()
        raw_data[bname] = arr
        branches_data[bname] = arr[np.isfinite(arr)]

    json_path = JSON_TMPL.format(ecm=ecm)
    if os.path.exists(json_path):
        with open(json_path) as fj:
            json_dict = json.load(fj)
        log.append(f"  Loaded JSON: {json_path}")
    else:
        log.append(f"  WARNING: JSON not found ({json_path}) — no PDF overlay for ecm{ecm}")
        json_dict = {}

    plot_per_ecm(ecm, branches_data, json_dict)
    plot_chi2_slices(ecm, raw_data, json_dict)
    plot_by_status(ecm, raw_data)

    return ecm, branches_data, json_dict, log


def run_kinfit_vars():
    os.makedirs(KINFIT_OUTDIR, exist_ok=True)

    all_data = {}
    all_json = {}

    with ProcessPoolExecutor(max_workers=len(ECM_LIST)) as pool:
        for ecm, branches_data, json_dict, log in pool.map(_process_ecm, ECM_LIST):
            for line in log:
                print(line)
            if branches_data is None:
                continue
            all_data[ecm] = branches_data
            all_json[ecm] = json_dict

    plot_kinfit_ecm_comparison(all_data, all_json)
    print(f"\nDone. All plots in {KINFIT_OUTDIR}/")
    publish(KINFIT_OUTDIR, os.environ.get("KINFIT_PUBSUB", "kinfit_vars"))


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print(" Stage 1 / 2 — mW overlay")
    print("=" * 60)
    run_mW_overlay()

    print()
    print("=" * 60)
    print(" Stage 2 / 2 — kinfit variables")
    print("=" * 60)
    run_kinfit_vars()


if __name__ == "__main__":
    main()
