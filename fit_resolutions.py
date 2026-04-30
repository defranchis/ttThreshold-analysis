#!/usr/bin/env python3
"""
Fit DCB (or DCB + Gaussian mixture) to resolution/diff branches.

v2 improvements over v1:
  - MAD-based robust sigma init (not dragged by long tails)
  - Mode-based mu init (histogram peak, not median)
  - Per-branch clip/bin overrides for asymmetric distributions
  - Two-component DCB+Gaussian model for jet resolution variables
    that show a narrow core + secondary shoulder

Outputs (per ECM)
  outputs/response/plots/ecm<N>/<branch>.{png,pdf}    one plot per branch with data + fit
  outputs/response/functions/dcb_params_ecm<N>.h      C++ header with evaluators + constexpr params
  outputs/response/functions/dcb_results_ecm<N>.json  full numerical fit results
"""

import os, json, math, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
from scipy.integrate import quad as _quad
from scipy.special import erf as _sp_erf
from eos_publish import publish

_SQRT2   = math.sqrt(2.0)
_LOG_MAX = math.log(np.finfo(np.float64).max)   # max safe float64 exponent ≈ 709.78
_LOG_MIN = -_LOG_MAX

os.makedirs("outputs/response/functions", exist_ok=True)

ECM_LIST    = [157, 160, 163]
INFILE_TMPL = "outputs/treemaker/lnuqq/step1/semihad/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
NBINS_DEF   = 100
CLIP_DEF  = (0.5, 99.5)

# ── Per-branch configuration overrides ─────────────────────────────────────
BRANCH_CONFIG = {
    # Jet p response: detector core + wide-radiation tail → DCB+G.
    "jet1_p_resp":          {"clip": (0.2, 99.8),  "nbins": 150, "model": "dcb2g"},
    "jet2_p_resp":          {"clip": (0.2, 99.8),  "nbins": 150, "model": "dcb2g"},
    "jet_p_resp":           {"clip": (0.2, 99.8),  "nbins": 150, "model": "dcb2g"},
    # Jet angular resolutions: detector core + wide-radiation tails.
    "jet1_theta_resol":     {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g"},
    "jet2_theta_resol":     {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g"},
    "jet_theta_resol":      {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g"},
    "jet1_phi_resol":       {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g"},
    "jet2_phi_resol":       {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g"},
    "jet_phi_resol":        {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g"},
    # Lepton p response: narrow detector core + exponential FSR left tail +
    # sharp right cliff just past 1. expleft2g (exp-left + Gaussian core +
    # power-right + wide Gauss). Right-side clip extended to 99.95%.
    "lep_p_resp":           {"clip": (0.2, 99.95), "nbins": 150, "model": "expleft2g"},
    # Lepton angular resolutions: tight detector core + wide-angle FSR tails → DCB+G.
    "lep_theta_resol":      {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g"},
    "lep_phi_resol":        {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g"},
    # MET (well-measured) → single DCB.
    "met_p_resp":           {"clip": (0.5, 99.5),  "nbins": 150},
    "met_theta_resol":      {"clip": (0.5, 99.5),  "nbins": 150},
    "met_phi_resol":        {"clip": (0.5, 99.5),  "nbins": 150},
    # Gen-level total momenta: ~85% of events have collinear (or no) ISR → spike at 0
    # narrower than any reasonable bin; the rest form a smooth ISR tail.
    # dcb2g (narrow DCB core + wide Gaussian) handles the unresolved spike + smooth
    # tail; dcbgb's box plateau does not match the data and gives chi²/ndf in the
    # hundreds for px/py.
    "gen_WW_px":           {"clip": (0.1, 99.9),  "nbins": 1500, "model": "dcb2g",
                             "zoom_xlim": (-0.5, 0.5)},
    "gen_WW_py":           {"clip": (0.1, 99.9),  "nbins": 1500, "model": "dcb2g",
                             "zoom_xlim": (-0.5, 0.5)},
    "gen_WW_pz":           {"clip": (0.1, 99.9),  "nbins": 1500, "model": "dcb2g",
                             "zoom_xlim": (-3.0, 3.0)},
    # Gen WW invariant mass minus ECM. Peak just below 0 (ISR), hard boundary at 0.
    "gen_WW_m_minus_ecm": {"clip": (0.5, 100.0), "nbins": 150, "model": "dcber2g"},
}

# Virtual combined branches: concatenate jet1 + jet2 into a single distribution
# for "pooled" jet fits and the jet1-vs-jet2 comparison plot.
COMBINED_BRANCHES = {
    "jet_p_resp":      ("jet1_p_resp",     "jet2_p_resp"),
    "jet_theta_resol": ("jet1_theta_resol", "jet2_theta_resol"),
    "jet_phi_resol":   ("jet1_phi_resol",   "jet2_phi_resol"),
}

# Branches used by the kinematic fit (also the full set of step1 outputs we fit).
KINFIT_BRANCHES = [
    "jet1_p_resp", "jet2_p_resp", "lep_p_resp", "met_p_resp",
    "jet1_theta_resol", "jet2_theta_resol", "jet1_phi_resol", "jet2_phi_resol",
    "lep_theta_resol",  "lep_phi_resol",
    "met_theta_resol",  "met_phi_resol",
    "gen_WW_px", "gen_WW_py", "gen_WW_pz",
    "gen_WW_m_minus_ecm",
]

# ── Model functions ──────────────────────────────────────────────────────────

def _dcb_core(t, aL, nL, aR, nR):
    """Unnormalized DCB shape in reduced variable t = (x-mu)/sigma.
    Log-space computation avoids overflow when nL/nR are large; np.where
    evaluates all branches eagerly so inactive branches must not overflow.
    """
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
    """Narrow DCB core + broad Gaussian component with independent centres."""
    core = _dcb_core((x - mu_c) / sigma_c, aL, nL, aR, nR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)


def _dcb_expleft_core(t, aL, kL, aR, nR):
    """DCB variant: exponential left tail (rate kL) + Gaussian core + power-law right tail.
    Left tail: exp(-0.5*aL^2 + kL*(aL+t)) for t < -aL.  kL > 0 gives faster decay than Gaussian.
    np.minimum caps the exponent at 0 in inactive branches to prevent overflow.
    """
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
    """Exp-left DCB core + broad Gaussian component."""
    core = _dcb_expleft_core((x - mu_c) / sigma_c, aL, kL, aR, nR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)


def _dcb_expright_core(t, aL, nL, aR, kR):
    """Power-law left tail + Gaussian core + exponential right tail.
    Left:  standard DCB power-law for t < -aL.
    Right: exp(-0.5*aR^2 - kR*(t-aR)) for t > aR (kR > 0 = fast right decay).
    Physically: heavy ISR tail on left, sharp kinematic cutoff on right.
    """
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
    """Power-law-left DCB core + broad Gaussian. kR: exponential right-tail decay rate."""
    core = _dcb_expright_core((x - mu_c) / sigma_c, aL, nL, aR, kR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)


def dcb_gaussbox(x, N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, p_max, sigma_box):
    """Narrow DCB core + Gaussian-smeared box wide component.
    Wide: Box(−p_max, p_max) convolved with Gaussian(sigma_box) — flat plateau, fast erf edges.
    Params: N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, p_max, sigma_box  (10 params, same as dcb_gauss).
    """
    core = _dcb_core((x - mu_c) / sigma_c, aL, nL, aR, nR)
    p_max = abs(p_max)
    sb = max(abs(sigma_box), 1e-10)
    sq2 = _SQRT2 * sb
    wide = 0.5 * (_sp_erf((x + p_max) / sq2) - _sp_erf((x - p_max) / sq2))
    wide_peak = max(float(_sp_erf(p_max / sq2)), 1e-10)
    return N * ((1.0 - f_wide) * core + f_wide * wide / wide_peak)


# ── Generic multi-start fitter ───────────────────────────────────────────────

def fit_dcb2g_iminuit(centers, counts, mu0, sig0):
    """
    Poisson binned NLL with iminuit for distributions where chi² gets trapped.
    Falls back gracefully if iminuit is not available or converges poorly.
    Returns (popt_list, chi2) in the same convention as _best_fit.
    """
    from iminuit import Minuit

    N0   = float(counts.max())
    errs = np.maximum(np.sqrt(counts), 1.0)

    # Param order: N, mu_c, sc, aL, nL, aR, nR, fw, muw, sw
    def nll(N, mu_c, sc, aL, nL, aR, nR, fw, muw, sw):
        if sc <= 0 or sw <= 0:
            return 1e15
        pred = dcb_gauss(centers, abs(N), mu_c, abs(sc), abs(aL), abs(nL),
                         abs(aR), abs(nR), abs(fw), muw, abs(sw))
        pred = np.maximum(pred, 1e-300)
        return 2.0 * float(np.sum(pred - counts * np.log(pred)))

    # Starting conditions tuned for narrow-core, left-bounded distributions:
    # small sigma_c, small aL (early left power-law), moderate aR for right tail
    starts = [
        [N0, mu0, sig0 * 0.5,  0.5,  3., 0.8, 3., 0.15, mu0 + 3*sig0,  8*sig0],
        [N0, mu0, sig0 * 0.4,  0.4,  2., 0.7, 3., 0.20, mu0 + 4*sig0, 10*sig0],
        [N0, mu0, sig0 * 0.5,  0.5,  4., 0.5, 2., 0.25, mu0,           8*sig0],
        [N0, mu0, sig0,        1.2,  5., 0.8, 3., 0.15, mu0 + 3*sig0,  8*sig0],
        # broad LEFT component (for left-heavy distributions like fromele responses)
        [N0, mu0, sig0 * 0.5,  0.5,  3., 1.5, 3., 0.30, mu0 - 3*sig0,  6*sig0],
        [N0, mu0, sig0 * 0.4,  0.4,  2., 1.2, 3., 0.40, mu0 - 4*sig0,  8*sig0],
        [N0, mu0, sig0 * 0.5,  0.5,  5., 1.5, 3., 0.50, mu0 - 6*sig0, 12*sig0],
        # very narrow spike + broad symmetric wings (ISR total-momentum distributions)
        [N0, mu0, sig0 * 0.04, 1.5,  5., 1.5,  5., 0.60, mu0, sig0 * 1.0],
        [N0, mu0, sig0 * 0.03, 1.0,  3., 1.0,  3., 0.70, mu0, sig0 * 0.8],
        [N0, mu0, sig0 * 0.05, 1.0,  5., 1.0,  5., 0.80, mu0, sig0 * 1.2],
        # ultra-narrow spike: wings extend further than MAD-based sig0 suggests
        [N0, mu0, sig0 * 0.02, 2.0, 10., 2.0, 10., 0.80, mu0, sig0 * 2.0],
        [N0, mu0, sig0 * 0.01, 1.5,  8., 1.5,  8., 0.88, mu0, sig0 * 1.5],
        [N0, mu0, sig0 * 0.03, 1.5,  7., 1.5,  7., 0.75, mu0, sig0 * 2.5],
    ]
    limits = [(1e-3, None), (None, None), (1e-6, None), (0.3, 8.),
              (1.01, 200.), (0.3, 8.), (1.01, 200.), (0.01, 0.95),
              (None, None), (1e-4, None)]
    names  = ['N','mu_c','sc','aL','nL','aR','nR','fw','muw','sw']

    best_popt, best_chi2 = None, np.inf
    for p0 in starts:
        try:
            m = Minuit(nll, *p0, name=names)
            for i, (lo_i, hi_i) in enumerate(limits):
                m.limits[i] = (lo_i, hi_i)
            m.migrad()
            if not m.valid:
                m.migrad()   # second pass
            if m.valid:
                popt = list(m.values)
                pred = dcb_gauss(centers, *[abs(v) if j not in (1, 8) else v
                                            for j, v in enumerate(popt)])
                chi2 = float(np.sum(((counts - pred) / errs) ** 2))
                if chi2 < best_chi2:
                    best_chi2, best_popt = chi2, popt
        except Exception:
            pass

    if best_popt is None:
        return None, None, False, np.inf
    # pcov not available from iminuit without HESSE — pass None; ok=True if converged
    return best_popt, None, True, best_chi2


def fit_dcb_gaussbox_iminuit(centers, counts, mu0, sig0):
    """Poisson NLL with iminuit for DCB + Gaussian-smeared box.
    Params: N, mu_c, sc, aL, nL, aR, nR, fw, p_max, sigma_box
    """
    from iminuit import Minuit
    N0 = float(counts.max())
    errs = np.maximum(np.sqrt(counts), 1.0)
    x_span = 0.5 * (centers[-1] - centers[0])

    def nll(N, mu_c, sc, aL, nL, aR, nR, fw, p_max, sw):
        if sc <= 0 or sw <= 0 or p_max <= 0:
            return 1e15
        pred = dcb_gaussbox(centers, abs(N), mu_c, abs(sc), abs(aL), abs(nL),
                             abs(aR), abs(nR), abs(fw), abs(p_max), abs(sw))
        pred = np.maximum(pred, 1e-300)
        return 2.0 * float(np.sum(pred - counts * np.log(pred)))

    starts = [
        [N0, mu0, sig0*0.005, 1.0, 50., 1.0, 50., 0.20, x_span*0.85, sig0*0.15],
        [N0, mu0, sig0*0.005, 1.5, 30., 1.5, 30., 0.18, x_span*0.80, sig0*0.15],
        [N0, mu0, sig0*0.010, 1.0,100., 1.0,100., 0.22, x_span*0.80, sig0*0.12],
        [N0, mu0, sig0*0.003, 1.2, 80., 1.2, 80., 0.16, x_span*0.75, sig0*0.10],
        [N0, mu0, sig0*0.007, 2.0, 20., 2.0, 20., 0.20, x_span*0.90, sig0*0.20],
        [N0, mu0, sig0*0.010, 1.5, 50., 1.5, 50., 0.25, x_span*0.85, sig0*0.18],
        [N0, mu0, sig0*0.003, 1.0,200., 1.0,200., 0.15, x_span*0.90, sig0*0.08],
        [N0, mu0, sig0*0.007, 1.0, 30., 1.0, 30., 0.30, x_span*0.80, sig0*0.20],
        [N0, mu0, sig0*0.015, 2.0, 15., 2.0, 15., 0.35, x_span*0.85, sig0*0.25],
        [N0, mu0, sig0*0.020, 1.5, 10., 1.5, 10., 0.40, x_span*0.80, sig0*0.30],
    ]
    limits = [(1e-3,None),(None,None),(1e-6,None),(0.3,8.),(1.01,200.),
              (0.3,8.),(1.01,200.),(0.01,0.40),(1.,None),(1e-4,None)]
    names = ['N','mu_c','sc','aL','nL','aR','nR','fw','p_max','sw']

    best_popt, best_chi2 = None, np.inf
    for p0 in starts:
        try:
            m = Minuit(nll, *p0, name=names)
            for i, (lo_i, hi_i) in enumerate(limits):
                m.limits[i] = (lo_i, hi_i)
            m.migrad()
            if not m.valid:
                m.migrad()
            if m.valid:
                popt = list(m.values)
                pred = dcb_gaussbox(centers, *[abs(v) if j != 1 else v
                                               for j, v in enumerate(popt)])
                chi2 = float(np.sum(((counts - pred) / errs) ** 2))
                if chi2 < best_chi2:
                    best_chi2, best_popt = chi2, popt
        except Exception:
            pass

    if best_popt is None:
        return None, None, False, np.inf
    return best_popt, None, True, best_chi2


def fit_dcb_expleft2g_iminuit(centers, counts, mu0, sig0, constrain_mu0=False):
    """Poisson binned NLL with iminuit for the expleft2g model.
    Params: N, mu_c, sc, aL, kL, aR, nR, fw, muw, sw
    constrain_mu0: restrict mu_c near 0 and aR small to force hard-right-cutoff solution.
    """
    from iminuit import Minuit

    N0   = float(counts.max())
    errs = np.maximum(np.sqrt(counts), 1.0)

    def nll(N, mu_c, sc, aL, kL, aR, nR, fw, muw, sw):
        if sc <= 0 or sw <= 0:
            return 1e15
        pred = dcb_expleft_gauss(centers, abs(N), mu_c, abs(sc), abs(aL), abs(kL),
                                 abs(aR), abs(nR), abs(fw), muw, abs(sw))
        pred = np.maximum(pred, 1e-300)
        return 2.0 * float(np.sum(pred - counts * np.log(pred)))

    starts = [
        [N0, mu0, sig0 * 0.5,  1.0,  8., 0.8,   3., 0.15, mu0 + 3*sig0,  8*sig0],
        [N0, mu0, sig0 * 0.5,  0.8, 10., 0.7,   3., 0.20, mu0 + 4*sig0, 10*sig0],
        # left-biased wide component
        [N0, mu0, sig0 * 0.05, 0.5, 10., 0.5,   3., 0.50, mu0 - 3*sig0,  2*sig0],
        [N0, mu0, sig0 * 0.03, 0.5, 15., 0.5,   3., 0.60, mu0 - 5*sig0,  4*sig0],
        [N0, mu0, sig0 * 0.10, 1.0,  5., 1.0,   5., 0.35, mu0 - 3*sig0,  2*sig0],
        # hard right cutoff: small aR, large nR
        [N0, mu0, sig0 * 0.3,  0.5,  5., 0.1, 100., 0.40, mu0 - 3*sig0,  2*sig0],
        [N0, mu0, sig0 * 0.2,  0.5,  8., 0.1, 200., 0.50, mu0 - 4*sig0,  3*sig0],
        [N0, mu0, sig0 * 0.1,  0.3, 10., 0.1, 300., 0.60, mu0 - 5*sig0,  4*sig0],
        [N0, mu0, sig0 * 0.5,  1.0,  5., 0.1, 100., 0.30, mu0 - 2*sig0,  2*sig0],
    ]
    if constrain_mu0:
        # Discard unconstrained starts; use large-sigma_c starts matching physical ISR slope.
        starts = [
            [N0, 0.0,  2.0, 0.30, 0.20, 0.10, 200., 0.5, mu0 - 2*sig0, 3*sig0],
            [N0, 0.0,  3.0, 0.30, 0.30, 0.10, 300., 0.4, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  1.5, 0.20, 0.15, 0.05, 200., 0.6, mu0 - 2*sig0, 3*sig0],
            [N0, 0.0,  4.0, 0.50, 0.40, 0.15, 200., 0.3, mu0 - 3*sig0, 5*sig0],
            [N0, -0.1, 2.5, 0.30, 0.25, 0.10, 300., 0.5, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  2.0, 0.50, 0.20, 0.08, 300., 0.6, mu0 - 3*sig0, 5*sig0],
            [N0, 0.0,  1.0, 0.20, 0.10, 0.05, 400., 0.7, mu0 - 3*sig0, 4*sig0],
            [N0, -0.2, 3.0, 0.40, 0.30, 0.10, 200., 0.4, mu0 - 2*sig0, 4*sig0],
            [N0, 0.0,  5.0, 0.50, 0.50, 0.20, 300., 0.3, mu0 - 2*sig0, 5*sig0],
            [N0, 0.0,  2.0, 0.10, 0.20, 0.10, 200., 0.5, mu0 - 3*sig0, 3*sig0],
            [N0, -0.3, 2.0, 0.30, 0.20, 0.10, 300., 0.5, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  1.5, 0.30, 0.15, 0.05, 250., 0.55, mu0 - 2*sig0, 3*sig0],
        ]
        limits = [(1e-3, None), (-0.5, 0.1), (0.5, 10.), (0.05, 3.),
                  (0.01, 5.), (0.02, 0.4), (1.01, 500.), (0.01, 0.95),
                  (None, None), (1e-4, None)]
    else:
        limits = [(1e-3, None), (None, None), (1e-6, None), (0.1, 6.),
                  (0.1, 50.), (0.1, 8.), (1.01, 500.), (0.01, 0.95),
                  (None, None), (1e-4, None)]
    names = ['N', 'mu_c', 'sc', 'aL', 'kL', 'aR', 'nR', 'fw', 'muw', 'sw']

    best_popt, best_chi2 = None, np.inf
    for p0 in starts:
        try:
            m = Minuit(nll, *p0, name=names)
            for i, (lo_i, hi_i) in enumerate(limits):
                m.limits[i] = (lo_i, hi_i)
            m.migrad()
            if not m.valid:
                m.migrad()
            if m.valid:
                popt = list(m.values)
                pred = dcb_expleft_gauss(centers, *[abs(v) if j not in (1, 8) else v
                                                    for j, v in enumerate(popt)])
                chi2 = float(np.sum(((counts - pred) / errs) ** 2))
                if chi2 < best_chi2:
                    best_chi2, best_popt = chi2, popt
        except Exception:
            pass

    if best_popt is None:
        return None, None, False, np.inf
    return best_popt, None, True, best_chi2


_GOOD_ENOUGH_CHI2_NDOF = 2.0   # stop trying more starts once we hit this

def _best_fit(fn, centers, counts, starts, lo, hi):
    errs = np.maximum(np.sqrt(counts), 1.0)
    nparams = len(starts[0])
    ndof = max(len(centers) - nparams, 1)
    best_popt, best_pcov, best_chi2 = None, None, np.inf
    for p0 in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(fn, centers, counts, p0=p0,
                                       bounds=(lo, hi), sigma=errs,
                                       maxfev=8_000, ftol=1e-7, xtol=1e-7)
            chi2 = float(np.sum(((counts - fn(centers, *popt)) / errs) ** 2))
            if chi2 < best_chi2:
                best_popt, best_pcov, best_chi2 = popt, pcov, chi2
            if best_chi2 / ndof < _GOOD_ENOUGH_CHI2_NDOF:
                break   # good enough — skip remaining starts
        except Exception:
            pass
    if best_popt is None:
        return None, None, False, np.inf
    ok = np.all(np.isfinite(np.sqrt(np.diag(best_pcov))))
    return best_popt, best_pcov, ok, best_chi2


def fit_dcb(centers, counts, mu0, sig0):
    N0 = float(counts.max())
    lo = [0, -np.inf, 1e-6, 0.3, 1.01, 0.3, 1.01]
    hi = [np.inf, np.inf, np.inf, 8., 200., 8., 200.]
    starts = [
        [N0, mu0, sig0,        1.5,  5.,  1.5,  5.],
        [N0, mu0, sig0,        1.0,  3.,  1.0,  3.],
        [N0, mu0, sig0,        2.0, 10.,  2.0, 10.],
        # asymmetric: gradual left, steep right
        [N0, mu0, sig0,        0.5,  2.,  2.0,  8.],
        # asymmetric: steep left, gradual right
        [N0, mu0, sig0,        2.0,  8.,  0.5,  2.],
        # narrow core
        [N0, mu0, sig0 * 0.5,  0.5,  3.,  0.5,  3.],
    ]
    return _best_fit(dcb, centers, counts, starts, lo, hi)


def fit_dcb2g(centers, counts, mu0, sig0):
    N0 = float(counts.max())
    # params: N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, mu_w, sigma_w
    lo = [0, -np.inf, 1e-6, 0.3, 1.01, 0.3, 1.01, 0.01, -np.inf, 1e-4]
    hi = [np.inf, np.inf, np.inf, 8., 200., 8., 200., 0.95, np.inf, np.inf]
    starts = [
        [N0, mu0, sig0,        1.2,  5., 0.8, 3., 0.15, mu0,           5 * sig0],
        [N0, mu0, sig0,        1.5,  8., 0.5, 2., 0.25, mu0 + 2*sig0,  8 * sig0],
        [N0, mu0, sig0,        2.0, 10., 0.5, 2., 0.10, mu0 + 3*sig0,  6 * sig0],
        # sharp-core: early power-law transitions
        [N0, mu0, sig0 * 0.5,  0.5,  3., 0.5, 2., 0.15, mu0,           5 * sig0],
        [N0, mu0, sig0 * 0.5,  0.4,  2., 0.4, 2., 0.20, mu0 + 2*sig0,  8 * sig0],
        # broad LEFT component (for left-heavy distributions like fromele responses)
        [N0, mu0, sig0 * 0.5,  0.5,  3., 1.5, 3., 0.30, mu0 - 3*sig0,  6 * sig0],
        [N0, mu0, sig0 * 0.4,  0.4,  2., 1.2, 3., 0.40, mu0 - 4*sig0,  8 * sig0],
        [N0, mu0, sig0 * 0.5,  0.3,  2., 1.0, 2., 0.35, mu0 - 5*sig0, 10 * sig0],
        [N0, mu0, sig0 * 0.5,  0.5,  5., 1.5, 3., 0.50, mu0 - 6*sig0, 12 * sig0],
        # very narrow spike + broad symmetric wings (ISR total-momentum distributions)
        [N0, mu0, sig0 * 0.04, 1.5,  5., 1.5,  5., 0.60, mu0, sig0 * 1.0],
        [N0, mu0, sig0 * 0.03, 1.0,  3., 1.0,  3., 0.70, mu0, sig0 * 0.8],
        [N0, mu0, sig0 * 0.05, 1.0,  5., 1.0,  5., 0.80, mu0, sig0 * 1.2],
        [N0, mu0, sig0 * 0.04, 1.5,  5., 1.5,  5., 0.55, mu0, sig0 * 0.9],
        # ultra-narrow spike: wings extend further than MAD-based sig0 suggests
        [N0, mu0, sig0 * 0.02, 2.0, 10., 2.0, 10., 0.80, mu0, sig0 * 2.0],
        [N0, mu0, sig0 * 0.01, 1.5,  8., 1.5,  8., 0.88, mu0, sig0 * 1.5],
        [N0, mu0, sig0 * 0.03, 1.5,  7., 1.5,  7., 0.75, mu0, sig0 * 2.5],
    ]
    return _best_fit(dcb_gauss, centers, counts, starts, lo, hi)


def fit_dcb_gaussbox(centers, counts, mu0, sig0):
    """DCB narrow core + Gaussian-smeared box wide component.
    Params: N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, p_max, sigma_box
    """
    N0 = float(counts.max())
    x_span = 0.5 * (centers[-1] - centers[0])
    # f_wide upper bound 0.40: plateau/peak ≈ 0.15–0.25 physically; high f_wide pulls N down
    # to match plateau, causing the optimizer to undershoot the spike peak.
    lo = [0, -np.inf, 1e-6, 0.3, 1.01, 0.3, 1.01, 0.01,  1., 1e-4]
    hi = [np.inf, np.inf, np.inf, 8., 200., 8., 200., 0.40, np.inf, np.inf]
    starts = [
        [N0, mu0, sig0*0.005, 1.0, 50., 1.0, 50., 0.20, x_span*0.85, sig0*0.15],
        [N0, mu0, sig0*0.005, 1.5, 30., 1.5, 30., 0.18, x_span*0.80, sig0*0.15],
        [N0, mu0, sig0*0.010, 1.0,100., 1.0,100., 0.22, x_span*0.80, sig0*0.12],
        [N0, mu0, sig0*0.003, 1.2, 80., 1.2, 80., 0.16, x_span*0.75, sig0*0.10],
        [N0, mu0, sig0*0.007, 2.0, 20., 2.0, 20., 0.20, x_span*0.90, sig0*0.20],
        [N0, mu0, sig0*0.010, 1.5, 50., 1.5, 50., 0.25, x_span*0.85, sig0*0.18],
        [N0, mu0, sig0*0.003, 1.0,200., 1.0,200., 0.15, x_span*0.90, sig0*0.08],
        [N0, mu0, sig0*0.007, 1.0, 30., 1.0, 30., 0.30, x_span*0.80, sig0*0.20],
        [N0, mu0, sig0*0.015, 2.0, 15., 2.0, 15., 0.35, x_span*0.85, sig0*0.25],
        [N0, mu0, sig0*0.020, 1.5, 10., 1.5, 10., 0.40, x_span*0.80, sig0*0.30],
    ]
    return _best_fit(dcb_gaussbox, centers, counts, starts, lo, hi)


def fit_dcb_expleft2g(centers, counts, mu0, sig0, constrain_mu0=False):
    """DCB with exponential left tail + Gaussian wide component.
    Params: N, mu_c, sigma_c, aL, kL, aR, nR, f_wide, mu_w, sigma_w
    kL > 0: exponential decay rate of left tail (kL > aL → faster than Gaussian).
    constrain_mu0: restrict mu_c near 0 and aR small for hard-right-cutoff distributions.
    """
    N0 = float(counts.max())
    if constrain_mu0:
        # Key insight: sigma_c must be LARGE (1-5 GeV) because the ISR exponential slope
        # in x-space is kL/sigma_c ≈ 0.10/GeV (1/e length ~10 GeV seen in data).
        # With sigma_c=2 GeV: kL ≈ 0.2. With sigma_c=3 GeV: kL ≈ 0.3.
        # mu_c near 0 (physical boundary) + hard right cutoff (small aR, large nR).
        lo = [0, -0.5, 0.5, 0.05, 0.01, 0.02, 1.01, 0.01, -np.inf, 1e-4]
        hi = [np.inf, 0.1, 10., 3., 5., 0.4, 500., 0.95, np.inf, np.inf]
        starts = [
            # sigma_c=1-5 GeV: kL=0.1-0.5 gives physical ISR slope ~0.10/GeV in x-space
            [N0, 0.0,  2.0, 0.30, 0.20, 0.10, 200., 0.5, mu0 - 2*sig0, 3*sig0],
            [N0, 0.0,  3.0, 0.30, 0.30, 0.10, 300., 0.4, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  1.5, 0.20, 0.15, 0.05, 200., 0.6, mu0 - 2*sig0, 3*sig0],
            [N0, 0.0,  4.0, 0.50, 0.40, 0.15, 200., 0.3, mu0 - 3*sig0, 5*sig0],
            [N0, -0.1, 2.5, 0.30, 0.25, 0.10, 300., 0.5, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  2.0, 0.50, 0.20, 0.08, 300., 0.6, mu0 - 3*sig0, 5*sig0],
            [N0, 0.0,  1.0, 0.20, 0.10, 0.05, 400., 0.7, mu0 - 3*sig0, 4*sig0],
            [N0, -0.2, 3.0, 0.40, 0.30, 0.10, 200., 0.4, mu0 - 2*sig0, 4*sig0],
            [N0, 0.0,  5.0, 0.50, 0.50, 0.20, 300., 0.3, mu0 - 2*sig0, 5*sig0],
            [N0, 0.0,  2.0, 0.10, 0.20, 0.10, 200., 0.5, mu0 - 3*sig0, 3*sig0],
            [N0, -0.3, 2.0, 0.30, 0.20, 0.10, 300., 0.5, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  1.5, 0.30, 0.15, 0.05, 250., 0.55, mu0 - 2*sig0, 3*sig0],
        ]
    else:
        # aR lower bound 0.1 (was 0.3): allows near-hard right cutoff when nR is large.
        # nR upper bound 500 (was 200): (nR/aR - aR + t)^{-nR} → 0 within one bin width.
        lo = [0, -np.inf, 1e-6, 0.1, 0.1,  0.1, 1.01, 0.01, -np.inf, 1e-4]
        hi = [np.inf, np.inf, np.inf, 6., 50., 8., 500., 0.95, np.inf, np.inf]
        starts = [
            # aL moderate, kL >> aL (sharp exponential cutoff), moderate right tail
            [N0, mu0, sig0 * 0.5,  1.0,  8., 0.8, 3., 0.15, mu0 + 3*sig0,  8*sig0],
            [N0, mu0, sig0 * 0.5,  0.8, 10., 0.7, 3., 0.20, mu0 + 4*sig0, 10*sig0],
            [N0, mu0, sig0 * 0.5,  0.5, 15., 0.5, 2., 0.25, mu0,           8*sig0],
            [N0, mu0, sig0,        2.0,  5., 0.8, 3., 0.15, mu0 + 3*sig0,  8*sig0],
            [N0, mu0, sig0 * 0.4,  1.0, 20., 0.5, 3., 0.30, mu0 + 2*sig0,  6*sig0],
            # left-biased: wide Gaussian to the left of core.
            [N0, mu0, sig0 * 0.05, 0.5, 10., 0.5, 3., 0.50, mu0 - 3*sig0,  2*sig0],
            [N0, mu0, sig0 * 0.03, 0.5, 15., 0.5, 3., 0.60, mu0 - 5*sig0,  4*sig0],
            [N0, mu0, sig0 * 0.05, 0.5, 10., 0.5, 5., 0.40, mu0 - 4*sig0,  3*sig0],
            [N0, mu0, sig0 * 0.10, 1.0,  5., 1.0, 5., 0.35, mu0 - 3*sig0,  2*sig0],
            [N0, mu0, sig0 * 0.02, 0.3, 20., 0.3, 3., 0.70, mu0 - 6*sig0,  6*sig0],
            # hard right cutoff: small aR + large nR
            [N0, mu0, sig0 * 0.3,  0.5,  5., 0.1, 100., 0.40, mu0 - 3*sig0,  2*sig0],
            [N0, mu0, sig0 * 0.2,  0.5,  8., 0.1, 200., 0.50, mu0 - 4*sig0,  3*sig0],
            [N0, mu0, sig0 * 0.1,  0.3, 10., 0.1, 300., 0.60, mu0 - 5*sig0,  4*sig0],
            [N0, mu0, sig0 * 0.5,  1.0,  5., 0.1, 100., 0.30, mu0 - 2*sig0,  2*sig0],
        ]
    return _best_fit(dcb_expleft_gauss, centers, counts, starts, lo, hi)


def fit_dcb_expright2g(centers, counts, mu0, sig0):
    """Scipy fit: power-law left DCB + exponential right tail + Gaussian wide component."""
    N0 = float(counts.max())
    # Anchor sigma_c to the narrow right-side width (peak → x_max), not to MAD which is
    # inflated by the heavy ISR left tail.
    x_right_span = max(float(centers[-1]) - mu0, 0.5)
    s = max(x_right_span * 0.40, 0.15)   # narrow Gaussian core

    lo = [0, -np.inf, 1e-6, 0.1, 1.01, 0.1, 0.05, 0.0, -np.inf, 1e-4]
    hi = [np.inf, np.inf, np.inf, 5., 50., 10., 100., 0.80, np.inf, np.inf]
    starts = [
        [N0, mu0, s,       0.5, 2.0, 1.5,  5.0, 0.05, mu0 - 4*s, 6*s],
        [N0, mu0, s,       0.3, 1.5, 1.0,  3.0, 0.10, mu0 - 5*s, 8*s],
        [N0, mu0, s*0.7,   0.5, 2.0, 2.0,  8.0, 0.05, mu0 - 5*s, 8*s],
        [N0, mu0, s*1.5,   0.4, 1.5, 1.5,  5.0, 0.05, mu0 - 4*s, 7*s],
        [N0, mu0, s,       0.7, 3.0, 2.0, 10.0, 0.05, mu0 - 5*s,10*s],
        [N0, mu0, s*0.5,   0.3, 1.2, 1.0,  5.0, 0.10, mu0 - 4*s, 6*s],
        [N0, mu0, s,       1.0, 2.0, 3.0, 15.0, 0.05, mu0 - 6*s,12*s],
        [N0, mu0, s*2.0,   0.5, 2.0, 2.0,  5.0, 0.10, mu0 - 4*s, 7*s],
        [N0, mu0, s*0.7,   0.4, 1.5, 1.5, 10.0, 0.20, mu0 - 6*s,12*s],
        [N0, mu0, s,       0.5, 1.5, 1.0,  3.0, 0.30, mu0 - 5*s,10*s],
    ]
    return _best_fit(dcb_expright_gauss, centers, counts, starts, lo, hi)


def fit_dcb_expright2g_iminuit(centers, counts, mu0, sig0):
    """Poisson NLL iminuit fit: power-law left DCB + exponential right + Gaussian wide."""
    from iminuit import Minuit
    N0   = float(counts.max())
    errs = np.maximum(np.sqrt(counts), 1.0)
    x_right_span = max(float(centers[-1]) - mu0, 0.5)
    s = max(x_right_span * 0.40, 0.15)

    def nll(N, mu_c, sc, aL, nL, aR, kR, fw, mu_w, sw):
        if sc <= 0 or nL < 1.0 or aL <= 0 or aR <= 0 or kR <= 0 or sw <= 0:
            return 1e15
        pred = dcb_expright_gauss(centers, abs(N), mu_c, abs(sc),
                                   abs(aL), abs(nL), abs(aR), abs(kR),
                                   abs(fw), mu_w, abs(sw))
        pred = np.maximum(pred, 1e-300)
        return 2.0 * float(np.sum(pred - counts * np.log(pred)))

    starts = [
        [N0, mu0, s,     0.5, 2.0, 1.5,  5.0, 0.05, mu0 - 4*s,  6*s],
        [N0, mu0, s,     0.3, 1.5, 1.0,  3.0, 0.10, mu0 - 5*s,  8*s],
        [N0, mu0, s*0.7, 0.5, 2.0, 2.0,  8.0, 0.05, mu0 - 5*s,  8*s],
        [N0, mu0, s,     0.7, 3.0, 2.0, 10.0, 0.05, mu0 - 5*s, 10*s],
        [N0, mu0, s*0.5, 0.3, 1.2, 1.0,  5.0, 0.10, mu0 - 4*s,  6*s],
        [N0, mu0, s,     1.0, 2.0, 3.0, 15.0, 0.05, mu0 - 6*s, 12*s],
    ]
    limits = [(1e-3,None),(None,None),(1e-6,None),(0.1,5.),(1.01,50.),
              (0.1,10.),(0.05,100.),(0.,0.80),(None,None),(1e-4,None)]
    names  = ['N','mu_c','sc','aL','nL','aR','kR','fw','mu_w','sw']

    best_popt, best_chi2 = None, np.inf
    for p0 in starts:
        try:
            m = Minuit(nll, *p0, name=names)
            for i, (lo_i, hi_i) in enumerate(limits):
                m.limits[i] = (lo_i, hi_i)
            m.migrad()
            if not m.valid:
                m.migrad()
            if m.valid:
                popt = list(m.values)
                pred = dcb_expright_gauss(centers, abs(popt[0]), popt[1], abs(popt[2]),
                                          abs(popt[3]), abs(popt[4]), abs(popt[5]), abs(popt[6]),
                                          abs(popt[7]), popt[8], abs(popt[9]))
                chi2 = float(np.sum(((counts - pred) / errs) ** 2))
                if chi2 < best_chi2:
                    best_chi2, best_popt = chi2, popt
        except Exception:
            pass

    if best_popt is None:
        return None, None, False, np.inf
    return best_popt, None, True, best_chi2


def _flatten_raw(raw):
    arr = np.asarray(raw)
    if arr.dtype == object:
        return np.concatenate([np.asarray(x, dtype=float).ravel() for x in arr])
    return arr.astype(float).ravel()


def _norm_curve(fn, xfine, x_lo, x_hi):
    integ = _quad(fn, x_lo, x_hi, limit=300)[0]
    return fn(xfine) / max(integ, 1e-300)


def _comparison_plot(ecm, _fitted, pairs, title_suffix, out_dir, label_a, label_b,
                     combined_label=None):
    os.makedirs(out_dir, exist_ok=True)
    for tag, (ba, bb) in pairs.items():
        if ba not in _fitted or bb not in _fitted:
            continue
        fn_a, edges_a, _ = _fitted[ba]
        fn_b, edges_b, _ = _fitted[bb]
        x_lo = min(edges_a[0], edges_b[0])
        x_hi = max(edges_a[-1], edges_b[-1])
        xfine = np.linspace(x_lo, x_hi, 600)
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.plot(xfine, _norm_curve(fn_a, xfine, x_lo, x_hi),
                color="tab:blue",   lw=2, label=label_a.format(ba))
        ax.plot(xfine, _norm_curve(fn_b, xfine, x_lo, x_hi),
                color="tab:orange", lw=2, label=label_b.format(bb))
        if combined_label and tag in _fitted:
            fn_c, _, _ = _fitted[tag]
            ax.plot(xfine, _norm_curve(fn_c, xfine, x_lo, x_hi),
                    color="crimson", lw=1.5, ls=":",
                    label=combined_label.format(tag))
        ax.set_xlabel(tag, fontsize=11)
        ax.set_ylabel("Normalised PDF", fontsize=11)
        ax.set_title(f"{tag}  [ecm{ecm}]  — {title_suffix}", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.set_ylim(bottom=0)
        for fmt in ("png", "pdf"):
            fig.savefig(f"{out_dir}/{tag}_comparison.{fmt}", dpi=150)
        plt.close(fig)


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.bool_):    return bool(o)
        return super().default(o)


def process_ecm(ecm):
    INFILE   = INFILE_TMPL.format(ecm=ecm)
    plot_dir = f"outputs/response/plots/ecm{ecm}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\n{'='*60}\nECM {ecm} GeV  —  {INFILE}\n{'='*60}")

    # ── Read tree (kinfit branches + virtual jet1+jet2 combined) ─────────────
    with uproot.open(INFILE) as f:
        tree = f["events"]
        available = set(tree.keys())

        branches = [b for b in KINFIT_BRANCHES if b in available]
        missing  = [b for b in KINFIT_BRANCHES if b not in available]
        if missing:
            print(f"WARNING: {len(missing)} kinfit branch(es) not in tree: {missing}")
        data_all = tree.arrays(branches, library="np")

        # Build virtual jet1+jet2 combined branches for the comparison plot.
        for cname, (b1, b2) in COMBINED_BRANCHES.items():
            if b1 in data_all and b2 in data_all:
                data_all[cname] = np.concatenate([_flatten_raw(data_all[b1]),
                                                  _flatten_raw(data_all[b2])])
                if cname not in branches:
                    branches.append(cname)

    print(f"Fitting {len(branches)} branches from {INFILE}\n")
    results = {}
    _fitted = {}   # bname → (yfn, edges, norm) for comparison plots

    for bname in branches:
        cfg              = BRANCH_CONFIG.get(bname, {})
        model            = cfg.get("model", "dcb")
        nbins            = cfg.get("nbins", NBINS_DEF)
        clip_lo, clip_hi = cfg.get("clip", CLIP_DEF)

        vals = _flatten_raw(data_all[bname])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 100:
            print(f"  [{ecm}]  SKIP {bname}: {len(vals)} entries"); continue

        lo_p, hi_p = np.percentile(vals, [clip_lo, clip_hi])
        vals_c = vals[(vals >= lo_p) & (vals <= hi_p)]

        counts, edges = np.histogram(vals_c, bins=nbins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mask    = counts > 0

        mu0  = float(centers[np.argmax(counts)])
        sig0 = float(median_abs_deviation(vals_c, scale="normal"))

        if model == "dcber2g":
            popt, pcov, fit_ok, chi2 = fit_dcb_expright2g(centers[mask], counts[mask], mu0, sig0)
            _ndof_est = max(int(mask.sum()) - 10, 1)
            if popt is None or chi2 / _ndof_est > 5.0:
                popt2, _, fit_ok2, chi2_2 = fit_dcb_expright2g_iminuit(
                    centers[mask], counts[mask], mu0, sig0)
                if popt2 is not None and (popt is None or chi2_2 < chi2):
                    popt, pcov, fit_ok, chi2 = popt2, None, fit_ok2, chi2_2
            nparams = 10
        elif model == "dcbgb":
            # Poisson NLL primary: correctly weights spike peak vs flat plateau.
            # chi² over-weights plateau bins (small sigma), forcing N down and undershooting peak.
            popt, pcov, fit_ok, chi2 = fit_dcb_gaussbox_iminuit(
                centers[mask], counts[mask], mu0, sig0)
            _ndof_est = max(int(mask.sum()) - 10, 1)
            if popt is None or chi2 / _ndof_est > 3.0:
                popt2, pcov2, fit_ok2, chi2_2 = fit_dcb_gaussbox(
                    centers[mask], counts[mask], mu0, sig0)
                if popt2 is not None and (popt is None or chi2_2 < chi2):
                    popt, pcov, fit_ok, chi2 = popt2, pcov2, fit_ok2, chi2_2
            nparams = 10
        elif model == "dcb2g":
            popt, pcov, fit_ok, chi2 = fit_dcb2g(centers[mask], counts[mask], mu0, sig0)
            # Only run iminuit (expensive) when scipy result is poor
            _ndof_est = max(int(mask.sum()) - 10, 1)
            if popt is None or chi2 / _ndof_est > 5.0:
                popt2, _, fit_ok2, chi2_2 = fit_dcb2g_iminuit(
                    centers[mask], counts[mask], mu0, sig0)
                if popt2 is not None and chi2_2 < chi2:
                    popt, pcov, fit_ok, chi2 = popt2, None, fit_ok2, chi2_2
            nparams = 10
        elif model == "expleft2g":
            constrain_mu0 = cfg.get("fix_mu0", False)
            popt, pcov, fit_ok, chi2 = fit_dcb_expleft2g(
                centers[mask], counts[mask], mu0, sig0, constrain_mu0=constrain_mu0)
            _ndof_est = max(int(mask.sum()) - 10, 1)
            # Lower threshold when constrained: scipy may settle in a suboptimal minimum
            _chi2_thr = 3.0 if constrain_mu0 else 5.0
            if popt is None or chi2 / _ndof_est > _chi2_thr:
                popt2, _, fit_ok2, chi2_2 = fit_dcb_expleft2g_iminuit(
                    centers[mask], counts[mask], mu0, sig0, constrain_mu0=constrain_mu0)
                if popt2 is not None and (popt is None or chi2_2 < chi2):
                    popt, pcov, fit_ok, chi2 = popt2, None, fit_ok2, chi2_2
            nparams = 10
        else:
            popt, pcov, fit_ok, chi2 = fit_dcb(centers[mask], counts[mask], mu0, sig0)
            nparams = 7

        ndof      = max(int(mask.sum()) - nparams, 1)
        chi2_ndof = chi2 / ndof

        if popt is None:
            print(f"  [{ecm}]  FAIL {bname}: all starts failed, using Gaussian-like fallback")
            popt = [float(counts.max()), mu0, sig0, 5., 100., 5., 100.]
            if model == "dcber2g":
                _s = max((float(centers[-1]) - mu0) * 0.4, 0.15)
                popt = [float(counts.max()), mu0, _s, 0.5, 2.0, 1.5, 5.0, 0.05,
                        mu0 - 4*_s, 6*_s]
            elif model == "dcbgb":
                x_span = 0.5 * (centers[-1] - centers[0])
                popt += [0.01, x_span * 0.8, sig0 * 0.2]
            elif model in ("dcb2g", "expleft2g"):
                popt += [0.01, mu0, sig0 * 5]
            fit_ok    = False
            chi2_ndof = np.inf

        # ── unpack and sanitise ──────────────────────────────────────────────
        if model == "dcber2g":
            N_f, mu_c, sc, aL, nL, aR, kR, fw, muw, sw = popt
            mu_c = float(mu_c);  sc  = abs(float(sc))
            aL   = abs(float(aL)); nL = abs(float(nL))
            aR   = abs(float(aR)); kR = abs(float(kR))
            fw   = abs(float(fw)); muw = float(muw); sw = abs(float(sw))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="dcber2g",
                mu=mu_c, sigma=sc, aL=aL, nL=nL, aR=aR, kR=kR,
                f_wide=fw, mu_wide=muw, sigma_wide=sw,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _mc=mu_c, _sc=sc, _aL=aL, _nL=nL, _aR=aR, _kR=kR,
                    _fw=fw, _mw=muw, _sw=sw):
                return dcb_expright_gauss(x, _N, _mc, _sc, _aL, _nL, _aR, _kR, _fw, _mw, _sw)
            _xscan = np.linspace(centers[0], centers[-1], 2000)
            _x_peak = float(_xscan[np.argmax(yfn(_xscan))])
            lbl = (rf"DCBExpRight+G: $x_{{peak}}$={_x_peak:+.3g}, $\mu_c$={mu_c:+.3g}"
                   "\n"
                   rf"$\sigma_c$={sc:.3g}, $\alpha_L$={aL:.2f}, $n_L$={nL:.2f}, $k_R$={kR:.3g}"
                   "\n"
                   rf"$f_w$={fw:.3f}, $\mu_w$={muw:.3g}, $\sigma_w$={sw:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "dcbgb":
            N_f, mu_c, sc, aL, nL, aR, nR, fw, p_max, sb = popt
            mu_c = float(mu_c);  sc   = abs(float(sc))
            aL   = abs(float(aL)); nL = abs(float(nL))
            aR   = abs(float(aR)); nR = abs(float(nR))
            fw   = abs(float(fw)); p_max = abs(float(p_max)); sb = abs(float(sb))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="dcbgb",
                mu=mu_c, sigma=sc, aL=aL, nL=nL, aR=aR, nR=nR,
                f_wide=fw, p_max=p_max, sigma_box=sb,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _mc=mu_c, _sc=sc, _aL=aL, _nL=nL, _aR=aR, _nR=nR,
                    _fw=fw, _pm=p_max, _sb=sb):
                return dcb_gaussbox(x, _N, _mc, _sc, _aL, _nL, _aR, _nR, _fw, _pm, _sb)
            lbl = (rf"DCB+Box: $\mu_c$={mu_c:+.3g}, $\sigma_c$={sc:.3g}"
                   "\n"
                   rf"$\alpha_L$={aL:.2f}, $n_L$={nL:.1f}, $\alpha_R$={aR:.2f}, $n_R$={nR:.1f}"
                   "\n"
                   rf"$f_w$={fw:.3f}, $p_{{max}}$={p_max:.2g}, $\sigma_{{box}}$={sb:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "dcb2g":
            N_f, mu_c, sc, aL, nL, aR, nR, fw, muw, sw = popt
            mu_c = float(mu_c);  sc  = abs(float(sc))
            aL   = abs(float(aL)); nL = abs(float(nL))
            aR   = abs(float(aR)); nR = abs(float(nR))
            fw   = abs(float(fw)); muw = float(muw); sw = abs(float(sw))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="dcb2g",
                mu=mu_c, sigma=sc, aL=aL, nL=nL, aR=aR, nR=nR,
                f_wide=fw, mu_wide=muw, sigma_wide=sw,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _mc=mu_c, _sc=sc, _aL=aL, _nL=nL, _aR=aR, _nR=nR,
                    _fw=fw, _mw=muw, _sw=sw):
                return dcb_gauss(x, _N, _mc, _sc, _aL, _nL, _aR, _nR, _fw, _mw, _sw)
            lbl = (rf"DCB+G: $\mu_c$={mu_c:+.3g}, $\sigma_c$={sc:.3g}"
                   "\n"
                   rf"$\alpha_L$={aL:.2f}, $n_L$={nL:.1f}, $\alpha_R$={aR:.2f}, $n_R$={nR:.1f}"
                   "\n"
                   rf"$f_w$={fw:.3f}, $\mu_w$={muw:.3g}, $\sigma_w$={sw:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "expleft2g":
            N_f, mu_c, sc, aL, kL, aR, nR, fw, muw, sw = popt
            mu_c = float(mu_c);  sc  = abs(float(sc))
            aL   = abs(float(aL)); kL = abs(float(kL))
            aR   = abs(float(aR)); nR = abs(float(nR))
            fw   = abs(float(fw)); muw = float(muw); sw = abs(float(sw))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="expleft2g",
                mu=mu_c, sigma=sc, aL=aL, kL=kL, aR=aR, nR=nR,
                f_wide=fw, mu_wide=muw, sigma_wide=sw,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _mc=mu_c, _sc=sc, _aL=aL, _kL=kL, _aR=aR, _nR=nR,
                    _fw=fw, _mw=muw, _sw=sw):
                return dcb_expleft_gauss(x, _N, _mc, _sc, _aL, _kL, _aR, _nR, _fw, _mw, _sw)
            # Report actual model maximum (physical peak), not the internal mu_c parameter.
            _xscan = np.linspace(centers[0], centers[-1], 2000)
            _x_peak = float(_xscan[np.argmax(yfn(_xscan))])
            lbl = (rf"ExpLeft+G: $x_{{peak}}$={_x_peak:+.3g}, $\mu_c$={mu_c:+.3g}"
                   "\n"
                   rf"$\sigma_c$={sc:.3g}, $k_L$={kL:.3g}, $\alpha_R$={aR:.2f}, $n_R$={nR:.1f}"
                   "\n"
                   rf"$f_w$={fw:.3f}, $\mu_w$={muw:.3g}, $\sigma_w$={sw:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        else:
            N_f, mu_f, sf, aL, nL, aR, nR = popt
            mu_f = float(mu_f); sf  = abs(float(sf))
            aL   = abs(float(aL)); nL = abs(float(nL))
            aR   = abs(float(aR)); nR = abs(float(nR))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="dcb",
                mu=float(mu_f), sigma=float(sf),
                aL=float(aL), nL=float(nL), aR=float(aR), nR=float(nR),
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _m=mu_f, _s=sf, _aL=aL, _nL=nL, _aR=aR, _nR=nR):
                return dcb(x, _N, _m, _s, _aL, _nL, _aR, _nR)
            lbl = (rf"DCB: $\mu$={mu_f:+.3g}, $\sigma$={sf:.3g}"
                   "\n"
                   rf"$\alpha_L$={aL:.2f}, $n_L$={nL:.1f}, $\alpha_R$={aR:.2f}, $n_R$={nR:.1f}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")

        # Normalise shape to unit integral; yfn includes N_f so divide it out.
        # Build integration range from fitted PDF shape (±10σ of core and wide
        # component) rather than histogram span, to capture broad wide-Gaussian tails.
        _p = results[bname]
        _sg = _p.get("sigma", 1.0)
        _half_core = 10.0 * max(abs(_p.get("aL", 2.0)), abs(_p.get("aR", 2.0)), 1.0) * _sg
        if "mu" in _p:
            _mu_ref = _p["mu"]
            _half = _half_core
            if "mu_wide" in _p and "sigma_wide" in _p:
                _half = max(_half, abs(_p["mu_wide"] - _mu_ref) + 10.0 * abs(_p["sigma_wide"]))
            _integ_lo, _integ_hi = _mu_ref - _half, _mu_ref + _half
        elif "x_cut" in _p:
            _xc = _p["x_cut"]; _sw = _p.get("sigma_wide", _sg)
            _integ_lo = _xc - 10.0 * max(_sw, _sg); _integ_hi = _xc + 0.5
        else:
            _integ_lo, _integ_hi = -_half_core, _half_core
        _integ, _ = _quad(lambda x: yfn(x) / N_f,
                          _integ_lo, _integ_hi,
                          limit=500, epsrel=1e-6)
        results[bname]["norm"] = float(1.0 / max(_integ, 1e-300))
        _fitted[bname] = (yfn, edges, results[bname]["norm"])

        print(f"  [{ecm}]  {bname:40s}  χ²/ndf={chi2_ndof:.2f}  "
              f"{'OK' if fit_ok else 'WARN'}  [{model}]")

        # ── plot ─────────────────────────────────────────────────────────────
        # Build xfine with a uniform global grid plus a dense sub-grid around
        # mu_c so very narrow cores (e.g. σ_c sub-mGeV for px/gen_WW_py) are
        # actually resolved by the displayed curve, not skipped between samples.
        xfine_uniform = np.linspace(edges[0], edges[-1], 600)
        _w = max(20.0 * sc, 5.0 * (edges[1] - edges[0]))
        xfine_peak = np.linspace(max(edges[0], mu_c - _w),
                                 min(edges[-1], mu_c + _w), 1500)
        xfine = np.unique(np.concatenate([xfine_uniform, xfine_peak]))
        yfine = yfn(xfine)
        bw    = np.diff(edges)[0]

        fig, (ax, ax_res) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            layout="constrained",
        )
        ax.bar(centers, counts, width=bw, color="steelblue", alpha=0.55, label="data")
        ax.plot(xfine, yfine, color="crimson", lw=2, label=lbl)

        if model == "dcber2g":
            y_core = dcb_expright_gauss(xfine, N_f, mu_c, sc, aL, nL, aR, kR, 0., muw, sw)
            y_wide = dcb_expright_gauss(xfine, N_f, mu_c, sc, aL, nL, aR, kR, 1., muw, sw)
            ax.plot(xfine, y_core * (1 - fw), color="tab:orange", lw=1.2, ls="--",
                    label="DCBExpRight core")
            ax.plot(xfine, y_wide * fw,       color="tab:green",  lw=1.2, ls=":",
                    label="Gaussian component")
        elif model == "dcbgb":
            y_core = dcb_gaussbox(xfine, N_f, mu_c, sc, aL, nL, aR, nR, 0., p_max, sb)
            y_wide = dcb_gaussbox(xfine, N_f, mu_c, sc, aL, nL, aR, nR, 1., p_max, sb)
            ax.plot(xfine, y_core * (1 - fw), color="tab:orange", lw=1.2, ls="--",
                    label="DCB core")
            ax.plot(xfine, y_wide * fw,       color="tab:green",  lw=1.2, ls=":",
                    label="GaussBox component")
        elif model == "dcb2g":
            y_core = dcb_gauss(xfine, N_f, mu_c, sc, aL, nL, aR, nR, 0., muw, sw)
            y_wide = dcb_gauss(xfine, N_f, mu_c, sc, aL, nL, aR, nR, 1., muw, sw)
            ax.plot(xfine, y_core * (1 - fw), color="tab:orange", lw=1.2, ls="--",
                    label="DCB core")
            ax.plot(xfine, y_wide * fw,       color="tab:green",  lw=1.2, ls=":",
                    label="Gaussian component")
        elif model == "expleft2g":
            y_core = dcb_expleft_gauss(xfine, N_f, mu_c, sc, aL, kL, aR, nR, 0., muw, sw)
            y_wide = dcb_expleft_gauss(xfine, N_f, mu_c, sc, aL, kL, aR, nR, 1., muw, sw)
            ax.plot(xfine, y_core * (1 - fw), color="tab:orange", lw=1.2, ls="--",
                    label="ExpLeft core")
            ax.plot(xfine, y_wide * fw,       color="tab:green",  lw=1.2, ls=":",
                    label="Gaussian component")

        ax.set_ylabel("Entries", fontsize=11)
        ax.set_title(f"{bname}  [ecm{ecm}]", fontsize=11)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax.set_ylim(bottom=0)

        pull = np.where(counts > 0, (counts - yfn(centers)) / np.sqrt(np.maximum(counts, 1)), 0)
        ax_res.bar(centers, pull, width=bw, color="steelblue", alpha=0.6)
        ax_res.axhline(0, color="crimson", lw=1)
        ax_res.set_ylabel("Pull", fontsize=10)
        ax_res.set_xlabel(bname, fontsize=11)
        ax_res.set_ylim(-5, 5)

        for fmt in ("png", "pdf"):
            fig.savefig(f"{plot_dir}/{bname}.{fmt}", dpi=150)

        # Optional zoomed-in view: same fit/binning, narrower x-range + log-y so
        # the unresolved peak and the wide tail are both visible.
        zoom_xlim = cfg.get("zoom_xlim")
        if zoom_xlim is not None:
            ax.set_xlim(*zoom_xlim)
            ax_res.set_xlim(*zoom_xlim)
            in_zoom = (centers >= zoom_xlim[0]) & (centers <= zoom_xlim[1]) & (counts > 0)
            if in_zoom.any():
                ax.set_ylim(max(0.5, float(counts[in_zoom].min()) * 0.5),
                            float(counts[in_zoom].max()) * 1.5)
                ax.set_yscale("log")
            for fmt in ("png", "pdf"):
                fig.savefig(f"{plot_dir}/{bname}_zoom.{fmt}", dpi=150)

        plt.close(fig)

    # ── Jet1 vs jet2 comparison plots ────────────────────────────────────────
    comp_dir = f"{plot_dir}/jet_comparisons"
    _comparison_plot(
        ecm, _fitted, COMBINED_BRANCHES, "jet1 vs jet2", comp_dir,
        label_a="{0}", label_b="{0}",
        combined_label="{0} (combined fit)",
    )
    print(f"  Jet comparison plots → {comp_dir}/")

    # ── JSON ──────────────────────────────────────────────────────────────────
    os.makedirs("outputs/response/functions", exist_ok=True)
    json_path = f"outputs/response/functions/dcb_results_ecm{ecm}.json"
    with open(json_path, "w") as fj:
        json.dump(results, fj, indent=2, cls=_NpEncoder)

    print(f"\nDone ecm{ecm}. Plots → {plot_dir}/  |  JSON → {json_path}")
    return ecm, results


def _cpp_param_line(bname, p, ecm):
    tag  = bname.upper()
    note = f"// chi2/ndf={p['chi2_ndof']:.2f}  {'OK' if p['fit_ok'] else 'WARN'}"
    if p["model"] == "dcb2g":
        return (
            f"constexpr DcbGaussParams DCBG_{tag}_{ecm} = "
            f"{{ {p['mu']:+.6f}, {p['sigma']:.6f}, "
            f"{p['aL']:.6f}, {p['nL']:.6f}, {p['aR']:.6f}, {p['nR']:.6f}, "
            f"{p['f_wide']:.6f}, {p['mu_wide']:+.6f}, {p['sigma_wide']:.6f}, "
            f"{p['norm']:.10e} }};  {note}"
        )
    elif p["model"] == "expleft2g":
        return (
            f"constexpr DcbExpLeftGaussParams DCBELG_{tag}_{ecm} = "
            f"{{ {p['mu']:+.6f}, {p['sigma']:.6f}, "
            f"{p['aL']:.6f}, {p['kL']:.6f}, {p['aR']:.6f}, {p['nR']:.6f}, "
            f"{p['f_wide']:.6f}, {p['mu_wide']:+.6f}, {p['sigma_wide']:.6f}, "
            f"{p['norm']:.10e} }};  {note}"
        )
    elif p["model"] == "dcber2g":
        return (
            f"constexpr DcbExpRightGaussParams DCBERG_{tag}_{ecm} = "
            f"{{ {p['mu']:+.6f}, {p['sigma']:.6f}, "
            f"{p['aL']:.6f}, {p['nL']:.6f}, {p['aR']:.6f}, {p['kR']:.6f}, "
            f"{p['f_wide']:.6f}, {p['mu_wide']:+.6f}, {p['sigma_wide']:.6f}, "
            f"{p['norm']:.10e} }};  {note}"
        )
    else:
        return (
            f"constexpr DcbParams DCB_{tag}_{ecm} = "
            f"{{ {p['mu']:+.6f}, {p['sigma']:.6f}, "
            f"{p['aL']:.6f}, {p['nL']:.6f}, {p['aR']:.6f}, {p['nR']:.6f}, "
            f"{p['norm']:.10e} }};  {note}"
        )


def write_combined_header(all_results):
    """
    Generate outputs/response/functions/dcb_params.h — a single self-contained header
    with structs, evaluators, and fitted parameters for all ECMs.
    all_results: dict  ecm -> {bname: params_dict}
    """
    lines = [
        "#pragma once",
        "// Auto-generated by fit_resolutions.py — do not edit by hand.",
        "// Contains PDF struct definitions, normalised evaluators, and fitted",
        "// parameters for all centre-of-mass energies.",
        "// Include this file instead of the individual per-ECM headers.",
        "",
        "#include <cmath>",
        "#include <algorithm>",
        "",
        "namespace WWFunctions {",
        "",
        "// ── Struct definitions ───────────────────────────────────────────────",
        "",
        "struct DcbParams {",
        "    double mu, sigma, aL, nL, aR, nR;",
        "    double norm;                           // 1/integral, shape integrates to 1",
        "};",
        "",
        "struct DcbGaussParams {",
        "    double mu, sigma, aL, nL, aR, nR;     // narrow DCB core",
        "    double f_wide, mu_wide, sigma_wide;    // broad Gaussian component",
        "    double norm;                           // 1/integral, shape integrates to 1",
        "};",
        "",
        "struct DcbExpLeftGaussParams {",
        "    double mu, sigma, aL, kL, aR, nR;     // exp-left DCB core (kL: exp decay rate)",
        "    double f_wide, mu_wide, sigma_wide;    // broad Gaussian component",
        "    double norm;                           // 1/integral, shape integrates to 1",
        "};",
        "",
        "struct DcbExpRightGaussParams {",
        "    double mu, sigma, aL, nL;              // power-law left tail (ISR heavy tail)",
        "    double aR, kR;                         // exponential right tail: exp(-kR*(t-aR))",
        "    double f_wide, mu_wide, sigma_wide;    // broad Gaussian component",
        "    double norm;                           // 1/integral, shape integrates to 1",
        "};",
        "",
        "// ── Unnormalised shape helpers ───────────────────────────────────────",
        "",
        "namespace detail {",
        "inline double dcb_unnorm(double t, double aL, double nL,",
        "                          double aR, double nR) {",
        "    if (t < -aL) {",
        "        double AL = std::pow(nL/aL, nL) * std::exp(-0.5*aL*aL);",
        "        double BL = nL/aL - aL;",
        "        return AL * std::pow(std::max(BL - t, 1e-10), -nL);",
        "    }",
        "    if (t > aR) {",
        "        double AR = std::pow(nR/aR, nR) * std::exp(-0.5*aR*aR);",
        "        double BR = nR/aR - aR;",
        "        return AR * std::pow(std::max(BR + t, 1e-10), -nR);",
        "    }",
        "    return std::exp(-0.5*t*t);",
        "}",
        "inline double dcb_expleft_unnorm(double t, double aL, double kL,",
        "                                  double aR, double nR) {",
        "    if (t < -aL) return std::exp(-0.5*aL*aL) * std::exp(kL*(aL + t));",
        "    if (t > aR) {",
        "        double AR = std::pow(nR/aR, nR) * std::exp(-0.5*aR*aR);",
        "        double BR = nR/aR - aR;",
        "        return AR * std::pow(std::max(BR + t, 1e-10), -nR);",
        "    }",
        "    return std::exp(-0.5*t*t);",
        "}",
        "} // namespace detail",
        "",
        "// ── Evaluators: return -2*log(P(x)), P(x) normalised to unit integral ─",
        "",
        "inline double dcb_neg2logpdf(double x, const DcbParams& p) {",
        "    double f = detail::dcb_unnorm((x - p.mu)/p.sigma, p.aL, p.nL, p.aR, p.nR);",
        "    return -2.0 * (std::log(std::max(f, 1e-300)) + std::log(p.norm));",
        "}",
        "",
        "inline double dcb_gauss_neg2logpdf(double x, const DcbGaussParams& p) {",
        "    double core = detail::dcb_unnorm((x - p.mu)/p.sigma, p.aL, p.nL, p.aR, p.nR);",
        "    double wide = std::exp(-0.5 * std::pow((x - p.mu_wide)/p.sigma_wide, 2));",
        "    double f    = (1.0 - p.f_wide) * core + p.f_wide * wide;",
        "    return -2.0 * (std::log(std::max(f, 1e-300)) + std::log(p.norm));",
        "}",
        "",
        "inline double dcb_expleft_gauss_neg2logpdf(double x, const DcbExpLeftGaussParams& p) {",
        "    double core = detail::dcb_expleft_unnorm((x - p.mu)/p.sigma, p.aL, p.kL, p.aR, p.nR);",
        "    double wide = std::exp(-0.5 * std::pow((x - p.mu_wide)/p.sigma_wide, 2));",
        "    double f    = (1.0 - p.f_wide) * core + p.f_wide * wide;",
        "    return -2.0 * (std::log(std::max(f, 1e-300)) + std::log(p.norm));",
        "}",
        "",
        "inline double dcb_expright_gauss_neg2logpdf(double x, const DcbExpRightGaussParams& p) {",
        "    double t    = (x - p.mu) / p.sigma;",
        "    double core;",
        "    if (t < -p.aL) {",
        "        double AL = std::pow(p.nL/p.aL, p.nL) * std::exp(-0.5*p.aL*p.aL);",
        "        double BL = p.nL/p.aL - p.aL;",
        "        core = AL * std::pow(std::max(BL - t, 1e-10), -p.nL);",
        "    } else if (t > p.aR) {",
        "        core = std::exp(-0.5*p.aR*p.aR - p.kR*(t - p.aR));",
        "    } else {",
        "        core = std::exp(-0.5*t*t);",
        "    }",
        "    double wide = std::exp(-0.5 * std::pow((x - p.mu_wide)/p.sigma_wide, 2));",
        "    double f    = (1.0 - p.f_wide) * core + p.f_wide * wide;",
        "    return -2.0 * (std::log(std::max(f, 1e-300)) + std::log(p.norm));",
        "}",
        "",
        "// ── Fitted parameters ────────────────────────────────────────────────",
    ]

    for ecm in sorted(all_results):
        results = all_results[ecm]
        lines.append(f"")
        lines.append(f"// ECM {ecm} GeV")
        for bname, p in sorted(results.items()):
            lines.append(_cpp_param_line(bname, p, ecm))

    lines += ["", "} // namespace WWFunctions", ""]

    out_path = "outputs/response/functions/dcb_params.h"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Written {out_path}")


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=len(ECM_LIST)) as pool:
        all_results = dict(pool.map(process_ecm, ECM_LIST))
    write_combined_header(all_results)
    publish("outputs/response/plots", "resolutions")
