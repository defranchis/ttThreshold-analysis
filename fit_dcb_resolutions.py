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
  response/plots/ecm<N>/<branch>.{png,pdf}    one plot per branch with data + fit
  response/functions/dcb_params_ecm<N>.h      C++ header with evaluators + constexpr params
  response/functions/dcb_results_ecm<N>.json  full numerical fit results
"""

import os, json, warnings
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
from scipy.integrate import quad as _quad

os.makedirs("response/functions", exist_ok=True)

ECM_LIST    = [157, 160, 163]
INFILE_TMPL = "outputs/treemaker/lnuqq/semihad/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
NBINS_DEF   = 100
CLIP_DEF  = (0.5, 99.5)

# ── Per-branch configuration overrides ─────────────────────────────────────
# Motivation for tight left-clip on the diff_RG_* variables:
#   these distributions have a narrow gaussian-like core but a very long,
#   flat left tail (ISR / neutrino mis-reco). Using std overestimates sigma
#   and shifts the fitted peak; clipping the extreme left tail + using MAD
#   focuses the fit on the physically relevant core region while the fitted
#   power-law extrapolates the tail.
#
# Motivation for "dcb2g" model on jet resolutions:
#   these show a sharp core peak (well-measured jets) plus a secondary
#   shoulder component (partially measured / split jets).  A single DCB
#   cannot describe both simultaneously.

BRANCH_CONFIG = {
    # Mass/momentum resolutions (absolute reco-gen differences)
    "m_lnu_resol":          {"clip": (0.5, 99.5), "nbins": 150},
    "p_lnu_resol":          {"clip": (3.0, 99.5), "nbins": 150},
    "pf_qq_m_resol":        {"clip": (2.0, 99.5), "nbins": 150},
    "m_qq_resol":           {"clip": (2.0, 99.5), "nbins": 150},
    "pf_qq_p_resol":        {"clip": (2.0, 99.5), "nbins": 150},
    "p_qq_resol":           {"clip": (2.0, 99.5), "nbins": 150},
    "m_lnuqq_resol":        {"clip": (1.0, 99.5), "nbins": 150},
    # Jet momentum responses (reco/gen ratios); two-component structure → DCB+G
    "jet1_p_resp":          {"clip": (0.2, 99.8), "nbins": 150, "model": "dcb2g"},
    "jet1_p_fromele_resp":  {"clip": (0.2, 99.8), "nbins": 150, "model": "dcb2g"},
    # Jet2: sharp core — single DCB describes it better than DCB+G
    "jet2_p_resp":          {"clip": (0.2, 99.8), "nbins": 150, "model": "dcb2g"},
    "jet2_p_fromele_resp":  {"clip": (1.0, 99.0), "nbins": 150, "model": "dcb2g"},
    # Combined (jet1+jet2) pooled responses
    "jet_p_resp":           {"clip": (0.2, 99.8), "nbins": 150, "model": "dcb2g"},
    "jet_p_fromele_resp":   {"clip": (0.2, 99.8), "nbins": 150, "model": "dcb2g"},
    # costheta resolutions need two-component model (DCB alone chi2/ndf ~5)
    "jet1_costheta_resol":  {"clip": (0.5, 99.5), "nbins": 150, "model": "dcb2g"},
    "jet2_costheta_resol":  {"clip": (0.5, 99.5), "nbins": 150, "model": "dcb2g"},
    "jet_costheta_resol":   {"clip": (0.5, 99.5), "nbins": 150, "model": "dcb2g"},
    # Gen-level total momenta: narrow ISR-free spike + broad wings → DCB+G
    "px_tot_gen":           {"clip": (0.5, 99.5), "nbins": 200, "model": "dcb2g"},
    "py_tot_gen":           {"clip": (0.5, 99.5), "nbins": 200, "model": "dcb2g"},
    "pz_tot_gen":           {"clip": (0.5, 99.5), "nbins": 200, "model": "dcb2g"},
}

# ── Virtual combined branches (concatenation of two tree branches) ──────────
COMBINED_BRANCHES = {
    "jet_p_resp":           ("jet1_p_resp",          "jet2_p_resp"),
    "jet_p_fromele_resp":   ("jet1_p_fromele_resp",   "jet2_p_fromele_resp"),
    "jet_costheta_resol":   ("jet1_costheta_resol",   "jet2_costheta_resol"),
    "jet_eta_resol":        ("jet1_eta_resol",         "jet2_eta_resol"),
    "jet_phi_resol":        ("jet1_phi_resol",         "jet2_phi_resol"),
    "jet_theta_resol":      ("jet1_theta_resol",       "jet2_theta_resol"),
}

# Standard vs fromele pairs (per-jet and combined)
FROMELE_PAIRS = {
    "jet1_p_resp": ("jet1_p_resp", "jet1_p_fromele_resp"),
    "jet2_p_resp": ("jet2_p_resp", "jet2_p_fromele_resp"),
    "jet_p_resp":  ("jet_p_resp",  "jet_p_fromele_resp"),
}

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
        np.exp(np.clip(log_AL - nL * np.log(np.maximum(BL - t, 1e-10)), -745., 709.)),
        np.where(
            t > aR,
            np.exp(np.clip(log_AR - nR * np.log(np.maximum(BR + t, 1e-10)), -745., 709.)),
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
            np.exp(np.clip(log_AR - nR * np.log(np.maximum(BR + t, 1e-10)), -745., 709.)),
            np.exp(-0.5 * t * t)
        )
    )


def dcb_expleft_gauss(x, N, mu_c, sigma_c, aL, kL, aR, nR, f_wide, mu_w, sigma_w):
    """Exp-left DCB core + broad Gaussian component."""
    core = _dcb_expleft_core((x - mu_c) / sigma_c, aL, kL, aR, nR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)


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
    ]
    return _best_fit(dcb_gauss, centers, counts, starts, lo, hi)


def fit_dcb_expleft2g(centers, counts, mu0, sig0):
    """DCB with exponential left tail + Gaussian wide component.
    Params: N, mu_c, sigma_c, aL, kL, aR, nR, f_wide, mu_w, sigma_w
    kL > 0: exponential decay rate of left tail (kL > aL → faster than Gaussian).
    """
    N0 = float(counts.max())
    lo = [0, -np.inf, 1e-6, 0.1, 0.1,  0.3, 1.01, 0.01, -np.inf, 1e-4]
    hi = [np.inf, np.inf, np.inf, 6., 50., 8., 200., 0.95, np.inf, np.inf]
    starts = [
        # aL moderate, kL >> aL (sharp exponential cutoff), moderate right tail
        [N0, mu0, sig0 * 0.5,  1.0,  8., 0.8, 3., 0.15, mu0 + 3*sig0,  8*sig0],
        [N0, mu0, sig0 * 0.5,  0.8, 10., 0.7, 3., 0.20, mu0 + 4*sig0, 10*sig0],
        [N0, mu0, sig0 * 0.5,  0.5, 15., 0.5, 2., 0.25, mu0,           8*sig0],
        [N0, mu0, sig0,        2.0,  5., 0.8, 3., 0.15, mu0 + 3*sig0,  8*sig0],
        [N0, mu0, sig0 * 0.4,  1.0, 20., 0.5, 3., 0.30, mu0 + 2*sig0,  6*sig0],
    ]
    return _best_fit(dcb_expleft_gauss, centers, counts, starts, lo, hi)


for ecm in ECM_LIST:
    INFILE   = INFILE_TMPL.format(ecm=ecm)
    plot_dir = f"response/plots/ecm{ecm}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\n{'='*60}\nECM {ecm} GeV  —  {INFILE}\n{'='*60}")

    # ── Read tree ────────────────────────────────────────────────────────────
    with uproot.open(INFILE) as f:
        tree = f["events"]
        available = set(tree.keys())
        EXTRA_BRANCHES   = {"px_tot_gen", "py_tot_gen", "pz_tot_gen"}
        EXCLUDE_BRANCHES = {"px_tot_resol", "py_tot_resol", "pz_tot_resol"}
        branches = sorted([b for b in available
                           if ((b.endswith("_resol") or b.endswith("_resp"))
                               and b not in EXCLUDE_BRANCHES)
                           or b in EXTRA_BRANCHES])
        data_all = tree.arrays(branches, library="np")

        def _flatten_raw(raw):
            arr = np.asarray(raw)
            if arr.dtype == object:
                return np.concatenate([np.asarray(x, dtype=float).ravel() for x in arr])
            return arr.astype(float).ravel()
        for cname, (b1, b2) in COMBINED_BRANCHES.items():
            if b1 in data_all and b2 in data_all:
                data_all[cname] = np.concatenate([_flatten_raw(data_all[b1]),
                                                  _flatten_raw(data_all[b2])])
                if cname not in branches:
                    branches = sorted(branches + [cname])

    missing_cfg = [b for b in BRANCH_CONFIG
                   if b not in available and b not in COMBINED_BRANCHES]
    if missing_cfg:
        print(f"WARNING: {len(missing_cfg)} configured branch(es) not found in {INFILE}:")
        for b in missing_cfg:
            print(f"  MISSING  {b}")

    print(f"Fitting {len(branches)} branches from {INFILE}\n")
    results = {}
    _fitted = {}   # bname → (yfn, edges, norm) for comparison plots

    for bname in branches:
        cfg              = BRANCH_CONFIG.get(bname, {})
        model            = cfg.get("model", "dcb")
        nbins            = cfg.get("nbins", NBINS_DEF)
        clip_lo, clip_hi = cfg.get("clip", CLIP_DEF)

        raw = data_all[bname]
        arr = np.asarray(raw)
        if arr.dtype == object:
            vals = np.concatenate([np.asarray(x, dtype=float).ravel() for x in arr])
        else:
            vals = arr.astype(float).ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals) < 100:
            print(f"  SKIP {bname}: {len(vals)} entries"); continue

        lo_p, hi_p = np.percentile(vals, [clip_lo, clip_hi])
        vals_c = vals[(vals >= lo_p) & (vals <= hi_p)]

        counts, edges = np.histogram(vals_c, bins=nbins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mask    = counts > 0

        mu0  = float(centers[np.argmax(counts)])
        sig0 = float(median_abs_deviation(vals_c, scale="normal"))

        if model == "dcb2g":
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
            popt, pcov, fit_ok, chi2 = fit_dcb_expleft2g(centers[mask], counts[mask], mu0, sig0)
            nparams = 10
        else:
            popt, pcov, fit_ok, chi2 = fit_dcb(centers[mask], counts[mask], mu0, sig0)
            nparams = 7

        ndof      = max(int(mask.sum()) - nparams, 1)
        chi2_ndof = chi2 / ndof

        if popt is None:
            print(f"  FAIL {bname}: all starts failed, using Gaussian-like fallback")
            popt = [float(counts.max()), mu0, sig0, 5., 100., 5., 100.]
            if model in ("dcb2g", "expleft2g"):
                popt += [0.01, mu0, sig0 * 5]
            fit_ok    = False
            chi2_ndof = np.inf

        # ── unpack and sanitise ──────────────────────────────────────────────
        if model == "dcb2g":
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
            lbl = (rf"ExpLeft+G: $\mu_c$={mu_c:+.3g}, $\sigma_c$={sc:.3g}"
                   "\n"
                   rf"$\alpha_L$={aL:.2f}, $k_L$={kL:.2f}, $\alpha_R$={aR:.2f}, $n_R$={nR:.1f}"
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

        # Normalise shape to unit integral; yfn includes N_f so divide it out
        _integ, _ = _quad(lambda x: yfn(x) / N_f, -np.inf, np.inf,
                          limit=500, epsabs=0, epsrel=1e-6)
        results[bname]["norm"] = float(1.0 / max(_integ, 1e-300))
        _fitted[bname] = (yfn, edges, results[bname]["norm"])

        print(f"  {bname:40s}  χ²/ndf={chi2_ndof:.2f}  "
              f"{'OK' if fit_ok else 'WARN'}  [{model}]")

        # ── plot ─────────────────────────────────────────────────────────────
        xfine = np.linspace(edges[0], edges[-1], 600)
        yfine = yfn(xfine)
        bw    = np.diff(edges)[0]

        fig, (ax, ax_res) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            layout="constrained",
        )
        ax.bar(centers, counts, width=bw, color="steelblue", alpha=0.55, label="data")
        ax.plot(xfine, yfine, color="crimson", lw=2, label=lbl)

        if model == "dcb2g":
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
        plt.close(fig)

    # ── Comparison plot helpers ───────────────────────────────────────────────
    def _norm_curve(fn, xfine, x_lo, x_hi):
        integ = _quad(fn, x_lo, x_hi, limit=300)[0]
        return fn(xfine) / max(integ, 1e-300)

    def _comparison_plot(pairs, title_suffix, out_dir, label_a, label_b,
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

    # ── Jet1 vs jet2 comparison plots ────────────────────────────────────────
    comp_dir = f"{plot_dir}/jet_comparisons"
    _comparison_plot(
        COMBINED_BRANCHES, "jet1 vs jet2", comp_dir,
        label_a="{0}", label_b="{0}",
        combined_label="{0} (combined fit)",
    )
    print(f"  Jet comparison plots → {comp_dir}/")

    # ── Standard vs fromele comparison plots ─────────────────────────────────
    fromele_dir = f"{plot_dir}/fromele_comparisons"
    _comparison_plot(
        FROMELE_PAIRS, "standard vs fromele", fromele_dir,
        label_a="{0} (standard)", label_b="{0} (fromele)",
    )
    print(f"  Fromele comparison plots → {fromele_dir}/")

    # ── JSON ──────────────────────────────────────────────────────────────────
    class _NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.bool_):    return bool(o)
            return super().default(o)

    json_path = f"response/functions/dcb_results_ecm{ecm}.json"
    with open(json_path, "w") as fj:
        json.dump(results, fj, indent=2, cls=_NpEncoder)

    # ── C++ header ────────────────────────────────────────────────────────────
    cpp = [
        "#pragma once",
        "// Auto-generated by fit_dcb_resolutions.py (v2)",
        f"// Fitted from wzp6_ee_munumuqq_noCut_ecm{ecm}.",
        "//",
        "// Usage:",
        "//   single DCB:          dcb_neg2logpdf(x, DCB_<BRANCH>)",
        "//   DCB+Gaussian:        dcb_gauss_neg2logpdf(x, DCBG_<BRANCH>)",
        "//   ExpLeft+Gaussian:    dcb_expleft_gauss_neg2logpdf(x, DCBELG_<BRANCH>)",
        "// All return -2*log(PDF(x)), drop-in for gaussian pull^2.",
        "",
        "#include <cmath>",
        "#include <algorithm>",
        "",
        "namespace WWFunctions {",
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
        "// ── Fitted parameters ────────────────────────────────────────────────",
    ]

    for bname, p in sorted(results.items()):
        tag  = bname.upper()
        note = f"// chi2/ndf={p['chi2_ndof']:.2f}  {'OK' if p['fit_ok'] else 'WARN'}"
        if p["model"] == "dcb2g":
            cpp.append(
                f"constexpr DcbGaussParams DCBG_{tag} = "
                f"{{ {p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['aL']:.6f}, {p['nL']:.6f}, {p['aR']:.6f}, {p['nR']:.6f}, "
                f"{p['f_wide']:.6f}, {p['mu_wide']:+.6f}, {p['sigma_wide']:.6f}, "
                f"{p['norm']:.10e} }};  {note}"
            )
        elif p["model"] == "expleft2g":
            cpp.append(
                f"constexpr DcbExpLeftGaussParams DCBELG_{tag} = "
                f"{{ {p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['aL']:.6f}, {p['kL']:.6f}, {p['aR']:.6f}, {p['nR']:.6f}, "
                f"{p['f_wide']:.6f}, {p['mu_wide']:+.6f}, {p['sigma_wide']:.6f}, "
                f"{p['norm']:.10e} }};  {note}"
            )
        else:
            cpp.append(
                f"constexpr DcbParams DCB_{tag} = "
                f"{{ {p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['aL']:.6f}, {p['nL']:.6f}, {p['aR']:.6f}, {p['nR']:.6f}, "
                f"{p['norm']:.10e} }};  {note}"
            )

    cpp += ["", "} // namespace WWFunctions", ""]
    h_path = f"response/functions/dcb_params_ecm{ecm}.h"
    with open(h_path, "w") as fh:
        fh.write("\n".join(cpp))

    print(f"\nDone ecm{ecm}. Plots → {plot_dir}/  |  C++ → {h_path}  |  JSON → {json_path}")
