#!/usr/bin/env python3
"""
Plot kinfit output variables (step2 treemaker) per ECM and compare across ECMs.
For branches with a fitted resolution PDF, overlay the input functional form
from the JSON produced by fit_resolutions.py.

Outputs:
  response/plots/kinfit_vars/ecm{N}/{branch}.{png,pdf}
  response/plots/kinfit_vars/ecm_comparison/{branch}.{png,pdf}
"""

import os, json, math
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad as _quad

# ── Constants ────────────────────────────────────────────────────────────────

ECM_LIST    = [157, 160, 163]
INFILE_TMPL = "outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
JSON_TMPL   = "response/functions/dcb_results_ecm{ecm}.json"
OUTDIR_BASE = "response/plots/kinfit_vars"

ECM_COLORS  = {157: "tab:purple", 160: "tab:orange", 163: "tab:cyan"}

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
    # WW system constraints
    "kinfit_WW_px":          "px_tot_gen",
    "kinfit_WW_py":          "py_tot_gen",
    "kinfit_WW_pz":          "pz_tot_gen",
    "kinfit_WW_m_minus_ecm": "m_gen_lnuqq_minus_ecm",
}

# All kinfit branches from treemaker_lnuqq_step2.py
KINFIT_BRANCHES = [
    "kinfit_mW", "kinfit_gW",
    "kinfit_s1", "kinfit_s2", "kinfit_sl", "kinfit_sn",
    "kinfit_t1", "kinfit_t2", "kinfit_tn", "kinfit_tl",
    "kinfit_p1", "kinfit_p2", "kinfit_pn", "kinfit_pl",
    "kinfit_chi2", "kinfit_valid",
    "kinfit_mWlep", "kinfit_mWhad",
    "kinfit_pt_j1", "kinfit_pt_j2", "kinfit_pt_lep", "kinfit_pt_nu",
    "kinfit_Wlep_px", "kinfit_Wlep_py", "kinfit_Wlep_pz",
    "kinfit_Whad_px", "kinfit_Whad_py", "kinfit_Whad_pz",
    "kinfit_theta_j1", "kinfit_theta_j2", "kinfit_theta_nu",
    "kinfit_phi_j1", "kinfit_phi_j2", "kinfit_phi_nu",
    "kinfit_deltaP",
    "kinfit_WW_px", "kinfit_WW_py", "kinfit_WW_pz",
    "kinfit_WW_m", "kinfit_WW_m_minus_ecm",
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

def exp_right_gauss(x, N, x_cut, kL, f_wide, mu_w, sigma_w):
    core = np.where(x <= x_cut, np.exp(kL * (x - x_cut)), np.float64(0.0))
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)

def gamma_right_gauss(x, N, x_cut, alpha, beta, f_wide, mu_w, sigma_w):
    y = x_cut - x
    log_core = np.where(
        (x <= x_cut) & (y > 0),
        (alpha - 1.0) * np.log(np.maximum(y, 1e-30)) - beta * y,
        np.float64(_LOG_MIN)
    )
    core = np.exp(np.minimum(np.maximum(log_core, _LOG_MIN), _LOG_MAX))
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


# ── Build normalized PDF callable from JSON params ────────────────────────────

def _make_pdf(p):
    """Return a callable f(x) → normalized PDF value, from JSON param dict."""
    norm = p["norm"]
    model = p["model"]

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

    elif model == "exprcut2g":
        xc, kL = p["x_cut"], p["kL"]
        fw, mw, sw = p["f_wide"], p["mu_wide"], p["sigma_wide"]
        def fn(x): return norm * exp_right_gauss(x, 1.0, xc, kL, fw, mw, sw)

    elif model == "gamright2g":
        xc, alpha, beta = p["x_cut"], p["alpha"], p["beta"]
        fw, mw, sw = p["f_wide"], p["mu_wide"], p["sigma_wide"]
        def fn(x): return norm * gamma_right_gauss(x, 1.0, xc, alpha, beta, fw, mw, sw)

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
_FIXED_RANGE = {
    "kinfit_mW":    (100, 50,   110),
    "kinfit_mWlep": (100, 50,   110),
    "kinfit_mWhad": (100, 50,   110),
    "kinfit_gW":    (100,  0,     5),
    "kinfit_valid": (  3, -0.5,  2.5),
    "kinfit_pt_j1": (100,  0,   100),
    "kinfit_pt_j2": (100,  0,   100),
    "kinfit_pt_lep":(100,  0,   100),
    "kinfit_pt_nu": (100,  0,   100),
    "kinfit_Wlep_px": (100, -100, 100),
    "kinfit_Wlep_py": (100, -100, 100),
    "kinfit_Wlep_pz": (100, -100, 100),
    "kinfit_Whad_px": (100, -100, 100),
    "kinfit_Whad_py": (100, -100, 100),
    "kinfit_Whad_pz": (100, -100, 100),
    "kinfit_theta_j1":  (100, 0,   3.2),
    "kinfit_theta_j2":  (100, 0,   3.2),
    "kinfit_theta_nu":  (100, 0,   3.2),
    "kinfit_phi_j1":    (100, -3.2, 3.2),
    "kinfit_phi_j2":    (100, -3.2, 3.2),
    "kinfit_phi_nu":    (100, -3.2, 3.2),
    # WW system post-fit: mass near ECM, mass-minus-ECM near 0 (slightly below, ISR)
    "kinfit_WW_m":           (100, 150, 170),
    "kinfit_WW_m_minus_ecm": (100,  -8,   1),
}

def _auto_range(vals):
    """Percentile-clipped range with a 10 % margin."""
    lo, hi = np.percentile(vals[np.isfinite(vals)], [0.5, 99.5])
    margin = max(abs(hi - lo) * 0.1, abs(lo) * 1e-3, 1e-12)
    return lo - margin, hi + margin

def _pdf_natural_range(p, nsigma=5):
    """±nsigma range covering the PDF core + power-law shoulders + wide component."""
    model = p.get("model", "dcb")
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
        # Use PDF range when data is much narrower than the prior (fit has over-constrained)
        data_w = data_hi - data_lo
        pdf_w  = pdf_hi  - pdf_lo
        if data_w < 0.2 * pdf_w:
            return (100, pdf_lo, pdf_hi)
        return (100, data_lo, data_hi)

    if data_lo is not None:
        return (100, data_lo, data_hi)
    return (100, -5, 5)


# ── Per-ECM single-branch plot ────────────────────────────────────────────────

def _pdf_integ_full(pdf_fn, pdf_params, nsigma=10):
    """Integrate pdf_fn over the full natural range (nsigma wide) for normalization."""
    lo, hi = _pdf_natural_range(pdf_params, nsigma=nsigma)
    xnorm = np.linspace(lo, hi, 5000)
    return float(np.trapz(pdf_fn(xnorm), xnorm))


def _plot_branch(ax, bname, vals, ecm, pdf_fn=None, pdf_params=None, color="steelblue"):
    nbins, xlo, xhi = _binning(bname, vals, pdf_params=pdf_params)
    vals_c = vals[(vals >= xlo) & (vals <= xhi)]
    counts, edges = np.histogram(vals_c, bins=nbins, range=(xlo, xhi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = np.diff(edges)[0]

    density = counts / max(counts.sum() * bw, 1e-300)
    ax.bar(centers, density, width=bw, color=color, alpha=0.55, label=f"ecm{ecm} data")

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
    out_dir = f"{OUTDIR_BASE}/ecm{ecm}"
    os.makedirs(out_dir, exist_ok=True)

    for bname, vals in branches_data.items():
        resol_name = KINFIT_TO_RESOL.get(bname)
        pdf_fn = None
        pdf_params = None
        if resol_name and resol_name in json_results:
            pdf_params = json_results[resol_name]
            pdf_fn = _make_pdf(pdf_params)

        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        _plot_branch(ax, bname, vals, ecm, pdf_fn=pdf_fn, pdf_params=pdf_params)
        title = f"{bname}  [ecm{ecm}]"
        if resol_name:
            title += f"\ninput PDF: {resol_name}"
            if resol_name in json_results:
                title += f"  ({json_results[resol_name]['model']}, χ²/ndf={json_results[resol_name]['chi2_ndof']:.2f})"
        ax.set_title(title, fontsize=10)

        for fmt in ("png", "pdf"):
            fig.savefig(f"{out_dir}/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    print(f"  [ecm{ecm}]  per-ECM plots → {out_dir}/")


# ── ECM comparison plots ──────────────────────────────────────────────────────

def plot_ecm_comparison(all_data, all_json):
    """One plot per branch showing all ECMs overlaid (normalized)."""
    out_dir = f"{OUTDIR_BASE}/ecm_comparison"
    os.makedirs(out_dir, exist_ok=True)

    branch_names = sorted({b for data in all_data.values() for b in data})

    for bname in branch_names:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

        resol_name = KINFIT_TO_RESOL.get(bname)

        # Determine pdf_params from first available ECM for binning heuristic.
        pdf_params_ref = None
        if resol_name:
            for _e in ECM_LIST:
                if _e in all_json and resol_name in all_json[_e]:
                    pdf_params_ref = all_json[_e][resol_name]
                    break

        # Union range across all ECMs so comparison is on a common axis.
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

            # overlay PDF per ECM (same colour, dashed) — integrate over full natural range
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

    print(f"  ECM comparison plots → {out_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTDIR_BASE, exist_ok=True)

    all_data = {}   # ecm → {bname: array}
    all_json = {}   # ecm → {resol_bname: params_dict}

    for ecm in ECM_LIST:
        infile = INFILE_TMPL.format(ecm=ecm)
        if not os.path.exists(infile):
            print(f"WARNING: {infile} not found — skipping ecm{ecm}")
            continue

        print(f"\n{'='*55}\nECM {ecm} GeV  —  {infile}\n{'='*55}")

        with uproot.open(infile) as f:
            tree = f["events"]
            available = set(tree.keys())
            to_load = [b for b in KINFIT_BRANCHES if b in available]
            missing  = [b for b in KINFIT_BRANCHES if b not in available]
            if missing:
                print(f"  WARNING: branches not in tree: {missing}")
            raw = tree.arrays(to_load, library="np")

        branches_data = {}
        for bname in to_load:
            arr = np.asarray(raw[bname], dtype=float).ravel()
            branches_data[bname] = arr[np.isfinite(arr)]

        all_data[ecm] = branches_data

        json_path = JSON_TMPL.format(ecm=ecm)
        if os.path.exists(json_path):
            with open(json_path) as fj:
                all_json[ecm] = json.load(fj)
            print(f"  Loaded JSON: {json_path}")
        else:
            print(f"  WARNING: JSON not found ({json_path}) — no PDF overlay for ecm{ecm}")
            all_json[ecm] = {}

        plot_per_ecm(ecm, branches_data, all_json[ecm])

    plot_ecm_comparison(all_data, all_json)
    print(f"\nDone. All plots in {OUTDIR_BASE}/")


if __name__ == "__main__":
    main()
