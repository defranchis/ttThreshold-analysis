#!/usr/bin/env python3
"""Kinfit diagnostics — write a markdown summary of the per-event fit quality
across the three ECMs.

Reads step2 outputs (one ROOT file per ECM) and the fit_resolutions JSONs
(for prior parameters). Computes:

  - convergence (strict / loose) per ECM, status-code distribution,
    cascade winner-pass histogram, n_passes_run mean;
  - mW and m(WW) posteriors (median / robust σ / closure vs gen);
  - BES nuisance posteriors (σ_post vs σ_prior, ratio, unique values);
  - ISR-via-balance posteriors (μ, robust σ).

Default I/O: reads `outputs/treemaker/lnuqq/step2/semihad/wzp6_*_ecm{ECM}.root`
and JSON `kinfit_inputs/dcb_results_ecm{ECM}.json`; writes
`outputs/diagnostics/kinfit_diagnostics.md`. Override via CLI:

    python3 kinfit_diagnostics.py [--indir DIR] [--out FILE]
"""
import argparse
import json
import os
import subprocess
from datetime import datetime
import numpy as np
import uproot

ECM_LIST = [157, 160, 163]


def robust_sigma(arr):
    """q[16, 84] / 2 — outlier-resistant σ estimate."""
    return float((np.percentile(arr, 84) - np.percentile(arr, 16)) / 2)


def fmt_pct(x): return f"{100*x:.2f}%"


def git_describe():
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--pretty=%h %s"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        return "(unknown)"


def diagnostics_one(ecm, infile, json_path):
    """Returns a dict of summary numbers for one ECM."""
    d = {}
    f = uproot.open(infile)
    t = f["events"]

    # Convergence + status + cascade
    valid = t["kinfit_valid"].array(library="np").astype(bool)
    valid_l = t["kinfit_valid_loose"].array(library="np").astype(bool)
    status = t["kinfit_status"].array(library="np")
    winner = t["kinfit_winner_pass"].array(library="np")
    npasses = t["kinfit_n_passes_run"].array(library="np")
    N = len(valid)
    d["N"] = N
    d["valid_frac"] = valid.mean()
    d["valid_loose_frac"] = valid_l.mean()
    d["status_distrib"] = {
        int(s): int(c) for s, c in zip(*np.unique(status, return_counts=True))
    }
    d["winner_distrib_valid"] = {
        int(p): int(c) for p, c in zip(*np.unique(winner[valid], return_counts=True))
    }
    d["n_passes_mean_valid"] = float(npasses[valid].mean())

    # mW posterior + chi2/ndf
    mw = t["kinfit_mW"].array(library="np")
    chi = t["kinfit_chi2_ndof"].array(library="np")
    d["mW_median"] = float(np.median(mw[valid]))
    d["mW_robust_sigma"] = robust_sigma(mw[valid])
    d["chi2_ndof_median"] = float(np.median(chi[valid]))

    # m(WW) closure
    if "kinfit_WW_m" in t.keys() and "gen_WW_m" in t.keys():
        wwm_kf = t["kinfit_WW_m"].array(library="np")
        wwm_gen = t["gen_WW_m"].array(library="np")
        res = wwm_kf[valid] - wwm_gen[valid]
        d["mWW_closure_mean"] = float(res.mean())
        d["mWW_closure_sigma"] = float(res.std())

    # BES nuisances
    priors = {}
    if json_path and os.path.exists(json_path):
        with open(json_path) as fh:
            priors = json.load(fh)
    bes_post = {}
    for kf_name, prior_key in [
        ("kinfit_bes_m_minus_ecm", "gen_ee_m_minus_ecm"),
        ("kinfit_bes_pz",          "gen_ee_pz"),
    ]:
        if kf_name not in t.keys():
            continue
        a = t[kf_name].array(library="np")[valid]
        sp = priors.get(prior_key, {}).get("sigma")
        bes_post[kf_name] = {
            "mu_post": float(np.mean(a)),
            "sigma_post_robust": robust_sigma(a),
            "sigma_prior": float(sp) if sp is not None else None,
            "n_unique": int(len(np.unique(np.round(a, 8)))),
        }
    d["bes_post"] = bes_post

    # ISR via balance: ISR_p = -WW_p (px,py) and bes_pz - WW_pz (pz)
    isr_post = {}
    if "kinfit_WW_px" in t.keys():
        ww_px = t["kinfit_WW_px"].array(library="np")[valid]
        ww_py = t["kinfit_WW_py"].array(library="np")[valid]
        ww_pz = t["kinfit_WW_pz"].array(library="np")[valid]
        bes_pz_post = (
            t["kinfit_bes_pz"].array(library="np")[valid]
            if "kinfit_bes_pz" in t.keys() else np.zeros_like(ww_pz)
        )
        isr_px = -ww_px
        isr_py = -ww_py
        isr_pz = bes_pz_post - ww_pz
        for name, a in [("isr_px", isr_px), ("isr_py", isr_py), ("isr_pz", isr_pz)]:
            isr_post[name] = {
                "mu_post": float(np.mean(a)),
                "sigma_post_robust": robust_sigma(a),
            }
    d["isr_post"] = isr_post

    return d


def render_markdown(data, args):
    """Format the per-ECM dicts into one markdown report."""
    lines = []
    lines.append(f"# Kinfit diagnostics — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"- input dir: `{args.indir}`")
    lines.append(f"- HEAD: {git_describe()}")
    lines.append("")
    lines.append("## Convergence")
    lines.append("")
    lines.append("| ECM | N | valid (strict) | valid (loose) | status distrib | n_passes mean |")
    lines.append("|---|---|---|---|---|---|")
    for ecm in ECM_LIST:
        d = data.get(ecm)
        if d is None:
            lines.append(f"| {ecm} | — | (missing) | | | |")
            continue
        sd = ", ".join(f"{k}:{v}" for k, v in sorted(d['status_distrib'].items()))
        lines.append(
            f"| {ecm} | {d['N']:,} | {fmt_pct(d['valid_frac'])} "
            f"| {fmt_pct(d['valid_loose_frac'])} | {sd} | {d['n_passes_mean_valid']:.2f} |"
        )

    lines.append("")
    lines.append("## Cascade winner pass (valid events)")
    lines.append("")
    lines.append("Pass legend: 1=Migrad-natural, 2=Migrad-swapped, 3=Simplex+Migrad-natural,")
    lines.append("4=Simplex+Migrad-swapped, 5=Hesse-refresh, 6=random-restart Migrad,")
    lines.append("7=Minimize-natural, 8=Minimize-swapped.")
    lines.append("")
    all_passes = sorted({p for d in data.values() for p in d.get("winner_distrib_valid", {})})
    if all_passes:
        header = "| ECM |" + "".join(f" pass{p} |" for p in all_passes)
        sep = "|---|" + "---|" * len(all_passes)
        lines.append(header)
        lines.append(sep)
        for ecm in ECM_LIST:
            d = data.get(ecm)
            if d is None:
                continue
            row = f"| {ecm} |"
            wp = d["winner_distrib_valid"]
            for p in all_passes:
                row += f" {wp.get(p, 0):,} |"
            lines.append(row)

    lines.append("")
    lines.append("## mW and m(WW) posterior")
    lines.append("")
    lines.append("| ECM | median(mW) [GeV] | robust σ(mW) [GeV] | median(χ²/ndf) | ⟨m(WW)kf−gen⟩ [GeV] | σ(closure) [GeV] |")
    lines.append("|---|---|---|---|---|---|")
    for ecm in ECM_LIST:
        d = data.get(ecm)
        if d is None:
            continue
        line = (f"| {ecm} | {d['mW_median']:.3f} | {d['mW_robust_sigma']:.3f} "
                f"| {d['chi2_ndof_median']:.3f} ")
        if "mWW_closure_mean" in d:
            line += f"| {d['mWW_closure_mean']:+.3f} | {d['mWW_closure_sigma']:.3f} |"
        else:
            line += "| — | — |"
        lines.append(line)

    lines.append("")
    lines.append("## BES nuisance posteriors (vs prior)")
    lines.append("")
    lines.append("| ECM | branch | μ post [MeV] | σ post robust [MeV] | σ prior [MeV] | ratio | unique values |")
    lines.append("|---|---|---|---|---|---|---|")
    for ecm in ECM_LIST:
        d = data.get(ecm)
        if d is None:
            continue
        for bn, bp in d.get("bes_post", {}).items():
            sp = bp["sigma_prior"]
            ratio = (bp["sigma_post_robust"] / sp) if sp else float("nan")
            sp_txt = f"{sp*1000:.2f}" if sp else "—"
            lines.append(
                f"| {ecm} | `{bn}` | {bp['mu_post']*1000:+.3f} "
                f"| {bp['sigma_post_robust']*1000:.3f} | {sp_txt} | {ratio:.3f} "
                f"| {bp['n_unique']:,} |"
            )

    lines.append("")
    lines.append("## ISR via balance — post-fit")
    lines.append("")
    lines.append("| ECM | branch | μ post [MeV] | σ post robust [MeV] |")
    lines.append("|---|---|---|---|")
    for ecm in ECM_LIST:
        d = data.get(ecm)
        if d is None:
            continue
        for bn, bp in d.get("isr_post", {}).items():
            lines.append(
                f"| {ecm} | `{bn}` | {bp['mu_post']*1000:+.3f} "
                f"| {bp['sigma_post_robust']*1000:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", default="outputs/treemaker/lnuqq/step2/semihad",
                   help="step2 output directory (default: production path)")
    p.add_argument("--json", default="kinfit_inputs/dcb_results_ecm{ecm}.json",
                   help="JSON template with {ecm} placeholder")
    p.add_argument("--out", default="outputs/diagnostics/kinfit_diagnostics.md",
                   help="markdown output path")
    args = p.parse_args()

    data = {}
    for ecm in ECM_LIST:
        infile = f"{args.indir}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
        if not os.path.exists(infile):
            print(f"[skip] ecm{ecm}: {infile} not found")
            continue
        json_path = args.json.format(ecm=ecm)
        try:
            data[ecm] = diagnostics_one(ecm, infile, json_path)
            print(f"[ok]  ecm{ecm}: N={data[ecm]['N']}  "
                  f"valid={fmt_pct(data[ecm]['valid_frac'])}")
        except Exception as e:
            print(f"[err] ecm{ecm}: {e}")

    md = render_markdown(data, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(md)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
