import math
import uproot
import ROOT
from array import array
import numpy as np
import vector
from iminuit import Minuit

ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

# ── global constants ───────────────────────────────────────────────────────
ECM            = 160.0
sigma_sqrtS    = 0.12
MAX_EVENTS     = 1000

jet1_theta_rms = 0.05
jet1_phi_rms   = 0.05
jet2_theta_rms = 0.05
jet2_phi_rms   = 0.06
met_theta_rms  = 0.04
met_phi_rms    = 0.05

# jet energy scale: bias at 1.01 (1% upward shift from resolution studies),
# MET scale unbiased at 1.0; sigmas from in-situ resolution measurements
JET_SCALE_BIAS  = 1.01
JET_SCALE_SIGMA = 0.05
LEP_SCALE_BIAS  = 1.0
LEP_SCALE_SIGMA = 0.002
MET_SCALE_BIAS  = 1.0
MET_SCALE_SIGMA = 0.015
# large weight to enforce Σp≈0 as a hard constraint relative to BW/resolution terms
MOMENTUM_PENALTY_WEIGHT = 100

fname = "/afs/cern.ch/work/a/anmehta/public/FCC_ver2/FCCAnalyses/ttThreshold-analysis/outputs/treemaker/lnuqq/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"

# ── resolution histograms via RDataFrame ───────────────────────────────────
outFile = ROOT.TFile("resolution_fits.root", "RECREATE")
df = ROOT.RDataFrame("events", fname)

h_mjj                 = df.Histo1D(("h_mjj",               "m_{jj} resolution; m_{jj}^{reco}-m_{jj}^{gen} [GeV];Events", 50, -5,   5),   "diff_RG_m_qq")
h_mlnu                = df.Histo1D(("h_mlnu",               "m_{lnu} resolution; m_{lnu}^{reco}-m_{lnu}^{gen} [GeV];Events", 50, -5, 5),   "diff_RG_m_lnu")
h_mlnujj              = df.Histo1D(("h_mlnujj",             "m_{lnuqq} resolution; m_{lnuqq}^{reco}-m_{lnuqq}^{gen} [GeV];Events", 50, -5, 5), "diff_RG_m_lnuqq")
h_p_iso_lnuexcljj     = df.Histo1D(("h_p_iso_lnuexcljj",   "", 50,  -5,   5),   "p_iso_lnuexcljj")
h_e_iso_lnuexcljj     = df.Histo1D(("h_e_iso_lnuexcljj",   "", 100,  0, 200),   "e_iso_lnuexcljj")
h_plnu                = df.Histo1D(("h_plnu",               "", 50,  -5,   5),   "diff_RG_p_lnu")
h_pjj                 = df.Histo1D(("h_pjj",               "", 50,  -5,   5),   "diff_RG_p_qq")
h_res_jet1_qq_fromele = df.Histo1D(("h_res_jet1_qq_fromele","", 50,  -0.5, 0.5), "res_jet1_qq_fromele")
h_res_jet2_qq_fromele = df.Histo1D(("h_res_jet2_qq_fromele","", 50,  -0.5, 0.5), "res_jet2_qq_fromele")
h_lep_res             = df.Histo1D(("h_lep_res",            "", 50,  -0.2, 0.2), "lep_res")
h_mp_res              = df.Histo1D(("h_met_res",            "", 50,  -0.3, 0.3), "missing_p_res")
h_jet1_theta_res      = df.Histo1D(("h_jet1_theta_res",     "", 50,  -0.4, 0.4), "jet1_dtheta")
h_jet2_theta_res      = df.Histo1D(("h_jet2_theta_res",     "", 50,  -0.4, 0.4), "jet2_dtheta")
h_jet1_phi_res        = df.Histo1D(("h_jet1_phi_res",       "", 50,  -0.4, 0.4), "jet1_dphi")
h_jet2_phi_res        = df.Histo1D(("h_jet2_phi_res",       "", 50,  -0.4, 0.4), "jet2_dphi")
h_met_phi_res         = df.Histo1D(("h_met_phi_res",        "", 50,  -0.4, 0.4), "met_dphi")
h_met_theta_res       = df.Histo1D(("h_met_theta_res",      "", 50,  -0.4, 0.4), "met_dtheta")

sigma_lep = h_lep_res.GetRMS()
sigma_mp  = h_mp_res.GetRMS()
sigma_j1  = h_res_jet1_qq_fromele.GetRMS()
sigma_j2  = h_res_jet2_qq_fromele.GetRMS()
mean_lep  = h_lep_res.GetMean()
mean_mp   = h_mp_res.GetMean()
mean_j1   = h_res_jet1_qq_fromele.GetMean()
mean_j2   = h_res_jet2_qq_fromele.GetMean()

for h in [h_mjj, h_mlnu, h_mlnujj, h_pjj, h_plnu,
          h_p_iso_lnuexcljj, h_e_iso_lnuexcljj,
          h_res_jet1_qq_fromele, h_res_jet2_qq_fromele,
          h_lep_res, h_mp_res]:
    h.GetValue().Write()

# ── pull statistics: precomputed once, not per event ──────────────────────
mean_s1  = 1.0 / (1.0 + mean_j1)
mean_s2  = 1.0 / (1.0 + mean_j2)
mean_sl  = 1.0 / (1.0 + mean_lep)
mean_sn  = 1.0 / (1.0 + mean_mp)
sigma_s1 = sigma_j1  / (1.0 + mean_j1)**2
sigma_s2 = sigma_j2  / (1.0 + mean_j2)**2
sigma_sl = sigma_lep / (1.0 + mean_lep)**2
sigma_sn = sigma_mp  / (1.0 + mean_mp)**2

h_pull_s1 = ROOT.TH1F("h_pull_s1", "Pull s1", 50, -5, 5)
h_pull_s2 = ROOT.TH1F("h_pull_s2", "Pull s2", 50, -5, 5)
h_pull_sl = ROOT.TH1F("h_pull_sl", "Pull sl", 50, -5, 5)
h_pull_sn = ROOT.TH1F("h_pull_sn", "Pull sn", 50, -5, 5)

# ── output tree ────────────────────────────────────────────────────────────
def _branch(tree, name, buf, spec):
    tree.Branch(name, buf, spec)
    return buf

tree_out = ROOT.TTree("fitResults", "fit results")

mW_arr  = _branch(tree_out, "mW",  array('f', [0.]), "mW/F")
gW_arr  = _branch(tree_out, "gW",  array('f', [0.]), "gW/F")
s1_arr  = _branch(tree_out, "s1",  array('f', [0.]), "s1/F")
s2_arr  = _branch(tree_out, "s2",  array('f', [0.]), "s2/F")
sl_arr  = _branch(tree_out, "sl",  array('f', [0.]), "sl/F")
sn_arr  = _branch(tree_out, "sn",  array('f', [0.]), "sn/F")
chi2_arr = _branch(tree_out, "chi2", array('f', [0.]), "chi2/F")
p1_arr  = _branch(tree_out, "p1",  array('f', [0.]), "p1/F")
p2_arr  = _branch(tree_out, "p2",  array('f', [0.]), "p2/F")
pn_arr  = _branch(tree_out, "pn",  array('f', [0.]), "pn/F")
t1_arr  = _branch(tree_out, "t1",  array('f', [0.]), "t1/F")
t2_arr  = _branch(tree_out, "t2",  array('f', [0.]), "t2/F")
tn_arr  = _branch(tree_out, "tn",  array('f', [0.]), "tn/F")

pt_lep_postfit  = _branch(tree_out, "pt_lep_postfit",  array('f', [0.]), "pt_lep_postfit/F")
pt_j1_postfit   = _branch(tree_out, "pt_j1_postfit",   array('f', [0.]), "pt_j1_postfit/F")
pt_j2_postfit   = _branch(tree_out, "pt_j2_postfit",   array('f', [0.]), "pt_j2_postfit/F")
pt_nu_postfit   = _branch(tree_out, "pt_nu_postfit",   array('f', [0.]), "pt_nu_postfit/F")
mWlep_postfit   = _branch(tree_out, "mWlep_postfit",   array('f', [0.]), "mWlep_postfit/F")
mWhad_postfit   = _branch(tree_out, "mWhad_postfit",   array('f', [0.]), "mWhad_postfit/F")
deltaP_postfit  = _branch(tree_out, "deltaP_postfit",  array('f', [0.]), "deltaP_postfit/F")
theta_j1_postfit = _branch(tree_out, "theta_j1_postfit", array('f', [0.]), "theta_j1_postfit/F")
theta_j2_postfit = _branch(tree_out, "theta_j2_postfit", array('f', [0.]), "theta_j2_postfit/F")
theta_nu_postfit = _branch(tree_out, "theta_nu_postfit", array('f', [0.]), "theta_nu_postfit/F")
phi_j1_postfit   = _branch(tree_out, "phi_j1_postfit",   array('f', [0.]), "phi_j1_postfit/F")
phi_j2_postfit   = _branch(tree_out, "phi_j2_postfit",   array('f', [0.]), "phi_j2_postfit/F")
phi_nu_postfit   = _branch(tree_out, "phi_nu_postfit",   array('f', [0.]), "phi_nu_postfit/F")
Wlep_px_postfit = _branch(tree_out, "Wlep_px_postfit", array('f', [0.]), "Wlep_px_postfit/F")
Wlep_py_postfit = _branch(tree_out, "Wlep_py_postfit", array('f', [0.]), "Wlep_py_postfit/F")
Wlep_pz_postfit = _branch(tree_out, "Wlep_pz_postfit", array('f', [0.]), "Wlep_pz_postfit/F")
Whad_px_postfit = _branch(tree_out, "Whad_px_postfit", array('f', [0.]), "Whad_px_postfit/F")
Whad_py_postfit = _branch(tree_out, "Whad_py_postfit", array('f', [0.]), "Whad_py_postfit/F")
Whad_pz_postfit = _branch(tree_out, "Whad_pz_postfit", array('f', [0.]), "Whad_pz_postfit/F")

pt_lep_prefit   = _branch(tree_out, "pt_lep_prefit",   array('f', [0.]), "pt_lep_prefit/F")
pt_j1_prefit    = _branch(tree_out, "pt_j1_prefit",    array('f', [0.]), "pt_j1_prefit/F")
pt_j2_prefit    = _branch(tree_out, "pt_j2_prefit",    array('f', [0.]), "pt_j2_prefit/F")
pt_nu_prefit    = _branch(tree_out, "pt_nu_prefit",    array('f', [0.]), "pt_nu_prefit/F")
mWlep_prefit    = _branch(tree_out, "mWlep_prefit",    array('f', [0.]), "mWlep_prefit/F")
mWhad_prefit    = _branch(tree_out, "mWhad_prefit",    array('f', [0.]), "mWhad_prefit/F")
deltaP_prefit   = _branch(tree_out, "deltaP_prefit",   array('f', [0.]), "deltaP_prefit/F")
theta_j1_prefit  = _branch(tree_out, "theta_j1_prefit",  array('f', [0.]), "theta_j1_prefit/F")
theta_j2_prefit  = _branch(tree_out, "theta_j2_prefit",  array('f', [0.]), "theta_j2_prefit/F")
theta_nu_prefit  = _branch(tree_out, "theta_nu_prefit",  array('f', [0.]), "theta_nu_prefit/F")
phi_j1_prefit    = _branch(tree_out, "phi_j1_prefit",    array('f', [0.]), "phi_j1_prefit/F")
phi_j2_prefit    = _branch(tree_out, "phi_j2_prefit",    array('f', [0.]), "phi_j2_prefit/F")
phi_nu_prefit    = _branch(tree_out, "phi_nu_prefit",    array('f', [0.]), "phi_nu_prefit/F")
Wlep_px_prefit  = _branch(tree_out, "Wlep_px_prefit",  array('f', [0.]), "Wlep_px_prefit/F")
Wlep_py_prefit  = _branch(tree_out, "Wlep_py_prefit",  array('f', [0.]), "Wlep_py_prefit/F")
Wlep_pz_prefit  = _branch(tree_out, "Wlep_pz_prefit",  array('f', [0.]), "Wlep_pz_prefit/F")
Whad_px_prefit  = _branch(tree_out, "Whad_px_prefit",  array('f', [0.]), "Whad_px_prefit/F")
Whad_py_prefit  = _branch(tree_out, "Whad_py_prefit",  array('f', [0.]), "Whad_py_prefit/F")
Whad_pz_prefit  = _branch(tree_out, "Whad_pz_prefit",  array('f', [0.]), "Whad_pz_prefit/F")


# ── kinematics helpers ─────────────────────────────────────────────────────

def _vec_from_spherical(p, theta, phi):
    st = math.sin(theta)
    ct = math.cos(theta)
    return vector.obj(px=p * st * math.cos(phi),
                      py=p * st * math.sin(phi),
                      pz=p * ct,
                      e=p)

def _build_fitted_particles(event, s1, s2, sl, sn, t1, t2, tn, p1, p2, pn):
    j1, j2, lep, nu = event
    j1f  = _vec_from_spherical(j1.p * s1, j1.theta + t1 * jet1_theta_rms, j1.phi + p1 * jet1_phi_rms)
    j2f  = _vec_from_spherical(j2.p * s2, j2.theta + t2 * jet2_theta_rms, j2.phi + p2 * jet2_phi_rms)
    nuf  = _vec_from_spherical(nu.p * sn, nu.theta + tn * met_theta_rms,   nu.phi + pn * met_phi_rms)
    lepf = lep * sl
    return j1f, j2f, lepf, nuf


def breit_wigner(m, MW, GW):
    mw_gw = MW * GW
    return mw_gw / ((m*m - MW*MW)**2 + mw_gw**2)


def event_chi2(scales, event, mW, gW):
    s1, s2, sl, sn, t1, t2, tn, p1, p2, pn = scales
    j1f, j2f, lepf, nuf = _build_fitted_particles(event, s1, s2, sl, sn, t1, t2, tn, p1, p2, pn)
    Wh  = j1f + j2f
    Wl  = lepf + nuf
    WW  = Wh + Wl
    bw  = -2.0 * (math.log(breit_wigner(Wh.mass, mW, gW)) +
                  math.log(breit_wigner(Wl.mass, mW, gW)))
    cons = ((WW.E - ECM)**2 / sigma_sqrtS**2 +
            ((Wl.px + Wh.px)**2 + (Wl.py + Wh.py)**2 + (Wl.pz + Wh.pz)**2) * MOMENTUM_PENALTY_WEIGHT)
    res  = ((s1 - JET_SCALE_BIAS)**2 / JET_SCALE_SIGMA**2 +
            (s2 - JET_SCALE_BIAS)**2 / JET_SCALE_SIGMA**2 +
            (sl - LEP_SCALE_BIAS)**2 / LEP_SCALE_SIGMA**2 +
            (sn - MET_SCALE_BIAS)**2 / MET_SCALE_SIGMA**2)
    angular = pn**2 + tn**2 + p1**2 + p2**2 + t1**2 + t2**2
    return bw + cons + res + angular


def fit_event(event):
    def f(s1, s2, sl, sn, t1, t2, tn, p1, p2, pn, mW, gW):
        return event_chi2((s1, s2, sl, sn, t1, t2, tn, p1, p2, pn), event, mW, gW)
    m = Minuit(f, s1=1, s2=1, sl=1, sn=1, t1=0, t2=1, tn=1, p1=1, p2=1, pn=1,
               mW=80.419, gW=2.049)
    m.limits["mW"] = (0, 200)
    m.limits["gW"] = (0, 10)
    m.fixed["gW"]  = True
    m.tol          = 1e-6
    m.strategy     = 2
    m.errordef     = Minuit.LEAST_SQUARES
    m.migrad(ncall=5000)
    m.migrad()
    if m.valid:
        print("mW =", m.values["mW"])
    return m


# ── load events (only up to MAX_EVENTS to avoid fitting unused data) ───────
arrays = uproot.open(fname)["events"].arrays([
    "jet1_pt", "jet1_eta", "jet1_phi",
    "jet2_pt", "jet2_eta", "jet2_phi",
    "Isolep_pt", "Isolep_eta", "Isolep_phi",
    "missing_p", "missing_p_phi", "missing_p_eta", "missing_pt",
], library="np", entry_stop=MAX_EVENTS)

N = len(arrays["jet1_pt"])
print("Loaded", N, "events")

# ── fit loop ───────────────────────────────────────────────────────────────
for i in range(N):
    j1  = vector.obj(pt=arrays["jet1_pt"][i],    eta=arrays["jet1_eta"][i],      phi=arrays["jet1_phi"][i],      mass=0)
    j2  = vector.obj(pt=arrays["jet2_pt"][i],    eta=arrays["jet2_eta"][i],      phi=arrays["jet2_phi"][i],      mass=0)
    lep = vector.obj(pt=arrays["Isolep_pt"][i],  eta=arrays["Isolep_eta"][i],    phi=arrays["Isolep_phi"][i],    mass=0)
    nu  = vector.obj(pt=arrays["missing_pt"][i], eta=arrays["missing_p_eta"][i], phi=arrays["missing_p_phi"][i], mass=0)
    event = (j1, j2, lep, nu)

    m = fit_event(event)
    if not m.valid or m.fval > 200:
        print("not saving", m.valid, "chi2", m.fval)
        continue

    vals = m.values
    s1, s2, sl, sn = vals["s1"], vals["s2"], vals["sl"], vals["sn"]
    t1, t2, tn     = vals["t1"], vals["t2"], vals["tn"]
    p1, p2, pn     = vals["p1"], vals["p2"], vals["pn"]

    mW_arr[0],  gW_arr[0]  = vals["mW"], vals["gW"]
    s1_arr[0],  s2_arr[0]  = s1, s2
    sl_arr[0],  sn_arr[0]  = sl, sn
    p1_arr[0],  p2_arr[0],  pn_arr[0] = p1, p2, pn
    t1_arr[0],  t2_arr[0],  tn_arr[0] = t1, t2, tn
    chi2_arr[0] = m.fval

    h_pull_s1.Fill((s1 - mean_s1) / sigma_s1)
    h_pull_s2.Fill((s2 - mean_s2) / sigma_s2)
    h_pull_sl.Fill((sl - mean_sl) / sigma_sl)
    h_pull_sn.Fill((sn - mean_sn) / sigma_sn)

    # post-fit kinematics (reuses _build_fitted_particles, no duplication)
    j1f, j2f, lepf, nuf = _build_fitted_particles(event, s1, s2, sl, sn, t1, t2, tn, p1, p2, pn)
    Wh = j1f + j2f
    Wl = lepf + nuf

    mWlep_postfit[0],   mWhad_postfit[0]   = Wl.mass, Wh.mass
    Wlep_px_postfit[0], Wlep_py_postfit[0], Wlep_pz_postfit[0] = Wl.px, Wl.py, Wl.pz
    Whad_px_postfit[0], Whad_py_postfit[0], Whad_pz_postfit[0] = Wh.px, Wh.py, Wh.pz
    pt_j1_postfit[0],   pt_j2_postfit[0]   = j1f.pt,  j2f.pt
    pt_lep_postfit[0],  pt_nu_postfit[0]   = lepf.pt, nuf.pt
    theta_j1_postfit[0], theta_j2_postfit[0], theta_nu_postfit[0] = j1f.theta, j2f.theta, nuf.theta
    phi_j1_postfit[0],   phi_j2_postfit[0],   phi_nu_postfit[0]   = j1f.phi,   j2f.phi,   nuf.phi
    deltaP_postfit[0] = math.sqrt((Wl.px+Wh.px)**2 + (Wl.py+Wh.py)**2 + (Wl.pz+Wh.pz)**2)

    # pre-fit kinematics
    Whad = j1 + j2
    Wlep = lep + nu

    mWlep_prefit[0],   mWhad_prefit[0]   = Wlep.mass, Whad.mass
    Wlep_px_prefit[0], Wlep_py_prefit[0], Wlep_pz_prefit[0] = Wlep.px, Wlep.py, Wlep.pz
    Whad_px_prefit[0], Whad_py_prefit[0], Whad_pz_prefit[0] = Whad.px, Whad.py, Whad.pz
    pt_j1_prefit[0],   pt_j2_prefit[0]   = j1.pt,  j2.pt
    pt_lep_prefit[0],  pt_nu_prefit[0]   = lep.pt, nu.pt
    theta_j1_prefit[0], theta_j2_prefit[0], theta_nu_prefit[0] = j1.theta,  j2.theta,  nu.theta
    phi_j1_prefit[0],   phi_j2_prefit[0],   phi_nu_prefit[0]   = j1.phi,    j2.phi,    nu.phi
    deltaP_prefit[0] = math.sqrt((Wlep.px+Whad.px)**2 + (Wlep.py+Whad.py)**2 + (Wlep.pz+Whad.pz)**2)

    tree_out.Fill()

# ── write output ───────────────────────────────────────────────────────────
tree_out.Write()
for h in [h_pull_s1, h_pull_s2, h_pull_sl, h_pull_sn]:
    h.Write()
outFile.Close()
