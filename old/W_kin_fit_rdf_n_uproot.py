import uproot
import ROOT
import numpy as np
import vector
from iminuit import Minuit
ROOT.gROOT.SetBatch(True)
ROOT.EnableImplicitMT()
outFile = ROOT.TFile("resolution_fits.root","RECREATE")

ECM = 160.0
fname = "/afs/cern.ch/work/a/anmehta/public/FCC_ver2/FCCAnalyses/ttThreshold-analysis/outputs/treemaker/lnuqq/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"

file = uproot.open(fname)
tree = file["events"]

df = ROOT.RDataFrame("events", fname)

# --------------------------------
# resolution histograms
# --------------------------------

h_mjj = df.Histo1D(
    ("h_mjj","m_{jj} resolution",200,-20,20),
    "diff_RG_m_qq"
)

h_mlnu = df.Histo1D(
    ("h_mlnu","m_{lnu} resolution",200,-20,20),
    "diff_RG_m_lnu"
)

h_mlnujj = df.Histo1D(
    ("h_mlnujj","m_{lnuqq} resolution",200,-20,20),
    "diff_RG_m_lnuqq"
)

h_dM = df.Histo1D(
    ("h_dM","deltaM",200,-20,20),
    "deltaM"
)

h_plnu=df.Histo1D(
    ("h_plnu","plnu resolution",200,-20,20),
    "diff_RG_p_lnu"
)

h_pjj=df.Histo1D(
    ("h_pjj","pjj resolution",200,-20,20),
    "diff_RG_p_qq"
)

def fit_resolution(hist):

    peak = hist.GetBinCenter(hist.GetMaximumBin())

    hist.Fit("gaus","QS","",peak-5,peak+5)

    f = hist.GetFunction("gaus")
    mean = f.GetParameter(1)
    sigma = f.GetParameter(2)

    hist.Fit("gaus","QS","",mean-2*sigma,mean+2*sigma)

    sigma = hist.GetFunction("gaus").GetParameter(2)

    return sigma


h_mjj.GetValue().Rebin(2)
# Fit resolution

#h_mjj.Fit("gaus","QS")
#h_mlnu.Fit("gaus","QS")
#h_mlnujj.Fit("gaus","QS")
#h_pjj.Fit("gaus","QS")
#h_plnu.Fit("gaus","QS")
#h_dM.Fit("gaus","QS")

sigma_mjj      = fit_resolution(h_mjj)
sigma_mlnu     = fit_resolution(h_mlnu)
sigma_mlnujj   = fit_resolution(h_mlnujj)
sigma_pjj      = fit_resolution(h_pjj)
sigma_plnu     = fit_resolution(h_plnu)
sigma_dM       = fit_resolution(h_dM)


#sigma_mjj = h_mjj.GetFunction("gaus").GetParameter(2)
#sigma_mlnu = h_mlnu.GetFunction("gaus").GetParameter(2)
#sigma_mlnujj = h_mlnujj.GetFunction("gaus").GetParameter(2)
#sigma_pjj = h_pjj.GetFunction("gaus").GetParameter(2)
#sigma_plnu = h_plnu.GetFunction("gaus").GetParameter(2)
#sigma_dM = h_dM.GetFunction("gaus").GetParameter(2)
print("Resolutions:")
print("sigma_mjj =",sigma_mjj)
print("sigma_mlnu =",sigma_mlnu)
print("sigma_mlnujj =",sigma_mlnujj)
print("sigma_pjj =",sigma_pjj)
print("sigma_plnu =",sigma_plnu)
print("sigma_dM =",sigma_dM)

h_mjj_hist    = h_mjj.GetValue()
h_mlnu_hist   = h_mlnu.GetValue()
h_mlnujj_hist = h_mlnujj.GetValue()
h_pjj_hist    = h_pjj.GetValue()
h_plnu_hist   = h_plnu.GetValue()
h_dM_hist     = h_dM.GetValue()
h_mjj_hist.Write()
h_mlnu_hist.Write()
h_mlnujj_hist.Write()
h_pjj_hist.Write()
h_plnu_hist.Write()
h_dM_hist.Write()
outFile.Close()
# --------------------------------
# read arrays with uproot
# --------------------------------

arrays = tree.arrays([
"jet1_pt","jet1_eta","jet1_phi","jet1_mass",
"jet2_pt","jet2_eta","jet2_phi","jet2_mass",
"Isolep_pt","Isolep_eta","Isolep_phi",
"missing_p","missing_p_phi"
], library="np")

N = len(arrays["jet1_pt"])
print("Loaded",N,"events")

events = []

for i in range(N):

    j1 = vector.obj(
        pt=arrays["jet1_pt"][i],
        eta=arrays["jet1_eta"][i],
        phi=arrays["jet1_phi"][i],
        mass=arrays["jet1_mass"][i]
    )

    j2 = vector.obj(
        pt=arrays["jet2_pt"][i],
        eta=arrays["jet2_eta"][i],
        phi=arrays["jet2_phi"][i],
        mass=arrays["jet2_mass"][i]
    )

    lep = vector.obj(
        pt=arrays["Isolep_pt"][i],
        eta=arrays["Isolep_eta"][i],
        phi=arrays["Isolep_phi"][i],
        mass=0
    )

    nu = vector.obj(
        pt=arrays["missing_p"][i],
        eta=0,
        phi=arrays["missing_p_phi"][i],
        mass=0
    )

    events.append((j1,j2,lep,nu))

print("Events stored:",len(events))

# --------------------------------
# detector scale uncertainties
# --------------------------------

sigma_j = 0.03
sigma_l = 0.001

sigma_E = 0.5
sigma_p = 0.5

# --------------------------------
# event chi2
# --------------------------------

def event_chi2(scales,event):

    s1,s2,sl,sn = scales

    j1,j2,lep,nu = event

    j1f = j1*s1
    j2f = j2*s2
    lepf = lep*sl
    nuf = nu*sn

    Wh = j1f + j2f
    Wl = lepf + nuf
    WW = Wh + Wl

    mjj = Wh.mass
    mlnu = Wl.mass
    
    cons = (
        (WW.E-ECM)**2/(sigma_E**2 + sigma_dM**2) +
        (mlnu-mjj)**2/(sigma_mlnu**2 + sigma_mjj**2) +
        (Wh.p-Wl.p)**2/(sigma_plnu**2 + sigma_pjj**2)
    )

    res = (
        (s1-1)**2/sigma_j**2 +
        (s2-1)**2/sigma_j**2 +
        (sl-1)**2/sigma_l**2
    )

    return cons #+ bw #res


# --------------------------------
# PREFIT EVENTS (only once)
# --------------------------------

print("Prefitting events...")

prefit_masses = []

for ievt,ev in enumerate(events):

    if ievt>1000:
        break

    def f(s1,s2,sl,sn):
        return event_chi2((s1,s2,sl,sn),ev)

    m = Minuit(f,s1=1,s2=1,sl=1,sn=1)
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()

    if not m.valid:
        continue

    s1,s2,sl,sn = m.values

    j1,j2,lep,nu = ev

    Wh = j1*s1 + j2*s2
    Wl = lep*sl + nu*sn

    prefit_masses.append((Wh.mass,Wl.mass))

print("Events used:",len(prefit_masses))

# --------------------------------
# GLOBAL FIT
# --------------------------------

def global_chi2(mW,gW):

    chi2 = 0

    for mjj,mlnu in prefit_masses:

        chi2 += np.log((mjj**2-mW**2)**2+(mW*gW)**2)
        chi2 += np.log((mlnu**2-mW**2)**2+(mW*gW)**2)

    return chi2


m = Minuit(global_chi2,mW=80.4,gW=2.1)

m.errordef = Minuit.LEAST_SQUARES
m.limits["mW"] = (79,82)
m.limits["gW"] = (1.5,3)

m.migrad()

print("Fit result")
print("mW =",m.values["mW"])
print("GammaW =",m.values["gW"])
