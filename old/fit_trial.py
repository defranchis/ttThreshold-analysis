import ROOT
import numpy as np
#from iminuit import Minuit
from ROOT import TLorentzVector
from treemaker_lnuqq import all_branches as cols
ROOT.EnableImplicitMT()
from array import array
minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit2","Migrad")
minimizer.SetMaxFunctionCalls(100000) # optional
minimizer.SetTolerance(1e-6)
minimizer.SetStrategy(1)


# ---------------------------------------
# Input
# ---------------------------------------

ifile = "/afs/cern.ch/work/a/anmehta/public/FCC_ver2/FCCAnalyses/ttThreshold-analysis/outputs/treemaker/lnuqq/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
df = ROOT.RDataFrame("events", ifile)
SQRTS = 160.0

# ---------------------------------------
# Load data with RDF
# ---------------------------------------

arrays = df.AsNumpy(cols)

# ---------------------------------------
# Compute mass resolutions
# ---------------------------------------
h_mjj = df.Histo1D(
    ("h_mjj","m_{jj} resolution; m_{jj}^{reco}-m_{jj}^{gen} [GeV];Events",200,-20,20),
    "diff_RG_m_qq"
)

h_mlnu = df.Histo1D(
    ("h_mlnu","m_{lnu} resolution; m_{lnu}^{reco}-m_{lnu}^{gen} [GeV];Events",200,-20,20),
    "diff_RG_m_lnu"
)

h_mlnujj = df.Histo1D(
    ("h_mlnujj","m_{lnuqq} resolution; m_{lnuqq}^{reco}-m_{lnuqq}^{gen} [GeV];Events",200,-20,20),
    "diff_RG_m_lnuqq"
)
h_dM = df.Histo1D(
    ("h_dM","delta M; sqrts - m_{lnuqq} [GeV];Events",200,-20,20),
    "deltaM"
)
h_plnu=df.Histo1D(
    ("h_plnu","p_{lnuqq} resolution; p_{lnu}^{reco}-p_{lnu}^{gen} [GeV];Events",200,-20,20),
    "diff_RG_p_lnu"
)

h_pjj=df.Histo1D(
    ("h_pjj","m_{qqqq} resolution; p_{qq}^{reco}-p_{qq}^{gen} [GeV];Events",200,-20,20),
    "diff_RG_p_qq"
)

fit_mjj      = h_mjj.Fit("gaus","QS")
fit_mlnu     = h_mlnu.Fit("gaus","QS")
fit_mlnujj   = h_mlnujj.Fit("gaus","QS")
fit_pjj      = h_pjj.Fit("gaus","QS")
fit_plnu     = h_plnu.Fit("gaus","QS")
fit_dM       = h_dM.Fit("gaus","QS")

sigma_mjj    = h_mjj.GetFunction("gaus").GetParameter(2)
sigma_mlnu   = h_mlnu.GetFunction("gaus").GetParameter(2)
sigma_mlnujj = h_mlnujj.GetFunction("gaus").GetParameter(2)
sigma_pjj    = h_pjj.GetFunction("gaus").GetParameter(2)
sigma_plnu   = h_plnu.GetFunction("gaus").GetParameter(2)
sigma_dM     = h_dM.GetFunction("gaus").GetParameter(2)
#print("Hadronic W resolution:",sigma_mjj)
#print("Leptonic W resolution:",sigma_mlnu)

# detector scale constraints
sigma_j = 0.03
sigma_l = 0.001

sigma_E = 0.5
sigma_p = 0.5

# ---------------------------------------
# Build event objects
# ---------------------------------------

events = []

N = len(arrays["jet1_p"])

for i in range(N):
    j1 = ROOT.TLorentzVector(0,0,0,0)
    j2 = ROOT.TLorentzVector(0,0,0,0)
    lep = ROOT.TLorentzVector(0,0,0,0)
    mp = ROOT.TLorentzVector(0,0,0,0)
    j1.SetPtEtaPhiM(arrays["jet1_pt"][i], arrays["jet1_eta"][i], arrays["jet1_phi"][i], arrays["jet1_mass"][i])
    j2.SetPtEtaPhiM(arrays["jet2_pt"][i], arrays["jet2_eta"][i], arrays["jet2_phi"][i], arrays["jet2_mass"][i])
    lep.SetPtEtaPhiM(arrays["Isolep_pt"][i], arrays["Isolep_eta"][i], arrays["Isolep_phi"][i],0)
    mp.SetPtEtaPhiM(arrays["missing_p"][i], 0, arrays["missing_p_phi"][i],0)
    events.append((j1,j2,lep,mp))

print("Loaded",len(events),"events")

# ---------------------------------------
# Event chi2
# ---------------------------------------

def event_chi2(mW,gW,event):
    j1,j2,lep,mp = event
    #print(j1,j2,lep,mp)
    def chi2(x):
        s1=x[0];        s2=x[1];        sl=x[2] ;       sm=x[3]
        j1f = TLorentzVector(j1)
        j1f *= s1
        j2f = TLorentzVector(j2)
        j2f *= s2
        lepf = TLorentzVector(lep)
        lepf *= sl
        mpf  = TLorentzVector(mp)
        mpf *=sm
        print("input scales",s1,s2,sl,sm)
        W_had = j1f + j2f
        W_lep = lepf + mpf
        lnuqq_sys = j1f + j2f + lepf + mpf
        mjj = W_had.M()
        mlnu = W_lep.M()

        # Breit-Wigner physics terms
        bw = 2*np.log((mjj*mjj - mW*mW)**2 + (mW*gW)**2)
        bw += 2*np.log((mlnu*mlnu - mW*mW)**2 + (mW*gW)**2)

        # detector mass constraints
        mass = (mjj-mW)**2/sigma_mjj**2
        mass += (mlnu-mW)**2/sigma_mlnu**2
        # energy momentum conservation
        cons =  (lnuqq_sys.E() - SQRTS)**2 / (sigma_E**2 +sigma_dM **2)
        cons += (lnuqq_sys.Px()**2 + lnuqq_sys.Py()**2 + lnuqq_sys.Pz()**2) / sigma_p**2
        cons += (W_lep.P() - W_had.P())**2 / (sigma_plnu**2+ sigma_pjj**2) #this
        cons += (mlnu - mjj)**2 / (sigma_mlnu**2+ sigma_mjj**2)
        # scale penalties
        res = ((s1-1)/sigma_j)**2
        res += ((s2-1)/sigma_j)**2
        res += ((sl-1)/sigma_l)**2

        return bw + mass + cons + res
    print("gooing to run minimisation")
    #m = minimizer(chi2,s1=1.0,s2=1.0,sl=1.0,sm=1.0)
    minimizer_ev = ROOT.Math.Factory.CreateMinimizer("Minuit2","Migrad")
    minimizer_ev.SetFunction(ROOT.Math.Functor(chi2,4))
    minimizer_ev.SetMaxFunctionCalls(10000)
    minimizer_ev.SetTolerance(1e-6)
    minimizer_ev.SetStrategy(1)
    minimizer_ev.SetVariable(0, "s1", 1.0, 0.01)
    minimizer_ev.SetVariable(1, "s2", 1.0, 0.01)
    minimizer_ev.SetVariable(2, "sl", 1.0, 0.01)
    minimizer_ev.SetVariable(3, "sm", 1.0, 0.01)
    minimizer_ev.Minimize()
    s1, s2, sl,sm = minimizer_ev.X()
    chi2_val = minimizer_ev.MinValue()
    j1f = ROOT.TLorentzVector(j1); j1f*=s1
    j2f = ROOT.TLorentzVector(j2); j2f*=s2
    lepf = ROOT.TLorentzVector(lep); lepf*=sl
    mpf  = ROOT.TLorentzVector(mp); mpf*=sm
    mjj = (j1f+j2f).M()
    mlnu = (lepf+mpf).M()
    return mjj, mlnu,chi2_val, s1, s2, sl

# ---------------------------------------
# Global chi2
# ---------------------------------------

def global_chi2(x):
    print("gloabl chisq",x)
    mW, gW = x[0], x[1]
    total = 0.0
    for ev in events:
        print(type(ev))
        mjj, mlnu, chi2, _, _, _ = event_chi2(mW,gW,ev)
        total += chi2
    return total


print("gooing to run minimisation 2")
minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit2","Migrad")
functor = ROOT.Math.Functor(global_chi2, 2)
minimizer.SetFunction(functor)
minimizer.SetMaxFunctionCalls(10000)
minimizer.SetTolerance(1e-6)
minimizer.SetStrategy(1)
minimizer.SetVariable(0, "mW", 80.4, 0.01)
minimizer.SetVariable(1, "gW", 2.1, 0.01)
# Optional limits
minimizer.SetLimitedVariable(1,"gW",2.1,0.01,0.5,5.0)
minimizer.Minimize()
mW_fit  = minimizer.X()[0]
gW_fit  = minimizer.X()[1]
err_mW  = minimizer.Errors()[0]
err_gW  = minimizer.Errors()[1]

print("----- Fit Result -----")
print("mW = ", mW_fit, "+/-", err_mW)
print("GammaW = ", gW_fit, "+/-", err_gW)
outFile = ROOT.TFile("fit_output.root","RECREATE")
outTree = ROOT.TTree("fit","fit results")
mjj_fit  = array('f',[0])
mlnu_fit = array('f',[0])
chi2_val = array('f',[0])
s1_out   = array('f',[0])
s2_out   = array('f',[0])
sl_out   = array('f',[0])
sm_out   = array('f',[0])
outTree.Branch("mjj_fit", mjj_fit, "mjj_fit/F")
outTree.Branch("mlnu_fit", mlnu_fit, "mlnu_fit/F")
outTree.Branch("chi2", chi2_val, "chi2/F")
outTree.Branch("s1", s1_out, "s1/F")
outTree.Branch("s2", s2_out, "s2/F")
outTree.Branch("sl", sl_out, "sl/F")
outTree.Branch("sm", sm_out, "sm/F")
for ev in events:
    mjj, mlnu, chi2v, s1, s2, sl,sm = event_fit(mW_fit, gW_fit, ev)
    mjj_fit[0] = mjj
    mlnu_fit[0] = mlnu
    chi2_val[0] = chi2v
    s1_out[0] = s1
    s2_out[0] = s2
    sl_out[0] = sl
    sm_out[0] = sm
    outTree.Fill()

outFile.Write()
outFile.Close()
print("Saved per-event fit output to fit_output.root")
