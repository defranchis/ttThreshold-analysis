import uproot
import ROOT
from array import array
import numpy as np
import vector
from iminuit import Minuit
ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

sigma_sqrtS=0.012
outFile = ROOT.TFile("resolution_fits.root","RECREATE")
tree_out = ROOT.TTree("fitResults", "fit results")

mW_arr  = array('f', [0.])
gW_arr  = array('f', [0.])
s1_arr  = array('f', [0.])
s2_arr  = array('f', [0.])
sl_arr  = array('f', [0.])
sn_arr  = array('f', [0.])
chi2_arr = array('f', [0.])

pt_lep_postfit = array('f', [0.])
pt_j1_postfit  = array('f', [0.])
pt_j2_postfit  = array('f', [0.])
pt_nu_postfit  = array('f', [0.])

mWlep_postfit  = array('f', [0.])
mWhad_postfit  = array('f', [0.])
deltaP_postfit = array('f', [0.])
tree_out.Branch("pt_lep_fit", pt_lep_postfit, "pt_lep_fit/F")
tree_out.Branch("pt_j1_fit",  pt_j1_postfit,  "pt_j1_fit/F")
tree_out.Branch("pt_j2_fit",  pt_j2_postfit,  "pt_j2_fit/F")
tree_out.Branch("pt_nu_fit",  pt_nu_postfit,  "pt_nu_fit/F")
tree_out.Branch("mWlep_fit", mWlep_postfit, "mWlep_fit/F")
tree_out.Branch("mWhad_fit", mWhad_postfit, "mWhad_fit/F")
tree_out.Branch("deltaP_fit", deltaP_postfit, "deltaP_fit/F")

tree_out.Branch("mW",  mW_arr,  "mW/F")
tree_out.Branch("gW",  gW_arr,  "gW/F")
tree_out.Branch("s1",  s1_arr,  "s1/F")
tree_out.Branch("s2",  s2_arr,  "s2/F")
tree_out.Branch("sl",  sl_arr,  "sl/F")
tree_out.Branch("sn",  sn_arr,  "sn/F")
tree_out.Branch("chi2",chi2_arr, "chi2/F")

ECM = 160.0
fname="/afs/cern.ch/work/a/anmehta/public/FCC_ver2/FCCAnalyses/ttThreshold-analysis/outputs/treemaker/lnuqq/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
file = uproot.open(fname)
tree = file["events"]

df = ROOT.RDataFrame("events", fname)

def fit_resolution(hist):
    peak = hist.GetBinCenter(hist.GetMaximumBin())
    hist.Fit("gaus","QS","",peak-1,peak+1)
    f = hist.GetFunction("gaus")
    mean = f.GetParameter(1)
    sigma = f.GetParameter(2)
    hist.Fit("gaus","QS","",mean-2*sigma,mean+2*sigma)
    sigma = hist.GetFunction("gaus").GetParameter(2)
    return mean,sigma


h_mjj = df.Histo1D(("h_mjj","m_{jj} resolution; m_{jj}^{reco}-m_{jj}^{gen} [GeV];Events",50,-5,5),    "diff_RG_m_qq")
h_mlnu = df.Histo1D(("h_mlnu","m_{lnu} resolution; m_{lnu}^{reco}-m_{lnu}^{gen} [GeV];Events",50,-5,5),    "diff_RG_m_lnu")
h_mlnujj = df.Histo1D(("h_mlnujj","m_{lnuqq} resolution; m_{lnuqq}^{reco}-m_{lnuqq}^{gen} [GeV];Events",50,-5,5),    "diff_RG_m_lnuqq")
h_p_iso_lnuexcljj = df.Histo1D(("h_p_iso_lnuexcljj","",50,-5,5),"p_iso_lnuexcljj")
h_e_iso_lnuexcljj = df.Histo1D(("h_e_iso_lnuexcljj","",100,0,200),"e_iso_lnuexcljj")
h_plnu=df.Histo1D(("h_plnu","",50,-5,5),    "diff_RG_p_lnu")
h_pjj=df.Histo1D(("h_pjj","",50,-5,5),    "diff_RG_p_qq")
h_res_jet2_qq_fromele=df.Histo1D(    ("h_res_jet2_qq_fromele","",50,-0.5,0.5),    "res_jet2_qq_fromele")
h_res_jet1_qq_fromele=df.Histo1D(    ("h_res_jet1_qq_fromele","",50,-0.5,0.5),    "res_jet1_qq_fromele")
h_lep_res=df.Histo1D(("h_lep_res","",50,-0.2,0.2),    "lep_res")
h_mp_res=df.Histo1D(("h_met_res","",50,-0.3,0.3),    "missing_p_res")

sigma_lep=h_lep_res.GetRMS();
sigma_mp=h_mp_res.GetRMS();
sigma_j1=h_res_jet1_qq_fromele.GetRMS();
sigma_j2=h_res_jet2_qq_fromele.GetRMS();
sigma_p_iso_lnuexcljj=h_p_iso_lnuexcljj.GetRMS();
sigma_plnu=h_plnu.GetRMS();
sigma_mjj=h_mjj.GetRMS();

mean_lep=h_lep_res.GetMean();
mean_mp=h_mp_res.GetMean();
mean_j1=h_res_jet1_qq_fromele.GetMean();
mean_j2=h_res_jet2_qq_fromele.GetMean();
mean_p_iso_lnuexcljj=h_p_iso_lnuexcljj.GetMean();
mean_plnu=h_plnu.GetMean();
mean_mjj=h_mjj.GetMean();


#mean_j1,sigma                   = fit_resolution(h_res_jet1_qq_fromele);
#mean_j2,sigma                   = fit_resolution(h_res_jet2_qq_fromele);
#mean_lep,sigma                 = fit_resolution(h_lep_res);
#mean_mp,sigma                   = fit_resolution(h_mp_res);
#sigma_lep=h_lep_res.GetRMS();
#mean,sigma_mjj                     = fit_resolution(h_mjj)
#mean,sigma_mlnu                    = fit_resolution(h_mlnu)
#mean,sigma_mlnujj                  = fit_resolution(h_mlnujj)
#mean,sigma                     = fit_resolution(h_pjj)
#mean,sigma                    = fit_resolution(h_plnu)
#mean,sigma_p_iso_lnuexcljj         = fit_resolution(h_p_iso_lnuexcljj)
#mean,sigma_e_iso_lnuexcljj         = fit_resolution(h_e_iso_lnuexcljj)

#h_e_iso_lnuexcljj.Fit("gaus", "QS") 
#fit_func = h_e_iso_lnuexcljj.GetFunction("gaus")
#mean_e_iso_lnuexcljj = fit_func.GetParameter(1)
#sigma_e_iso_lnuexcljj = fit_func.GetParameter(2)
#print("RMS (sigma) of total momentum:", sigma_e_iso_lnuexcljj)


hist_mjj                     = h_mjj.GetValue()
hist_mlnu                    = h_mlnu.GetValue()
hist_mlnujj                  = h_mlnujj.GetValue()
hist_pjj                     = h_pjj.GetValue()
hist_plnu                    = h_plnu.GetValue()
hist_p_iso_lnuexcljj         = h_p_iso_lnuexcljj.GetValue()
hist_e_iso_lnuexcljj         = h_e_iso_lnuexcljj.GetValue()
hist_sigma_j1                = h_res_jet1_qq_fromele.GetValue()
hist_sigma_j2                = h_res_jet2_qq_fromele.GetValue()
hist_sigma_lep               = h_lep_res.GetValue()
hist_sigma_mp                = h_mp_res.GetValue()
hist_sigma_p_iso_lnuexcljj   = h_p_iso_lnuexcljj.GetValue()
hist_sigma_e_iso_lnuexcljj   = h_e_iso_lnuexcljj.GetValue()

hist_mjj.Write()
hist_mlnu.Write()
hist_mlnujj.Write()
hist_pjj.Write()
hist_plnu.Write()
hist_p_iso_lnuexcljj.Write()
hist_sigma_j1.Write()              
hist_sigma_j2.Write()              
hist_sigma_lep.Write()             
hist_sigma_mp.Write()              
hist_sigma_p_iso_lnuexcljj.Write()
hist_sigma_e_iso_lnuexcljj.Write()
h_pull_s1 = ROOT.TH1F("h_pull_s1", "Pull s1", 50, -5, 5)
h_pull_s2 = ROOT.TH1F("h_pull_s2", "Pull s2", 50, -5, 5)
h_pull_sl = ROOT.TH1F("h_pull_sl", "Pull sl", 50, -5, 5)
h_pull_sn = ROOT.TH1F("h_pull_sn", "Pull sn", 50, -5, 5)


arrays = tree.arrays([
"jet1_pt","jet1_eta","jet1_phi","jet1_mass",
"jet2_pt","jet2_eta","jet2_phi","jet2_mass",
"Isolep_pt","Isolep_eta","Isolep_phi",
    "missing_p","missing_p_phi","missing_p_eta","missing_pt",
], library="np")

N = len(arrays["jet1_pt"])
print("Loaded",N,"events")

events = []

for i in range(N):

    j1 = vector.obj(
        pt=arrays["jet1_pt"][i],
        eta=arrays["jet1_eta"][i],
        phi=arrays["jet1_phi"][i],
        mass=0
    )

    j2 = vector.obj(
        pt=arrays["jet2_pt"][i],
        eta=arrays["jet2_eta"][i],
        phi=arrays["jet2_phi"][i],
        mass=0 #arrays["jet2_mass"][i] #mass of u/d/s/c replace 
    )

    lep = vector.obj(
        pt=arrays["Isolep_pt"][i],
        eta=arrays["Isolep_eta"][i],
        phi=arrays["Isolep_phi"][i],
        mass=0 #mass of muon GeV
    )

    nu = vector.obj(
        pt=arrays["missing_pt"][i],
        eta=arrays["missing_p_eta"][i],
        phi=arrays["missing_p_phi"][i],
        mass=0
    )
#    print(lep.pt,lep.eta,j1.pt)
    events.append((j1,j2,lep,nu))

print("Events stored:",len(events))


def breit_wigner(m, MW, GW):
    return (MW * GW) / ((m**2 - MW**2)**2 + (MW * GW)**2)

def event_chi2(scales,event,mW,gW):
    s1,s2,sl,sn = scales
    j1,j2,lep,nu = event
#    print(lep.pt)
    j1f = j1 * s1
    j2f = j2 * s2
    lepf = lep * sl
    nuf = nu * sn

    Wh = j1f + j2f
    Wl = lepf + nuf
    WW = Wh + Wl

    mjj = Wh.mass
    mlnu = Wl.mass
    #    BW = MG/ [ (m2-M2)2 + M2G2]
    bw = -2 * (np.log(breit_wigner(mjj, mW, gW)) + np.log(breit_wigner(mlnu, mW, gW)))
    cons = (
        (WW.E-ECM)**2/(sigma_sqrtS **2) 
        #((Wl.px - Wh.px)**2 + (Wl.py - Wh.py)**2 + (Wl.pz - Wh.pz)**2) /0.001  #(sigma_plnu**2 + sigma_pjj**2) #resolution for numerator
        #((WW.px **2 + WW.py **2 + WW.pz**2)/(sigma_plnuqq**2)
    )
    
    mean_s1 = 1.0 / (1.0 + mean_j1)
    mean_s2 = 1.0 / (1.0 + mean_j2)
    mean_sl = 1.0 / (1.0  + mean_lep)
    mean_sn = 1.0 / (1.0 + mean_mp)
    res = (
    #(s1 - 1.02)**2 / 0.005**2 + #sigma_j1**2 +
    #(s2 - 1.02)**2 / 0.005**2 + #sigma_j2**2 +
    #(sl - 1.0)**2 / 0.0048**2 +  #sigma_lep**2 +
    #(sn - 1.0)**2 / 0.01446**2 #sigma_mp**2
    (s1 - mean_s1)**2 / sigma_j1**2 +
    (s2 - mean_s2)**2 / sigma_j2**2 +
    (sl - mean_sl)**2 / sigma_lep**2 +
    (sn - mean_sn)**2 / sigma_mp**2
    )
    #print("mean_s1 =", mean_s1 ,"verusu", mean_j2,sigma_j1)
    #print("mean_s2 =", mean_s2,"verusu", mean_j2,sigma_j2)
    #print("mean_sl =", mean_sl,"verusu", mean_lep, sigma_lep)
    #print("mean_sl =", mean_sl)
    #print("mean_sn =", mean_sn)
    #print("expected scale =", 1/(1+mean_j1))
    #pull_s1 = (vals["s1"] - mean_s1) / sigma_s1

    return bw + cons + res #mass_term + cons #+ res





def fit_event(event):
    
    def f(s1,s2,sl,sn,mW,gW):
        return event_chi2((s1,s2,sl,sn),event,mW,gW)

    m = Minuit(f,s1=1,s2=1,sl=1,sn=1,mW=80.419,gW=2.049)# missing mW and gamma W #one fit fxn
    m.limits["mW"] = (0, 200)
    m.limits["gW"] = (0,10)
    #m.fixed["gW"] = True
    #    m.fixed["mW"] = True
    #m.print_level=1
    m.tol=1E-6
    m.strategy=2
    m.migrad(ncall=5000)
    #m.limits["s1"] = (0.9, 1.0)
    #m.limits["s2"] = (0.9, 1.0)
    #m.limits["sl"] = (0.95, 1.0)
    #m.limits["sn"] = (0.8, 1.0)
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    if m.valid:
        #print("Valid:", m.valid)
        #print("EDM:", m.fmin.edm)
        #print("Nfcn:", m.nfcn)
        #print("fval",m.fval)
        print("mW =", m.values["mW"])
        print("gW =", m.values["gW"])
        #print("s1 =", m.values["s1"])
        #print("s2 =", m.values["s2"])
        #print("sl =", m.values["sl"])
        #print("sn =", m.values["sn"])


    return m

for i, event in enumerate(events):
    m = fit_event(event)
    if not m.valid or m.fval > 200: continue
    if i > 10000: break
    #print("converged")
    #print("Strategy:", m.strategy)
    #print("Tolerance:", m.tol)
    #print("Print level:", m.print_level)
    #    print("Errors (initial):", m.errors)
    #print("Limits:", m.limits)
    vals = m.values
    mW_arr[0] =vals["mW"]
    gW_arr[0] =vals["gW"]
    s1_arr[0] =vals["s1"]
    s2_arr[0] =vals["s2"]
    sl_arr[0] =vals["sl"]
    sn_arr[0] =vals["sn"]
    chi2_arr[0]=m.fval
    s1 = vals["s1"]
    s2 = vals["s2"]
    sl = vals["sl"]
    sn = vals["sn"]
    mean_s1 = 1.0 / (1.0 + mean_j1)
    mean_s2 = 1.0 / (1.0 + mean_j2)
    mean_sl = 1.0 / (1.0 + mean_lep)
    mean_sn = 1.0 / (1.0 + mean_mp)
    
    sigma_s1 = sigma_j1 / (1.0 + mean_j1)**2
    sigma_s2 = sigma_j2 / (1.0 + mean_j2)**2
    sigma_sl = sigma_lep / (1.0 + mean_lep)**2
    sigma_sn = sigma_mp / (1.0 + mean_mp)**2
    
    h_pull_s1.Fill((s1 - mean_s1) / sigma_s1)
    h_pull_s2.Fill((s2 - mean_s2) / sigma_s2)
    h_pull_sl.Fill((sl - mean_sl) / sigma_sl)
    h_pull_sn.Fill((sn - mean_sn) / sigma_sn)
    j1, j2, lep, nu = event
    j1f  = j1 * s1
    j2f  = j2 * s2
    lepf = lep * sl
    nuf  = nu * sn
    Wh = j1f + j2f
    Wl = lepf + nuf
    mWhad_postfit[0] = Wh.mass
    mWlep_postfit[0] = Wl.mass
    pt_j1_postfit[0] = j1f.pt
    pt_j2_postfit[0] = j2f.pt
    pt_lep_postfit[0] = lepf.pt
    pt_nu_postfit[0] = nuf.pt
    deltaP_postfit[0] = (Wl.px + Wh.px)**2 + (Wl.py + Wh.py)**2 + (Wl.pz +Wh.pz)**2
    
    tree_out.Fill()

#outFile.cd()
tree_out.Write()
outFile.Close()
