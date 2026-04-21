import uproot
import ROOT
from array import array
import numpy as np
import vector
from iminuit import Minuit
ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

sigma_sqrtS=0.12
outFile = ROOT.TFile("resolution_fits.root","RECREATE")
tree_out = ROOT.TTree("fitResults", "fit results")

mW_arr  = array('f', [0.])
gW_arr  = array('f', [0.])
s1_arr  = array('f', [0.])
s2_arr  = array('f', [0.])
sl_arr  = array('f', [0.])
sn_arr  = array('f', [0.])
chi2_arr = array('f', [0.])

p1_arr  = array('f', [0.])
p2_arr  = array('f', [0.])
pn_arr  = array('f', [0.])
t1_arr  = array('f', [0.])
t2_arr  = array('f', [0.])
tn_arr  = array('f', [0.])


pt_lep_postfit = array('f', [0.])
pt_j1_postfit  = array('f', [0.])
pt_j2_postfit  = array('f', [0.])
pt_nu_postfit  = array('f', [0.])
mWlep_postfit  = array('f', [0.])
mWhad_postfit  = array('f', [0.])
deltaP_postfit = array('f', [0.])

pt_lep_prefit = array('f', [0.])
pt_j1_prefit  = array('f', [0.])
pt_j2_prefit  = array('f', [0.])
pt_nu_prefit  = array('f', [0.])
mWlep_prefit  = array('f', [0.])
mWhad_prefit  = array('f', [0.])
deltaP_prefit = array('f', [0.])


phi_j1_prefit  = array('f', [0.])
phi_j2_prefit  = array('f', [0.])
phi_nu_prefit  = array('f', [0.])
theta_j1_prefit  = array('f', [0.])
theta_j2_prefit  = array('f', [0.])
theta_nu_prefit  = array('f', [0.])

phi_j1_postfit  = array('f', [0.])
phi_j2_postfit  = array('f', [0.])
phi_nu_postfit  = array('f', [0.])
theta_j1_postfit  = array('f', [0.])
theta_j2_postfit  = array('f', [0.])
theta_nu_postfit  = array('f', [0.])


Wlep_px_prefit = array('f', [0.])
Wlep_py_prefit = array('f', [0.])
Wlep_pz_prefit = array('f', [0.])
Whad_px_prefit = array('f', [0.])
Whad_py_prefit = array('f', [0.])
Whad_pz_prefit = array('f', [0.])

Wlep_px_postfit = array('f', [0.])
Wlep_py_postfit = array('f', [0.])
Wlep_pz_postfit = array('f', [0.])
Whad_px_postfit = array('f', [0.])
Whad_py_postfit = array('f', [0.])
Whad_pz_postfit = array('f', [0.])



tree_out.Branch("pt_lep_postfit", pt_lep_postfit, "pt_lep_postfit/F")
tree_out.Branch("pt_j1_postfit",  pt_j1_postfit,  "pt_j1_postfit/F")
tree_out.Branch("pt_j2_postfit",  pt_j2_postfit,  "pt_j2_postfit/F")
tree_out.Branch("pt_nu_postfit",  pt_nu_postfit,  "pt_nu_postfit/F")
tree_out.Branch("mWlep_postfit",  mWlep_postfit, "mWlep_postfit/F")
tree_out.Branch("mWhad_postfit",  mWhad_postfit, "mWhad_postfit/F")
tree_out.Branch("deltaP_postfit", deltaP_postfit, "deltaP_postfit/F")

tree_out.Branch("pt_lep_prefit", pt_lep_prefit, "pt_lep_prefit/F")
tree_out.Branch("pt_j1_prefit",  pt_j1_prefit,  "pt_j1_prefit/F")
tree_out.Branch("pt_j2_prefit",  pt_j2_prefit,  "pt_j2_prefit/F")
tree_out.Branch("pt_nu_prefit",  pt_nu_prefit,  "pt_nu_prefit/F")
tree_out.Branch("mWlep_prefit",  mWlep_prefit, "mWlep_prefit/F")
tree_out.Branch("mWhad_prefit",  mWhad_prefit, "mWhad_prefit/F")
tree_out.Branch("deltaP_prefit", deltaP_prefit, "deltaP_prefit/F")


tree_out.Branch("theta_j1_prefit",  theta_j1_prefit,  "theta_j1_prefit/F")
tree_out.Branch("theta_j2_prefit",  theta_j2_prefit,  "theta_j2_prefit/F")
tree_out.Branch("theta_nu_prefit",  theta_nu_prefit,  "theta_nu_prefit/F")

tree_out.Branch("phi_j1_prefit",  phi_j1_prefit,  "phi_j1_prefit/F")
tree_out.Branch("phi_j2_prefit",  phi_j2_prefit,  "phi_j2_prefit/F")
tree_out.Branch("phi_nu_prefit",  phi_nu_prefit,  "phi_nu_prefit/F")


tree_out.Branch("Wlep_px_postfit", Wlep_px_postfit, "Wlep_px_postfit/F")
tree_out.Branch("Wlep_py_postfit", Wlep_py_postfit, "Wlep_py_postfit/F")
tree_out.Branch("Wlep_pz_postfit", Wlep_pz_postfit, "Wlep_pz_postfit/F")
tree_out.Branch("Whad_px_postfit", Whad_px_postfit, "Whad_px_postfit/F")
tree_out.Branch("Whad_py_postfit", Whad_py_postfit, "Whad_py_postfit/F")
tree_out.Branch("Whad_pz_postfit", Whad_pz_postfit, "Whad_pz_postfit/F")

tree_out.Branch("Wlep_px_prefit", Wlep_px_prefit, "Wlep_px_prefit/F")
tree_out.Branch("Wlep_py_prefit", Wlep_py_prefit, "Wlep_py_prefit/F")
tree_out.Branch("Wlep_pz_prefit", Wlep_pz_prefit, "Wlep_pz_prefit/F")
tree_out.Branch("Whad_px_prefit", Whad_px_prefit, "Whad_px_prefit/F")
tree_out.Branch("Whad_py_prefit", Whad_py_prefit, "Whad_py_prefit/F")
tree_out.Branch("Whad_pz_prefit", Whad_pz_prefit, "Whad_pz_prefit/F")


tree_out.Branch("mW",  mW_arr,  "mW/F")
tree_out.Branch("gW",  gW_arr,  "gW/F")
tree_out.Branch("s1",  s1_arr,  "s1/F")
tree_out.Branch("s2",  s2_arr,  "s2/F")
tree_out.Branch("sl",  sl_arr,  "sl/F")
tree_out.Branch("sn",  sn_arr,  "sn/F")
tree_out.Branch("chi2",chi2_arr, "chi2/F")



tree_out.Branch("p1",  p1_arr,  "p1/F")
tree_out.Branch("p2",  p2_arr,  "p2/F")
tree_out.Branch("pn",  pn_arr,  "pn/F")
tree_out.Branch("t1",  t1_arr,  "t1/F")
tree_out.Branch("t2",  t2_arr,  "t2/F")
tree_out.Branch("tn",  tn_arr,  "tn/F")


ECM = 160.0
fname="/afs/cern.ch/work/a/anmehta/public/FCC_ver2/FCCAnalyses/ttThreshold-analysis/outputs/treemaker/lnuqq/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
file = uproot.open(fname)
tree = file["events"]

df = ROOT.RDataFrame("events", fname)

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

h_jet1_theta_res=df.Histo1D(("h_jet1_theta_res","",50,-0.4,0.4),"jet1_dtheta")
h_jet2_theta_res=df.Histo1D(("h_jet2_theta_res","",50,-0.4,0.4),"jet2_dtheta")
h_jet1_phi_res=df.Histo1D(("h_jet1_phi_res","",50,-0.4,0.4),"jet1_dphi")
h_jet2_phi_res=df.Histo1D(("h_jet2_phi_res","",50,-0.4,0.4),"jet2_dphi")
h_met_phi_res=df.Histo1D(("h_met_phi_res","",50,-0.4,0.4),"met_dphi")
h_met_theta_res=df.Histo1D(("h_met_theta_res","",50,-0.4,0.4),"met_dtheta")


sigma_lep=h_lep_res.GetRMS();
sigma_mp=h_mp_res.GetRMS();
sigma_j1=h_res_jet1_qq_fromele.GetRMS();
sigma_j2=h_res_jet2_qq_fromele.GetRMS();
sigma_p_iso_lnuexcljj=h_p_iso_lnuexcljj.GetRMS();
sigma_plnu=h_plnu.GetRMS();
sigma_mjj=h_mjj.GetRMS();
#jet1_theta_rms = h_jet1_theta_res.GetRMS();
#jet2_theta_rms = h_jet2_theta_res.GetRMS();
#jet1_phi_rms = h_jet1_phi_res.GetRMS();
#jet2_phi_rms = h_jet2_phi_res.GetRMS();
#met_phi_rms = h_met_phi_res.GetRMS();
#met_theta_rms = h_met_theta_res.GetRMS();

jet1_theta_rms=0.05
jet1_phi_rms=0.05
jet2_theta_rms=0.05
jet2_phi_rms=0.06
met_theta_rms=0.04
met_phi_rms=0.05

mean_lep=h_lep_res.GetMean();
mean_mp=h_mp_res.GetMean();
mean_j1=h_res_jet1_qq_fromele.GetMean();
mean_j2=h_res_jet2_qq_fromele.GetMean();
mean_p_iso_lnuexcljj=h_p_iso_lnuexcljj.GetMean();
mean_plnu=h_plnu.GetMean();
mean_mjj=h_mjj.GetMean();


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
    s1,s2,sl,sn,t1,t2,tn,p1,p2,pn = scales
    j1,j2,lep,nu = event
    
    j1f_theta = j1.theta + t1* jet1_theta_rms
    j2f_theta = j2.theta + t2* jet2_theta_rms
    j1f_phi   = j1.phi   + p1* jet1_phi_rms
    j2f_phi   = j2.phi   + p2* jet2_phi_rms
    nuf_phi   = nu.phi   + pn* met_phi_rms
    nuf_theta = nu.theta + tn* met_theta_rms
    j1f_p     = j1.p * s1
    j2f_p     = j2.p * s2    
    lepf      = lep * sl
    nuf_p     = nu.p * sn

    nuf = vector.obj(
        px= nuf_p * np.sin(nuf_theta) * np.cos(nuf_phi),
        py= nuf_p * np.sin(nuf_theta) * np.sin(nuf_phi),
        pz= nuf_p * np.cos(nuf_theta),
        e=nuf_p
    )

    j1f = vector.obj(
        px= j1f_p * np.sin(j1f_theta) * np.cos(j1f_phi),
        py= j1f_p * np.sin(j1f_theta) * np.sin(j1f_phi),
        pz= j1f_p * np.cos(j1f_theta),
        e=j1f_p
    )

    j2f = vector.obj(
        px= j2f_p * np.sin(j2f_theta) * np.cos(j2f_phi),
        py= j2f_p * np.sin(j2f_theta) * np.sin(j2f_phi),
        pz= j2f_p * np.cos(j2f_theta),
        e=j2f_p
    )
    
    Wh = j1f + j2f
    Wl = lepf + nuf
    WW = Wh + Wl

    mjj = Wh.mass
    mlnu = Wl.mass
    bw = -2 * (np.log(breit_wigner(mjj, mW, gW)) + np.log(breit_wigner(mlnu, mW, gW)))
    cons = (
        (WW.E-ECM)**2/(sigma_sqrtS **2) +
        ((Wl.px + Wh.px)**2 + (Wl.py + Wh.py)**2 + (Wl.pz + Wh.pz)**2)*100

    )
    

    res=( (s1 - 1.01)**2 / 0.05**2 + #sigma_j1**2 +                                                                                                                                                     
    (s2 - 1.01)**2 / 0.05**2 + #sigma_j2**2 +                                                                                                                                                
    (sl - 1.0)**2 / 0.002**2 +  #sigma_lep**2 +                                                                                                                                                           
    (sn - 1.0)**2 / 0.015**2 #sigma_mp**2
    )

    angular=(pn**2+tn**2+p1**2+p2**2+t1**2+t2**2)

    
    #print( mean_sl ,"verusu", mean_j2,mean_j1,mean_sn)
    #print("mean_s2 =", mean_s2,"verusu", mean_j2,sigma_j2)
    #print("mean_sl =", mean_sl,"verusu", mean_lep, sigma_lep)
    #print("mean_sl =", mean_sl)
    #print("mean_sn =", mean_sn)
    #print("expected scale =", 1/(1+mean_j1))
    #pull_s1 = (vals["s1"] - mean_s1) / sigma_s1
    #print(s1,s2,sn,p1,p2,pn,t1,t2,tn)
    #print(met_phi_rms,met_theta_rms,jet1_theta_rms,jet1_phi_rms,jet2_phi_rms,jet2_theta_rms)
    return bw + cons + res + angular





def fit_event(event):
    
    def f(s1,s2,sl,sn,t1,t2,tn,p1,p2,pn,mW,gW):
        return event_chi2((s1,s2,sl,sn,t1,t2,tn,p1,p2,pn),event,mW,gW)
    m = Minuit(f,s1=1,s2=1,sl=1,sn=1,t1=0,t2=1,tn=1,p1=1,p2=1,pn=1,mW=80.419,gW=2.049)# missing mW and gamma W #one fit fxn
    m.limits["mW"] = (0, 200)
    m.limits["gW"] = (0,10)
    m.fixed["gW"] = True
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
#        print("gW =", m.values["gW"])
        #print("s1 =", m.values["s1"])
        #print("s2 =", m.values["s2"])
        #print("sl =", m.values["sl"])
        #print("sn =", m.values["sn"])


    return m

for i, event in enumerate(events):
    m = fit_event(event)
    if not m.valid or m.fval > 200: 
        print("not saving",  m.valid ,"chi2",m.fval);
        continue
    if i > 20000: break
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
    p1_arr[0] =vals["p1"]
    p2_arr[0] =vals["p2"]
    pn_arr[0] =vals["pn"]
    t1_arr[0] =vals["t1"]
    t2_arr[0] =vals["t2"]
    tn_arr[0] =vals["tn"]

    
    chi2_arr[0]=m.fval
    s1 = vals["s1"]
    s2 = vals["s2"]
    sl = vals["sl"]
    sn = vals["sn"]
    t1 = vals["t1"]
    t2 = vals["t2"]
    tn = vals["tn"]
    p1 = vals["p1"]
    p2 = vals["p2"]
    pn = vals["pn"]
    
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
    j1f_p  = j1.p * s1
    j2f_p  = j2.p * s2
    lepf   = lep * sl
    nuf_p  = nu.p * sn    
    j1f_theta = j1.theta+t1*jet1_theta_rms
    j2f_theta = j2.theta+t2*jet2_theta_rms
    j1f_phi   = j1.phi+p1*jet1_phi_rms
    j2f_phi   = j2.phi+p2*jet2_phi_rms
    nuf_phi   = nu.phi+pn*met_phi_rms
    nuf_theta = nu.theta+t2*met_theta_rms
    #print(nuf_p)
    nuf = vector.obj(
        px= nuf_p * np.sin(nuf_theta) * np.cos(nuf_phi),
        py= nuf_p * np.sin(nuf_theta) * np.sin(nuf_phi),
        pz= nuf_p * np.cos(nuf_theta),
        e=nuf_p
    )

    j1f = vector.obj(
        px= j1f_p * np.sin(j1f_theta) * np.cos(j1f_phi),
        py= j1f_p * np.sin(j1f_theta) * np.sin(j1f_phi),
        pz= j1f_p * np.cos(j1f_theta),
        e=j1f_p
    )

    j2f = vector.obj(
        px= j2f_p * np.sin(j2f_theta) * np.cos(j2f_phi),
        py= j2f_p * np.sin(j2f_theta) * np.sin(j2f_phi),
        pz= j2f_p * np.cos(j2f_theta),
        e=j2f_p
    )

    
    Wh = j1f + j2f
    Wl = lepf + nuf
    mWhad_postfit[0] = Wh.mass
    mWlep_postfit[0] = Wl.mass
    Wlep_px_postfit[0]=Wl.px
    Wlep_py_postfit[0]=Wl.py
    Wlep_pz_postfit[0]=Wl.pz
    Whad_px_postfit[0]=Wh.px
    Whad_py_postfit[0]=Wh.py
    Whad_pz_postfit[0]=Wh.pz

    pt_j1_postfit[0] = j1f.pt
    pt_j2_postfit[0] = j2f.pt
    pt_lep_postfit[0] = lepf.pt
    pt_nu_postfit[0] = nuf.pt

    theta_j1_postfit[0] = j1f.theta
    theta_j2_postfit[0] = j2f.theta
    theta_nu_postfit[0] = nuf.theta

    phi_j1_postfit[0] = j1f.phi
    phi_j2_postfit[0] = j2f.phi
    phi_nu_postfit[0] = nuf.phi

    deltaP_postfit[0] = np.sqrt((Wl.px + Wh.px)**2 + (Wl.py + Wh.py)**2 + (Wl.pz +Wh.pz)**2)
    Whad = j1 + j2
    Wlep = lep + nu
    mWhad_prefit[0] = Whad.mass
    mWlep_prefit[0] = Wlep.mass
    pt_j1_prefit[0] = j1.pt
    pt_j2_prefit[0] = j2.pt
    pt_lep_prefit[0] = lep.pt
    pt_nu_prefit[0] = nu.pt
    deltaP_prefit[0] = np.sqrt((Wlep.px + Whad.px)**2 + (Wlep.py + Whad.py)**2 + (Wlep.pz +Whad.pz)**2)
    Wlep_px_prefit[0]=Wlep.px
    Wlep_py_prefit[0]=Wlep.py
    Wlep_pz_prefit[0]=Wlep.pz
    Whad_px_prefit[0]=Whad.px
    Whad_py_prefit[0]=Whad.py
    Whad_pz_prefit[0]=Whad.pz
    
    theta_j1_prefit[0] = j1.theta
    theta_j2_prefit[0] = j2.theta
    theta_nu_prefit[0] = nu.theta
    phi_j1_prefit[0] = j1.phi
    phi_j2_prefit[0] = j2.phi
    phi_nu_prefit[0] = nu.phi

    
    tree_out.Fill()

#outFile.cd()
tree_out.Write()
outFile.Close()


#######TO DO run with a mean value of 1 for jets
