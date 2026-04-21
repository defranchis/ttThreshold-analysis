import os, sys, json, math, iminuit
import numpy as np
from datetime import datetime
from iminuit import Minuit

ifile = "outputs/treemaker/lnuqq/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
df = ROOT.RDataFrame("events", ifile)

df = df.Define(
    "lep_p4",
    "ROOT::Math::PtEtaPhiMVector(lep_pt, lep_eta, lep_phi, 0.0)"
)
df = df.Define(
    "jet1_p4",
    "ROOT::Math::PtEtaPhiMVector(jet1_pt, jet1_eta, jet1_phi, jet1_mass)"
)

df = df.Define(
    "jet2_p4",
    "ROOT::Math::PtEtaPhiMVector(jet2_pt, jet2_eta, jet2_phi, jet2_mass)"
)


df = df.Define(
    "metx",
    "met*cos(met_phi)"
)

df = df.Define(
    "mety",
    "met*sin(met_phi)"
)
ROOT.gInterpreter.Declare("""

#include <Math/Vector4D.h>

double chi2_full(double al, double aj1, double aj2, double pznu,
                 ROOT::Math::PtEtaPhiMVector lep,
                 ROOT::Math::PtEtaPhiMVector j1,
                 ROOT::Math::PtEtaPhiMVector j2,
                 double metx, double mety)
{

    const double mW = 80.379;
    const double gW = 2.085;
    const double sqrts = 160.0;

    auto lep_f = lep * al;
    auto j1_f  = j1 * aj1;
    auto j2_f  = j2 * aj2;

    double pxnu = metx;
    double pynu = mety;

    double Enu = sqrt(pxnu*pxnu + pynu*pynu + pznu*pznu);

    ROOT::Math::PxPyPzEVector nu(pxnu,pynu,pznu,Enu);

    auto Wlep = lep_f + nu;
    auto Whad = j1_f + j2_f;

    double mlep = Wlep.M();
    double mhad = Whad.M();

    double chiW =
        pow((mlep-mW)/gW,2) +
        pow((mhad-mW)/gW,2);

    double chiRes =
        pow((al-1)/0.02,2) +
        pow((aj1-1)/0.05,2) +
        pow((aj2-1)/0.05,2);

    double px_sum = lep_f.Px() + j1_f.Px() + j2_f.Px() + pxnu;
    double py_sum = lep_f.Py() + j1_f.Py() + j2_f.Py() + pynu;
    double pz_sum = lep_f.Pz() + j1_f.Pz() + j2_f.Pz() + pznu;

    double E_sum =
        lep_f.E() + j1_f.E() + j2_f.E() + Enu;

    double chiCons =
        1000*px_sum*px_sum +
        1000*py_sum*py_sum +
        1000*pz_sum*pz_sum +
        1000*(E_sum - sqrts)*(E_sum - sqrts);

    return chiW + chiRes + chiCons;
}

""")

ROOT.gInterpreter.Declare("""

#include <Math/Factory.h>
#include <Math/Functor.h>

double fit_event(ROOT::Math::PtEtaPhiMVector lep,
                 ROOT::Math::PtEtaPhiMVector j1,
                 ROOT::Math::PtEtaPhiMVector j2,
                 double metx,
                 double mety)
{

    auto f = [&](const double *x){

        return chi2_full(x[0],x[1],x[2],x[3],
                         lep,j1,j2,
                         metx,mety);

    };

    ROOT::Math::Functor functor(f,4);

    auto minimizer =
        ROOT::Math::Factory::CreateMinimizer("Minuit2","Migrad");

    minimizer->SetFunction(functor);

    minimizer->SetVariable(0,"al",1,0.01);
    minimizer->SetVariable(1,"aj1",1,0.02);
    minimizer->SetVariable(2,"aj2",1,0.02);
    minimizer->SetVariable(3,"pznu",0,1);

    minimizer->Minimize();

    return minimizer->MinValue();
}

""")
df = df.Define(
    "chi2",
    "fit_event(lep_p4, j1_p4, j2_p4, metx, mety)"
)


df.Snapshot(
    "fit_tree",
    "w_fit_output.root",
    ["chi2"]
)

print("Fit finished. Output written to w_fit_output.root")


def chi2_lnuqq(params, event):
    mW, gW, c_plep, c_pmet, c_pjet1, c_pjet2, c_massW = params
    plep = event["lep_p"] * c_plep
    pmet = event["pmet_p"] * c_pmet
    pjet1 = event["jet1_p"] * c_pjet1
    pjet2 = event["jet2_p"] * c_pjet2
    massW = event["massW"] * c_massW
    lep_f  = ROOT.TLorentzVector(0,0,0,0)
    pmet_f = ROOT.TLorentzVector(0,0,0,0)
    pjet1_f  = ROOT.TLorentzVector(0,0,0,0)
    pjet2_f  = ROOT.TLorentzVector(0,0,0,0)
    lep_f.SetPxPyPzE(plep*cos(event["lep_phi"])*sin(event["lep_theta"]), plep*sin(event["lep_phi"])*sin(event["lep_theta"]),plep*cos(event["lep_theta"]), plep)
    pmet_f.SetPxPyPzE(pmet_f*cos(event["missin_p_phi"])*sin(event["missin_p_theta"]), pmet*sin(event["missin_p_phi"])*sin(event["missin_p_theta"]),pmet*cos(event["missin_p_theta"]), pmet)
    pjet1.SetPxPyPzE(pjet1*cos(event["jet1_phi"])*sin(event["jet1_theta"]), pjet1*sin(event["jet1_phi"])*sin(event["jet1_theta"]),pjet1*cos(event["jet1_theta"]), pjet1)
    pjet2.SetPxPyPzE(pjet2*cos(event["jet2_phi"])*sin(event["jet2_theta"]), pjet2*sin(event["jet2_phi"])*sin(event["jet2_theta"]),pjet2*cos(event["jet2_theta"]), pjet2)
    m_lnu  = (lep_f+pmet_f).M()
    m_qq   = (pjet1_f+pjet2_f).M()
    p_tot = (lep_f+pmet_f+pjet1_f+pjet2_f).P()
    p_lnu = (lep_f+pmet_f).P()
    p_qq = (pjet1_f+pjet2_f).P()
    
    sqrts = 160.0 #+ dE
    sigma_lnuqq=1.0
    dE=0.0001 # BE spread 1keV
    sigma_plnu = df.StdDev("delta_plnu").GetValue()
    sigma_pqq  = df.StdDev("delta_pqq").GetValue() #reco -gen
    sigma_mlnuqq  = df.StdDev("delta_mlnuqq").GetValue()
    sigma_mqq = df.StdDev("delta_mqq").GetValue()
    sigma_mlnu = df.StdDev("delta_mlnu").GetValue()
    chi2_mW1 = (m_qq-mW)**2/(sigma_m_qq**2+dW**2)
    chi2_mW2 = (m_lnu-mW)**2/(sigma_m_lnu**2+dW**2)
    chi2_Econs = (m_lnuqq - sqrts)**2/(sigma_lnuqq**2+dE**2)
    chi2_Pcons = (p_lnu -p_qq)**2/(sigma_p_lnu**2+sigma_p_qq**2)
    bw = 2*np.log((m_jj*m_jj - mW*mW)**2 + (mW*gW)**2)
    bw += 2*np.log((m_lnu*m_lnu - mW*mW)**2 + (mW*gW)**2)
    cons = ((p_tot.E() - SQRTS)**2) / sigma_p**2
    
def chi2_event(params, event):
    mW, gW, dE, dj1, dj2, dl = params

    # Beam energy
    sqrts = 160.0 + dE

    # Scale momenta
    p_j1 = event.p_j1 * (1 + dj1)
    p_j2 = event.p_j2 * (1 + dj2)
    p_l  = event.p_l  * (1 + dl)

    # Reconstructed hadronic W
    p_W = p_j1 + p_j2
    m_qq = p_W.mass()

    # Breit-Wigner term
    chi2_bw = ((m_qq - mW)**2) / (gW**2)

    # Energy-momentum conservation
    E_tot = p_j1.E + p_j2.E + p_l.E + event.p_nu.E
    chi2_Econs = ((E_tot - sqrts)/event.sigma_Econs)**2

    # Constraints
    chi2_constraints = (
        (dE / event.sigma_E)**2 +
        (dj1 / event.sigma_j)**2 +
        (dj2 / event.sigma_j)**2 +
        (dl  / event.sigma_l)**2
    )

    return chi2_bw + chi2_Econs + chi2_constraints
