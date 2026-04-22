#ifndef WWFunctions_H
#define WWFunctions_H

#include <cmath>
#include <memory>
#include <functional>
#include "TMath.h"
#include "TVector2.h"
#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"
#include "Math/Vector4D.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"

namespace FCCAnalyses { namespace WWFunctions {

static constexpr float ECM = 160.0f;

// ── angle helpers ──────────────────────────────────────────────────────────

float deltaTheta3D(const ROOT::Math::PxPyPzEVector& r,
                   const ROOT::Math::PxPyPzEVector& g) {
    double dot = r.Px()*g.Px() + r.Py()*g.Py() + r.Pz()*g.Pz();
    double mag = r.P() * g.P();
    if (mag <= 0) return -1.0;
    double cosang = dot / mag;
    if (cosang >  1.0) cosang =  1.0;
    if (cosang < -1.0) cosang = -1.0;
    return acos(cosang);
}

// used only by matchJetsAndComputeResolution
static float jetAngle(float px1, float py1, float pz1,
                      float px2, float py2, float pz2) {
    float dot  = px1*px2 + py1*py2 + pz1*pz2;
    float mag1 = sqrt(px1*px1 + py1*py1 + pz1*pz1);
    float mag2 = sqrt(px2*px2 + py2*py2 + pz2*pz2);
    return acos(dot / (mag1*mag2));
}

// ── jet matching / resolution helpers ─────────────────────────────────────

ROOT::VecOps::RVec<float> matchJetsAndComputeResolution(
    const ROOT::VecOps::RVec<float>& reco_px,
    const ROOT::VecOps::RVec<float>& reco_py,
    const ROOT::VecOps::RVec<float>& reco_pz,
    const ROOT::VecOps::RVec<float>& reco_E,
    const ROOT::VecOps::RVec<float>& truth_px,
    const ROOT::VecOps::RVec<float>& truth_py,
    const ROOT::VecOps::RVec<float>& truth_pz,
    const ROOT::VecOps::RVec<float>& truth_E) {

    ROOT::VecOps::RVec<float> resolution;
    for (size_t i = 0; i < reco_E.size(); ++i) {
        float bestAngle = 999.;
        int   bestMatch = -1;
        for (size_t j = 0; j < truth_E.size(); ++j) {
            float ang = jetAngle(reco_px[i], reco_py[i], reco_pz[i],
                                 truth_px[j], truth_py[j], truth_pz[j]);
            if (ang < bestAngle) { bestAngle = ang; bestMatch = j; }
        }
        if (bestMatch >= 0) {
            float resp = (reco_E[i] - truth_E[bestMatch]) / truth_E[bestMatch];
            resolution.push_back(resp);
        }
    }
    return resolution;
}

std::pair<ROOT::Math::PxPyPzEVector, ROOT::Math::PxPyPzEVector>
matchJets2(const ROOT::Math::PxPyPzEVector& r1,
           const ROOT::Math::PxPyPzEVector& r2,
           const ROOT::Math::PxPyPzEVector& g1,
           const ROOT::Math::PxPyPzEVector& g2) {

    auto dR = [](const ROOT::Math::PxPyPzEVector& a,
                 const ROOT::Math::PxPyPzEVector& b) {
        double deta = a.Eta() - b.Eta();
        double dphi = TVector2::Phi_mpi_pi(a.Phi() - b.Phi());
        return sqrt(deta*deta + dphi*dphi);
    };

    double dR_A = dR(r1, g1) + dR(r2, g2);
    double dR_B = dR(r1, g2) + dR(r2, g1);
    return (dR_A < dR_B) ? std::make_pair(g1, g2) : std::make_pair(g2, g1);
}

ROOT::VecOps::RVec<ROOT::Math::PxPyPzEVector>
build_p4(const ROOT::VecOps::RVec<float>& px,
         const ROOT::VecOps::RVec<float>& py,
         const ROOT::VecOps::RVec<float>& pz,
         const ROOT::VecOps::RVec<float>& e) {
    ROOT::VecOps::RVec<ROOT::Math::PxPyPzEVector> out;
    for (size_t i = 0; i < px.size(); ++i)
        out.emplace_back(px[i], py[i], pz[i], e[i]);
    return out;
}

// ── shared building block ──────────────────────────────────────────────────

// Builds W → lep + nu from spherical-coordinate RVec columns (index [0] used).
static TLorentzVector _Wlep_from_spherical(
    const ROOT::VecOps::RVec<float>& lep_p,   const ROOT::VecOps::RVec<float>& lep_phi,
    const ROOT::VecOps::RVec<float>& lep_theta, const ROOT::VecOps::RVec<float>& lep_e,
    const ROOT::VecOps::RVec<float>& nu_p,    const ROOT::VecOps::RVec<float>& nu_phi,
    const ROOT::VecOps::RVec<float>& nu_theta,  const ROOT::VecOps::RVec<float>& nu_e) {
    TLorentzVector lep, nu;
    lep.SetPxPyPzE(lep_p[0] * cos(lep_phi[0]) * sin(lep_theta[0]),
                   lep_p[0] * sin(lep_phi[0]) * sin(lep_theta[0]),
                   lep_p[0] * cos(lep_theta[0]),
                   lep_e[0]);
    nu.SetPxPyPzE(nu_p[0] * cos(nu_phi[0]) * sin(nu_theta[0]),
                  nu_p[0] * sin(nu_phi[0]) * sin(nu_theta[0]),
                  nu_p[0] * cos(nu_theta[0]),
                  nu_e[0]);
    return lep + nu;
}

// Builds massless p4 from first element of pt/eta/phi RVec columns.
static TLorentzVector _p4_from_rvec(const ROOT::VecOps::RVec<float>& pt,
                                     const ROOT::VecOps::RVec<float>& eta,
                                     const ROOT::VecOps::RVec<float>& phi) {
    TLorentzVector v;
    v.SetPtEtaPhiM(pt[0], eta[0], phi[0], 0);
    return v;
}

// ── reco-level kinematics ──────────────────────────────────────────────────

TLorentzVector Wlep_reco(float Isolep_p, float Isolep_phi, float Isolep_theta, float Isolep_e,
                          float missing_p, float missing_p_phi, float missing_p_theta) {
    TLorentzVector lep, nu;
    lep.SetPxPyPzE(Isolep_p * cos(Isolep_phi) * sin(Isolep_theta),
                   Isolep_p * sin(Isolep_phi) * sin(Isolep_theta),
                   Isolep_p * cos(Isolep_theta),
                   Isolep_e);
    nu.SetPxPyPzE(missing_p * cos(missing_p_phi) * sin(missing_p_theta),
                  missing_p * sin(missing_p_phi) * sin(missing_p_theta),
                  missing_p * cos(missing_p_theta),
                  missing_p);
    return lep + nu;
}

TLorentzVector Isoleps_p4_reco(float p, float phi, float theta, float e) {
    TLorentzVector v;
    v.SetPxPyPzE(p * cos(phi) * sin(theta),
                 p * sin(phi) * sin(theta),
                 p * cos(theta),
                 e);
    return v;
}

TLorentzVector missing_p_p4(float p, float phi, float theta) {
    TLorentzVector v;
    v.SetPxPyPzE(p * cos(phi) * sin(theta),
                 p * sin(phi) * sin(theta),
                 p * cos(theta),
                 p);
    return v;
}

TLorentzVector Whad_reco(const ROOT::VecOps::RVec<TLorentzVector>& jets_p4) {
    TLorentzVector j1, j2;
    j1.SetPxPyPzE(jets_p4[0].Px(), jets_p4[0].Py(), jets_p4[0].Pz(), jets_p4[0].E());
    j2.SetPxPyPzE(jets_p4[1].Px(), jets_p4[1].Py(), jets_p4[1].Pz(), jets_p4[1].E());
    return j1 + j2;
}

float deltaM(int nIsolep, int nRecoJets,
             const TLorentzVector& Wlep, const TLorentzVector& Whad) {
    if (nIsolep < 1 || nRecoJets < 2) return -1.0;
    TLorentzVector P_initial(0, 0, 0, ECM);
    return (P_initial - (Wlep + Whad)).M();
}

// ── gen-level kinematics ───────────────────────────────────────────────────

TLorentzVector lep_p4_gen(const ROOT::VecOps::RVec<float>& pt,
                           const ROOT::VecOps::RVec<float>& eta,
                           const ROOT::VecOps::RVec<float>& phi) {
    return _p4_from_rvec(pt, eta, phi);
}

TLorentzVector nu_p4_gen(const ROOT::VecOps::RVec<float>& pt,
                          const ROOT::VecOps::RVec<float>& eta,
                          const ROOT::VecOps::RVec<float>& phi) {
    return _p4_from_rvec(pt, eta, phi);
}

TLorentzVector Wlep_gen(const ROOT::VecOps::RVec<float>& lep_pt,
                         const ROOT::VecOps::RVec<float>& lep_eta,
                         const ROOT::VecOps::RVec<float>& lep_phi,
                         const ROOT::VecOps::RVec<float>& nu_pt,
                         const ROOT::VecOps::RVec<float>& nu_eta,
                         const ROOT::VecOps::RVec<float>& nu_phi) {
    return _p4_from_rvec(lep_pt, lep_eta, lep_phi) + _p4_from_rvec(nu_pt, nu_eta, nu_phi);
}

TLorentzVector Wlep_gen_old(const ROOT::VecOps::RVec<float>& lep_p,
                             const ROOT::VecOps::RVec<float>& lep_phi,
                             const ROOT::VecOps::RVec<float>& lep_theta,
                             const ROOT::VecOps::RVec<float>& lep_e,
                             const ROOT::VecOps::RVec<float>& nu_p,
                             const ROOT::VecOps::RVec<float>& nu_phi,
                             const ROOT::VecOps::RVec<float>& nu_theta,
                             const ROOT::VecOps::RVec<float>& nu_e) {
    return _Wlep_from_spherical(lep_p, lep_phi, lep_theta, lep_e,
                                nu_p,  nu_phi,  nu_theta,  nu_e);
}

TLorentzVector Wlep_gen_status2(const ROOT::VecOps::RVec<float>& lep_p,
                                 const ROOT::VecOps::RVec<float>& lep_phi,
                                 const ROOT::VecOps::RVec<float>& lep_theta,
                                 const ROOT::VecOps::RVec<float>& lep_e,
                                 const ROOT::VecOps::RVec<float>& nu_p,
                                 const ROOT::VecOps::RVec<float>& nu_phi,
                                 const ROOT::VecOps::RVec<float>& nu_theta,
                                 const ROOT::VecOps::RVec<float>& nu_e) {
    return _Wlep_from_spherical(lep_p, lep_phi, lep_theta, lep_e,
                                nu_p,  nu_phi,  nu_theta,  nu_e);
}

TLorentzVector Whad_gen_status2(const ROOT::VecOps::RVec<float>& p,
                                 const ROOT::VecOps::RVec<float>& phi,
                                 const ROOT::VecOps::RVec<float>& theta,
                                 const ROOT::VecOps::RVec<float>& e) {
    TLorentzVector q1, q2;
    q1.SetPxPyPzE(p[1] * cos(phi[1]) * sin(theta[1]),
                  p[1] * sin(phi[1]) * sin(theta[1]),
                  p[1] * cos(theta[1]),
                  e[1]);
    q2.SetPxPyPzE(p[0] * cos(phi[0]) * sin(theta[0]),
                  p[0] * sin(phi[0]) * sin(theta[0]),
                  p[0] * cos(theta[0]),
                  e[0]);
    return q1 + q2;
}

TLorentzVector Whad_gen_qq_fromele(const ROOT::VecOps::RVec<float>& p,
                                    const ROOT::VecOps::RVec<float>& phi,
                                    const ROOT::VecOps::RVec<float>& theta,
                                    const ROOT::VecOps::RVec<float>& e) {
    TLorentzVector j1, j2;
    j1.SetPxPyPzE(p[1] * cos(phi[1]) * sin(theta[1]),
                  p[1] * sin(phi[1]) * sin(theta[1]),
                  p[1] * cos(theta[1]),
                  e[1]);
    j2.SetPxPyPzE(p[0] * cos(phi[0]) * sin(theta[0]),
                  p[0] * sin(phi[0]) * sin(theta[0]),
                  p[0] * cos(theta[0]),
                  e[0]);
    return j1 + j2;
}

float m_qq_fromele(const ROOT::VecOps::RVec<float>& pt,
                   const ROOT::VecOps::RVec<float>& eta,
                   const ROOT::VecOps::RVec<float>& phi) {
    TLorentzVector j1, j2;
    j1.SetPtEtaPhiM(pt[0], eta[0], phi[0], 0);
    j2.SetPtEtaPhiM(pt[1], eta[1], phi[1], 0);
    return (j1 + j2).M();
}

float Whad_gen_old(const ROOT::VecOps::RVec<float>& px,
                   const ROOT::VecOps::RVec<float>& py,
                   const ROOT::VecOps::RVec<float>& pz,
                   const ROOT::VecOps::RVec<float>& e) {
    TLorentzVector j1, j2;
    j1.SetPxPyPzE(px[0], py[0], pz[0], e[0]);
    j2.SetPxPyPzE(px[1], py[1], pz[1], e[1]);
    return (j1 + j2).M();
}

TLorentzVector Whad_gen(const ROOT::VecOps::RVec<float>& pt,
                         const ROOT::VecOps::RVec<float>& eta,
                         const ROOT::VecOps::RVec<float>& phi) {
    TLorentzVector j1, j2;
    j1.SetPtEtaPhiM(pt[0], eta[0], phi[0], 0);
    j2.SetPtEtaPhiM(pt[1], eta[1], phi[1], 0);
    return j1 + j2;
}

float m_gen_lnuqq(const ROOT::VecOps::RVec<float>& lep_status2_p,
                   const ROOT::VecOps::RVec<float>& q_fromele_p,
                   const ROOT::VecOps::RVec<float>& lep_px,
                   const ROOT::VecOps::RVec<float>& lep_py,
                   const ROOT::VecOps::RVec<float>& lep_pz,
                   const ROOT::VecOps::RVec<float>& lep_p,
                   const ROOT::VecOps::RVec<float>& nu_px,
                   const ROOT::VecOps::RVec<float>& nu_py,
                   const ROOT::VecOps::RVec<float>& nu_pz,
                   const ROOT::VecOps::RVec<float>& nu_p,
                   const ROOT::VecOps::RVec<float>& q_px,
                   const ROOT::VecOps::RVec<float>& q_py,
                   const ROOT::VecOps::RVec<float>& q_pz,
                   const ROOT::VecOps::RVec<float>& q_e) {
    if (lep_status2_p.size() < 1 || q_fromele_p.size() < 2) return -1.0;
    TLorentzVector lep, nu, j1, j2;
    lep.SetPxPyPzE(lep_px[0], lep_py[0], lep_pz[0], lep_p[0]);
    nu.SetPxPyPzE( nu_px[0],  nu_py[0],  nu_pz[0],  nu_p[0]);
    j1.SetPxPyPzE( q_px[0],   q_py[0],   q_pz[0],   q_e[0]);
    j2.SetPxPyPzE( q_px[1],   q_py[1],   q_pz[1],   q_e[1]);
    return (lep + nu + j1 + j2).M();
}

// Takes pre-computed W 4-vectors (already defined as RDF columns before this).
float sumP_gen_new(const TLorentzVector& Wlep, const TLorentzVector& Whad) {
    return Wlep.Pz() + Whad.Pz() + Wlep.Px() + Whad.Px() + Wlep.Py() + Whad.Py();
}

// ── kinematic fit ──────────────────────────────────────────────────────────

// Fit constants (matching W_kin_fit.py)
static constexpr double KF_SIGMA_SQRTS      = 0.12;
static constexpr double KF_JET_SCALE_BIAS   = 1.01;
static constexpr double KF_JET_SCALE_SIGMA  = 0.05;
static constexpr double KF_LEP_SCALE_BIAS   = 1.0;
static constexpr double KF_LEP_SCALE_SIGMA  = 0.002;
static constexpr double KF_MET_SCALE_BIAS   = 1.0;
static constexpr double KF_MET_SCALE_SIGMA  = 0.015;
static constexpr double KF_MOMENTUM_PENALTY = 100.0;
static constexpr double KF_JET1_THETA_RMS   = 0.05;
static constexpr double KF_JET1_PHI_RMS     = 0.05;
static constexpr double KF_JET2_THETA_RMS   = 0.05;
static constexpr double KF_JET2_PHI_RMS     = 0.06;
static constexpr double KF_MET_THETA_RMS    = 0.04;
static constexpr double KF_MET_PHI_RMS      = 0.05;
static constexpr double KF_MW_INIT          = 80.419;
static constexpr double KF_GW_FIXED         = 2.049;

struct KinFitResult {
    float mW, gW;
    float s1, s2, sl, sn;
    float t1, t2, tn;
    float p1, p2, pn;
    float chi2;
    int   valid;
    float mWlep_postfit, mWhad_postfit;
    float pt_j1_postfit, pt_j2_postfit, pt_lep_postfit, pt_nu_postfit;
    float Wlep_px_postfit, Wlep_py_postfit, Wlep_pz_postfit;
    float Whad_px_postfit, Whad_py_postfit, Whad_pz_postfit;
    float theta_j1_postfit, theta_j2_postfit, theta_nu_postfit;
    float phi_j1_postfit,   phi_j2_postfit,   phi_nu_postfit;
    float deltaP_postfit;
};

// Massless 4-vector from spherical coordinates.
static TLorentzVector _vec_spherical(double p, double theta, double phi) {
    double st = std::sin(theta), ct = std::cos(theta);
    TLorentzVector v;
    v.SetPxPyPzE(p * st * std::cos(phi), p * st * std::sin(phi), p * ct, p);
    return v;
}

KinFitResult kinFit(float jet1_p,    float jet1_theta,    float jet1_phi,
                    float jet2_p,    float jet2_theta,    float jet2_phi,
                    float Isolep_p,  float Isolep_theta,  float Isolep_phi,
                    float missing_p, float missing_p_theta, float missing_p_phi) {

    KinFitResult result{};
    result.gW    = KF_GW_FIXED;
    result.valid = 0;
    result.chi2  = 999.0f;

    if (Isolep_p < 0 || jet1_p <= 0 || jet2_p <= 0 || missing_p <= 0)
        return result;

    // Parameters: x[0]=mW, x[1]=s1, x[2]=s2, x[3]=sl, x[4]=sn,
    //             x[5]=t1, x[6]=t2, x[7]=tn, x[8]=p1, x[9]=p2, x[10]=pn
    auto chi2fn = [=](const double* x) -> double {
        const double mW = x[0];
        const double s1 = x[1], s2 = x[2], sl = x[3], sn = x[4];
        const double t1 = x[5], t2 = x[6], tn = x[7];
        const double p1 = x[8], p2 = x[9], pn = x[10];

        TLorentzVector j1f = _vec_spherical(jet1_p*s1,    jet1_theta    + t1*KF_JET1_THETA_RMS, jet1_phi    + p1*KF_JET1_PHI_RMS);
        TLorentzVector j2f = _vec_spherical(jet2_p*s2,    jet2_theta    + t2*KF_JET2_THETA_RMS, jet2_phi    + p2*KF_JET2_PHI_RMS);
        TLorentzVector lf  = _vec_spherical(Isolep_p*sl,  Isolep_theta,                         Isolep_phi);
        TLorentzVector nf  = _vec_spherical(missing_p*sn, missing_p_theta + tn*KF_MET_THETA_RMS, missing_p_phi + pn*KF_MET_PHI_RMS);

        TLorentzVector Wh = j1f + j2f;
        TLorentzVector Wl = lf  + nf;
        TLorentzVector WW = Wh  + Wl;

        double mh = Wh.M(), ml = Wl.M();
        double mwgw = mW * KF_GW_FIXED;
        double dh   = mh*mh - mW*mW,   dl = ml*ml - mW*mW;
        double bw_h = mwgw / (dh*dh + mwgw*mwgw);
        double bw_l = mwgw / (dl*dl + mwgw*mwgw);
        double bw_term = -2.0 * (std::log(bw_h) + std::log(bw_l));

        double cons = std::pow(WW.E() - ECM, 2) / (KF_SIGMA_SQRTS * KF_SIGMA_SQRTS)
                    + (std::pow(Wh.Px()+Wl.Px(), 2) + std::pow(Wh.Py()+Wl.Py(), 2) + std::pow(Wh.Pz()+Wl.Pz(), 2)) * KF_MOMENTUM_PENALTY;

        double scale_pen = std::pow((s1-KF_JET_SCALE_BIAS)/KF_JET_SCALE_SIGMA, 2)
                         + std::pow((s2-KF_JET_SCALE_BIAS)/KF_JET_SCALE_SIGMA, 2)
                         + std::pow((sl-KF_LEP_SCALE_BIAS)/KF_LEP_SCALE_SIGMA, 2)
                         + std::pow((sn-KF_MET_SCALE_BIAS)/KF_MET_SCALE_SIGMA, 2);

        double angular = t1*t1 + t2*t2 + tn*tn + p1*p1 + p2*p2 + pn*pn;

        return bw_term + cons + scale_pen + angular;
    };

    std::function<double(const double*)> fObj = chi2fn;
    ROOT::Math::Functor functor(fObj, 11);

    std::unique_ptr<ROOT::Math::Minimizer> minimizer(
        ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad")
    );
    minimizer->SetFunction(functor);
    minimizer->SetMaxFunctionCalls(10000);
    minimizer->SetTolerance(1e-6);
    minimizer->SetStrategy(2);
    minimizer->SetPrintLevel(0);

    minimizer->SetVariable(0,  "mW", KF_MW_INIT, 0.1);
    minimizer->SetVariableLimits(0, 0.0, 200.0);
    minimizer->SetVariable(1,  "s1", 1.0,  0.01);
    minimizer->SetVariable(2,  "s2", 1.0,  0.01);
    minimizer->SetVariable(3,  "sl", 1.0,  0.001);
    minimizer->SetVariable(4,  "sn", 1.0,  0.01);
    minimizer->SetVariable(5,  "t1", 0.0,  0.1);
    minimizer->SetVariable(6,  "t2", 0.0,  0.1);
    minimizer->SetVariable(7,  "tn", 0.0,  0.1);
    minimizer->SetVariable(8,  "p1", 0.0,  0.1);
    minimizer->SetVariable(9,  "p2", 0.0,  0.1);
    minimizer->SetVariable(10, "pn", 0.0,  0.1);

    minimizer->Minimize();
    minimizer->Minimize();  // second pass for better convergence

    result.valid = (minimizer->Status() == 0) ? 1 : 0;
    result.chi2  = minimizer->MinValue();

    const double* x = minimizer->X();
    result.mW = x[0];
    result.s1 = x[1]; result.s2 = x[2]; result.sl = x[3]; result.sn = x[4];
    result.t1 = x[5]; result.t2 = x[6]; result.tn = x[7];
    result.p1 = x[8]; result.p2 = x[9]; result.pn = x[10];

    // Post-fit kinematics
    TLorentzVector j1f = _vec_spherical(jet1_p*result.s1,    jet1_theta    + result.t1*KF_JET1_THETA_RMS, jet1_phi    + result.p1*KF_JET1_PHI_RMS);
    TLorentzVector j2f = _vec_spherical(jet2_p*result.s2,    jet2_theta    + result.t2*KF_JET2_THETA_RMS, jet2_phi    + result.p2*KF_JET2_PHI_RMS);
    TLorentzVector lf  = _vec_spherical(Isolep_p*result.sl,  Isolep_theta,                                Isolep_phi);
    TLorentzVector nf  = _vec_spherical(missing_p*result.sn, missing_p_theta + result.tn*KF_MET_THETA_RMS, missing_p_phi + result.pn*KF_MET_PHI_RMS);

    TLorentzVector Wh = j1f + j2f;
    TLorentzVector Wl = lf  + nf;

    result.mWlep_postfit    = Wl.M();       result.mWhad_postfit    = Wh.M();
    result.pt_j1_postfit    = j1f.Pt();     result.pt_j2_postfit    = j2f.Pt();
    result.pt_lep_postfit   = lf.Pt();      result.pt_nu_postfit    = nf.Pt();
    result.Wlep_px_postfit  = Wl.Px();      result.Wlep_py_postfit  = Wl.Py();  result.Wlep_pz_postfit = Wl.Pz();
    result.Whad_px_postfit  = Wh.Px();      result.Whad_py_postfit  = Wh.Py();  result.Whad_pz_postfit = Wh.Pz();
    result.theta_j1_postfit = j1f.Theta();  result.theta_j2_postfit = j2f.Theta(); result.theta_nu_postfit = nf.Theta();
    result.phi_j1_postfit   = j1f.Phi();    result.phi_j2_postfit   = j2f.Phi();   result.phi_nu_postfit   = nf.Phi();
    result.deltaP_postfit   = std::sqrt(std::pow(Wl.Px()+Wh.Px(), 2)
                                      + std::pow(Wl.Py()+Wh.Py(), 2)
                                      + std::pow(Wl.Pz()+Wh.Pz(), 2));
    return result;
}

}}

#endif
