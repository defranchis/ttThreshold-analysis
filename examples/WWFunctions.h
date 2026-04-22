#ifndef WWFunctions_H
#define WWFunctions_H

#include <cmath>
#include "TMath.h"
#include "TVector2.h"
#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"
#include "Math/Vector4D.h"

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

}}

#endif
