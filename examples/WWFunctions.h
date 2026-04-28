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
#include "response/functions/dcb_params_ecm160.h"

namespace FCCAnalyses { namespace WWFunctions {

// Pull in DCB evaluators and fitted params from the generated header.
using namespace ::WWFunctions;

inline float ECM = 160.0f;

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

// Kinematic fit constants.
// Momentum scale params (s1,s2,sl,sn) are now response = p_reco/p_gen;
// angular params (t1,t2,tn,p1,p2,pn) are now absolute shifts in radians.
// Constraints use DCB/DCB+G PDFs from dcb_params_ecm160.h.
// sqrts constraint uses DCBG_M_GEN_LNUQQ_MINUS_ECM: fitted PDF of (m_gen_WW - ECM),
// which is the physically correct ISR-induced invariant-mass deficit of the final state.
static constexpr double KF_MW_INIT = 80.419;
static constexpr double KF_GW_FIXED = 2.049;
static constexpr int    KF_NDIM    = 13;   // free parameters when gW is fixed (added tl, pl)

struct KinFitResult {
    float mW, gW;
    float s1, s2, sl, sn;
    float t1, t2, tn, tl;   // theta shifts: jet1, jet2, MET, lepton
    float p1, p2, pn, pl;   // phi shifts:   jet1, jet2, MET, lepton
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

// ── lightweight massless 4-vector (E = p) for chi2 evaluations ───────────
// Avoids TLorentzVector construction/virtual-dispatch overhead in the hot path.
struct Vec4 {
    double px, py, pz, E;
    static Vec4 spherical(double p, double th, double ph) {
        double sth = std::sin(th), cth = std::cos(th);
        return {p * sth * std::cos(ph), p * sth * std::sin(ph), p * cth, p};
    }
    Vec4 operator+(const Vec4& o) const { return {px+o.px, py+o.py, pz+o.pz, E+o.E}; }
    double M2() const { return E*E - px*px - py*py - pz*pz; }
    double M()  const { double m2 = M2(); return m2 > 0.0 ? std::sqrt(m2) : 0.0; }
};

// Shared chi2 kernel — called by both kinFit and kinFitBFGS lambdas.
static double _chi2_eval(
    double mW, double gW,
    double s1, double s2, double sl, double sn,
    double t1, double t2, double tn,
    double p1, double p2, double pn,
    double jet1_p, double jet1_theta, double jet1_phi,
    double jet2_p, double jet2_theta, double jet2_phi,
    double lep_p,  double lep_theta,  double lep_phi,
    double nu_p,   double nu_theta,   double nu_phi)
{
    if (gW <= 0.0 || s1 <= 0.0 || s2 <= 0.0 || sl <= 0.0 || sn <= 0.0) return 1e10;

    // s = response = p_reco/p_gen  →  p_fit = p_meas / s
    // t, p = absolute angular shifts in radians  →  angle_fit = angle_meas - shift
    Vec4 j1 = Vec4::spherical(jet1_p/s1, jet1_theta - t1, jet1_phi - p1);
    Vec4 j2 = Vec4::spherical(jet2_p/s2, jet2_theta - t2, jet2_phi - p2);
    Vec4 lf = Vec4::spherical(lep_p/sl,  lep_theta,        lep_phi);
    Vec4 nf = Vec4::spherical(nu_p/sn,   nu_theta   - tn,  nu_phi  - pn);

    Vec4 Wh = j1 + j2;
    Vec4 Wl = lf + nf;

    double mh = Wh.M(), ml = Wl.M();
    double mwgw = mW * gW;
    double dh   = mh*mh - mW*mW,  dl = ml*ml - mW*mW;
    double bw_h = mwgw / (dh*dh + mwgw*mwgw);
    double bw_l = mwgw / (dl*dl + mwgw*mwgw);
    double bw_term = -2.0 * (std::log(bw_h) + std::log(bw_l));

    Vec4 WW = Wh + Wl;
    // sqrts: gen WW mass minus ECM (ISR energy deficit); 3-momentum: gen-level ISR distributions
    double cons = dcb_gauss_neg2logpdf(WW.M() - ECM, DCBG_M_GEN_LNUQQ_MINUS_ECM)
                + dcb_gauss_neg2logpdf(WW.px, DCBG_PX_TOT_GEN)
                + dcb_gauss_neg2logpdf(WW.py, DCBG_PY_TOT_GEN)
                + dcb_gauss_neg2logpdf(WW.pz, DCBG_PZ_TOT_GEN);

    double scale_pen = dcb_gauss_neg2logpdf(s1, DCBG_JET1_P_RESP)
                     + dcb_gauss_neg2logpdf(s2, DCBG_JET2_P_RESP)
                     + dcb_neg2logpdf(sl, DCB_LEP_P_RESP)
                     + dcb_neg2logpdf(sn, DCB_MET_P_RESP);

    double angular = dcb_neg2logpdf(t1, DCB_JET1_THETA_RESOL)
                   + dcb_neg2logpdf(t2, DCB_JET2_THETA_RESOL)
                   + dcb_neg2logpdf(tn, DCB_MET_THETA_RESOL)
                   + dcb_neg2logpdf(p1, DCB_JET1_PHI_RESOL)
                   + dcb_neg2logpdf(p2, DCB_JET2_PHI_RESOL)
                   + dcb_neg2logpdf(pn, DCB_MET_PHI_RESOL);

    return bw_term + cons + scale_pen + angular;
}

// ── BFGS minimizer (option C+A) ───────────────────────────────────────────
// Template avoids std::function overhead (no heap, no virtual dispatch).
// All work arrays are on the stack, so the function is inherently thread-safe
// without any thread_local annotation — suitable for multithreaded RDataFrame.
//
// Uses central finite differences for the gradient and Armijo backtracking
// for the line search. Resets H to identity on non-descent or line-search
// failure so it never gets permanently stuck.
//
// Returns 0 on convergence (||grad|| < GTOL), 1 if max iterations reached.
template<typename Func, int N>
static int _bfgs_minimize(const Func& f, double* x, double& fmin) {
    constexpr int    MAXITER = 300;
    constexpr double GTOL    = 1e-5;
    constexpr double C1      = 1e-4;    // Armijo sufficient-decrease constant

    double g[N], gn[N], d[N], s[N], y[N], Hy[N], xn[N];
    double H[N*N];  // inverse Hessian approximation, row-major (121 doubles ~1 kB)

    // H = I
    for (int i = 0; i < N*N; ++i) H[i] = 0.0;
    for (int i = 0; i < N;   ++i) H[i*N+i] = 1.0;

    // Central finite-difference gradient; step scales with |x_i| to handle
    // parameters with very different magnitudes (mW~80 vs angular pulls~0).
    auto grad_fn = [&](const double* xp, double* gp) {
        double xc[N];
        for (int i = 0; i < N; ++i) xc[i] = xp[i];
        for (int i = 0; i < N; ++i) {
            double h  = 1e-4 * (std::abs(xc[i]) > 1.0 ? std::abs(xc[i]) : 1.0);
            double xi = xc[i];
            xc[i] = xi + h;  double fp = f(xc);
            xc[i] = xi - h;  double fm = f(xc);
            xc[i] = xi;
            gp[i] = (fp - fm) / (2.0 * h);
        }
    };

    fmin = f(x);
    grad_fn(x, g);

    for (int iter = 0; iter < MAXITER; ++iter) {
        // Convergence check on gradient norm
        double gnorm2 = 0;
        for (int i = 0; i < N; ++i) gnorm2 += g[i]*g[i];
        if (gnorm2 < GTOL*GTOL) return 0;

        // Search direction: d = -H * g
        for (int i = 0; i < N; ++i) {
            d[i] = 0;
            for (int j = 0; j < N; ++j) d[i] -= H[i*N+j] * g[j];
        }

        // If d is not a descent direction (can happen if H drifted non-PD),
        // reset H to identity and fall back to steepest descent for this step.
        double slope = 0;
        for (int i = 0; i < N; ++i) slope += g[i] * d[i];
        if (slope >= 0) {
            for (int i = 0; i < N*N; ++i) H[i] = 0.0;
            for (int i = 0; i < N;   ++i) { H[i*N+i] = 1.0; d[i] = -g[i]; }
            slope = -gnorm2;
        }

        // Armijo backtracking line search
        double alpha = 1.0;
        bool   ls_ok = false;
        for (int ls = 0; ls < 40; ++ls) {
            for (int i = 0; i < N; ++i) xn[i] = x[i] + alpha * d[i];
            double fn = f(xn);
            if (fn <= fmin + C1 * alpha * slope) { fmin = fn; ls_ok = true; break; }
            alpha *= 0.5;
            if (alpha < 1e-14) break;
        }
        if (!ls_ok) break;  // line search failed — stop

        // Accept step: s = alpha*d, update x
        for (int i = 0; i < N; ++i) { s[i] = alpha * d[i]; x[i] = xn[i]; }

        // Gradient at new point, y = g_new - g_old
        grad_fn(x, gn);
        double sy = 0;
        for (int i = 0; i < N; ++i) { y[i] = gn[i] - g[i]; sy += s[i] * y[i]; g[i] = gn[i]; }

        // BFGS rank-2 inverse Hessian update (skip if curvature condition fails).
        // Formula: H += -rho*(s*Hy^T + Hy*s^T) + (rho^2*yHy + rho)*s*s^T
        // where Hy = H*y, yHy = y^T*Hy, rho = 1/(s^T*y)
        if (sy <= 1e-14) continue;
        double rho = 1.0 / sy;

        for (int i = 0; i < N; ++i) {
            Hy[i] = 0;
            for (int j = 0; j < N; ++j) Hy[i] += H[i*N+j] * y[j];
        }
        double yHy = 0;
        for (int i = 0; i < N; ++i) yHy += y[i] * Hy[i];
        double fac  = rho*rho*yHy + rho;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                H[i*N+j] += -rho*(s[i]*Hy[j] + Hy[i]*s[j]) + fac*s[i]*s[j];
    }
    return 1;  // max iterations reached without meeting GTOL
}

KinFitResult kinFitBFGS(float jet1_p,    float jet1_theta,    float jet1_phi,
                         float jet2_p,    float jet2_theta,    float jet2_phi,
                         float Isolep_p,  float Isolep_theta,  float Isolep_phi,
                         float missing_p, float missing_p_theta, float missing_p_phi,
                         bool fit_gW = false) {

    KinFitResult result{};
    result.gW    = KF_GW_FIXED;
    result.valid = 0;
    result.chi2  = 999.0f;

    if (Isolep_p < 0 || jet1_p <= 0 || jet2_p <= 0 || missing_p <= 0)
        return result;

    double fmin = 0;

    if (fit_gW) {
        // 14 free parameters: x[0]=mW, x[1]=gW, x[2..5]=scales, x[6..8]=jet/MET theta, x[9..11]=jet/MET phi, x[12..13]=lep angles
        auto chi2fn = [=](const double* x) -> double {
            const double mW = x[0], gW = x[1];
            if (gW <= 0.0) return 1e10;
            const double s1 = x[2], s2 = x[3], sl = x[4], sn = x[5];
            const double t1 = x[6], t2 = x[7], tn = x[8];
            const double p1 = x[9], p2 = x[10], pn = x[11];
            const double tl = x[12], pl = x[13];
            if (s1 <= 0.0 || s2 <= 0.0 || sl <= 0.0 || sn <= 0.0) return 1e10;

            TLorentzVector j1f = _vec_spherical(jet1_p/s1,    jet1_theta    - t1, jet1_phi    - p1);
            TLorentzVector j2f = _vec_spherical(jet2_p/s2,    jet2_theta    - t2, jet2_phi    - p2);
            TLorentzVector lf  = _vec_spherical(Isolep_p/sl,  Isolep_theta  - tl, Isolep_phi  - pl);
            TLorentzVector nf  = _vec_spherical(missing_p/sn, missing_p_theta - tn, missing_p_phi - pn);

            TLorentzVector Wh = j1f + j2f;
            TLorentzVector Wl = lf  + nf;
            TLorentzVector WW = Wh  + Wl;

            double mh = Wh.M(), ml = Wl.M();
            double mwgw = mW * gW;
            double dh   = mh*mh - mW*mW,  dl = ml*ml - mW*mW;
            double bw_h = mwgw / (dh*dh + mwgw*mwgw);
            double bw_l = mwgw / (dl*dl + mwgw*mwgw);
            double bw_term = -2.0 * (std::log(bw_h) + std::log(bw_l));

            double cons = dcb_gauss_neg2logpdf(WW.M() - ECM, DCBG_M_GEN_LNUQQ_MINUS_ECM)
                        + dcb_gauss_neg2logpdf(WW.Px(), DCBG_PX_TOT_GEN)
                        + dcb_gauss_neg2logpdf(WW.Py(), DCBG_PY_TOT_GEN)
                        + dcb_gauss_neg2logpdf(WW.Pz(), DCBG_PZ_TOT_GEN);

            double scale_pen = dcb_gauss_neg2logpdf(s1, DCBG_JET1_P_RESP)
                             + dcb_gauss_neg2logpdf(s2, DCBG_JET2_P_RESP)
                             + dcb_neg2logpdf(sl, DCB_LEP_P_RESP)
                             + dcb_neg2logpdf(sn, DCB_MET_P_RESP);

            double angular = dcb_neg2logpdf(t1, DCB_JET1_THETA_RESOL)
                           + dcb_neg2logpdf(t2, DCB_JET2_THETA_RESOL)
                           + dcb_neg2logpdf(tn, DCB_MET_THETA_RESOL)
                           + dcb_neg2logpdf(tl, DCB_LEP_THETA_RESOL)
                           + dcb_neg2logpdf(p1, DCB_JET1_PHI_RESOL)
                           + dcb_neg2logpdf(p2, DCB_JET2_PHI_RESOL)
                           + dcb_neg2logpdf(pn, DCB_MET_PHI_RESOL)
                           + dcb_neg2logpdf(pl, DCB_LEP_PHI_RESOL);

            return bw_term + cons + scale_pen + angular;
        };

        // s-params init at DCB mode (jet resp ≈ 0.97, lep/MET resp ≈ 1.0)
        double x0[14] = {KF_MW_INIT, KF_GW_FIXED, 0.97, 0.97, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int status = _bfgs_minimize<decltype(chi2fn), 14>(chi2fn, x0, fmin);
        result.valid = (status == 0) ? 1 : 0;
        result.chi2  = fmin;
        result.mW = x0[0]; result.gW = x0[1];
        result.s1 = x0[2]; result.s2 = x0[3]; result.sl = x0[4]; result.sn = x0[5];
        result.t1 = x0[6]; result.t2 = x0[7]; result.tn = x0[8];
        result.p1 = x0[9]; result.p2 = x0[10]; result.pn = x0[11];
        result.tl = x0[12]; result.pl = x0[13];
    } else {
        // 13 free parameters (KF_NDIM): x[0]=mW, x[1..4]=scales, x[5..7]=jet/MET theta, x[8..10]=jet/MET phi, x[11..12]=lep angles
        auto chi2fn = [=](const double* x) -> double {
            const double mW = x[0];
            const double s1 = x[1], s2 = x[2], sl = x[3], sn = x[4];
            const double t1 = x[5], t2 = x[6], tn = x[7];
            const double p1 = x[8], p2 = x[9], pn = x[10];
            const double tl = x[11], pl = x[12];
            if (s1 <= 0.0 || s2 <= 0.0 || sl <= 0.0 || sn <= 0.0) return 1e10;

            TLorentzVector j1f = _vec_spherical(jet1_p/s1,    jet1_theta    - t1, jet1_phi    - p1);
            TLorentzVector j2f = _vec_spherical(jet2_p/s2,    jet2_theta    - t2, jet2_phi    - p2);
            TLorentzVector lf  = _vec_spherical(Isolep_p/sl,  Isolep_theta  - tl, Isolep_phi  - pl);
            TLorentzVector nf  = _vec_spherical(missing_p/sn, missing_p_theta - tn, missing_p_phi - pn);

            TLorentzVector Wh = j1f + j2f;
            TLorentzVector Wl = lf  + nf;
            TLorentzVector WW = Wh  + Wl;

            double mh = Wh.M(), ml = Wl.M();
            double mwgw = mW * KF_GW_FIXED;
            double dh   = mh*mh - mW*mW,  dl = ml*ml - mW*mW;
            double bw_h = mwgw / (dh*dh + mwgw*mwgw);
            double bw_l = mwgw / (dl*dl + mwgw*mwgw);
            double bw_term = -2.0 * (std::log(bw_h) + std::log(bw_l));

            double cons = dcb_gauss_neg2logpdf(WW.M() - ECM, DCBG_M_GEN_LNUQQ_MINUS_ECM)
                        + dcb_gauss_neg2logpdf(WW.Px(), DCBG_PX_TOT_GEN)
                        + dcb_gauss_neg2logpdf(WW.Py(), DCBG_PY_TOT_GEN)
                        + dcb_gauss_neg2logpdf(WW.Pz(), DCBG_PZ_TOT_GEN);

            double scale_pen = dcb_gauss_neg2logpdf(s1, DCBG_JET1_P_RESP)
                             + dcb_gauss_neg2logpdf(s2, DCBG_JET2_P_RESP)
                             + dcb_neg2logpdf(sl, DCB_LEP_P_RESP)
                             + dcb_neg2logpdf(sn, DCB_MET_P_RESP);

            double angular = dcb_neg2logpdf(t1, DCB_JET1_THETA_RESOL)
                           + dcb_neg2logpdf(t2, DCB_JET2_THETA_RESOL)
                           + dcb_neg2logpdf(tn, DCB_MET_THETA_RESOL)
                           + dcb_neg2logpdf(tl, DCB_LEP_THETA_RESOL)
                           + dcb_neg2logpdf(p1, DCB_JET1_PHI_RESOL)
                           + dcb_neg2logpdf(p2, DCB_JET2_PHI_RESOL)
                           + dcb_neg2logpdf(pn, DCB_MET_PHI_RESOL)
                           + dcb_neg2logpdf(pl, DCB_LEP_PHI_RESOL);

            return bw_term + cons + scale_pen + angular;
        };

        double x0[KF_NDIM] = {KF_MW_INIT, 0.97, 0.97, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int status = _bfgs_minimize<decltype(chi2fn), KF_NDIM>(chi2fn, x0, fmin);
        result.valid = (status == 0) ? 1 : 0;
        result.chi2  = fmin;
        result.mW = x0[0];
        result.s1 = x0[1]; result.s2 = x0[2]; result.sl = x0[3]; result.sn = x0[4];
        result.t1 = x0[5]; result.t2 = x0[6]; result.tn = x0[7];
        result.p1 = x0[8]; result.p2 = x0[9]; result.pn = x0[10];
        result.tl = x0[11]; result.pl = x0[12];
    }

    // Post-fit kinematics (shared — uses result fields filled above)
    TLorentzVector j1f = _vec_spherical(jet1_p/result.s1,    jet1_theta    - result.t1, jet1_phi    - result.p1);
    TLorentzVector j2f = _vec_spherical(jet2_p/result.s2,    jet2_theta    - result.t2, jet2_phi    - result.p2);
    TLorentzVector lf  = _vec_spherical(Isolep_p/result.sl,  Isolep_theta - result.tl, Isolep_phi - result.pl);
    TLorentzVector nf  = _vec_spherical(missing_p/result.sn, missing_p_theta - result.tn, missing_p_phi - result.pn);

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

KinFitResult kinFit(float jet1_p,    float jet1_theta,    float jet1_phi,
                    float jet2_p,    float jet2_theta,    float jet2_phi,
                    float Isolep_p,  float Isolep_theta,  float Isolep_phi,
                    float missing_p, float missing_p_theta, float missing_p_phi,
                    bool fit_gW = false) {

    KinFitResult result{};
    result.gW    = KF_GW_FIXED;
    result.valid = 0;
    result.chi2  = 999.0f;

    if (Isolep_p < 0 || jet1_p <= 0 || jet2_p <= 0 || missing_p <= 0)
        return result;

    // 14 parameters: x[0]=mW, x[1]=gW, x[2..5]=scales, x[6..8]=jet/MET theta, x[9..11]=jet/MET phi, x[12..13]=lep angles.
    // When fit_gW=false, gW is pinned to KF_GW_FIXED via FixVariable(1).
    auto chi2fn = [=](const double* x) -> double {
        const double mW = x[0], gW = x[1];
        if (gW <= 0.0) return 1e10;
        const double s1 = x[2], s2 = x[3], sl = x[4], sn = x[5];
        const double t1 = x[6], t2 = x[7], tn = x[8];
        const double p1 = x[9], p2 = x[10], pn = x[11];
        const double tl = x[12], pl = x[13];
        if (s1 <= 0.0 || s2 <= 0.0 || sl <= 0.0 || sn <= 0.0) return 1e10;

        TLorentzVector j1f = _vec_spherical(jet1_p/s1,    jet1_theta    - t1, jet1_phi    - p1);
        TLorentzVector j2f = _vec_spherical(jet2_p/s2,    jet2_theta    - t2, jet2_phi    - p2);
        TLorentzVector lf  = _vec_spherical(Isolep_p/sl,  Isolep_theta  - tl, Isolep_phi  - pl);
        TLorentzVector nf  = _vec_spherical(missing_p/sn, missing_p_theta - tn, missing_p_phi - pn);

        TLorentzVector Wh = j1f + j2f;
        TLorentzVector Wl = lf  + nf;
        TLorentzVector WW = Wh  + Wl;

        double mh = Wh.M(), ml = Wl.M();
        double mwgw = mW * gW;
        double dh   = mh*mh - mW*mW,  dl = ml*ml - mW*mW;
        double bw_h = mwgw / (dh*dh + mwgw*mwgw);
        double bw_l = mwgw / (dl*dl + mwgw*mwgw);
        double bw_term = -2.0 * (std::log(bw_h) + std::log(bw_l));

        double cons = dcb_gauss_neg2logpdf(WW.E() - ECM, DCBG_PZ_TOT_GEN)
                    + dcb_gauss_neg2logpdf(WW.Px(), DCBG_PX_TOT_GEN)
                    + dcb_gauss_neg2logpdf(WW.Py(), DCBG_PY_TOT_GEN)
                    + dcb_gauss_neg2logpdf(WW.Pz(), DCBG_PZ_TOT_GEN);

        double scale_pen = dcb_gauss_neg2logpdf(s1, DCBG_JET1_P_RESP)
                         + dcb_gauss_neg2logpdf(s2, DCBG_JET2_P_RESP)
                         + dcb_neg2logpdf(sl, DCB_LEP_P_RESP)
                         + dcb_neg2logpdf(sn, DCB_MET_P_RESP);

        double angular = dcb_neg2logpdf(t1, DCB_JET1_THETA_RESOL)
                       + dcb_neg2logpdf(t2, DCB_JET2_THETA_RESOL)
                       + dcb_neg2logpdf(tn, DCB_MET_THETA_RESOL)
                       + dcb_neg2logpdf(tl, DCB_LEP_THETA_RESOL)
                       + dcb_neg2logpdf(p1, DCB_JET1_PHI_RESOL)
                       + dcb_neg2logpdf(p2, DCB_JET2_PHI_RESOL)
                       + dcb_neg2logpdf(pn, DCB_MET_PHI_RESOL)
                       + dcb_neg2logpdf(pl, DCB_LEP_PHI_RESOL);

        return bw_term + cons + scale_pen + angular;
    };

    std::function<double(const double*)> fObj = chi2fn;
    ROOT::Math::Functor functor(fObj, 14);

    std::unique_ptr<ROOT::Math::Minimizer> minimizer(
        ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad")
    );
    minimizer->SetFunction(functor);
    minimizer->SetMaxFunctionCalls(10000);
    minimizer->SetTolerance(1e-6);
    minimizer->SetStrategy(2);
    minimizer->SetPrintLevel(0);

    // s-params: response = p_reco/p_gen; t/p-params: absolute shift in radians
    minimizer->SetVariable(0,  "mW", KF_MW_INIT,  0.1);   minimizer->SetVariableLimits(0,  0.0, 200.0);
    minimizer->SetVariable(1,  "gW", KF_GW_FIXED, 0.01);  minimizer->SetVariableLimits(1,  0.01, 10.0);
    minimizer->SetVariable(2,  "s1", 0.97,  0.01);   // jet1 p-response, DCB mu ≈ 0.974
    minimizer->SetVariable(3,  "s2", 0.97,  0.01);   // jet2 p-response, DCB mu ≈ 0.970
    minimizer->SetVariable(4,  "sl", 1.0,   0.001);  // lep  p-response, DCB mu ≈ 1.000
    minimizer->SetVariable(5,  "sn", 1.0,   0.005);  // MET  p-response, DCB mu ≈ 1.000
    minimizer->SetVariable(6,  "t1", 0.0,   0.01);   // jet1 dtheta [rad], sigma ≈ 0.014
    minimizer->SetVariable(7,  "t2", 0.0,   0.01);   // jet2 dtheta [rad], sigma ≈ 0.017
    minimizer->SetVariable(8,  "tn", 0.0,   0.005);  // MET  dtheta [rad], sigma ≈ 0.008
    minimizer->SetVariable(9,  "p1", 0.0,   0.01);   // jet1 dphi   [rad], sigma ≈ 0.016
    minimizer->SetVariable(10, "p2", 0.0,   0.01);   // jet2 dphi   [rad], sigma ≈ 0.020
    minimizer->SetVariable(11, "pn", 0.0,   0.003);  // MET  dphi   [rad], sigma ≈ 0.004
    minimizer->SetVariable(12, "tl", 0.0,   0.001);  // lep  dtheta [rad]
    minimizer->SetVariable(13, "pl", 0.0,   0.001);  // lep  dphi   [rad]

    if (!fit_gW) minimizer->FixVariable(1);

    minimizer->Minimize();
    minimizer->Minimize();

    result.valid = (minimizer->Status() == 0) ? 1 : 0;
    result.chi2  = minimizer->MinValue();
    const double* x = minimizer->X();
    result.mW = x[0]; result.gW = x[1];
    result.s1 = x[2]; result.s2 = x[3]; result.sl = x[4]; result.sn = x[5];
    result.t1 = x[6]; result.t2 = x[7]; result.tn = x[8];
    result.p1 = x[9]; result.p2 = x[10]; result.pn = x[11];
    result.tl = x[12]; result.pl = x[13];

    // Post-fit kinematics (shared — uses result fields filled above)
    TLorentzVector j1f = _vec_spherical(jet1_p/result.s1,    jet1_theta    - result.t1, jet1_phi    - result.p1);
    TLorentzVector j2f = _vec_spherical(jet2_p/result.s2,    jet2_theta    - result.t2, jet2_phi    - result.p2);
    TLorentzVector lf  = _vec_spherical(Isolep_p/result.sl,  Isolep_theta - result.tl, Isolep_phi - result.pl);
    TLorentzVector nf  = _vec_spherical(missing_p/result.sn, missing_p_theta - result.tn, missing_p_phi - result.pn);

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
