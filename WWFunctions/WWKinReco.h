#ifndef WWKinReco_H
#define WWKinReco_H

#include <cmath>
#include <memory>
#include <functional>
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include "outputs/response/functions/dcb_params.h"
#include "WWFunctions/WWFunctions.h"

namespace FCCAnalyses { namespace WWFunctions {

// Pull in DCB evaluators and fitted params from the generated headers.
using namespace ::WWFunctions;

// ── Per-ECM parameter bundles ───────────────────────────────────────────────
struct KinFitParamSet {
    DcbGaussParams         jet1_p_resp;
    DcbGaussParams         jet2_p_resp;
    DcbExpLeftGaussParams  lep_p_resp;            // expleft2g
    DcbParams              met_p_resp;
    DcbGaussParams         jet1_phi_resol;        // dcb2g
    DcbGaussParams         jet1_theta_resol;      // dcb2g
    DcbGaussParams         jet2_phi_resol;        // dcb2g
    DcbGaussParams         jet2_theta_resol;      // dcb2g
    DcbGaussParams         lep_phi_resol;         // dcb2g
    DcbGaussParams         lep_theta_resol;       // dcb2g
    DcbParams              met_phi_resol;
    DcbParams              met_theta_resol;
    DcbExpRightGaussParams m_gen_lnuqq_minus_ecm;
    DcbGaussParams         px_tot_gen;            // dcb2g
    DcbGaussParams         py_tot_gen;            // dcb2g
    DcbGaussParams         pz_tot_gen;            // dcb2g
};

static const KinFitParamSet KF_PARAMS_157 = {
    DCBG_JET1_P_RESP_157, DCBG_JET2_P_RESP_157,
    DCBELG_LEP_P_RESP_157, DCB_MET_P_RESP_157,
    DCBG_JET1_PHI_RESOL_157, DCBG_JET1_THETA_RESOL_157,
    DCBG_JET2_PHI_RESOL_157, DCBG_JET2_THETA_RESOL_157,
    DCBG_LEP_PHI_RESOL_157,  DCBG_LEP_THETA_RESOL_157,
    DCB_MET_PHI_RESOL_157,   DCB_MET_THETA_RESOL_157,
    DCBERG_GEN_WW_M_MINUS_ECM_157,
    DCBG_GEN_WW_PX_157, DCBG_GEN_WW_PY_157, DCBG_GEN_WW_PZ_157,
};
static const KinFitParamSet KF_PARAMS_160 = {
    DCBG_JET1_P_RESP_160, DCBG_JET2_P_RESP_160,
    DCBELG_LEP_P_RESP_160, DCB_MET_P_RESP_160,
    DCBG_JET1_PHI_RESOL_160, DCBG_JET1_THETA_RESOL_160,
    DCBG_JET2_PHI_RESOL_160, DCBG_JET2_THETA_RESOL_160,
    DCBG_LEP_PHI_RESOL_160,  DCBG_LEP_THETA_RESOL_160,
    DCB_MET_PHI_RESOL_160,   DCB_MET_THETA_RESOL_160,
    DCBERG_GEN_WW_M_MINUS_ECM_160,
    DCBG_GEN_WW_PX_160, DCBG_GEN_WW_PY_160, DCBG_GEN_WW_PZ_160,
};
static const KinFitParamSet KF_PARAMS_163 = {
    DCBG_JET1_P_RESP_163, DCBG_JET2_P_RESP_163,
    DCBELG_LEP_P_RESP_163, DCB_MET_P_RESP_163,
    DCBG_JET1_PHI_RESOL_163, DCBG_JET1_THETA_RESOL_163,
    DCBG_JET2_PHI_RESOL_163, DCBG_JET2_THETA_RESOL_163,
    DCBG_LEP_PHI_RESOL_163,  DCBG_LEP_THETA_RESOL_163,
    DCB_MET_PHI_RESOL_163,   DCB_MET_THETA_RESOL_163,
    DCBERG_GEN_WW_M_MINUS_ECM_163,
    DCBG_GEN_WW_PX_163, DCBG_GEN_WW_PY_163, DCBG_GEN_WW_PZ_163,
};

// ── Active kinfit parameters (set per-dataset via setKinFitParams) ─────────
inline DcbGaussParams         kf_jet1_p_resp           = DCBG_JET1_P_RESP_160;
inline DcbGaussParams         kf_jet2_p_resp           = DCBG_JET2_P_RESP_160;
inline DcbExpLeftGaussParams  kf_lep_p_resp            = DCBELG_LEP_P_RESP_160;
inline DcbParams              kf_met_p_resp            = DCB_MET_P_RESP_160;
inline DcbGaussParams         kf_jet1_phi_resol        = DCBG_JET1_PHI_RESOL_160;
inline DcbGaussParams         kf_jet1_theta_resol      = DCBG_JET1_THETA_RESOL_160;
inline DcbGaussParams         kf_jet2_phi_resol        = DCBG_JET2_PHI_RESOL_160;
inline DcbGaussParams         kf_jet2_theta_resol      = DCBG_JET2_THETA_RESOL_160;
inline DcbGaussParams         kf_lep_phi_resol         = DCBG_LEP_PHI_RESOL_160;
inline DcbGaussParams         kf_lep_theta_resol       = DCBG_LEP_THETA_RESOL_160;
inline DcbParams              kf_met_phi_resol         = DCB_MET_PHI_RESOL_160;
inline DcbParams              kf_met_theta_resol       = DCB_MET_THETA_RESOL_160;
inline DcbExpRightGaussParams kf_m_gen_lnuqq_minus_ecm = DCBERG_GEN_WW_M_MINUS_ECM_160;
inline DcbGaussParams         kf_px_tot_gen            = DCBG_GEN_WW_PX_160;
inline DcbGaussParams         kf_py_tot_gen            = DCBG_GEN_WW_PY_160;
inline DcbGaussParams         kf_pz_tot_gen            = DCBG_GEN_WW_PZ_160;

inline void setKinFitParams(int ecm) {
    ECM = static_cast<float>(ecm);
    const KinFitParamSet* p =
        ecm == 157 ? &KF_PARAMS_157 :
        ecm == 160 ? &KF_PARAMS_160 :
        ecm == 163 ? &KF_PARAMS_163 : nullptr;
    if (!p) return;
    kf_jet1_p_resp           = p->jet1_p_resp;
    kf_jet2_p_resp           = p->jet2_p_resp;
    kf_lep_p_resp            = p->lep_p_resp;
    kf_met_p_resp            = p->met_p_resp;
    kf_jet1_phi_resol        = p->jet1_phi_resol;
    kf_jet1_theta_resol      = p->jet1_theta_resol;
    kf_jet2_phi_resol        = p->jet2_phi_resol;
    kf_jet2_theta_resol      = p->jet2_theta_resol;
    kf_lep_phi_resol         = p->lep_phi_resol;
    kf_lep_theta_resol       = p->lep_theta_resol;
    kf_met_phi_resol         = p->met_phi_resol;
    kf_met_theta_resol       = p->met_theta_resol;
    kf_m_gen_lnuqq_minus_ecm = p->m_gen_lnuqq_minus_ecm;
    kf_px_tot_gen            = p->px_tot_gen;
    kf_py_tot_gen            = p->py_tot_gen;
    kf_pz_tot_gen            = p->pz_tot_gen;
}

// ── kinematic fit ──────────────────────────────────────────────────────────

// Kinematic fit constants.
// Momentum scale params (s1,s2,sl,sn) are now response = p_reco/p_gen;
// angular params (t1,t2,tn,p1,p2,pn) are now absolute shifts in radians.
// Constraints use DCB/DCB+G PDFs from dcb_params_ecm<N>.h (selected per-dataset by setKinFitParams).
// WW momentum and mass constraints use kf_px/py/pz_tot_gen and kf_m_gen_lnuqq_minus_ecm.
static constexpr double KF_MW_INIT = 80.419;
static constexpr double KF_GW_FIXED = 2.049;
static constexpr int    KF_NDIM    = 13;   // free parameters when gW is fixed (added tl, pl)
// Number of constraint terms in chi2: 4 momentum-response + 8 angular-resolution
// + 4 WW-system (Px,Py,Pz,M-ECM) + 2 BW. The phase-space factor is part of the
// joint BW/PS density and is not counted separately.
static constexpr int    KF_N_CONSTR = 18;

struct KinFitResult {
    float mW, gW;
    float s1, s2, sl, sn;
    float t1, t2, tn, tl;   // theta shifts: jet1, jet2, MET, lepton
    float p1, p2, pn, pl;   // phi shifts:   jet1, jet2, MET, lepton
    float chi2;
    float chi2_ndof;        // chi2 / (KF_N_CONSTR - n_free_params)
    int   valid;
    // Post-fit 4-vectors. All scalar projections (P, Pt, M, Px, ...) and the
    // Wlep/Whad/WW sums are derived in the consumer.
    TLorentzVector j1, j2, lep, nu;
};

// Massless 4-vector from spherical coordinates.
static TLorentzVector _vec_spherical(double p, double theta, double phi) {
    double st = std::sin(theta), ct = std::cos(theta);
    TLorentzVector v;
    v.SetPxPyPzE(p * st * std::cos(phi), p * st * std::sin(phi), p * ct, p);
    return v;
}



// ── BFGS minimizer ────────────────────────────────────────────────────────
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
    result.chi2_ndof = 999.0f;

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
            double s_ww = WW.M2();
            double lam  = (s_ww - (mh+ml)*(mh+ml)) * (s_ww - (mh-ml)*(mh-ml));
            // Floor lam at 1e-12 instead of bailing — keeps the gradient continuous
            // at the kinematic boundary (lam = 0 corresponds to W masses summing to √s_WW).
            lam = std::max(lam, 1e-12);
            // Joint BW × phase-space PDF (each BW normalised: ∫ BW dm² = π,
            // so BW_norm = BW/π → -2log adds 2log(π) per W). Phase space ∝ √λ/s_WW.
            double bw_term = -2.0 * (std::log(bw_h) + std::log(bw_l))
                           + 4.0 * std::log(M_PI)
                           - std::log(lam) + 2.0 * std::log(s_ww);

            double cons = dcb_gauss_neg2logpdf(WW.Px(), kf_px_tot_gen)
                        + dcb_gauss_neg2logpdf(WW.Py(), kf_py_tot_gen)
                        + dcb_gauss_neg2logpdf(WW.Pz(), kf_pz_tot_gen)
                        + dcb_expright_gauss_neg2logpdf(WW.M() - ECM, kf_m_gen_lnuqq_minus_ecm);

            double scale_pen = dcb_gauss_neg2logpdf(s1, kf_jet1_p_resp)
                             + dcb_gauss_neg2logpdf(s2, kf_jet2_p_resp)
                             + dcb_expleft_gauss_neg2logpdf(sl, kf_lep_p_resp)
                             + dcb_neg2logpdf(sn, kf_met_p_resp);

            double angular = dcb_gauss_neg2logpdf(t1, kf_jet1_theta_resol)
                           + dcb_gauss_neg2logpdf(t2, kf_jet2_theta_resol)
                           + dcb_neg2logpdf(tn, kf_met_theta_resol)
                           + dcb_gauss_neg2logpdf(tl, kf_lep_theta_resol)
                           + dcb_gauss_neg2logpdf(p1, kf_jet1_phi_resol)
                           + dcb_gauss_neg2logpdf(p2, kf_jet2_phi_resol)
                           + dcb_neg2logpdf(pn, kf_met_phi_resol)
                           + dcb_gauss_neg2logpdf(pl, kf_lep_phi_resol);

            return bw_term + cons + scale_pen + angular;
        };

        // s-params init at DCB mode (jet resp ≈ 0.97, lep/MET resp ≈ 1.0)
        double x0[14] = {KF_MW_INIT, KF_GW_FIXED, 0.97, 0.97, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int status = _bfgs_minimize<decltype(chi2fn), 14>(chi2fn, x0, fmin);
        result.valid = (status == 0) ? 1 : 0;
        result.chi2  = fmin;
        result.chi2_ndof = (KF_N_CONSTR > 14) ? fmin / float(KF_N_CONSTR - 14) : -1.0f;
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
            double s_ww = WW.M2();
            double lam  = (s_ww - (mh+ml)*(mh+ml)) * (s_ww - (mh-ml)*(mh-ml));
            // Floor lam to keep -log(lam) finite and gradient smooth across the boundary.
            lam = std::max(lam, 1e-12);
            // Joint BW × phase-space PDF (BW_norm = BW/π; phase space ∝ √λ/s_WW).
            double bw_term = -2.0 * (std::log(bw_h) + std::log(bw_l))
                           + 4.0 * std::log(M_PI)
                           - std::log(lam) + 2.0 * std::log(s_ww);

            double cons = dcb_gauss_neg2logpdf(WW.Px(), kf_px_tot_gen)
                        + dcb_gauss_neg2logpdf(WW.Py(), kf_py_tot_gen)
                        + dcb_gauss_neg2logpdf(WW.Pz(), kf_pz_tot_gen)
                        + dcb_expright_gauss_neg2logpdf(WW.M() - ECM, kf_m_gen_lnuqq_minus_ecm);

            double scale_pen = dcb_gauss_neg2logpdf(s1, kf_jet1_p_resp)
                             + dcb_gauss_neg2logpdf(s2, kf_jet2_p_resp)
                             + dcb_expleft_gauss_neg2logpdf(sl, kf_lep_p_resp)
                             + dcb_neg2logpdf(sn, kf_met_p_resp);

            double angular = dcb_gauss_neg2logpdf(t1, kf_jet1_theta_resol)
                           + dcb_gauss_neg2logpdf(t2, kf_jet2_theta_resol)
                           + dcb_neg2logpdf(tn, kf_met_theta_resol)
                           + dcb_gauss_neg2logpdf(tl, kf_lep_theta_resol)
                           + dcb_gauss_neg2logpdf(p1, kf_jet1_phi_resol)
                           + dcb_gauss_neg2logpdf(p2, kf_jet2_phi_resol)
                           + dcb_neg2logpdf(pn, kf_met_phi_resol)
                           + dcb_gauss_neg2logpdf(pl, kf_lep_phi_resol);

            return bw_term + cons + scale_pen + angular;
        };

        double x0[KF_NDIM] = {KF_MW_INIT, 0.97, 0.97, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int status = _bfgs_minimize<decltype(chi2fn), KF_NDIM>(chi2fn, x0, fmin);
        result.valid = (status == 0) ? 1 : 0;
        result.chi2  = fmin;
        result.chi2_ndof = (KF_N_CONSTR > KF_NDIM) ? fmin / float(KF_N_CONSTR - KF_NDIM) : -1.0f;
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

    result.j1  = j1f;
    result.j2  = j2f;
    result.lep = lf;
    result.nu  = nf;
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
    result.chi2_ndof = 999.0f;

    if (Isolep_p < 0 || jet1_p <= 0 || jet2_p <= 0 || missing_p <= 0)
        return result;

    // 14 parameters: x[0]=mW, x[1]=gW, x[2..5]=scales, x[6..8]=jet/MET theta, x[9..11]=jet/MET phi, x[12..13]=lep angles.
    // When fit_gW=false, gW is pinned to KF_GW_FIXED via FixVariable(1).
    auto chi2fn = [=](const double* x) -> double {
        // gW has limits (0.01, 10.0) and s_i guard against accidental negative regions
        // during line search (s_i are unbounded in this Minuit setup).
        const double mW = x[0], gW = x[1];
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
        double s_ww = WW.M2();
        double lam  = (s_ww - (mh+ml)*(mh+ml)) * (s_ww - (mh-ml)*(mh-ml));
        // Floor lam to keep -log(lam) finite and gradient smooth across the boundary.
        lam = std::max(lam, 1e-12);
        // Joint BW × phase-space PDF (BW_norm = BW/π; phase space ∝ √λ/s_WW).
        double bw_term = -2.0 * (std::log(bw_h) + std::log(bw_l))
                       + 4.0 * std::log(M_PI)
                       - std::log(lam) + 2.0 * std::log(s_ww);

        double cons = dcb_gauss_neg2logpdf(WW.Px(), kf_px_tot_gen)
                    + dcb_gauss_neg2logpdf(WW.Py(), kf_py_tot_gen)
                    + dcb_gauss_neg2logpdf(WW.Pz(), kf_pz_tot_gen)
                    + dcb_expright_gauss_neg2logpdf(WW.M() - ECM, kf_m_gen_lnuqq_minus_ecm);

        double scale_pen = dcb_gauss_neg2logpdf(s1, kf_jet1_p_resp)
                         + dcb_gauss_neg2logpdf(s2, kf_jet2_p_resp)
                         + dcb_expleft_gauss_neg2logpdf(sl, kf_lep_p_resp)
                         + dcb_neg2logpdf(sn, kf_met_p_resp);

        double angular = dcb_gauss_neg2logpdf(t1, kf_jet1_theta_resol)
                       + dcb_gauss_neg2logpdf(t2, kf_jet2_theta_resol)
                       + dcb_neg2logpdf(tn, kf_met_theta_resol)
                       + dcb_gauss_neg2logpdf(tl, kf_lep_theta_resol)
                       + dcb_gauss_neg2logpdf(p1, kf_jet1_phi_resol)
                       + dcb_gauss_neg2logpdf(p2, kf_jet2_phi_resol)
                       + dcb_neg2logpdf(pn, kf_met_phi_resol)
                       + dcb_gauss_neg2logpdf(pl, kf_lep_phi_resol);

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
    minimizer->SetPrintLevel(-1);

    // Step sizes matched to ECM-160 PDF sigmas so Migrad's finite-difference probes
    // see chi2 changes of O(1) per step — critical for the lepton angular shifts
    // whose σ ~ 1e-4 rad (was 1e-3, ~30× too coarse).
    minimizer->SetVariable(0,  "mW", KF_MW_INIT,  0.1);     minimizer->SetVariableLimits(0,  0.0, 200.0);
    minimizer->SetVariable(1,  "gW", KF_GW_FIXED, 0.01);    minimizer->SetVariableLimits(1,  0.01, 10.0);
    minimizer->SetVariable(2,  "s1", 0.97,  0.014);    // jet1 p-resp, σ ≈ 0.014
    minimizer->SetVariable(3,  "s2", 0.97,  0.014);    // jet2 p-resp, σ ≈ 0.018
    minimizer->SetVariable(4,  "sl", 1.0,   0.004);    // lep  p-resp, σ ≈ 0.004
    minimizer->SetVariable(5,  "sn", 1.0,   0.012);    // MET  p-resp, σ ≈ 0.012
    minimizer->SetVariable(6,  "t1", 0.0,   0.014);    // jet1 dtheta, σ ≈ 0.013
    minimizer->SetVariable(7,  "t2", 0.0,   0.017);    // jet2 dtheta, σ ≈ 0.016
    minimizer->SetVariable(8,  "tn", 0.0,   0.007);    // MET  dtheta, σ ≈ 0.007
    minimizer->SetVariable(9,  "p1", 0.0,   0.016);    // jet1 dphi,   σ ≈ 0.015
    minimizer->SetVariable(10, "p2", 0.0,   0.020);    // jet2 dphi,   σ ≈ 0.018
    minimizer->SetVariable(11, "pn", 0.0,   0.004);    // MET  dphi,   σ ≈ 0.004
    minimizer->SetVariable(12, "tl", 0.0,   0.0001);   // lep  dtheta, σ ≈ 2e-5
    minimizer->SetVariable(13, "pl", 0.0,   0.0003);   // lep  dphi,   σ ≈ 3e-4

    if (!fit_gW) minimizer->FixVariable(1);

    minimizer->Minimize();
    minimizer->Minimize();

    // Accept status 0 ("minimum found") and 1 ("covariance forced positive-definite").
    // The latter is common with non-Gaussian PDFs where the local Hessian estimate
    // needs PD adjustment — the postfit values are still valid.
    int status = minimizer->Status();
    result.valid = (status == 0 || status == 1) ? 1 : 0;
    result.chi2  = minimizer->MinValue();
    int n_par = fit_gW ? 14 : KF_NDIM;
    result.chi2_ndof = (KF_N_CONSTR > n_par) ? result.chi2 / float(KF_N_CONSTR - n_par) : -1.0f;
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

    result.j1  = j1f;
    result.j2  = j2f;
    result.lep = lf;
    result.nu  = nf;
    return result;
}

}}  // namespace FCCAnalyses::WWFunctions

#endif
