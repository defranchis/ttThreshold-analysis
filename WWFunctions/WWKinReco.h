#ifndef WWKinReco_H
#define WWKinReco_H

#include <cmath>
#include <memory>
#include <functional>
#include <string>
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

// Two jet-prior conventions, selected at setKinFitParams() time:
//  - POOL: jet1 and jet2 share the pooled prior DCBG_JET_*_{ECM}. No pT-ordering
//          dependence; pass-1 already covers both orderings, swap fallback off.
//  - SEP : jet1 / jet2 use the separately-fit DCBG_JET{1,2}_*_{ECM} priors. The
//          per-jet ordering disagrees on some events, so the kinFit() pass-2
//          swap fallback (jet1↔jet2 priors) is enabled to rescue them.
static const KinFitParamSet KF_PARAMS_157_POOL = {
    DCBG_JET_P_RESP_157, DCBG_JET_P_RESP_157,
    DCBELG_LEP_P_RESP_157, DCB_MET_P_RESP_157,
    DCBG_JET_PHI_RESOL_157, DCBG_JET_THETA_RESOL_157,
    DCBG_JET_PHI_RESOL_157, DCBG_JET_THETA_RESOL_157,
    DCBG_LEP_PHI_RESOL_157,  DCBG_LEP_THETA_RESOL_157,
    DCB_MET_PHI_RESOL_157,   DCB_MET_THETA_RESOL_157,
    DCBERG_GEN_WW_M_MINUS_ECM_157,
    DCBG_GEN_WW_PX_157, DCBG_GEN_WW_PY_157, DCBG_GEN_WW_PZ_157,
};
static const KinFitParamSet KF_PARAMS_160_POOL = {
    DCBG_JET_P_RESP_160, DCBG_JET_P_RESP_160,
    DCBELG_LEP_P_RESP_160, DCB_MET_P_RESP_160,
    DCBG_JET_PHI_RESOL_160, DCBG_JET_THETA_RESOL_160,
    DCBG_JET_PHI_RESOL_160, DCBG_JET_THETA_RESOL_160,
    DCBG_LEP_PHI_RESOL_160,  DCBG_LEP_THETA_RESOL_160,
    DCB_MET_PHI_RESOL_160,   DCB_MET_THETA_RESOL_160,
    DCBERG_GEN_WW_M_MINUS_ECM_160,
    DCBG_GEN_WW_PX_160, DCBG_GEN_WW_PY_160, DCBG_GEN_WW_PZ_160,
};
static const KinFitParamSet KF_PARAMS_163_POOL = {
    DCBG_JET_P_RESP_163, DCBG_JET_P_RESP_163,
    DCBELG_LEP_P_RESP_163, DCB_MET_P_RESP_163,
    DCBG_JET_PHI_RESOL_163, DCBG_JET_THETA_RESOL_163,
    DCBG_JET_PHI_RESOL_163, DCBG_JET_THETA_RESOL_163,
    DCBG_LEP_PHI_RESOL_163,  DCBG_LEP_THETA_RESOL_163,
    DCB_MET_PHI_RESOL_163,   DCB_MET_THETA_RESOL_163,
    DCBERG_GEN_WW_M_MINUS_ECM_163,
    DCBG_GEN_WW_PX_163, DCBG_GEN_WW_PY_163, DCBG_GEN_WW_PZ_163,
};
static const KinFitParamSet KF_PARAMS_157_SEP = {
    DCBG_JET1_P_RESP_157, DCBG_JET2_P_RESP_157,
    DCBELG_LEP_P_RESP_157, DCB_MET_P_RESP_157,
    DCBG_JET1_PHI_RESOL_157, DCBG_JET1_THETA_RESOL_157,
    DCBG_JET2_PHI_RESOL_157, DCBG_JET2_THETA_RESOL_157,
    DCBG_LEP_PHI_RESOL_157,  DCBG_LEP_THETA_RESOL_157,
    DCB_MET_PHI_RESOL_157,   DCB_MET_THETA_RESOL_157,
    DCBERG_GEN_WW_M_MINUS_ECM_157,
    DCBG_GEN_WW_PX_157, DCBG_GEN_WW_PY_157, DCBG_GEN_WW_PZ_157,
};
static const KinFitParamSet KF_PARAMS_160_SEP = {
    DCBG_JET1_P_RESP_160, DCBG_JET2_P_RESP_160,
    DCBELG_LEP_P_RESP_160, DCB_MET_P_RESP_160,
    DCBG_JET1_PHI_RESOL_160, DCBG_JET1_THETA_RESOL_160,
    DCBG_JET2_PHI_RESOL_160, DCBG_JET2_THETA_RESOL_160,
    DCBG_LEP_PHI_RESOL_160,  DCBG_LEP_THETA_RESOL_160,
    DCB_MET_PHI_RESOL_160,   DCB_MET_THETA_RESOL_160,
    DCBERG_GEN_WW_M_MINUS_ECM_160,
    DCBG_GEN_WW_PX_160, DCBG_GEN_WW_PY_160, DCBG_GEN_WW_PZ_160,
};
static const KinFitParamSet KF_PARAMS_163_SEP = {
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

// True when SEP (per-jet) priors are active → kinFit() runs the jet1↔jet2 swap
// fallback on non-converged events. False under POOL (priors are jet-symmetric,
// swap is a no-op so we skip the second Migrad pass).
inline bool kf_jet_swap_enabled = false;

inline void setKinFitParams(int ecm, const std::string& jet_mode = "pool") {
    ECM = static_cast<float>(ecm);
    const bool use_pool = (jet_mode == "pool");
    kf_jet_swap_enabled = !use_pool;
    const KinFitParamSet* p =
        ecm == 157 ? (use_pool ? &KF_PARAMS_157_POOL : &KF_PARAMS_157_SEP) :
        ecm == 160 ? (use_pool ? &KF_PARAMS_160_POOL : &KF_PARAMS_160_SEP) :
        ecm == 163 ? (use_pool ? &KF_PARAMS_163_POOL : &KF_PARAMS_163_SEP) : nullptr;
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
// + 4 WW-system (Px,Py,Pz,M-ECM) + 2 BW. When fit_gW=true a Gaussian prior on gW
// adds +1 constraint, applied at chi2_ndof time.
static constexpr int    KF_N_CONSTR = 18;

// Gaussian prior on gW (only active when fit_gW=true).
static constexpr double KF_GW_PRIOR_SIGMA_REL = 0.01;
static constexpr double KF_GW_PRIOR_SIGMA     = KF_GW_PRIOR_SIGMA_REL * KF_GW_FIXED;
static constexpr double KF_GW_PRIOR_INV_SIGMA = 1.0 / KF_GW_PRIOR_SIGMA;
// std::log isn't constexpr until C++26, so this is a runtime const initialized once.
inline const double KF_GW_PRIOR_LOG_NORM =
        std::log(2.0 * M_PI * KF_GW_PRIOR_SIGMA * KF_GW_PRIOR_SIGMA);

// −2·log G(gW; KF_GW_FIXED, KF_GW_PRIOR_SIGMA).
static inline double _gw_prior_neg2logpdf(double gW) {
    const double dgw = (gW - KF_GW_FIXED) * KF_GW_PRIOR_INV_SIGMA;
    return dgw * dgw + KF_GW_PRIOR_LOG_NORM;
}

struct KinFitResult {
    float mW, gW;
    float s1, s2, sl, sn;
    float t1, t2, tn, tl;   // theta shifts: jet1, jet2, MET, lepton
    float p1, p2, pn, pl;   // phi shifts:   jet1, jet2, MET, lepton
    float chi2;
    float chi2_ndof;        // chi2 / (KF_N_CONSTR - n_free_params)
    int   status;           // raw minimizer status code (Minuit2: 0=OK, 1=PD-forced cov,
                            //   2=Hesse failed, 3=EDM>tol, 4=max calls, 5=other;
                            //   BFGS: 0=converged, 1=max-iter/LS-fail).
                            //   −1 if early-returned without fitting (invalid input p).
    int   valid;            // currently (status == 0 || status == 1)
    // Diagnostics: which of the 4 passes won and how many actually ran.
    //   winner_pass: 1=Migrad-natural, 2=Migrad-swapped,
    //                3=Simplex+Migrad-natural, 4=Simplex+Migrad-swapped.
    //   n_passes_run: total passes executed before stopping (1..4).
    //   priors_swapped: 1 if the winning pass used jet1↔jet2-swapped priors.
    int   winner_pass;
    int   n_passes_run;
    int   priors_swapped;
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

// Decode standardized y-coord to physical value: x = μ_prior + σ_prior · y.
// All prior PDFs (DcbParams, DcbGaussParams, DcbExpLeftGaussParams,
// DcbExpRightGaussParams) expose .mu/.sigma as their first two fields, so this
// template works for every kf_* struct. Preconditions the Hessian: in y-space
// all 12 nuisance directions have unit RMS, so Migrad's EDM tolerance is uniform.
// mW (and gW when free) stay in physical units.
template<typename PdfT>
static inline double _y2x(double y, const PdfT& p) { return p.mu + p.sigma * y; }

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
    result.status = -1;
    result.chi2  = 999.0f;
    result.chi2_ndof = 999.0f;

    if (Isolep_p < 0 || jet1_p <= 0 || jet2_p <= 0 || missing_p <= 0)
        return result;

    double fmin = 0;

    if (fit_gW) {
        // 14 free params: x[0]=mW, x[1]=gW (physical); x[2..13] standardized y-coords.
        auto chi2fn = [=](const double* x) -> double {
            const double mW = x[0], gW = x[1];
            if (gW <= 0.0) return 1e10;
            const double s1 = _y2x(x[2],  kf_jet1_p_resp);
            const double s2 = _y2x(x[3],  kf_jet2_p_resp);
            const double sl = _y2x(x[4],  kf_lep_p_resp);
            const double sn = _y2x(x[5],  kf_met_p_resp);
            const double t1 = _y2x(x[6],  kf_jet1_theta_resol);
            const double t2 = _y2x(x[7],  kf_jet2_theta_resol);
            const double tn = _y2x(x[8],  kf_met_theta_resol);
            const double p1 = _y2x(x[9],  kf_jet1_phi_resol);
            const double p2 = _y2x(x[10], kf_jet2_phi_resol);
            const double pn = _y2x(x[11], kf_met_phi_resol);
            const double tl = _y2x(x[12], kf_lep_theta_resol);
            const double pl = _y2x(x[13], kf_lep_phi_resol);
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
            lam = std::sqrt(lam*lam + 1e-24);  // smooth |λ| floor — derivative continuous through 0
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

            double gw_term = _gw_prior_neg2logpdf(gW);

            return bw_term + cons + scale_pen + angular + gw_term;
        };

        // y-coords start at 0 → physical starts at prior μ for each nuisance param.
        double x0[14] = {KF_MW_INIT, KF_GW_FIXED, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int status = _bfgs_minimize<decltype(chi2fn), 14>(chi2fn, x0, fmin);
        result.status = status;
        result.valid = (status == 0) ? 1 : 0;
        result.chi2  = fmin;
        result.winner_pass    = 1;
        result.n_passes_run   = 1;
        result.priors_swapped = 0;
        // +1 constraint from the gW Gaussian prior.
        result.chi2_ndof = (KF_N_CONSTR + 1 > 14) ? fmin / float(KF_N_CONSTR + 1 - 14) : -1.0f;
        result.mW = x0[0]; result.gW = x0[1];
        result.s1 = _y2x(x0[2],  kf_jet1_p_resp);
        result.s2 = _y2x(x0[3],  kf_jet2_p_resp);
        result.sl = _y2x(x0[4],  kf_lep_p_resp);
        result.sn = _y2x(x0[5],  kf_met_p_resp);
        result.t1 = _y2x(x0[6],  kf_jet1_theta_resol);
        result.t2 = _y2x(x0[7],  kf_jet2_theta_resol);
        result.tn = _y2x(x0[8],  kf_met_theta_resol);
        result.p1 = _y2x(x0[9],  kf_jet1_phi_resol);
        result.p2 = _y2x(x0[10], kf_jet2_phi_resol);
        result.pn = _y2x(x0[11], kf_met_phi_resol);
        result.tl = _y2x(x0[12], kf_lep_theta_resol);
        result.pl = _y2x(x0[13], kf_lep_phi_resol);
    } else {
        // 13 free params: x[0]=mW (physical); x[1..12] standardized y-coords.
        auto chi2fn = [=](const double* x) -> double {
            const double mW = x[0];
            const double s1 = _y2x(x[1],  kf_jet1_p_resp);
            const double s2 = _y2x(x[2],  kf_jet2_p_resp);
            const double sl = _y2x(x[3],  kf_lep_p_resp);
            const double sn = _y2x(x[4],  kf_met_p_resp);
            const double t1 = _y2x(x[5],  kf_jet1_theta_resol);
            const double t2 = _y2x(x[6],  kf_jet2_theta_resol);
            const double tn = _y2x(x[7],  kf_met_theta_resol);
            const double p1 = _y2x(x[8],  kf_jet1_phi_resol);
            const double p2 = _y2x(x[9],  kf_jet2_phi_resol);
            const double pn = _y2x(x[10], kf_met_phi_resol);
            const double tl = _y2x(x[11], kf_lep_theta_resol);
            const double pl = _y2x(x[12], kf_lep_phi_resol);
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
            lam = std::sqrt(lam*lam + 1e-24);  // smooth |λ| floor — derivative continuous through 0
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

        // y-coords start at 0 → physical starts at prior μ for each nuisance param.
        double x0[KF_NDIM] = {KF_MW_INIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int status = _bfgs_minimize<decltype(chi2fn), KF_NDIM>(chi2fn, x0, fmin);
        result.status = status;
        result.valid = (status == 0) ? 1 : 0;
        result.chi2  = fmin;
        result.winner_pass    = 1;
        result.n_passes_run   = 1;
        result.priors_swapped = 0;
        result.chi2_ndof = (KF_N_CONSTR > KF_NDIM) ? fmin / float(KF_N_CONSTR - KF_NDIM) : -1.0f;
        result.mW = x0[0];
        result.s1 = _y2x(x0[1],  kf_jet1_p_resp);
        result.s2 = _y2x(x0[2],  kf_jet2_p_resp);
        result.sl = _y2x(x0[3],  kf_lep_p_resp);
        result.sn = _y2x(x0[4],  kf_met_p_resp);
        result.t1 = _y2x(x0[5],  kf_jet1_theta_resol);
        result.t2 = _y2x(x0[6],  kf_jet2_theta_resol);
        result.tn = _y2x(x0[7],  kf_met_theta_resol);
        result.p1 = _y2x(x0[8],  kf_jet1_phi_resol);
        result.p2 = _y2x(x0[9],  kf_jet2_phi_resol);
        result.pn = _y2x(x0[10], kf_met_phi_resol);
        result.tl = _y2x(x0[11], kf_lep_theta_resol);
        result.pl = _y2x(x0[12], kf_lep_phi_resol);
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
    result.status = -1;
    result.chi2  = 999.0f;
    result.chi2_ndof = 999.0f;

    if (Isolep_p < 0 || jet1_p <= 0 || jet2_p <= 0 || missing_p <= 0)
        return result;

    // Local copies of the jet priors. chi2fn captures these by reference, so the
    // jet1↔jet2 swap fallback below can repoint them via std::swap without
    // rebuilding the lambda.
    DcbGaussParams p_jet1_p_resp      = kf_jet1_p_resp;
    DcbGaussParams p_jet2_p_resp      = kf_jet2_p_resp;
    DcbGaussParams p_jet1_theta_resol = kf_jet1_theta_resol;
    DcbGaussParams p_jet2_theta_resol = kf_jet2_theta_resol;
    DcbGaussParams p_jet1_phi_resol   = kf_jet1_phi_resol;
    DcbGaussParams p_jet2_phi_resol   = kf_jet2_phi_resol;

    // 14 parameters: x[0]=mW, x[1]=gW, x[2..5]=scales, x[6..8]=jet/MET theta, x[9..11]=jet/MET phi, x[12..13]=lep angles.
    // When fit_gW=false, gW is pinned to KF_GW_FIXED via FixVariable(1).
    auto chi2fn = [=, &p_jet1_p_resp, &p_jet2_p_resp,
                       &p_jet1_theta_resol, &p_jet2_theta_resol,
                       &p_jet1_phi_resol,   &p_jet2_phi_resol](const double* x) -> double {
        // x[0]=mW, x[1]=gW kept physical. x[2..13] are standardized y-coords
        // (y = (x_phys − μ_prior)/σ_prior); decoded back to physical via _y2x.
        // s_i guard handles transient negative regions during Migrad line search.
        const double mW = x[0], gW = x[1];
        const double s1 = _y2x(x[2],  p_jet1_p_resp);
        const double s2 = _y2x(x[3],  p_jet2_p_resp);
        const double sl = _y2x(x[4],  kf_lep_p_resp);
        const double sn = _y2x(x[5],  kf_met_p_resp);
        const double t1 = _y2x(x[6],  p_jet1_theta_resol);
        const double t2 = _y2x(x[7],  p_jet2_theta_resol);
        const double tn = _y2x(x[8],  kf_met_theta_resol);
        const double p1 = _y2x(x[9],  p_jet1_phi_resol);
        const double p2 = _y2x(x[10], p_jet2_phi_resol);
        const double pn = _y2x(x[11], kf_met_phi_resol);
        const double tl = _y2x(x[12], kf_lep_theta_resol);
        const double pl = _y2x(x[13], kf_lep_phi_resol);
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
        lam = std::sqrt(lam*lam + 1e-24);  // smooth |λ| floor — derivative continuous through 0
        // Joint BW × phase-space PDF (BW_norm = BW/π; phase space ∝ √λ/s_WW).
        double bw_term = -2.0 * (std::log(bw_h) + std::log(bw_l))
                       + 4.0 * std::log(M_PI)
                       - std::log(lam) + 2.0 * std::log(s_ww);

        double cons = dcb_gauss_neg2logpdf(WW.Px(), kf_px_tot_gen)
                    + dcb_gauss_neg2logpdf(WW.Py(), kf_py_tot_gen)
                    + dcb_gauss_neg2logpdf(WW.Pz(), kf_pz_tot_gen)
                    + dcb_expright_gauss_neg2logpdf(WW.M() - ECM, kf_m_gen_lnuqq_minus_ecm);

        double scale_pen = dcb_gauss_neg2logpdf(s1, p_jet1_p_resp)
                         + dcb_gauss_neg2logpdf(s2, p_jet2_p_resp)
                         + dcb_expleft_gauss_neg2logpdf(sl, kf_lep_p_resp)
                         + dcb_neg2logpdf(sn, kf_met_p_resp);

        double angular = dcb_gauss_neg2logpdf(t1, p_jet1_theta_resol)
                       + dcb_gauss_neg2logpdf(t2, p_jet2_theta_resol)
                       + dcb_neg2logpdf(tn, kf_met_theta_resol)
                       + dcb_gauss_neg2logpdf(tl, kf_lep_theta_resol)
                       + dcb_gauss_neg2logpdf(p1, p_jet1_phi_resol)
                       + dcb_gauss_neg2logpdf(p2, p_jet2_phi_resol)
                       + dcb_neg2logpdf(pn, kf_met_phi_resol)
                       + dcb_gauss_neg2logpdf(pl, kf_lep_phi_resol);

        double gw_term = fit_gW ? _gw_prior_neg2logpdf(gW) : 0.0;

        return bw_term + cons + scale_pen + angular + gw_term;
    };

    std::function<double(const double*)> fObj = chi2fn;
    ROOT::Math::Functor functor(fObj, 14);

    // Configure a Minuit2 minimizer (Migrad or Simplex) with a 14-D starting point.
    // Variable layout: 0=mW, 1=gW (physical, optionally fixed); 2..13=y-coords.
    auto configure = [&](ROOT::Math::Minimizer* m, const double* x0, bool with_strategy) {
        m->SetFunction(functor);
        m->SetMaxFunctionCalls(100000);
        m->SetTolerance(1e-3);
        if (with_strategy) m->SetStrategy(2);
        m->SetPrintLevel(-1);
        m->SetVariable(0,  "mW",   x0[0],  0.1);  m->SetVariableLimits(0, 0.0, 200.0);
        m->SetVariable(1,  "gW",   x0[1],  0.01); m->SetVariableLimits(1, 0.01, 10.0);
        m->SetVariable(2,  "y_s1", x0[2],  0.1);
        m->SetVariable(3,  "y_s2", x0[3],  0.1);
        m->SetVariable(4,  "y_sl", x0[4],  0.1);
        m->SetVariable(5,  "y_sn", x0[5],  0.1);
        m->SetVariable(6,  "y_t1", x0[6],  0.1);
        m->SetVariable(7,  "y_t2", x0[7],  0.1);
        m->SetVariable(8,  "y_tn", x0[8],  0.1);
        m->SetVariable(9,  "y_p1", x0[9],  0.1);
        m->SetVariable(10, "y_p2", x0[10], 0.1);
        m->SetVariable(11, "y_pn", x0[11], 0.1);
        m->SetVariable(12, "y_tl", x0[12], 0.1);
        m->SetVariable(13, "y_pl", x0[13], 0.1);
        if (!fit_gW) m->FixVariable(1);
    };

    double x_default[14] = {KF_MW_INIT, KF_GW_FIXED, 0,0,0,0, 0,0,0, 0,0,0, 0,0};
    std::unique_ptr<ROOT::Math::Minimizer> minimizer(
        ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad")
    );

    // Two minimizer "modes": cheap Migrad-only, and Simplex pre-pass + Migrad.
    // Status 0 = minimum found; 1 = covariance forced positive-definite (still
    // valid postfit); 3 = EDM > tol (the dominant residual failure). Tolerance
    // was loosened from 1e-6 to 1e-3 (Migrad default). The Simplex pre-pass
    // helps descend through non-quadratic regions at ~2× cost.
    auto migrad_only = [&](const double* x_init) {
        configure(minimizer.get(), x_init, /*with_strategy=*/true);
        minimizer->Minimize();
        minimizer->Minimize();
        return minimizer->Status();
    };
    auto simplex_then_migrad = [&](const double* x_init) {
        std::unique_ptr<ROOT::Math::Minimizer> simplex(
            ROOT::Math::Factory::CreateMinimizer("Minuit2", "Simplex")
        );
        configure(simplex.get(), x_init, /*with_strategy=*/false);
        simplex->Minimize();
        configure(minimizer.get(), simplex->X(), /*with_strategy=*/true);
        minimizer->Minimize();
        minimizer->Minimize();
        return minimizer->Status();
    };

    // Toggle the jet1↔jet2 prior assignment (momentum + theta + phi). Tests the
    // alternative jet-to-prior pairing for events whose pT-ordering disagrees
    // with the one used when the priors were fitted.
    auto swap_jet_priors = [&]() {
        std::swap(p_jet1_p_resp,      p_jet2_p_resp);
        std::swap(p_jet1_theta_resol, p_jet2_theta_resol);
        std::swap(p_jet1_phi_resol,   p_jet2_phi_resol);
    };

    struct PassResult { int status; double chi2; double x[14]; bool swapped; int pass_id; };
    auto snapshot = [&](int s, bool swapped_now, int pass_id) {
        PassResult r{};
        r.status   = s;
        r.chi2     = minimizer->MinValue();
        r.swapped  = swapped_now;
        r.pass_id  = pass_id;
        const double* xref = minimizer->X();
        for (int i = 0; i < 14; ++i) r.x[i] = xref[i];
        return r;
    };
    auto converged = [](int s) { return s == 0 || s == 1; };
    // Converged beats non-converged; otherwise lower chi² wins.
    auto pick_better = [&](PassResult& best, const PassResult& cand) {
        bool ob = converged(best.status), oc = converged(cand.status);
        if (oc && !ob)             { best = cand; return; }
        if (!oc && ob)             return;
        if (cand.chi2 < best.chi2) best = cand;
    };

    // Pass order (when swap is enabled, i.e. SEP priors): try cheap Migrad over
    // both prior orderings before paying the ~2× Simplex cost.
    //   1. Migrad, natural
    //   2. Migrad, swapped
    //   3. Simplex+Migrad, natural
    //   4. Simplex+Migrad, swapped
    // Stop as soon as a pass converges. With swap disabled (POOL priors are
    // jet-symmetric) the swap passes are skipped → Migrad → Simplex+Migrad on
    // natural priors only.
    int n_passes_run = 1;
    bool priors_swapped = false;
    PassResult best = snapshot(migrad_only(x_default), priors_swapped, /*pass=*/1);

    if (!converged(best.status) && kf_jet_swap_enabled) {
        swap_jet_priors(); priors_swapped = true;
        ++n_passes_run;
        pick_better(best, snapshot(migrad_only(x_default), priors_swapped, /*pass=*/2));
    }
    if (!converged(best.status)) {
        if (priors_swapped) { swap_jet_priors(); priors_swapped = false; }
        ++n_passes_run;
        pick_better(best, snapshot(simplex_then_migrad(x_default), priors_swapped, /*pass=*/3));
    }
    if (!converged(best.status) && kf_jet_swap_enabled) {
        swap_jet_priors(); priors_swapped = true;
        ++n_passes_run;
        pick_better(best, snapshot(simplex_then_migrad(x_default), priors_swapped, /*pass=*/4));
    }

    // Sync in-scope priors to the winner's orientation — result extraction
    // below reads them by reference.
    if (priors_swapped != best.swapped) swap_jet_priors();

    int    status   = best.status;
    double chi2     = best.chi2;
    const double* x_final = best.x;

    result.status         = status;
    result.valid          = (status == 0 || status == 1) ? 1 : 0;
    result.chi2           = chi2;
    result.winner_pass    = best.pass_id;
    result.n_passes_run   = n_passes_run;
    result.priors_swapped = best.swapped ? 1 : 0;
    int n_par    = fit_gW ? 14 : KF_NDIM;
    // +1 constraint from the gW Gaussian prior when fit_gW=true.
    int n_constr = KF_N_CONSTR + (fit_gW ? 1 : 0);
    result.chi2_ndof = (n_constr > n_par) ? result.chi2 / float(n_constr - n_par) : -1.0f;
    result.mW = x_final[0]; result.gW = x_final[1];
    result.s1 = _y2x(x_final[2],  p_jet1_p_resp);
    result.s2 = _y2x(x_final[3],  p_jet2_p_resp);
    result.sl = _y2x(x_final[4],  kf_lep_p_resp);
    result.sn = _y2x(x_final[5],  kf_met_p_resp);
    result.t1 = _y2x(x_final[6],  p_jet1_theta_resol);
    result.t2 = _y2x(x_final[7],  p_jet2_theta_resol);
    result.tn = _y2x(x_final[8],  kf_met_theta_resol);
    result.p1 = _y2x(x_final[9],  p_jet1_phi_resol);
    result.p2 = _y2x(x_final[10], p_jet2_phi_resol);
    result.pn = _y2x(x_final[11], kf_met_phi_resol);
    result.tl = _y2x(x_final[12], kf_lep_theta_resol);
    result.pl = _y2x(x_final[13], kf_lep_phi_resol);

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
