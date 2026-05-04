#ifndef WWKinReco_H
#define WWKinReco_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <functional>
#include <random>
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
//
// All resolutions (jet/lep p_resp, θ/φ resolutions) are now BINNED — each holds
// a 5-element std::array of DCB params plus the equal-occupancy quantile edges
// (6 doubles). The per-event prior is selected at kinFit() entry by feeding the
// reco kinematic into pick_bin(): jet*_p / lep_p drive p_resp + θ_resol; jet
// |cosθ| drives jet φ_resol; lep_p drives lep φ_resol. MET and gen-level
// constraints remain unbinned (no kinematics dependence in the resol study).
constexpr std::size_t KF_NBINS = 5;

struct KinFitParamSet {
    // Binned variants (per-bin DCB params + equal-occupancy quantile edges).
    std::array<DcbGaussParams,        KF_NBINS> jet1_p_resp_bins;
    std::array<double,                KF_NBINS + 1> jet1_p_resp_edges;
    std::array<DcbGaussParams,        KF_NBINS> jet2_p_resp_bins;
    std::array<double,                KF_NBINS + 1> jet2_p_resp_edges;
    std::array<DcbExpLeftGaussParams, KF_NBINS> lep_p_resp_bins;        // expleft2g
    std::array<double,                KF_NBINS + 1> lep_p_resp_edges;
    DcbParams                                    met_p_resp;
    std::array<DcbGaussParams,        KF_NBINS> jet1_phi_resol_bins;    // dcb2g
    std::array<double,                KF_NBINS + 1> jet1_phi_resol_edges;
    std::array<DcbGaussParams,        KF_NBINS> jet1_theta_resol_bins;  // dcb2g
    std::array<double,                KF_NBINS + 1> jet1_theta_resol_edges;
    std::array<DcbGaussParams,        KF_NBINS> jet2_phi_resol_bins;    // dcb2g
    std::array<double,                KF_NBINS + 1> jet2_phi_resol_edges;
    std::array<DcbGaussParams,        KF_NBINS> jet2_theta_resol_bins;  // dcb2g
    std::array<double,                KF_NBINS + 1> jet2_theta_resol_edges;
    std::array<DcbGaussParams,        KF_NBINS> lep_phi_resol_bins;     // dcb2g
    std::array<double,                KF_NBINS + 1> lep_phi_resol_edges;
    std::array<DcbGaussParams,        KF_NBINS> lep_theta_resol_bins;   // dcb2g
    std::array<double,                KF_NBINS + 1> lep_theta_resol_edges;
    DcbParams                                    met_phi_resol;
    DcbParams                                    met_theta_resol;
    DcbExpRightGaussParams                       m_gen_lnuqq_minus_ecm;
    DcbGaussParams                               px_tot_gen;
    DcbGaussParams                               py_tot_gen;
    DcbGaussParams                               pz_tot_gen;
    // Inclusive (kinematics-averaged) variants — used when kf_use_binned_priors=false.
    DcbGaussParams                               jet1_p_resp_incl;
    DcbGaussParams                               jet2_p_resp_incl;
    DcbExpLeftGaussParams                        lep_p_resp_incl;
    DcbGaussParams                               jet1_phi_resol_incl;
    DcbGaussParams                               jet1_theta_resol_incl;
    DcbGaussParams                               jet2_phi_resol_incl;
    DcbGaussParams                               jet2_theta_resol_incl;
    DcbGaussParams                               lep_phi_resol_incl;
    DcbGaussParams                               lep_theta_resol_incl;
};

// Three jet-prior conventions, selected at setKinFitParams() time:
//  - POOL  : jet1 and jet2 share the pooled prior DCBG_JET_*_BINS_{ECM}. No
//            pT-ordering dependence; jet1↔jet2 swap fallback is a no-op so it's
//            disabled.
//  - SWAP  : per-jet priors DCBG_JET{1,2}_*_BINS_{ECM}; swap fallback enabled
//            so events whose pT ordering disagrees with the prior fit ordering
//            can still converge.
//  - FIXED : per-jet priors DCBG_JET{1,2}_*_BINS_{ECM}; swap fallback disabled
//            (sanity check: how often do we lose convergence without swap?).
#define KF_POOL_BUNDLE(E) \
    DCBG_JET_P_RESP_BINS_##E,        DCBG_JET_P_RESP_EDGES_##E, \
    DCBG_JET_P_RESP_BINS_##E,        DCBG_JET_P_RESP_EDGES_##E, \
    DCBELG_LEP_P_RESP_BINS_##E,      DCBELG_LEP_P_RESP_EDGES_##E, \
    DCB_MET_P_RESP_##E, \
    DCBG_JET_PHI_RESOL_BINS_##E,     DCBG_JET_PHI_RESOL_EDGES_##E, \
    DCBG_JET_THETA_RESOL_BINS_##E,   DCBG_JET_THETA_RESOL_EDGES_##E, \
    DCBG_JET_PHI_RESOL_BINS_##E,     DCBG_JET_PHI_RESOL_EDGES_##E, \
    DCBG_JET_THETA_RESOL_BINS_##E,   DCBG_JET_THETA_RESOL_EDGES_##E, \
    DCBG_LEP_PHI_RESOL_BINS_##E,     DCBG_LEP_PHI_RESOL_EDGES_##E, \
    DCBG_LEP_THETA_RESOL_BINS_##E,   DCBG_LEP_THETA_RESOL_EDGES_##E, \
    DCB_MET_PHI_RESOL_##E,           DCB_MET_THETA_RESOL_##E, \
    DCBERG_GEN_WW_M_MINUS_ECM_##E, \
    DCBG_GEN_WW_PX_##E, DCBG_GEN_WW_PY_##E, DCBG_GEN_WW_PZ_##E, \
    /* inclusive — POOL: jet1 and jet2 share the pooled scalar */ \
    DCBG_JET_P_RESP_##E, DCBG_JET_P_RESP_##E, DCBELG_LEP_P_RESP_##E, \
    DCBG_JET_PHI_RESOL_##E, DCBG_JET_THETA_RESOL_##E, \
    DCBG_JET_PHI_RESOL_##E, DCBG_JET_THETA_RESOL_##E, \
    DCBG_LEP_PHI_RESOL_##E, DCBG_LEP_THETA_RESOL_##E

#define KF_SEP_BUNDLE(E) \
    DCBG_JET1_P_RESP_BINS_##E,       DCBG_JET1_P_RESP_EDGES_##E, \
    DCBG_JET2_P_RESP_BINS_##E,       DCBG_JET2_P_RESP_EDGES_##E, \
    DCBELG_LEP_P_RESP_BINS_##E,      DCBELG_LEP_P_RESP_EDGES_##E, \
    DCB_MET_P_RESP_##E, \
    DCBG_JET1_PHI_RESOL_BINS_##E,    DCBG_JET1_PHI_RESOL_EDGES_##E, \
    DCBG_JET1_THETA_RESOL_BINS_##E,  DCBG_JET1_THETA_RESOL_EDGES_##E, \
    DCBG_JET2_PHI_RESOL_BINS_##E,    DCBG_JET2_PHI_RESOL_EDGES_##E, \
    DCBG_JET2_THETA_RESOL_BINS_##E,  DCBG_JET2_THETA_RESOL_EDGES_##E, \
    DCBG_LEP_PHI_RESOL_BINS_##E,     DCBG_LEP_PHI_RESOL_EDGES_##E, \
    DCBG_LEP_THETA_RESOL_BINS_##E,   DCBG_LEP_THETA_RESOL_EDGES_##E, \
    DCB_MET_PHI_RESOL_##E,           DCB_MET_THETA_RESOL_##E, \
    DCBERG_GEN_WW_M_MINUS_ECM_##E, \
    DCBG_GEN_WW_PX_##E, DCBG_GEN_WW_PY_##E, DCBG_GEN_WW_PZ_##E, \
    /* inclusive — SEP: per-jet scalar */ \
    DCBG_JET1_P_RESP_##E, DCBG_JET2_P_RESP_##E, DCBELG_LEP_P_RESP_##E, \
    DCBG_JET1_PHI_RESOL_##E, DCBG_JET1_THETA_RESOL_##E, \
    DCBG_JET2_PHI_RESOL_##E, DCBG_JET2_THETA_RESOL_##E, \
    DCBG_LEP_PHI_RESOL_##E, DCBG_LEP_THETA_RESOL_##E

static const KinFitParamSet KF_PARAMS_157_POOL = { KF_POOL_BUNDLE(157) };
static const KinFitParamSet KF_PARAMS_160_POOL = { KF_POOL_BUNDLE(160) };
static const KinFitParamSet KF_PARAMS_163_POOL = { KF_POOL_BUNDLE(163) };
static const KinFitParamSet KF_PARAMS_157_SEP  = { KF_SEP_BUNDLE(157) };
static const KinFitParamSet KF_PARAMS_160_SEP  = { KF_SEP_BUNDLE(160) };
static const KinFitParamSet KF_PARAMS_163_SEP  = { KF_SEP_BUNDLE(163) };

// ── Active kinfit parameters (set per-dataset via setKinFitParams) ─────────
// Binned priors live in std::array<...> bundles plus matching edge arrays.
// kinFit() picks the per-event prior via pick_bin() at the function entry.
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet1_p_resp_bins      = DCBG_JET1_P_RESP_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet1_p_resp_edges      = DCBG_JET1_P_RESP_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet2_p_resp_bins      = DCBG_JET2_P_RESP_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet2_p_resp_edges      = DCBG_JET2_P_RESP_EDGES_160;
inline std::array<DcbExpLeftGaussParams, KF_NBINS> kf_lep_p_resp_bins       = DCBELG_LEP_P_RESP_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_lep_p_resp_edges       = DCBELG_LEP_P_RESP_EDGES_160;
inline DcbParams                                    kf_met_p_resp            = DCB_MET_P_RESP_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet1_phi_resol_bins   = DCBG_JET1_PHI_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet1_phi_resol_edges   = DCBG_JET1_PHI_RESOL_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet1_theta_resol_bins = DCBG_JET1_THETA_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet1_theta_resol_edges = DCBG_JET1_THETA_RESOL_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet2_phi_resol_bins   = DCBG_JET2_PHI_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet2_phi_resol_edges   = DCBG_JET2_PHI_RESOL_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet2_theta_resol_bins = DCBG_JET2_THETA_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet2_theta_resol_edges = DCBG_JET2_THETA_RESOL_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_lep_phi_resol_bins    = DCBG_LEP_PHI_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_lep_phi_resol_edges    = DCBG_LEP_PHI_RESOL_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_lep_theta_resol_bins  = DCBG_LEP_THETA_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_lep_theta_resol_edges  = DCBG_LEP_THETA_RESOL_EDGES_160;
inline DcbParams                                    kf_met_phi_resol         = DCB_MET_PHI_RESOL_160;
inline DcbParams                                    kf_met_theta_resol       = DCB_MET_THETA_RESOL_160;
inline DcbExpRightGaussParams                       kf_m_gen_lnuqq_minus_ecm = DCBERG_GEN_WW_M_MINUS_ECM_160;
inline DcbGaussParams                               kf_px_tot_gen            = DCBG_GEN_WW_PX_160;
inline DcbGaussParams                               kf_py_tot_gen            = DCBG_GEN_WW_PY_160;
inline DcbGaussParams                               kf_pz_tot_gen            = DCBG_GEN_WW_PZ_160;

// Inclusive (kinematics-averaged) scalars — used when kf_use_binned_priors=false.
// Same data as the bin arrays would collapse to with one bin spanning all events.
inline DcbGaussParams         kf_jet1_p_resp_incl      = DCBG_JET1_P_RESP_160;
inline DcbGaussParams         kf_jet2_p_resp_incl      = DCBG_JET2_P_RESP_160;
inline DcbExpLeftGaussParams  kf_lep_p_resp_incl       = DCBELG_LEP_P_RESP_160;
inline DcbGaussParams         kf_jet1_phi_resol_incl   = DCBG_JET1_PHI_RESOL_160;
inline DcbGaussParams         kf_jet1_theta_resol_incl = DCBG_JET1_THETA_RESOL_160;
inline DcbGaussParams         kf_jet2_phi_resol_incl   = DCBG_JET2_PHI_RESOL_160;
inline DcbGaussParams         kf_jet2_theta_resol_incl = DCBG_JET2_THETA_RESOL_160;
inline DcbGaussParams         kf_lep_phi_resol_incl    = DCBG_LEP_PHI_RESOL_160;
inline DcbGaussParams         kf_lep_theta_resol_incl  = DCBG_LEP_THETA_RESOL_160;

// True when the jet1↔jet2 swap fallback should be tried on non-converged
// events. Enabled only in "swap" mode; "pool" priors are jet-symmetric so swap
// is a no-op, and "fixed" mode deliberately skips the swap to measure the
// natural-ordering convergence rate alone.
inline bool kf_jet_swap_enabled = false;
// True → kinFit picks per-event prior from the binned arrays via pick_bin().
// False → use the inclusive (kinematics-averaged) scalar priors.
inline bool kf_use_binned_priors = true;

inline void setKinFitParams(int ecm, const std::string& jet_mode = "swap",
                             bool use_binned = true) {
    ECM = static_cast<float>(ecm);
    const bool use_pool  = (jet_mode == "pool");
    const bool use_fixed = (jet_mode == "fixed");
    // Swap fallback only in "swap" (default) mode.
    kf_jet_swap_enabled  = !use_pool && !use_fixed;
    kf_use_binned_priors = use_binned;
    const KinFitParamSet* p =
        ecm == 157 ? (use_pool ? &KF_PARAMS_157_POOL : &KF_PARAMS_157_SEP) :
        ecm == 160 ? (use_pool ? &KF_PARAMS_160_POOL : &KF_PARAMS_160_SEP) :
        ecm == 163 ? (use_pool ? &KF_PARAMS_163_POOL : &KF_PARAMS_163_SEP) : nullptr;
    if (!p) return;
    kf_jet1_p_resp_bins       = p->jet1_p_resp_bins;
    kf_jet1_p_resp_edges      = p->jet1_p_resp_edges;
    kf_jet2_p_resp_bins       = p->jet2_p_resp_bins;
    kf_jet2_p_resp_edges      = p->jet2_p_resp_edges;
    kf_lep_p_resp_bins        = p->lep_p_resp_bins;
    kf_lep_p_resp_edges       = p->lep_p_resp_edges;
    kf_met_p_resp             = p->met_p_resp;
    kf_jet1_phi_resol_bins    = p->jet1_phi_resol_bins;
    kf_jet1_phi_resol_edges   = p->jet1_phi_resol_edges;
    kf_jet1_theta_resol_bins  = p->jet1_theta_resol_bins;
    kf_jet1_theta_resol_edges = p->jet1_theta_resol_edges;
    kf_jet2_phi_resol_bins    = p->jet2_phi_resol_bins;
    kf_jet2_phi_resol_edges   = p->jet2_phi_resol_edges;
    kf_jet2_theta_resol_bins  = p->jet2_theta_resol_bins;
    kf_jet2_theta_resol_edges = p->jet2_theta_resol_edges;
    kf_lep_phi_resol_bins     = p->lep_phi_resol_bins;
    kf_lep_phi_resol_edges    = p->lep_phi_resol_edges;
    kf_lep_theta_resol_bins   = p->lep_theta_resol_bins;
    kf_lep_theta_resol_edges  = p->lep_theta_resol_edges;
    kf_met_phi_resol          = p->met_phi_resol;
    kf_met_theta_resol        = p->met_theta_resol;
    kf_m_gen_lnuqq_minus_ecm  = p->m_gen_lnuqq_minus_ecm;
    kf_px_tot_gen             = p->px_tot_gen;
    kf_py_tot_gen             = p->py_tot_gen;
    kf_pz_tot_gen             = p->pz_tot_gen;
    kf_jet1_p_resp_incl       = p->jet1_p_resp_incl;
    kf_jet2_p_resp_incl       = p->jet2_p_resp_incl;
    kf_lep_p_resp_incl        = p->lep_p_resp_incl;
    kf_jet1_phi_resol_incl    = p->jet1_phi_resol_incl;
    kf_jet1_theta_resol_incl  = p->jet1_theta_resol_incl;
    kf_jet2_phi_resol_incl    = p->jet2_phi_resol_incl;
    kf_jet2_theta_resol_incl  = p->jet2_theta_resol_incl;
    kf_lep_phi_resol_incl     = p->lep_phi_resol_incl;
    kf_lep_theta_resol_incl   = p->lep_theta_resol_incl;
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

// Loose-valid EDM cap. Migrad tolerance is 1e-3 (set in configure); any status=3
// event with finite EDM under 10× that lands a fit point essentially indistinguishable
// from a converged one (matched chi²/ndof distributions; verified 2026-05-04).
static constexpr double KF_LOOSE_EDM_MAX = 1e-2;

// Random-restart 6th pass. After the 5-pass cascade has failed, re-run Migrad
// from up to N draws of x_default jittered by 0.5σ in y-space. Seeded from the
// event's reco kinematics → reproducible across runs.
static constexpr int    KF_RESTART_N      = 2;
static constexpr double KF_RESTART_SIGMA  = 0.5;

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
    int   valid;            // strict: status ∈ {0,1} (Migrad+Hesse clean)
    int   valid_loose;      // loose: valid OR (status==3 AND finite EDM AND edm<KF_LOOSE_EDM_MAX)
                            //   — Migrad-near-converged events whose EDM is within one
                            //   decade of tolerance. Empirically clean (~57-61% of
                            //   status=3 events; no chi²/ndof tail vs strict valid).
    // Diagnostics: which pass won and how many actually ran.
    //   winner_pass: 1=Migrad-natural, 2=Migrad-swapped,
    //                3=Simplex+Migrad-natural, 4=Simplex+Migrad-swapped,
    //                5=Hesse-refresh, 6=random-restart Migrad.
    //   n_passes_run: total passes executed before stopping (1..6+restarts).
    //   priors_swapped: 1 if the winning pass used jet1↔jet2-swapped priors.
    int   winner_pass;
    int   n_passes_run;
    int   priors_swapped;
    float edm;              // estimated distance to minimum at convergence (Minuit2)
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

    // Pick per-event priors. Jet priors are taken by VALUE so chi2fn can
    // capture them by reference. Each jet ALSO has an "alt" pick — the other
    // jet's prior at the SAME jet's own kinematics (binned: other binset, own
    // bin index; inclusive: other jet's scalar). swap_jet_priors() toggles
    // active <-> alt via std::swap (involution).
    const double j1_acth = std::abs(std::cos(jet1_theta));
    const double j2_acth = std::abs(std::cos(jet2_theta));
    DcbGaussParams p_jet1_p_resp      = kf_use_binned_priors ? pick_bin(kf_jet1_p_resp_bins,      kf_jet1_p_resp_edges,      jet1_p)   : kf_jet1_p_resp_incl;
    DcbGaussParams p_jet2_p_resp      = kf_use_binned_priors ? pick_bin(kf_jet2_p_resp_bins,      kf_jet2_p_resp_edges,      jet2_p)   : kf_jet2_p_resp_incl;
    DcbGaussParams p_jet1_theta_resol = kf_use_binned_priors ? pick_bin(kf_jet1_theta_resol_bins, kf_jet1_theta_resol_edges, jet1_p)   : kf_jet1_theta_resol_incl;
    DcbGaussParams p_jet2_theta_resol = kf_use_binned_priors ? pick_bin(kf_jet2_theta_resol_bins, kf_jet2_theta_resol_edges, jet2_p)   : kf_jet2_theta_resol_incl;
    DcbGaussParams p_jet1_phi_resol   = kf_use_binned_priors ? pick_bin(kf_jet1_phi_resol_bins,   kf_jet1_phi_resol_edges,   j1_acth)  : kf_jet1_phi_resol_incl;
    DcbGaussParams p_jet2_phi_resol   = kf_use_binned_priors ? pick_bin(kf_jet2_phi_resol_bins,   kf_jet2_phi_resol_edges,   j2_acth)  : kf_jet2_phi_resol_incl;
    DcbGaussParams p_jet1_p_resp_alt{};
    DcbGaussParams p_jet2_p_resp_alt{};
    DcbGaussParams p_jet1_theta_resol_alt{};
    DcbGaussParams p_jet2_theta_resol_alt{};
    DcbGaussParams p_jet1_phi_resol_alt{};
    DcbGaussParams p_jet2_phi_resol_alt{};
    if (kf_jet_swap_enabled) {
        p_jet1_p_resp_alt      = kf_use_binned_priors ? pick_bin(kf_jet2_p_resp_bins,      kf_jet2_p_resp_edges,      jet1_p)   : kf_jet2_p_resp_incl;
        p_jet2_p_resp_alt      = kf_use_binned_priors ? pick_bin(kf_jet1_p_resp_bins,      kf_jet1_p_resp_edges,      jet2_p)   : kf_jet1_p_resp_incl;
        p_jet1_theta_resol_alt = kf_use_binned_priors ? pick_bin(kf_jet2_theta_resol_bins, kf_jet2_theta_resol_edges, jet1_p)   : kf_jet2_theta_resol_incl;
        p_jet2_theta_resol_alt = kf_use_binned_priors ? pick_bin(kf_jet1_theta_resol_bins, kf_jet1_theta_resol_edges, jet2_p)   : kf_jet1_theta_resol_incl;
        p_jet1_phi_resol_alt   = kf_use_binned_priors ? pick_bin(kf_jet2_phi_resol_bins,   kf_jet2_phi_resol_edges,   j1_acth)  : kf_jet2_phi_resol_incl;
        p_jet2_phi_resol_alt   = kf_use_binned_priors ? pick_bin(kf_jet1_phi_resol_bins,   kf_jet1_phi_resol_edges,   j2_acth)  : kf_jet1_phi_resol_incl;
    }
    const DcbExpLeftGaussParams kf_lep_p_resp     = kf_use_binned_priors ? pick_bin(kf_lep_p_resp_bins,      kf_lep_p_resp_edges,      Isolep_p) : kf_lep_p_resp_incl;
    const DcbGaussParams        kf_lep_theta_resol = kf_use_binned_priors ? pick_bin(kf_lep_theta_resol_bins, kf_lep_theta_resol_edges, Isolep_p) : kf_lep_theta_resol_incl;
    const DcbGaussParams        kf_lep_phi_resol   = kf_use_binned_priors ? pick_bin(kf_lep_phi_resol_bins,   kf_lep_phi_resol_edges,   Isolep_p) : kf_lep_phi_resol_incl;

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

    // Toggle the jet1↔jet2 prior assignment (momentum + theta + phi). Tests
    // the alternative jet-to-prior pairing for events whose pT-ordering
    // disagrees with the prior-fit ordering. Each std::swap exchanges the
    // active prior with its precomputed alt (other-jet binset, own kinematics)
    // — so the bin index per jet stays fixed; only the binset name flips.
    auto swap_jet_priors = [&]() {
        std::swap(p_jet1_p_resp,      p_jet1_p_resp_alt);
        std::swap(p_jet2_p_resp,      p_jet2_p_resp_alt);
        std::swap(p_jet1_theta_resol, p_jet1_theta_resol_alt);
        std::swap(p_jet2_theta_resol, p_jet2_theta_resol_alt);
        std::swap(p_jet1_phi_resol,   p_jet1_phi_resol_alt);
        std::swap(p_jet2_phi_resol,   p_jet2_phi_resol_alt);
    };

    struct PassResult { int status; double chi2; double edm; double x[14]; bool swapped; int pass_id; };
    auto snapshot = [&](int s, bool swapped_now, int pass_id) {
        PassResult r{};
        r.status   = s;
        r.chi2     = minimizer->MinValue();
        r.edm      = minimizer->Edm();
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

    // Pass 5: Hesse + re-Migrad recovery from the best.x found so far. Refreshes
    // Hessian numerically (breaks stale-Davidon plateaus). Also fires when the
    // Davidon estimate is NaN/Inf — Hesse recomputes from scratch by finite
    // differences and doesn't depend on the broken estimate.
    if (!converged(best.status) && (!std::isfinite(best.edm) || best.edm < 1.0)) {
        if (priors_swapped != best.swapped) {
            swap_jet_priors();
            priors_swapped = best.swapped;
        }
        configure(minimizer.get(), best.x, /*with_strategy=*/true);
        minimizer->Hesse();
        minimizer->Minimize();
        ++n_passes_run;
        pick_better(best, snapshot(minimizer->Status(), priors_swapped, /*pass=*/5));
    }

    // Pass 6: deterministic random-restart Migrad. Catches events the 5-pass
    // cascade leaves with large or non-finite EDM (the residual ~20% of
    // status=3 events that pass 5 can't touch). Seed is hashed from event
    // kinematics so the same event gets the same jitter on re-runs.
    if (!converged(best.status)) {
        if (priors_swapped != best.swapped) {
            swap_jet_priors();
            priors_swapped = best.swapped;
        }
        auto bits_of = [](double d) -> uint64_t {
            uint64_t b; std::memcpy(&b, &d, sizeof(b)); return b;
        };
        uint64_t s = bits_of(jet1_p) ^ (bits_of(jet2_p) << 1) ^ (bits_of(Isolep_p) << 2);
        std::mt19937_64 rng(s);
        std::normal_distribution<double> jitter(0.0, KF_RESTART_SIGMA);
        for (int t = 0; t < KF_RESTART_N; ++t) {
            double x_jitter[14];
            for (int i = 0; i < 14; ++i) x_jitter[i] = x_default[i];
            for (int i = 2; i < 14; ++i) x_jitter[i] += jitter(rng);  // jitter y-coords only
            ++n_passes_run;
            pick_better(best, snapshot(migrad_only(x_jitter), priors_swapped, /*pass=*/6));
            if (converged(best.status)) break;
        }
    }

    // Sync in-scope priors to the winner's orientation — result extraction
    // below reads them by reference.
    if (priors_swapped != best.swapped) swap_jet_priors();

    int    status   = best.status;
    double chi2     = best.chi2;
    const double* x_final = best.x;

    result.status         = status;
    result.valid          = (status == 0 || status == 1) ? 1 : 0;
    result.valid_loose    = (result.valid
                             || (status == 3 && std::isfinite(best.edm)
                                 && best.edm < KF_LOOSE_EDM_MAX)) ? 1 : 0;
    result.chi2           = chi2;
    result.winner_pass    = best.pass_id;
    result.n_passes_run   = n_passes_run;
    result.priors_swapped = best.swapped ? 1 : 0;
    result.edm            = static_cast<float>(best.edm);
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
