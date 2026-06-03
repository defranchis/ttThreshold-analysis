// ─────────────────────────────────────────────────────────────────────────────
// ARCHIVED: BFGS minimizer + kinFitBFGS variant of the WW kinematic fit.
// Removed from production code 2026-05-04. Kept here for future reference only;
// **NOT compiled into the build**.
//
// === Why this was an alternative to Minuit2 ===
// Stack-only, template-inlined, no std::function heap traffic and no need for
// thread_local annotations to be safe under ROOT::RDataFrame multithreading.
// At the time it was written, the chi² landscape was simpler and BFGS-with-
// Armijo handled it adequately. Faster than Minuit2 (~50-150 chi² evaluations
// per event vs ~200-500) when convergence was easy.
//
// === Why it was removed ===
// As the chi² evolved (BW × phase-space + λ floor, dcber2g priors with heavy
// ISR right tail, binned per-event priors, hard sentinel returns at s_i ≤ 0
// and gW ≤ 0, etc.), naive line-search BFGS started failing at almost every
// event. Tested on 2026-05-04 with the canonical chi²:
//   - Without trust-radius cap on initial α: ~100% line-search failure at
//     iter 0 (steepest-descent step at α=1 overshoots into hard-sentinel
//     barriers, Armijo halves α 40 times and gives up).
//   - With trust-radius cap STEP_MAX=0.1 on initial α: iter-0 stalls fixed,
//     but events still LS-fail at iter ~16-32 when BFGS drifts into NaN-
//     producing regions of the chi² (suspected: DCB power-law tail underflow
//     or BW pole near mh = mW). valid_frac = 0.00 / 0.00 / 0.02 % at
//     ECM 157/160/163 vs 97 / 92 / 61 % for Minuit2 with the 5-pass cascade.
//   - Adding an mW>0 guard in the chi² lambda did not change the result —
//     mW staying positive was confirmed; the NaN comes from elsewhere.
//
// Reviving this would require either a trust-region BFGS, an H-reset-on-stall
// fallback, or a Simplex pre-pass — i.e., reproducing the structural
// robustness of Minuit2's Migrad cascade. Not worth the engineering effort
// against the current chi² unless thread-safe minimization becomes a
// real bottleneck.
//
// See `project_kinfit_object_correlations.md` and the 2026-05-04 BFGS test
// session for full context.
//
// === Snapshot below: the code as it existed in WWFunctions/WWKinReco.h
// === just before removal. Includes the diagnostic instrumentation
// === (n_iter + final ‖grad‖ out-params, status codes 0/1/2) and the
// === STEP_MAX=0.1 trust-radius cap added during the same investigation.
// ─────────────────────────────────────────────────────────────────────────────

// ── BFGS minimizer ────────────────────────────────────────────────────────
// Template avoids std::function overhead (no heap, no virtual dispatch).
// All work arrays are on the stack, so the function is inherently thread-safe
// without any thread_local annotation — suitable for multithreaded RDataFrame.
//
// Uses central finite differences for the gradient and Armijo backtracking
// for the line search. Resets H to identity on non-descent or line-search
// failure so it never gets permanently stuck.
//
// Status codes:
//   0 = converged (||grad|| < GTOL)
//   1 = max iterations reached (||grad|| still > GTOL)
//   2 = line-search failure (no Armijo-acceptable step found)
// Diagnostics (out params): n_iter = iterations actually run; final_grad_norm =
// ||grad|| at exit (analog to Minuit2's EDM — single-number stationarity proxy).
template<typename Func, int N>
static int _bfgs_minimize(const Func& f, double* x, double& fmin,
                          int& n_iter, double& final_grad_norm) {
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

    int iter = 0;
    double gnorm2 = 0;
    for (; iter < MAXITER; ++iter) {
        // Convergence check on gradient norm
        gnorm2 = 0;
        for (int i = 0; i < N; ++i) gnorm2 += g[i]*g[i];
        if (gnorm2 < GTOL*GTOL) {
            n_iter = iter; final_grad_norm = std::sqrt(gnorm2);
            return 0;
        }

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

        // Armijo backtracking line search. Cap initial α with a trust-radius
        // bound: ‖α·d‖_∞ ≤ STEP_MAX. Without this, iter 0 (H=I → d=-g, α=1)
        // overshoots into the chi² barriers (s_i ≤ 0, gW ≤ 0, λ < 0, DCB
        // power-law tails) and Armijo backtracking cannot recover. STEP_MAX
        // matches Minuit2's SetVariable() initial step size (0.1) — small
        // enough to stay in-bounds, large enough not to crawl. The cap is a
        // no-op once H accumulates curvature and ‖d‖_∞ shrinks naturally.
        constexpr double STEP_MAX = 0.1;
        double max_d = 0;
        for (int i = 0; i < N; ++i)
            if (std::abs(d[i]) > max_d) max_d = std::abs(d[i]);
        double alpha = (max_d > STEP_MAX) ? STEP_MAX / max_d : 1.0;
        bool   ls_ok = false;
        for (int ls = 0; ls < 40; ++ls) {
            for (int i = 0; i < N; ++i) xn[i] = x[i] + alpha * d[i];
            double fn = f(xn);
            if (fn <= fmin + C1 * alpha * slope) { fmin = fn; ls_ok = true; break; }
            alpha *= 0.5;
            if (alpha < 1e-14) break;
        }
        if (!ls_ok) {
            // Line search failed — exit with status 2.
            n_iter = iter; final_grad_norm = std::sqrt(gnorm2);
            return 2;
        }

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
    // Max iterations reached without meeting GTOL.
    n_iter = iter; final_grad_norm = std::sqrt(gnorm2);
    return 1;
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

    // Pick per-event priors. When kf_use_binned_priors is true: bin via
    // pick_bin() — p_resp / θ_resol on object p; jet φ_resol on |cos θ|; lep
    // φ_resol on lep_p. Otherwise fall back to the inclusive scalar priors.
    const double j1_acth = std::abs(std::cos(jet1_theta));
    const double j2_acth = std::abs(std::cos(jet2_theta));
    const DcbGaussParams         kf_jet1_p_resp      = kf_use_binned_priors ? pick_bin(kf_jet1_p_resp_bins,      kf_jet1_p_resp_edges,      jet1_p)   : kf_jet1_p_resp_incl;
    const DcbGaussParams         kf_jet2_p_resp      = kf_use_binned_priors ? pick_bin(kf_jet2_p_resp_bins,      kf_jet2_p_resp_edges,      jet2_p)   : kf_jet2_p_resp_incl;
    const DcbExpLeftGaussParams  kf_lep_p_resp       = kf_use_binned_priors ? pick_bin(kf_lep_p_resp_bins,       kf_lep_p_resp_edges,       Isolep_p) : kf_lep_p_resp_incl;
    const DcbGaussParams         kf_jet1_theta_resol = kf_use_binned_priors ? pick_bin(kf_jet1_theta_resol_bins, kf_jet1_theta_resol_edges, jet1_p)   : kf_jet1_theta_resol_incl;
    const DcbGaussParams         kf_jet2_theta_resol = kf_use_binned_priors ? pick_bin(kf_jet2_theta_resol_bins, kf_jet2_theta_resol_edges, jet2_p)   : kf_jet2_theta_resol_incl;
    const DcbGaussParams         kf_jet1_phi_resol   = kf_use_binned_priors ? pick_bin(kf_jet1_phi_resol_bins,   kf_jet1_phi_resol_edges,   j1_acth)  : kf_jet1_phi_resol_incl;
    const DcbGaussParams         kf_jet2_phi_resol   = kf_use_binned_priors ? pick_bin(kf_jet2_phi_resol_bins,   kf_jet2_phi_resol_edges,   j2_acth)  : kf_jet2_phi_resol_incl;
    const DcbGaussParams         kf_lep_phi_resol    = kf_use_binned_priors ? pick_bin(kf_lep_phi_resol_bins,    kf_lep_phi_resol_edges,    Isolep_p) : kf_lep_phi_resol_incl;
    const DcbGaussParams         kf_lep_theta_resol  = kf_use_binned_priors ? pick_bin(kf_lep_theta_resol_bins,  kf_lep_theta_resol_edges,  Isolep_p) : kf_lep_theta_resol_incl;

    double fmin = 0;

    if (fit_gW) {
        // 14 free params: x[0]=mW, x[1]=gW (physical); x[2..13] standardized y-coords.
        auto chi2fn = [=, &kf_jet1_p_resp, &kf_jet2_p_resp, &kf_lep_p_resp,
                          &kf_jet1_theta_resol, &kf_jet2_theta_resol, &kf_lep_theta_resol,
                          &kf_jet1_phi_resol,   &kf_jet2_phi_resol,   &kf_lep_phi_resol](const double* x) -> double {
            const double mW = x[0], gW = x[1];
            // Guard against unphysical W mass / width — Minuit2 has SetVariableLimits
            // for these, BFGS does not. Without this, log(bw_h) for mW<0 yields NaN
            // that poisons the gradient and stalls the line search.
            if (mW <= 0.0 || gW <= 0.0) return 1e10;
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
        int n_iter = 0; double grad_norm = 0.0;
        int status = _bfgs_minimize<decltype(chi2fn), 14>(chi2fn, x0, fmin, n_iter, grad_norm);
        result.status = status;
        result.valid = (status == 0) ? 1 : 0;
        result.chi2  = fmin;
        result.winner_pass    = 1;
        result.n_passes_run   = n_iter;                       // BFGS: iterations actually run
        result.priors_swapped = 0;
        result.edm            = static_cast<float>(grad_norm); // BFGS: ||grad|| at exit (analog to EDM)
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
        auto chi2fn = [=, &kf_jet1_p_resp, &kf_jet2_p_resp, &kf_lep_p_resp,
                          &kf_jet1_theta_resol, &kf_jet2_theta_resol, &kf_lep_theta_resol,
                          &kf_jet1_phi_resol,   &kf_jet2_phi_resol,   &kf_lep_phi_resol](const double* x) -> double {
            const double mW = x[0];
            if (mW <= 0.0) return 1e10;
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
        int n_iter = 0; double grad_norm = 0.0;
        int status = _bfgs_minimize<decltype(chi2fn), KF_NDIM>(chi2fn, x0, fmin, n_iter, grad_norm);
        result.status = status;
        result.valid = (status == 0) ? 1 : 0;
        result.chi2  = fmin;
        result.winner_pass    = 1;
        result.n_passes_run   = n_iter;
        result.priors_swapped = 0;
        result.edm            = static_cast<float>(grad_norm);
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
