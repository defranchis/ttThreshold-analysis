"""
Correlated double Breit-Wigner term for a WW hadronic kinematic fit.

Standalone, dependency-light (numpy only) port of the production term in
WWFunctions/WWKinReco.h (bw_phasespace_neg2ll + log_Z_bw_phasespace_ontf,
around lines 474-696 there). This file is meant to be read end to end -- see
examples/double_bw_pairing/README.md for the physics writeup, and
demo_pairing_and_normalization.py for a runnable demo on real WW->4q jets.

Physics
-------
A WW->4q (or WW->lnuqq) event gives two "W candidate" invariant masses
(m_h, m_l) built from a jet pairing. The -2 ln L contribution from the W
lineshape is NOT the sum of two independent single-W terms: both masses come
from the same parent system of invariant mass m_WW = sqrt(s_WW), and are tied
together by

  * the two-body-decay phase-space factor sqrt(lambda(s_WW, m_h^2, m_l^2))
    (Kallen function) -- a function of BOTH masses jointly, not separable
    into f(m_h) * g(m_l);
  * the kinematic triangle constraint m_h + m_l < m_WW, which caps how far
    apart the two masses' allowed ranges can be.

That coupling is the "correlation": the joint density is
BW(m_h) * BW(m_l) * sqrt(lambda) / s_WW, and its normalization is a genuine
2-D integral over the (m_h, m_l) triangle -- it does not factorize into the
product of two independent 1-D Breit-Wigner normalizations.
"""
import numpy as np

# Generator W pole/width for the winter2023 IDEA WW sample this repo uses
# (see memory project_sample_whizard_truth_winter2023.md). Swap for your own
# sample's values -- everything below is generic in (mW, gW).
MW_PDG = 80.419  # GeV
GW_PDG = 2.049   # GeV


def bw(m, mW, gW):
    """Constant-width (non-relativistic-normalization) Breit-Wigner density in m."""
    mwgw = mW * gW
    d = m * m - mW * mW
    return mwgw / (d * d + mwgw * mwgw)


def kallen(a, b, c):
    """Kallen (triangle) function: lambda(a,b,c) = a^2+b^2+c^2-2ab-2bc-2ca."""
    return a * a + b * b + c * c - 2 * a * b - 2 * b * c - 2 * c * a


def neg2ll_bw_phasespace(mh, ml, m_WW, mW, gW):
    """
    Un-normalized -2 ln[ BW(mh) * BW(ml) * sqrt(lambda(s_WW,mh^2,ml^2)) / s_WW ].

    Mirrors WWKinReco.h::bw_phasespace_neg2ll up to the +2*log_Z piece added
    by neg2ll_bw_phasespace_normalized() below. One deliberate difference:
    the forbidden region (lambda <= 0) gets a flat large penalty here rather
    than C++'s continuous |lambda| floor, since that floor is even in lambda
    and would let a badly-mispaired (lambda very negative) pairing tie a
    well-populated allowed one in the discrete argmin used by
    choose_pairing() below -- harmless in C++'s gradient-based fit, not here.

    Symmetric under mh<->ml swap, so the two dijet masses of a pairing can be
    passed in either order.

    This term alone is enough to RANK jet pairings within one event (see
    choose_pairing()): the missing normalization depends only on
    (mW, gW, m_WW), which is identical for every pairing of the same event,
    so it cancels in an argmin/argmax comparison and can be skipped.
    """
    s_WW = m_WW * m_WW
    bw_h = np.maximum(bw(mh, mW, gW), 1e-300)
    bw_l = np.maximum(bw(ml, mW, gW), 1e-300)
    lam = kallen(s_WW, mh * mh, ml * ml)  # Kallen function; > 0 inside the kinematic triangle
    term = (-2.0 * (np.log(bw_h) + np.log(bw_l))
            + 4.0 * np.log(np.pi)
            - np.log(np.maximum(lam, 1e-300)) + 2.0 * np.log(s_WW))
    return np.where(lam > 0, term, 1e6)  # kinematically forbidden (mh+ml>=m_WW) -> huge penalty


_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)  # matches KF_GL_N in WWKinReco.h


def log_z_bw_phasespace(m_WW, mW, gW):
    """
    log Z(mW, gW, m_WW): joint normalization of BW(mh)*BW(ml)*sqrt(lambda)/s_WW
    over the 2-D kinematic triangle {mh, ml > 0, mh + ml < m_WW}.

    Substituting t = atan((m^2 - mW^2) / (mW*gW)) turns each BW peak into a
    flat measure in t, so a plain Gauss-Legendre rule on t_h, t_l integrates
    it accurately with few nodes. This is a direct, readability-first port of
    WWKinReco.h::log_Z_bw_phasespace_ontf (the *_ontf on-the-fly evaluator;
    the production kinfit interpolates a precomputed table instead, for
    speed -- see log_Z_bw_phasespace in the same header).

    m_WW: array-like, one value per event. mW, gW: scalars.
    Returns: array of log Z, same shape as m_WW.
    """
    m_WW = np.atleast_1d(np.asarray(m_WW, dtype=float))
    mwgw, mW2 = mW * gW, mW * mW
    s_WW = m_WW * m_WW
    t_min = np.arctan(-mW2 / mwgw)
    t_max = np.arctan((s_WW - mW2) / mwgw)
    half_d = 0.5 * (t_max - t_min)
    half_s = 0.5 * (t_max + t_min)

    t = half_d[:, None] * _GL_X[None, :] + half_s[:, None]        # (nEvt, nGL)
    m = np.sqrt(np.maximum(mW2 + mwgw * np.tan(t), 1e-12))
    inv_m = 1.0 / m

    mh, ml = m[:, :, None], m[:, None, :]                          # (nEvt, nGL, 1) / (nEvt, 1, nGL)
    ih, il = inv_m[:, :, None], inv_m[:, None, :]
    s = s_WW[:, None, None]
    lam = kallen(s, mh * mh, ml * ml)
    integrand = np.where(lam > 0, np.sqrt(np.maximum(lam, 0.0)) * ih * il / (4.0 * s), 0.0)

    w2 = (_GL_W[:, None] * _GL_W[None, :])[None, :, :]
    Z = np.sum(w2 * integrand, axis=(1, 2)) * half_d * half_d
    return np.log(np.maximum(Z, 1e-300))


def neg2ll_bw_phasespace_normalized(mh, ml, m_WW, mW, gW):
    """
    Full, properly-normalized -2 ln L term: neg2ll_bw_phasespace + 2*log_Z.

    Use THIS one (not the un-normalized term above) whenever mW (or gW) is a
    FLOATED fit parameter -- e.g. in a kinfit chi2/likelihood that fits mW
    per event or in an ensemble, or in a scan over mW hypotheses. Without the
    2*log_Z(mW, gW, m_WW) piece, the term is not a genuine -2 ln L in mW: the
    unnormalized density's peak height itself changes with mW, which biases
    the fit toward whichever mW happens to make BW(mh)*BW(ml) largest rather
    than the mW that best describes the data. See demo_pairing_and_normalization.py
    for a numerical illustration of the size of this bias.
    """
    return neg2ll_bw_phasespace(mh, ml, m_WW, mW, gW) + 2.0 * log_z_bw_phasespace(m_WW, mW, gW)


def choose_pairing(mA, mB, m_WW, mW=MW_PDG, gW=GW_PDG):
    """
    Pick, per event, which jet pairing (column of mA/mB) minimizes the
    correlated double-BW*phase-space -2lnL term.

    mA, mB: (nEvt, nPairing) dijet masses of the two W candidates for each
        pairing hypothesis (e.g. nPairing=3 for WW->4q's 3 ways to split 4
        jets into 2 dijets).
    m_WW: (nEvt,) per-event parent invariant mass sqrt(s_WW) for the
        sqrt(lambda) term -- pairing-invariant (e.g. the invariant mass of
        the sum of all 4 jets), so compute it once per event, not per pairing.

    Normalization is deliberately NOT added here: log_Z(mW, gW, m_WW) does
    not depend on the pairing, so it cancels in the argmin (see
    neg2ll_bw_phasespace's docstring) and would only cost cycles.
    """
    n2ll = neg2ll_bw_phasespace(mA, mB, m_WW[:, None], mW, gW)
    return np.argmin(n2ll, axis=1)
