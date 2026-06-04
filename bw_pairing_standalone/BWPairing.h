#ifndef BWPairing_H
#define BWPairing_H

// ── Standalone BW jet→W pairing discriminant ────────────────────────────────
//
// Given 4 jets, decide which of the 3 partitions into 2 di-jets is most likely
// to be the two W's — using ONLY the Breit-Wigner compatibility of the two
// di-jet masses with the W resonance. No kinematic fit, no Minuit: it is pure
// arithmetic on the raw jet 4-vectors and runs in ~microseconds.
//
// This is the discriminating core of the full WW→4q kinematic fit (where the
// per-term study showed the BW term is the ONLY part that separates pairings)
// distilled into a self-contained tool, usable as (a) a fast pairing chooser and
// (b) a WW-vs-background (e.g. ZZ→4q) χ²-like discriminant under the W hypothesis.
//
// Outputs per call:
//   pairing  — most probable partition ∈ {0,1,2}
//   gof[k]   — "goodness of fit" of partition k: −2·log[BW(m_a)·BW(m_b)] referenced
//              to the W pole (both di-jets exactly on mW), so gof ≥ 0 and gof = 0
//              means both masses sit on the pole. Lower = more W-like.
//   prob[k]  — posterior probability that partition k is the correct one, under a
//              flat prior: prob[k] = L_k / Σ_j L_j with L_k = BW(m_a)·BW(m_b).
//              The three probabilities sum to 1 by construction.
//   m_a[k], m_b[k] — the two di-jet masses of partition k (a = first pair).
//   dgof     — gof(2nd best) − gof(best): the pairing separation.
//
// Pairing index convention matches kinFit4q_bestpairing / pairing_index_from_groups:
//   0: (j1 j2)(j3 j4)   1: (j1 j3)(j2 j4)   2: (j1 j4)(j2 j3)
//
// ── Why a BARE, pole-referenced BW (no normalization table, no phase space) ──
// The full WW→4q kinematic fit uses a phase-space-normalised BW term:
//     BW(m_a)·BW(m_b)·PS(m_a,m_b,M_WW) / Z(M_WW,mW,Γ),   PS = √λ/M_WW²  (Källén)
// For *choosing the pairing* both extra factors are unnecessary or harmful:
//   • Z (the normalization integral) depends only on M_WW = the invariant mass of
//     all 4 jets, which is the SAME for the 3 partitions. So Z is a common offset
//     → it cancels in argmin(gof) and in the prob softmax. No log-Z table needed
//     (the fit needs it only because M_WW floats there; here it is fixed/event).
//   • PS (the shared phase space of the two W's) IS pairing-dependent and is a
//     closed form (no table) — but at √s≈160 the true both-on-shell pairing has
//     m_a+m_b ≈ M_WW ≈ 2mW, i.e. it sits at the λ→0 threshold where PS is tiny, so
//     including PS PENALISES the correct assignment. Measured: adding PS drops the
//     correct-pairing fraction 91.6%→89.2% (matched events). So PS is omitted.
// The bare −2·log[BW(m_a)·BW(m_b)] is therefore both the simplest AND the best
// pairing discriminant here.
//
// (Consequence: prob[k] is the pure W-lineshape posterior. It is NOT a calibrated
// P(correct) — the natural width Γ≈2 GeV is narrower than the ~few-GeV di-jet mass
// resolution, so the probabilities are over-confident. Resolution is deliberately
// NOT folded into Γ; a calibrated number would need a Voigtian or the full fit.)

#include <TLorentzVector.h>
#include <cmath>
#include <limits>

namespace FCCAnalyses { namespace WWFunctions {

// PDG-ish defaults (kept independent of the Minuit-pulling WWKinReco.h so this
// header stays dependency-light). Override per call if desired.
static constexpr double BWPAIR_MW    = 80.385;
static constexpr double BWPAIR_GAMMA = 2.085;

struct BWPairingResult {
    int   pairing;        // most probable partition ∈ {0,1,2}
    float gof[3];         // pole-referenced −2 log[BW_a·BW_b] (≥ 0)
    float prob[3];        // posterior over partitions, Σ = 1
    float m_a[3], m_b[3]; // di-jet masses (a = first pair, b = second pair)
    float gof_best;       // gof[pairing]
    float prob_best;      // prob[pairing]
    float dgof;           // gof(2nd best) − gof(best)
};

// Relativistic Breit-Wigner shape value at mass m (peak = 1/(mW·Γ) at m = mW).
// Relativistic Breit-Wigner shape (unnormalised), peaks at m = mW.
inline double _bwpair_val(double m, double mW, double Gamma) {
    const double mwg = mW * Gamma;
    const double d   = m * m - mW * mW;             // 0 when m is exactly on the pole
    return mwg / (d * d + mwg * mwg);
}

inline BWPairingResult bwPairing(const TLorentzVector& j1, const TLorentzVector& j2,
                                 const TLorentzVector& j3, const TLorentzVector& j4,
                                 double mW = BWPAIR_MW, double Gamma = BWPAIR_GAMMA) {
    // The 3 ways to split {j1,j2,j3,j4} into two di-jets (Wa = first pair, Wb = second):
    //   k=0:(j1 j2)(j3 j4)   k=1:(j1 j3)(j2 j4)   k=2:(j1 j4)(j2 j3)
    static const int order[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
    const TLorentzVector* J[4] = {&j1, &j2, &j3, &j4};

    BWPairingResult R{};
    const double mwg       = mW * Gamma;
    const double pole_ref  = 4.0 * std::log(mwg);   // gof value when both di-jets are on the pole

    // --- score each of the 3 splits ---
    double L[3], gof[3];
    for (int k = 0; k < 3; ++k) {
        const TLorentzVector Wa = *J[order[k][0]] + *J[order[k][1]];   // the two W candidates
        const TLorentzVector Wb = *J[order[k][2]] + *J[order[k][3]];
        const double ma = Wa.M(), mb = Wb.M();                        // their di-jet masses
        R.m_a[k] = static_cast<float>(ma);
        R.m_b[k] = static_cast<float>(mb);
        const double bwa = _bwpair_val(ma, mW, Gamma);                // BW compatibility of each
        const double bwb = _bwpair_val(mb, mW, Gamma);
        // goodness-of-fit: -2 log(BW_a * BW_b), offset so gof=0 means both on the pole, gof>=0.
        gof[k]   = -2.0 * (std::log(bwa) + std::log(bwb)) - pole_ref;
        L[k]     = bwa * bwb;                                         // un-normalised likelihood
        R.gof[k] = static_cast<float>(gof[k]);
    }

    // --- turn the 3 scores into probabilities (they sum to 1) ---
    // prob[k] = L_k / sum_j L_j = softmax(-gof/2). Done in the numerically-stable
    // softmax form (subtract the smallest gof first so exp() never overflows).
    double gmin = gof[0];
    for (int k = 1; k < 3; ++k) gmin = std::min(gmin, gof[k]);
    double w[3], wsum = 0.0;
    for (int k = 0; k < 3; ++k) { w[k] = std::exp(-0.5 * (gof[k] - gmin)); wsum += w[k]; }
    for (int k = 0; k < 3; ++k) R.prob[k] = static_cast<float>(w[k] / wsum);

    // --- pick the winner (lowest gof = highest prob) and the gap to 2nd best ---
    int best = 0;
    for (int k = 1; k < 3; ++k) if (gof[k] < gof[best]) best = k;
    double second = std::numeric_limits<double>::infinity();
    for (int k = 0; k < 3; ++k) if (k != best) second = std::min(second, gof[k]);

    R.pairing   = best;
    R.gof_best  = static_cast<float>(gof[best]);
    R.prob_best = R.prob[best];
    R.dgof      = static_cast<float>(second - gof[best]);   // separation: 2nd-best gof − best gof
    return R;
}

}}  // namespace FCCAnalyses::WWFunctions

#endif
