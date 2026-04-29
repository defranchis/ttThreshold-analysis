#ifndef WWFunctions_H
#define WWFunctions_H

#include <cmath>
#include "TMath.h"
#include "TVector2.h"
#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"
#include "Math/Vector4D.h"
#include "edm4hep/MCParticleData.h"

namespace FCCAnalyses { namespace WWFunctions {

inline float ECM = 160.0f;

// ── selectors keeping particles whose mother is e± (proxy for "from W decay"
//    in samples where the W is not stored in the MC history). Walk the proper
//    parents_begin/parents_end relation range — robust against the flat
//    Particle0[i] indexing assumption used in FCCAnalyses::MCParticle::sel_*.

namespace _selectors_detail {
inline bool _has_electron_parent(const edm4hep::MCParticleData& p,
                                  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& in,
                                  const ROOT::VecOps::RVec<int>& parents_relation) {
    for (unsigned j = p.parents_begin; j < p.parents_end; ++j) {
        if (j >= parents_relation.size()) break;
        int parent_idx = parents_relation[j];
        if (parent_idx < 0 || parent_idx >= (int)in.size()) continue;
        if (std::abs(in[parent_idx].PDG) == 11) return true;
    }
    return false;
}
}

struct sel_genleps_fromele {
    int m_pdg;
    sel_genleps_fromele(int pdg) : m_pdg(pdg) {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) != m_pdg) continue;
            if (_selectors_detail::_has_electron_parent(p, in, parents_relation)) result.emplace_back(p);
        }
        return result;
    }
};

// Light quarks (|PDG|<=5) with e± parent — like FCCAnalyses::MCParticle::sel_lightQuarks_fromele
// but using the robust parents-relation walk.
struct sel_lightQuarks_fromele {
    sel_lightQuarks_fromele() {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) > 5 || std::abs(p.PDG) == 0) continue;
            if (_selectors_detail::_has_electron_parent(p, in, parents_relation)) result.emplace_back(p);
        }
        return result;
    }
};

// ── jet matching ───────────────────────────────────────────────────────────

template<typename V>
std::pair<V, V>
matchJets2(const V& r1, const V& r2, const V& g1, const V& g2) {
    auto dR = [](const V& a, const V& b) {
        double deta = a.Eta() - b.Eta();
        double dphi = TVector2::Phi_mpi_pi(a.Phi() - b.Phi());
        return sqrt(deta*deta + dphi*dphi);
    };
    double dR_A = dR(r1, g1) + dR(r2, g2);
    double dR_B = dR(r1, g2) + dR(r2, g1);
    return (dR_A < dR_B) ? std::make_pair(g1, g2) : std::make_pair(g2, g1);
}

// ── 4-vector sums ─────────────────────────────────────────────────────────

TLorentzVector sum_p4(const ROOT::VecOps::RVec<TLorentzVector>& ps) {
    TLorentzVector total;
    for (const auto& p : ps) total += p;
    return total;
}

TLorentzVector sum_p4(std::initializer_list<TLorentzVector> ps) {
    TLorentzVector total;
    for (const auto& p : ps) total += p;
    return total;
}

float deltaM(int nIsolep, int nRecoJets,
             const TLorentzVector& Wlep, const TLorentzVector& Whad) {
    if (nIsolep < 1 || nRecoJets < 2) return -1.0;
    TLorentzVector P_initial(0, 0, 0, ECM);
    return (P_initial - (Wlep + Whad)).M();
}

}}  // namespace FCCAnalyses::WWFunctions

#endif
