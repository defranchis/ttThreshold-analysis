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

// Post-BES, pre-ISR beam e± (depth=1 in the e± chain).
// In Winter2023 wzp6 samples the chain is:
//   depth 0 = nominal beams (no parents, m(ee)=ECM exactly, no BES)
//   depth 1 = e± with an e± parent that itself has no parents — BES applied
//             here (σ_BES ≈ 84 MeV per beam, m(ee) symmetric around ECM)
//   depth≥2 = post-ISR (energy-loss tail, m(ee) ≪ ECM with ISR-photon recoil)
// We pick depth=1 to isolate BES from ISR.
struct sel_beam_electrons {
    sel_beam_electrons() {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(2);
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) != 11) continue;
            if (p.parents_begin == p.parents_end) continue;  // skip depth 0
            int parent_e = -1;
            for (unsigned j = p.parents_begin; j < p.parents_end; ++j) {
                if (j >= parents_relation.size()) break;
                int idx = parents_relation[j];
                if (idx < 0 || idx >= (int)in.size()) continue;
                if (std::abs(in[idx].PDG) == 11) { parent_e = idx; break; }
            }
            if (parent_e < 0) continue;
            const auto& pa = in[parent_e];
            if (pa.parents_begin != pa.parents_end) continue;  // require depth-0 parent
            result.emplace_back(p);
        }
        return result;
    }
};

// Post-ISR beam e± (depth=2 in the e± chain) — the e± entering the hard
// process, after the ISR photons have been emitted. Identified as e± with an
// e± parent whose own first e± parent has no parents (= depth-0 beams).
// (gen_ee_p4_depth1 − gen_ee_p4_depth2) gives the total ISR 4-momentum.
struct sel_post_isr_electrons {
    sel_post_isr_electrons() {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        auto first_e_parent = [&](const edm4hep::MCParticleData& p) -> int {
            for (unsigned j = p.parents_begin; j < p.parents_end; ++j) {
                if (j >= parents_relation.size()) break;
                int idx = parents_relation[j];
                if (idx < 0 || idx >= (int)in.size()) continue;
                if (std::abs(in[idx].PDG) == 11) return idx;
            }
            return -1;
        };
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(2);
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) != 11) continue;
            int parent = first_e_parent(p);
            if (parent < 0) continue;
            const auto& pa = in[parent];
            if (pa.parents_begin == pa.parents_end) continue;       // parent is depth-0; we want depth-2
            int grandp = first_e_parent(pa);
            if (grandp < 0) continue;
            if (in[grandp].parents_begin != in[grandp].parents_end) continue;  // grandparent must be depth-0
            result.emplace_back(p);
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

inline TLorentzVector tlv_setmass(const TLorentzVector& p, double m) {
    double e = std::sqrt(p.P()*p.P() + m*m);
    return TLorentzVector(p.Px(), p.Py(), p.Pz(), e);
}

float deltaM(int nIsolep, int nRecoJets,
             const TLorentzVector& Wlep, const TLorentzVector& Whad) {
    if (nIsolep < 1 || nRecoJets < 2) return -1.0;
    TLorentzVector P_initial(0, 0, 0, ECM);
    return (P_initial - (Wlep + Whad)).M();
}

}}  // namespace FCCAnalyses::WWFunctions

#endif
