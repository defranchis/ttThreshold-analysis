#ifndef BWMatching_H
#define BWMatching_H

// ── Gen-truth ingredients for the BW jet→W pairing (standalone) ─────────────
// Self-contained copies of the only three truth helpers the pipeline needs:
//   sel_quarks_fromBoson : the 4 quarks of a VV→4q event, grouped by parent boson
//   matchJets4           : global 4-jet ↔ 4-quark assignment (min total ΔR)
//   pairing_index_from_groups : W-labels of the 4 jets → pairing index {0,1,2}
// These are used ONLY to define the truth so the pairing efficiency can be
// measured; the BW pairing itself (BWPairing.h) needs none of this.

#include <edm4hep/MCParticleData.h>
#include <ROOT/RVec.hxx>
#include <TLorentzVector.h>
#include <TVector2.h>
#include <map>
#include <array>
#include <algorithm>
#include <limits>
#include <cmath>

namespace FCCAnalyses { namespace WWFunctions {

// The 4 quarks of a VV→4q event, grouped by their parent boson.
//   boson_pdg = 24 → WW→4q,  23 → ZZ→4q.
// Returns size 4 ([0,1] = boson A, [2,3] = boson B) only for a clean
// 2-boson × 2-quark topology, else an empty vector (the caller filters size==4).
struct sel_quarks_fromBoson {
    int boson_pdg;
    explicit sel_quarks_fromBoson(int pdg = 24) : boson_pdg(pdg) {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        std::map<int, ROOT::VecOps::RVec<edm4hep::MCParticleData>> groups;
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) > 5 || p.PDG == 0) continue;
            int bparent = -1;
            for (unsigned j = p.parents_begin; j < p.parents_end; ++j) {
                if (j >= parents_relation.size()) break;
                int idx = parents_relation[j];
                if (idx < 0 || idx >= (int)in.size()) continue;
                if (std::abs(in[idx].PDG) == boson_pdg) { bparent = idx; break; }
            }
            if (bparent < 0) continue;
            groups[bparent].emplace_back(p);
        }
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        if (groups.size() != 2) return result;
        for (auto& kv : groups) {
            if (kv.second.size() != 2) { result.clear(); return result; }
            for (auto& q : kv.second) result.emplace_back(q);
        }
        return result;   // size 4, [0,1] = boson A, [2,3] = boson B
    }
};

// Global 4-jet ↔ 4-quark assignment. Returns a length-4 permutation: out[i] is
// the gen-quark index (0..3) matched to reco jet i, chosen to minimise the total
// ΔR over all 24 bijections.
template<typename V>
ROOT::VecOps::RVec<int>
matchJets4(const V& j0, const V& j1, const V& j2, const V& j3,
           const V& q0, const V& q1, const V& q2, const V& q3) {
    auto dR = [](const V& a, const V& b) {
        double deta = a.Eta() - b.Eta();
        double dphi = TVector2::Phi_mpi_pi(a.Phi() - b.Phi());
        return std::sqrt(deta * deta + dphi * dphi);
    };
    const V* jets[4] = {&j0, &j1, &j2, &j3};
    const V* qs[4]   = {&q0, &q1, &q2, &q3};
    double dRm[4][4];
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k) dRm[i][k] = dR(*jets[i], *qs[k]);
    std::array<int, 4> p = {0, 1, 2, 3};
    std::array<int, 4> best = p;
    double best_sum = std::numeric_limits<double>::max();
    do {
        double s = dRm[0][p[0]] + dRm[1][p[1]] + dRm[2][p[2]] + dRm[3][p[3]];
        if (s < best_sum) { best_sum = s; best = p; }
    } while (std::next_permutation(p.begin(), p.end()));
    return ROOT::VecOps::RVec<int>(best.begin(), best.end());
}

// W-labels of the 4 jets (0 = boson A, 1 = boson B) → pairing index {0,1,2},
// matching the bwPairing convention. -1 if the labels don't split 2-2.
inline int pairing_index_from_groups(int w0, int w1, int w2, int w3) {
    if (w0 == w1 && w2 == w3) return 0;
    if (w0 == w2 && w1 == w3) return 1;
    if (w0 == w3 && w1 == w2) return 2;
    return -1;
}

}}  // namespace FCCAnalyses::WWFunctions

#endif
