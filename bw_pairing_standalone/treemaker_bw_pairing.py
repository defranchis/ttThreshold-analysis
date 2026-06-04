# =============================================================================
#  Standalone single-step pipeline for the Breit-Wigner jet -> W pairing.
# =============================================================================
#  For each event it:
#    1. clusters the reconstructed particles into exactly 4 jets (Durham/ee-kt),
#    2. keeps only genuine 4-jet events  (sqrt(d_45) < BW_SQRTD45_MAX),
#    3. runs the BW pairing (BWPairing.h): which 2+2 split of the 4 jets best
#       matches two W's -> chosen pairing, per-pairing gof and probability,
#    4. for the WW signal, adds the gen-truth jet->quark matching so the
#       pairing efficiency can be measured downstream.
#
#  This is the ONLY processing step. Output is a flat TTree ('events').
#
#  Run it (from a key4hep / FCCAnalyses environment):
#    # WW signal (with gen-truth pairing):
#    BW_BOSON=W BW_SAMPLE=p8_ee_WW_ecm160 \
#        fccanalysis run bw_pairing_standalone/treemaker_bw_pairing.py
#    # ZZ control (WW hypothesis applied to ZZ->4q, no truth):
#    BW_BOSON=Z BW_SAMPLE=p8_ee_ZZ_ecm160 \
#        fccanalysis run bw_pairing_standalone/treemaker_bw_pairing.py
#
#  Environment knobs (all optional):
#    BW_SAMPLE        dataset name        (default p8_ee_WW_ecm160)
#    BW_BOSON         "W" or "Z"          (default W) -> gen-4q filter + truth
#    BW_SQRTD45_MAX   genuine-4-jet cut   (default 7.0 GeV; 0 disables)
#    BW_FRACTION      fraction of sample  (default 0.001)
#    BW_OUTDIR        output directory    (default outputs/bw_pairing/<BOSON>)
# =============================================================================
import os
from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper

BW_SAMPLE  = os.environ.get("BW_SAMPLE", "p8_ee_WW_ecm160")
BW_BOSON   = os.environ.get("BW_BOSON", "W").strip().upper()
BW_SQRTD45 = float(os.environ.get("BW_SQRTD45_MAX", "7.0"))
_frac      = float(os.environ.get("BW_FRACTION", "0.001"))

assert BW_BOSON in ("W", "Z"), "BW_BOSON must be 'W' (WW->4q) or 'Z' (ZZ->4q)"
_boson_pdg  = {"W": 24, "Z": 23}[BW_BOSON]
_with_truth = (BW_BOSON == "W")    # only WW gets the pairing gen-truth

# --- FCCAnalyses driver configuration ---------------------------------------
processList  = {BW_SAMPLE: {"fraction": _frac, "crossSection": 1}}
prodTag      = "FCCee/winter2023/IDEA/"
outputDir    = os.environ.get("BW_OUTDIR", "outputs/bw_pairing/" + BW_BOSON)
# includePaths are resolved relative to THIS script's directory.
includePaths = ["JetQuarkMatching.h", "BWPairing.h"]

# --- output branches ---------------------------------------------------------
branches = ["d_45"]
for i in (1, 2, 3, 4):
    branches += [f"reco_jet{i}_p", f"reco_jet{i}_theta", f"reco_jet{i}_phi", f"reco_jet{i}_mass"]
branches += ["bwpair_pairing", "bwpair_gof_best", "bwpair_prob_best", "bwpair_dgof",
             "bwpair_gof0", "bwpair_gof1", "bwpair_gof2",
             "bwpair_prob0", "bwpair_prob1", "bwpair_prob2",
             "bwpair_ma0", "bwpair_mb0", "bwpair_ma1", "bwpair_mb1", "bwpair_ma2", "bwpair_mb2"]
if _with_truth:
    branches += ["gen_pairing_true", "bwpair_correct",
                 "jet1_matched_q_dR", "jet2_matched_q_dR", "jet3_matched_q_dR", "jet4_matched_q_dR"]


class RDFanalysis:
    def analysers(df):
        print(f"[bw_pairing] sample={BW_SAMPLE} boson={BW_BOSON} sqrt(d45)<{BW_SQRTD45}")

        # 1) SIGNAL DEFINITION FIRST: gen-level VV->4q (4 quarks from the 2 bosons).
        #    Applying it up front means every reco cut below is an efficiency
        #    measured ON the signal (so "4 reco jets" is ~100% for real 4q events).
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Define("gen_quarks",
            f"FCCAnalyses::WWFunctions::sel_quarks_fromBoson({_boson_pdg})(Particle, Particle0)")
        df = df.Filter("gen_quarks.size() == 4", f"gen: 4 quarks from 2 {BW_BOSON} (signal)")

        # 2) cluster into exactly 4 jets (exclusive ee-kt / Durham)
        helper = ExclusiveJetClusteringHelper("ReconstructedParticles", 4)
        df = helper.define(df)
        df = df.Define("jets_p4", f"JetConstituentsUtils::compute_tlv_jets({helper.jets})")
        for i in (1, 2, 3, 4):
            df = df.Define(f"jet{i}", f"jets_p4[{i-1}]")
        df = df.Define("n_reco_jets", "(int)jets_p4.size()")
        df = df.Filter("n_reco_jets == 4", "exactly 4 reco jets")

        # 3) genuine-4-jet cut: reject hard 5th-jet / radiative events
        df = df.Define("d_45", "JetClusteringUtils::get_exclusive_dmerge(_jet, 4)")
        if BW_SQRTD45 > 0:
            df = df.Filter(f"d_45 < {BW_SQRTD45 * BW_SQRTD45}",
                           f"genuine 4-jet: sqrt(d_45) < {BW_SQRTD45:g} GeV")

        for i in (1, 2, 3, 4):
            df = df.Define(f"reco_jet{i}_p",     f"jet{i}.P()")
            df = df.Define(f"reco_jet{i}_theta", f"jet{i}.Theta()")
            df = df.Define(f"reco_jet{i}_phi",   f"jet{i}.Phi()")
            df = df.Define(f"reco_jet{i}_mass",  f"jet{i}.M()")

        # 4) BW pairing (the actual tool) -> pairing / gof / probabilities
        df = df.Define("bwpair", "FCCAnalyses::WWFunctions::bwPairing(jet1, jet2, jet3, jet4)")
        df = df.Define("bwpair_pairing",   "bwpair.pairing")
        df = df.Define("bwpair_gof_best",  "bwpair.gof_best")
        df = df.Define("bwpair_prob_best", "bwpair.prob_best")
        df = df.Define("bwpair_dgof",      "bwpair.dgof")
        for k in range(3):
            df = df.Define(f"bwpair_gof{k}",  f"bwpair.gof[{k}]")
            df = df.Define(f"bwpair_prob{k}", f"bwpair.prob[{k}]")
            df = df.Define(f"bwpair_ma{k}",   f"bwpair.m_a[{k}]")
            df = df.Define(f"bwpair_mb{k}",   f"bwpair.m_b[{k}]")

        # 5) gen-truth jet->quark matching (WW signal only) -> pairing efficiency
        if _with_truth:
            df = df.Define("gen_quarks_tlv", "FCCAnalyses::MCParticle::get_tlv(gen_quarks)")
            for k in range(4):
                df = df.Define(f"gen_q{k}", f"gen_quarks_tlv[{k}]")
            df = df.Define("perm",
                "FCCAnalyses::WWFunctions::matchJets4(jet1,jet2,jet3,jet4,"
                "gen_q0,gen_q1,gen_q2,gen_q3)")
            for i in (1, 2, 3, 4):
                k = i - 1
                df = df.Define(f"gen_quark{i}", f"gen_quarks_tlv[perm[{k}]]")
                df = df.Define(f"jet{i}_matched_q_dR", f"(double)jet{i}.DeltaR(gen_quark{i})")
                df = df.Define(f"jet{i}_wlab", f"(int)(perm[{k}] >= 2)")
            df = df.Define("gen_pairing_true",
                "FCCAnalyses::WWFunctions::pairing_index_from_groups("
                "jet1_wlab, jet2_wlab, jet3_wlab, jet4_wlab)")
            df = df.Define("bwpair_correct",
                "(int)(gen_pairing_true >= 0 && bwpair.pairing == gen_pairing_true)")

        print("\n[cutflow]")
        df.Report().Print()
        print()
        return df

    def output():
        return branches
