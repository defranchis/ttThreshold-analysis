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

        # 2) cluster the event into EXACTLY 4 jets with the exclusive ee-kt
        #    (Durham) algorithm. "Exclusive, N=4" forces the event into 4 jets
        #    (no jet-radius / no leftover particles) — the natural choice for a
        #    4-quark final state at an e+e- collider.
        helper = ExclusiveJetClusteringHelper("ReconstructedParticles", 4)
        df = helper.define(df)                  # adds the clustering result (the "_jet" object)
        # turn the 4 jets into TLorentzVectors and give each its own handle jet1..jet4
        df = df.Define("jets_p4", f"JetConstituentsUtils::compute_tlv_jets({helper.jets})")
        for i in (1, 2, 3, 4):
            df = df.Define(f"jet{i}", f"jets_p4[{i-1}]")
        df = df.Define("n_reco_jets", "(int)jets_p4.size()")
        df = df.Filter("n_reco_jets == 4", "exactly 4 reco jets")   # safety (very rare to fail)

        # 3) GENUINE-4-JET cut. d_45 is the Durham distance at which the event would
        #    resolve a 5th jet: large d_45 means a hard gluon was radiated, so the
        #    "4 jets" no longer correspond to the 4 quarks. Keep sqrt(d_45) < ~7 GeV.
        #    This is a standard ee 4-jet selection and needs NO truth -> works on data.
        df = df.Define("d_45", "JetClusteringUtils::get_exclusive_dmerge(_jet, 4)")
        if BW_SQRTD45 > 0:
            df = df.Filter(f"d_45 < {BW_SQRTD45 * BW_SQRTD45}",
                           f"genuine 4-jet: sqrt(d_45) < {BW_SQRTD45:g} GeV")

        # store the jet kinematics (so the analysis can be redone without re-clustering)
        for i in (1, 2, 3, 4):
            df = df.Define(f"reco_jet{i}_p",     f"jet{i}.P()")
            df = df.Define(f"reco_jet{i}_theta", f"jet{i}.Theta()")
            df = df.Define(f"reco_jet{i}_phi",   f"jet{i}.Phi()")
            df = df.Define(f"reco_jet{i}_mass",  f"jet{i}.M()")

        # 4) THE ACTUAL TOOL. bwPairing tries the 3 ways to split the 4 jets into
        #    2+2 and returns: the chosen split (.pairing), and for each split k its
        #    goodness-of-fit (.gof[k], low = W-like), posterior probability
        #    (.prob[k], the 3 sum to 1) and the two di-jet masses (.m_a/.m_b).
        #    See BWPairing.h for the maths.
        df = df.Define("bwpair", "FCCAnalyses::WWFunctions::bwPairing(jet1, jet2, jet3, jet4)")
        df = df.Define("bwpair_pairing",   "bwpair.pairing")      # 0/1/2  <- the answer
        df = df.Define("bwpair_gof_best",  "bwpair.gof_best")     # discriminant of the winner
        df = df.Define("bwpair_prob_best", "bwpair.prob_best")    # confidence of the winner
        df = df.Define("bwpair_dgof",      "bwpair.dgof")         # gap to the 2nd-best split
        for k in range(3):                                       # ...and the per-split details
            df = df.Define(f"bwpair_gof{k}",  f"bwpair.gof[{k}]")
            df = df.Define(f"bwpair_prob{k}", f"bwpair.prob[{k}]")
            df = df.Define(f"bwpair_ma{k}",   f"bwpair.m_a[{k}]")
            df = df.Define(f"bwpair_mb{k}",   f"bwpair.m_b[{k}]")

        # 5) GEN-TRUTH (WW signal only): work out the *true* pairing so we can
        #    measure how often bwPairing got it right.
        if _with_truth:
            # match the 4 reco jets to the 4 gen quarks (global min-total-dR), then
            # label each jet by which W its matched quark came from -> true pairing.
            df = df.Define("gen_quarks_tlv", "FCCAnalyses::MCParticle::get_tlv(gen_quarks)")
            for k in range(4):
                df = df.Define(f"gen_q{k}", f"gen_quarks_tlv[{k}]")
            df = df.Define("perm",            # perm[i] = gen-quark index matched to jet i
                "FCCAnalyses::WWFunctions::matchJets4(jet1,jet2,jet3,jet4,"
                "gen_q0,gen_q1,gen_q2,gen_q3)")
            for i in (1, 2, 3, 4):
                k = i - 1
                df = df.Define(f"gen_quark{i}", f"gen_quarks_tlv[perm[{k}]]")
                # how close jet i is to its quark (used to call an event "matched")
                df = df.Define(f"jet{i}_matched_q_dR", f"(double)jet{i}.DeltaR(gen_quark{i})")
                # W-label of jet i: quarks 0,1 -> boson A (0); quarks 2,3 -> boson B (1)
                df = df.Define(f"jet{i}_wlab", f"(int)(perm[{k}] >= 2)")
            # the true pairing index {0,1,2} from the 4 jet W-labels
            df = df.Define("gen_pairing_true",
                "FCCAnalyses::WWFunctions::pairing_index_from_groups("
                "jet1_wlab, jet2_wlab, jet3_wlab, jet4_wlab)")
            # did the BW tool pick the true pairing? (1/0)  <- the efficiency numerator
            df = df.Define("bwpair_correct",
                "(int)(gen_pairing_true >= 0 && bwpair.pairing == gen_pairing_true)")

        print("\n[cutflow]")
        df.Report().Print()
        print()
        return df

    def output():
        return branches
