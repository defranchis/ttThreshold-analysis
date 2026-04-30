# Step 2 of 2 — runs kinematic fit using DCB params from fit_dcb_resolutions.py.
# Requires response/functions/dcb_params_ecm<N>.h headers to be present before compiling.
import os, re, ROOT
import urllib
processList = {
    "wzp6_ee_munumuqq_noCut_ecm160": {
        "fraction": 1,
        "crossSection": 1,
    },
    "wzp6_ee_munumuqq_noCut_ecm157":{
        "fraction": 1,
        "crossSection": 1,
    },
    "wzp6_ee_munumuqq_noCut_ecm163":{
        "fraction": 1,
        "crossSection": 1,
    },
}

available_ecm = ['157', '160', '163'] #for a redundant check

def _parse_ecm(name):
    m = re.search(r'_ecm(\d+)', name)
    if not m:
        raise ValueError(f"Cannot parse ecm from sample name: {name}")
    return int(m.group(1))



# ── kinematic fit method ───────────────────────────────────────────────────
# "minuit" → ROOT Minuit2 (robust, ~200-500 function evaluations per event)
# "bfgs"   → custom BFGS, stack-only, no heap, template-inlined chi2
#             (~50-150 evaluations, thread-safe without thread_local)
KIN_FIT_METHOD  = "minuit"
# True → fit gW as a free parameter (12-dim); False → fix gW = KF_GW_FIXED (11-dim)
KIN_FIT_FREE_GW = False

channel = "CHANNELNAMEHERE"

if  channel not in ["lep","semihad","had"]:
    channel="semihad"
print(channel)

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
prodTag     = "FCCee/winter2023/IDEA/"

#Optional: output directory, default is local running directoryp
outputDir   = "outputs/treemaker/lnuqq/step2/{}".format(channel)


# additional/costom C++ functions, defined in header files (optional)
includePaths = ["examples/functions.h", "WWFunctions/WWFunctions.h", "WWFunctions/WWKinReco.h"]

## latest particle transformer model, trained on 9M jets in winter2023 samples
model_name = "fccee_flavtagging_edm4hep_wc" #"fccee_flavtagging_edm4hep_wc_v1"

## model files locally stored on /eos
eos_dir ="/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"
model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)

url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

## get local file, else download from url
def get_file_path(url, filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)

weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
)

jetFlavourHelper = None
jetClusteringHelper = None

all_branches = [
    # ── constituent kinematics: jet1 / jet2 / lep / nu ─────────────────
    # reco
    "reco_jet1_p",  "reco_jet1_pt", "reco_jet1_theta", "reco_jet1_phi", "reco_jet1_eta", "reco_jet1_costheta", "reco_jet1_mass",
    "reco_jet2_p",  "reco_jet2_pt", "reco_jet2_theta", "reco_jet2_phi", "reco_jet2_eta", "reco_jet2_costheta", "reco_jet2_mass",
    "reco_lep_p",   "reco_lep_pt",  "reco_lep_theta",  "reco_lep_phi",  "reco_lep_eta",  "reco_lep_costheta",
    "reco_met_p",    "reco_met_pt",   "reco_met_theta",   "reco_met_phi",   "reco_met_eta",   "reco_met_costheta",
    # gen
    "gen_quark1_p", "gen_quark1_pt","gen_quark1_theta","gen_quark1_phi","gen_quark1_eta","gen_quark1_costheta",
    "gen_quark2_p", "gen_quark2_pt","gen_quark2_theta","gen_quark2_phi","gen_quark2_eta","gen_quark2_costheta",
    "gen_lep_p",    "gen_lep_pt",   "gen_lep_theta",   "gen_lep_phi",   "gen_lep_eta",   "gen_lep_costheta",
    "gen_nu_p",     "gen_nu_pt",    "gen_nu_theta",    "gen_nu_phi",    "gen_nu_eta",    "gen_nu_costheta",
    # kinfit (no eta / costheta / mass)
    "kinfit_jet1_p", "kinfit_jet1_pt", "kinfit_jet1_theta", "kinfit_jet1_phi",
    "kinfit_jet2_p", "kinfit_jet2_pt", "kinfit_jet2_theta", "kinfit_jet2_phi",
    "kinfit_lep_p",  "kinfit_lep_pt",  "kinfit_lep_theta",  "kinfit_lep_phi",
    "kinfit_nu_p",   "kinfit_nu_pt",   "kinfit_nu_theta",   "kinfit_nu_phi",

    # ── W kinematics: Wlep / Whad ──────────────────────────────────────
    "reco_Wlep_m",   "reco_Wlep_p",   "reco_Wlep_pt",  "reco_Wlep_px",  "reco_Wlep_py",  "reco_Wlep_pz",  "reco_Wlep_costheta",  "reco_Wlep_phi",
    "gen_Wlep_m",    "gen_Wlep_p",    "gen_Wlep_pt",   "gen_Wlep_px",   "gen_Wlep_py",   "gen_Wlep_pz",   "gen_Wlep_costheta",   "gen_Wlep_phi",
    "kinfit_Wlep_m", "kinfit_Wlep_p", "kinfit_Wlep_pt","kinfit_Wlep_px","kinfit_Wlep_py","kinfit_Wlep_pz",
    "reco_Whad_m",   "reco_Whad_p",   "reco_Whad_pt",  "reco_Whad_px",  "reco_Whad_py",  "reco_Whad_pz",  "reco_Whad_costheta",  "reco_Whad_phi",
    "gen_Whad_m",    "gen_Whad_p",    "gen_Whad_pt",   "gen_Whad_px",   "gen_Whad_py",   "gen_Whad_pz",   "gen_Whad_costheta",   "gen_Whad_phi",
    "kinfit_Whad_m", "kinfit_Whad_p", "kinfit_Whad_pt","kinfit_Whad_px","kinfit_Whad_py","kinfit_Whad_pz",

    # ── WW system ──────────────────────────────────────────────────────
    "reco_WW_m",   "reco_WW_m_minus_ecm",   "reco_WW_px",   "reco_WW_py",   "reco_WW_pz",   "reco_WW_p_imbalance_tot",
    "gen_WW_m",    "gen_WW_m_minus_ecm",    "gen_WW_px",    "gen_WW_py",    "gen_WW_pz",    "gen_WW_p_imbalance_tot",
    "kinfit_WW_m", "kinfit_WW_m_minus_ecm", "kinfit_WW_px", "kinfit_WW_py", "kinfit_WW_pz", "kinfit_WW_p_imbalance_tot",

    # ── resolutions / responses (cross-level by construction) ──────────
    "jet1_p_resp", "jet1_theta_resol", "jet1_phi_resol", "jet1_eta_resol", "jet1_costheta_resol",
    "jet2_p_resp", "jet2_theta_resol", "jet2_phi_resol", "jet2_eta_resol", "jet2_costheta_resol",
    "lep_p_resp",  "lep_theta_resol",  "lep_phi_resol",  "lep_eta_resol",  "lep_costheta_resol",
    "met_p_resp",   "met_theta_resol",   "met_phi_resol",   "met_eta_resol",   "met_costheta_resol",
    "Wlep_m_resol", "Wlep_p_resol",
    "Whad_m_resol", "Whad_p_resol",
    "WW_m_resol", "WW_px_resol", "WW_py_resol", "WW_pz_resol",

    # ── kinfit-internal: parameters and fit quality ────────────────────
    "kinfit_mW", "kinfit_gW",
    "kinfit_s1", "kinfit_s2", "kinfit_sl", "kinfit_sn",
    "kinfit_t1", "kinfit_t2", "kinfit_tn", "kinfit_tl",
    "kinfit_p1", "kinfit_p2", "kinfit_pn", "kinfit_pl",
    "kinfit_chi2", "kinfit_chi2_ndof", "kinfit_valid",

    # ── misc derived ───────────────────────────────────────────────────
    "n_lep_reco", "n_reco_jets", "deltaM",
    "d_12", "d_32",
]

_dataset_iter = iter(processList.keys())

class RDFanalysis:

    def analysers(df):

        _dataset = next(_dataset_iter)
        _ecm = _parse_ecm(_dataset)
        print(f"[treemaker] dataset={_dataset}  ecm={_ecm}")
        if str(_ecm) not in available_ecm:
            raise ValueError(f"ecm={_ecm} parsed from '{_dataset}' not in available_ecm={available_ecm}")
        ROOT.gInterpreter.ProcessLine(f"FCCAnalyses::WWFunctions::setKinFitParams({_ecm});")
        ROOT.gInterpreter.ProcessLine('std::cout << "[DEBUG step2] ECM from WWFunctions = " << FCCAnalyses::WWFunctions::ECM << std::endl;')

        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Electron0","Electron#0.index")

        df = df.Define(
            "muons_all",
            "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)",
        )
        df = df.Define(
            "electrons_all",
            "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)",
        )

        df = df.Define(
            "muons_sel",
            "FCCAnalyses::ReconstructedParticle::sel_p(12)(muons_all)",
        )

        df = df.Define(
            "electrons_sel",
            "FCCAnalyses::ReconstructedParticle::sel_p(12)(electrons_all)",
        )

        df = df.Define(
            "muons_iso",
            "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(muons_sel, ReconstructedParticles)",
        )
        df = df.Define(
            "electrons_iso",
            "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(electrons_sel, ReconstructedParticles)",
        )

        df = df.Define(
            "muons_sel_iso",
            "FCCAnalyses::ZHfunctions::sel_iso(0.7)(muons_sel, muons_iso)",
        )

        df = df.Define(
            "electrons_sel_iso",
            "FCCAnalyses::ZHfunctions::sel_iso(0.7)(electrons_sel, electrons_iso)",
        )

        if channel == "had":
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 0")
        elif  channel == "semihad":
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 1")
        else:
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 2")

            df = df.Define(
                "muons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons_all)"
            )
            df = df.Define(
                "muons_theta",
                "FCCAnalyses::ReconstructedParticle::get_theta(muons_all)",
            )
            df = df.Define(
                "muons_phi",
                "FCCAnalyses::ReconstructedParticle::get_phi(muons_all)",
            )
            df = df.Define(
                "muons_q",
                "FCCAnalyses::ReconstructedParticle::get_charge(muons_all)",
            )
            df = df.Define(
                "muons_n", "FCCAnalyses::ReconstructedParticle::get_n(muons_all)",
            )
            df = df.Define(
                "Isomuons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons_sel_iso)"
            )
            df = df.Define(
                "Isomuons_theta",
                "FCCAnalyses::ReconstructedParticle::get_theta(muons_sel_iso)",
            )
            df = df.Define(
                "Isomuons_phi",
                "FCCAnalyses::ReconstructedParticle::get_phi(muons_sel_iso)",
            )
            df = df.Define(
                "Isomuons_q",
                "FCCAnalyses::ReconstructedParticle::get_charge(muons_sel_iso)",
            )
            df = df.Define(
                "Isomuons_n", "FCCAnalyses::ReconstructedParticle::get_n(muons_sel_iso)",
            )
            df = df.Define(
                "Isoelectrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons_sel_iso)"
            )
            df = df.Define(
                "Isoelectrons_theta",
                "FCCAnalyses::ReconstructedParticle::get_theta(electrons_sel_iso)",
            )
            df = df.Define(
                "Isoelectrons_phi",
                "FCCAnalyses::ReconstructedParticle::get_phi(electrons_sel_iso)",
                )
            df = df.Define(
                "Isoelectrons_q",
                "FCCAnalyses::ReconstructedParticle::get_charge(electrons_sel_iso)",
                )
            df = df.Define(
                "Isoelectrons_n", "FCCAnalyses::ReconstructedParticle::get_n(electrons_sel_iso)",
            )

        df = df.Define("Isoleps", "ROOT::VecOps::Concatenate(muons_sel_iso, electrons_sel_iso)")

        df = df.Define(
            "ReconstructedParticlesNoMuons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles,muons_sel_iso)",
        )
        df = df.Define(
            "ReconstructedParticlesNoMuNoEl",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoMuons,electrons_sel_iso)",
        )

        global jetClusteringHelper
        global jetFlavourHelper

        collections = {
            "GenParticles": "Particle",
            "PFParticles": "ReconstructedParticles",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "EFlowTrack_1",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "dNdx": "EFlowTrack_2",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }

        nJets = 2 if  channel == "semihad" else 4

        collections_noleps = dict(collections)
        collections_noleps["PFParticles"] = "ReconstructedParticlesNoMuNoEl"

        jetClusteringHelper = ExclusiveJetClusteringHelper(
            collections_noleps["PFParticles"], nJets
        )

        df = jetClusteringHelper.define(df)

        jetFlavourHelper = JetFlavourHelper(
            collections_noleps,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )
        df = jetFlavourHelper.define(df)
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)


        df = df.Define("Isoleps_p4_reco", "FCCAnalyses::ReconstructedParticle::get_tlv(Isoleps, 0)")
        df = df.Define("reco_lep_p",     "Isoleps_p4_reco.P()")
        df = df.Define("reco_lep_pt",    "Isoleps_p4_reco.Pt()")
        df = df.Define("reco_lep_eta",   "Isoleps_p4_reco.Eta()")
        df = df.Define("reco_lep_theta", "Isoleps_p4_reco.Theta()")
        df = df.Define("reco_lep_phi",   "Isoleps_p4_reco.Phi()")
        df = df.Define("n_lep_reco",      "(int)Isoleps.size()")

        df = df.Define("missing_p_p4",    "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET, 0)")
        df = df.Define("reco_met_p",       "missing_p_p4.P()")
        df = df.Define("reco_met_pt",      "missing_p_p4.Pt()")
        df = df.Define("reco_met_theta", "missing_p_p4.Theta()")
        df = df.Define("reco_met_phi",   "missing_p_p4.Phi()")
        df = df.Define("reco_met_eta",   "missing_p_p4.Eta()")
        df = df.Define("Wlep_reco",
            "FCCAnalyses::WWFunctions::sum_p4({Isoleps_p4_reco, missing_p_p4})")

        df = df.Define("reco_Wlep_m", "Wlep_reco.M()");
        df = df.Define("reco_Wlep_p", "Wlep_reco.P()");

        df = df.Define(
            "jets_p4",
            "JetConstituentsUtils::compute_tlv_jets({})".format(
                jetClusteringHelper.jets
            ),
        )

        df = df.Alias("Particle0", "Particle#0.index")

        # ── gen-level W-daughter selectors (parent==e±, since W is not in MC history) ─
        df = df.Define("gen_leps_fromele",        "FCCAnalyses::WWFunctions::sel_genleps_fromele(13)(Particle, Particle0)")
        df = df.Define("gen_neutrinos_fromele",   "FCCAnalyses::WWFunctions::sel_genleps_fromele(14)(Particle, Particle0)")
        df = df.Define("gen_lightquarks_fromele", "FCCAnalyses::WWFunctions::sel_lightQuarks_fromele()(Particle, Particle0)")

        df = df.Filter("gen_leps_fromele.size() == 1 && gen_neutrinos_fromele.size() == 1 && gen_lightquarks_fromele.size() == 2")

        # ── fromele kinematics ────────────────────────────────────────────────
        df = df.Define("gen_leps_fromele_tlv",        "FCCAnalyses::MCParticle::get_tlv(gen_leps_fromele)")
        df = df.Define("gen_neutrinos_fromele_tlv",   "FCCAnalyses::MCParticle::get_tlv(gen_neutrinos_fromele)")
        df = df.Define("gen_lightquarks_fromele_tlv", "FCCAnalyses::MCParticle::get_tlv(gen_lightquarks_fromele)")

        # ── gen-level per-particle TLVs ───────────────────────────────────────
        df = df.Define("lep_p4_gen", "gen_leps_fromele_tlv[0]")
        df = df.Define("nu_p4_gen",  "gen_neutrinos_fromele_tlv[0]")
        df = df.Define("gen_q1_p4",  "gen_lightquarks_fromele_tlv[0]")
        df = df.Define("gen_q2_p4",  "gen_lightquarks_fromele_tlv[1]")

        # ── gen-level lepton / neutrino scalars ───────────────────────────────
        df = df.Define("gen_lep_p",        "lep_p4_gen.P()")
        df = df.Define("gen_lep_pt",       "lep_p4_gen.Pt()")
        df = df.Define("gen_lep_theta",    "lep_p4_gen.Theta()")
        df = df.Define("gen_lep_phi",      "lep_p4_gen.Phi()")
        df = df.Define("gen_lep_eta",      "lep_p4_gen.Eta()")
        df = df.Define("gen_lep_costheta", "lep_p4_gen.CosTheta()")
        df = df.Define("gen_nu_p",         "nu_p4_gen.P()")
        df = df.Define("gen_nu_pt",        "nu_p4_gen.Pt()")
        df = df.Define("gen_nu_theta",     "nu_p4_gen.Theta()")
        df = df.Define("gen_nu_phi",       "nu_p4_gen.Phi()")
        df = df.Define("gen_nu_eta",       "nu_p4_gen.Eta()")
        df = df.Define("gen_nu_costheta",  "nu_p4_gen.CosTheta()")
        df = df.Define("reco_lep_costheta","Isoleps_p4_reco.CosTheta()")
        df = df.Define("reco_met_costheta", "missing_p_p4.CosTheta()")

        # ── lep / nu reco-vs-gen resolutions and responses ────────────────────
        df = df.Define("lep_p_resp",         "reco_lep_p / gen_lep_p")
        df = df.Define("lep_theta_resol",    "reco_lep_theta - gen_lep_theta")
        df = df.Define("lep_phi_resol",      "TVector2::Phi_mpi_pi(reco_lep_phi - gen_lep_phi)")
        df = df.Define("lep_eta_resol",      "reco_lep_eta - gen_lep_eta")
        df = df.Define("lep_costheta_resol", "reco_lep_costheta - gen_lep_costheta")
        df = df.Define("met_p_resp",          "reco_met_p / gen_nu_p")
        df = df.Define("met_theta_resol",     "reco_met_theta - gen_nu_theta")
        df = df.Define("met_phi_resol",       "TVector2::Phi_mpi_pi(reco_met_phi - gen_nu_phi)")
        df = df.Define("met_eta_resol",       "reco_met_eta - gen_nu_eta")
        df = df.Define("met_costheta_resol",  "reco_met_costheta - gen_nu_costheta")

        # ── gen W candidates ──────────────────────────────────────────────────
        # Convention: gen quarks keep their native (MC) masses; reco jets are massless.
        df = df.Define("Wlep_gen", "FCCAnalyses::WWFunctions::sum_p4({lep_p4_gen, nu_p4_gen})")
        df = df.Define("Whad_gen", "FCCAnalyses::WWFunctions::sum_p4({gen_q1_p4, gen_q2_p4})")
        df = df.Define("gen_Wlep_m",        "Wlep_gen.M()")
        df = df.Define("gen_Wlep_p",        "Wlep_gen.P()")
        df = df.Define("gen_Whad_m",        "Whad_gen.M()")
        df = df.Define("gen_Whad_p",        "Whad_gen.P()")
        df = df.Define("gen_Whad_costheta", "Whad_gen.CosTheta()")
        df = df.Define("gen_Whad_phi",      "Whad_gen.Phi()")
        df = df.Define("Wlnuqq_gen",
            "FCCAnalyses::WWFunctions::sum_p4({lep_p4_gen, nu_p4_gen, gen_q1_p4, gen_q2_p4})")
        df = df.Define("gen_WW_m",           "Wlnuqq_gen.M()")
        df = df.Define("gen_WW_m_minus_ecm", "gen_WW_m - FCCAnalyses::WWFunctions::ECM")

        df = df.Define("jet1", "jets_p4[0]")
        df = df.Define("jet2", "jets_p4[1]")

        df = df.Define("reco_jet1_p",        "jet1.P()")
        df = df.Define("reco_jet1_pt",       "jet1.Pt()")
        df = df.Define("reco_jet1_theta",    "jet1.Theta()")
        df = df.Define("reco_jet1_phi",      "jet1.Phi()")
        df = df.Define("reco_jet1_eta",      "jet1.Eta()")
        df = df.Define("reco_jet1_costheta", "jet1.CosTheta()")
        df = df.Define("reco_jet1_mass",     "jet1.M()")
        df = df.Define("reco_jet2_p",        "jet2.P()")
        df = df.Define("reco_jet2_pt",       "jet2.Pt()")
        df = df.Define("reco_jet2_theta",    "jet2.Theta()")
        df = df.Define("reco_jet2_phi",      "jet2.Phi()")
        df = df.Define("reco_jet2_eta",      "jet2.Eta()")
        df = df.Define("reco_jet2_costheta", "jet2.CosTheta()")
        df = df.Define("reco_jet2_mass",     "jet2.M()")
        df = df.Define("n_reco_jets", "jets_p4.size()")
        df = df.Filter("n_reco_jets == 2")
        df = df.Define("d_12", "JetClusteringUtils::get_exclusive_dmerge(_jet, 1)")
        df = df.Define("d_32", "JetClusteringUtils::get_exclusive_dmerge(_jet, 2)")
        # Reco jets treated as massless (jet mass is dominated by clustering/detector,
        # not the parton mass); matches the kinfit's (p, theta, phi) jet parameterization.
        df = df.Define("jet1_massless",
            "FCCAnalyses::WWFunctions::tlv_setmass(jet1, 0.)")
        df = df.Define("jet2_massless",
            "FCCAnalyses::WWFunctions::tlv_setmass(jet2, 0.)")
        df = df.Define("Whad_reco",
            "FCCAnalyses::WWFunctions::sum_p4({jet1_massless, jet2_massless})")
        df = df.Define("reco_Whad_m",        "Whad_reco.M()")
        df = df.Define("reco_Whad_p",        "Whad_reco.P()")
        df = df.Define("reco_Whad_costheta", "Whad_reco.CosTheta()")
        df = df.Define("reco_Whad_phi",      "Whad_reco.Phi()")
        df = df.Define("reco_Wlep_costheta", "Wlep_reco.CosTheta()")
        df = df.Define("reco_Wlep_phi",      "Wlep_reco.Phi()")
        df = df.Define("gen_Wlep_costheta",  "Wlep_gen.CosTheta()")
        df = df.Define("gen_Wlep_phi",       "Wlep_gen.Phi()")

        df = df.Define("WW_reco","(Wlep_reco+Whad_reco)")
        df = df.Define("reco_WW_m",              "WW_reco.M()")
        df = df.Define("reco_WW_m_minus_ecm",    "reco_WW_m - FCCAnalyses::WWFunctions::ECM")
        df = df.Define("reco_WW_p_imbalance_tot","WW_reco.P()")

        df = df.Define("WW_gen", "(Wlep_gen + Whad_gen)")

        df = df.Define("gen_Wlep_pt", "Wlep_gen.Pt()")
        df = df.Define("gen_Whad_pt", "Whad_gen.Pt()")
        df = df.Define("reco_Wlep_pt","Wlep_reco.Pt()")
        df = df.Define("reco_Whad_pt","Whad_reco.Pt()")

        df = df.Define("gen_WW_px",  "Wlnuqq_gen.Px()")
        df = df.Define("gen_WW_py",  "Wlnuqq_gen.Py()")
        df = df.Define("gen_WW_pz",  "Wlnuqq_gen.Pz()")
        df = df.Define("gen_WW_p_imbalance_tot", "Wlnuqq_gen.P()")
        df = df.Define("reco_WW_px", "WW_reco.Px()")
        df = df.Define("reco_WW_py", "WW_reco.Py()")
        df = df.Define("reco_WW_pz", "WW_reco.Pz()")
        df = df.Define("WW_px_resol", "reco_WW_px - gen_WW_px")
        df = df.Define("WW_py_resol", "reco_WW_py - gen_WW_py")
        df = df.Define("WW_pz_resol", "reco_WW_pz - gen_WW_pz")

        # ── W boson per-component momenta (reco / gen) ───────────────────
        df = df.Define("reco_Wlep_px", "Wlep_reco.Px()")
        df = df.Define("reco_Wlep_py", "Wlep_reco.Py()")
        df = df.Define("reco_Wlep_pz", "Wlep_reco.Pz()")
        df = df.Define("gen_Wlep_px",  "Wlep_gen.Px()")
        df = df.Define("gen_Wlep_py",  "Wlep_gen.Py()")
        df = df.Define("gen_Wlep_pz",  "Wlep_gen.Pz()")
        df = df.Define("reco_Whad_px", "Whad_reco.Px()")
        df = df.Define("reco_Whad_py", "Whad_reco.Py()")
        df = df.Define("reco_Whad_pz", "Whad_reco.Pz()")
        df = df.Define("gen_Whad_px",  "Whad_gen.Px()")
        df = df.Define("gen_Whad_py",  "Whad_gen.Py()")
        df = df.Define("gen_Whad_pz",  "Whad_gen.Pz()")
        df = df.Define(
            "deltaM",
            "FCCAnalyses::WWFunctions::deltaM(n_lep_reco, n_reco_jets, Wlep_reco, Whad_reco)"
        )

        # ── jet ↔ gen-quark matching (no dR cut: matchJets2 always assigns) ────
        df = df.Define("matched_gen_quarks","FCCAnalyses::WWFunctions::matchJets2(jet1, jet2, gen_q1_p4, gen_q2_p4)")
        df = df.Define("jet1_matched_q_p4", "matched_gen_quarks.first")
        df = df.Define("jet2_matched_q_p4", "matched_gen_quarks.second")

        # ── gen-quark scalars (matched to jet1, jet2) ─────────────────────────
        df = df.Define("gen_quark1_p",        "jet1_matched_q_p4.P()")
        df = df.Define("gen_quark1_pt",       "jet1_matched_q_p4.Pt()")
        df = df.Define("gen_quark1_theta",    "jet1_matched_q_p4.Theta()")
        df = df.Define("gen_quark1_phi",      "jet1_matched_q_p4.Phi()")
        df = df.Define("gen_quark1_eta",      "jet1_matched_q_p4.Eta()")
        df = df.Define("gen_quark1_costheta", "jet1_matched_q_p4.CosTheta()")
        df = df.Define("gen_quark2_p",        "jet2_matched_q_p4.P()")
        df = df.Define("gen_quark2_pt",       "jet2_matched_q_p4.Pt()")
        df = df.Define("gen_quark2_theta",    "jet2_matched_q_p4.Theta()")
        df = df.Define("gen_quark2_phi",      "jet2_matched_q_p4.Phi()")
        df = df.Define("gen_quark2_eta",      "jet2_matched_q_p4.Eta()")
        df = df.Define("gen_quark2_costheta", "jet2_matched_q_p4.CosTheta()")

        # ── jet reco-vs-gen-quark resolutions and responses ───────────────────
        df = df.Define("jet1_p_resp",         "reco_jet1_p / gen_quark1_p")
        df = df.Define("jet1_theta_resol",    "reco_jet1_theta - gen_quark1_theta")
        df = df.Define("jet1_phi_resol",      "TVector2::Phi_mpi_pi(reco_jet1_phi - gen_quark1_phi)")
        df = df.Define("jet1_eta_resol",      "reco_jet1_eta - gen_quark1_eta")
        df = df.Define("jet1_costheta_resol", "reco_jet1_costheta - gen_quark1_costheta")
        df = df.Define("jet2_p_resp",         "reco_jet2_p / gen_quark2_p")
        df = df.Define("jet2_theta_resol",    "reco_jet2_theta - gen_quark2_theta")
        df = df.Define("jet2_phi_resol",      "TVector2::Phi_mpi_pi(reco_jet2_phi - gen_quark2_phi)")
        df = df.Define("jet2_eta_resol",      "reco_jet2_eta - gen_quark2_eta")
        df = df.Define("jet2_costheta_resol", "reco_jet2_costheta - gen_quark2_costheta")

        df = df.Define("Wlep_m_resol",     "reco_Wlep_m - gen_Wlep_m")
        df = df.Define("Wlep_p_resol",     "reco_Wlep_p - gen_Wlep_p")
        df = df.Define("Whad_m_resol",     "reco_Whad_m - gen_Whad_m")
        df = df.Define("Whad_p_resol",     "reco_Whad_p - gen_Whad_p")
        df = df.Define("WW_m_resol",       "reco_WW_m - gen_WW_m")

        # ── kinematic fit ──────────────────────────────────────────────────
        _kinfit_funcs = {
            "minuit": "FCCAnalyses::WWFunctions::kinFit",
            "bfgs":   "FCCAnalyses::WWFunctions::kinFitBFGS",
        }
        _kinfit_call   = _kinfit_funcs[KIN_FIT_METHOD]
        _kinfit_free_gw = "true" if KIN_FIT_FREE_GW else "false"
        df = df.Define("kinfit",
            _kinfit_call + "("
            "reco_jet1_p, reco_jet1_theta, reco_jet1_phi,"
            "reco_jet2_p, reco_jet2_theta, reco_jet2_phi,"
            "reco_lep_p, reco_lep_theta, reco_lep_phi,"
            f"reco_met_p, reco_met_theta, reco_met_phi, {_kinfit_free_gw})"
        )
        df = df.Define("kinfit_mW",       "kinfit.mW")
        df = df.Define("kinfit_gW",       "kinfit.gW")
        df = df.Define("kinfit_s1",       "kinfit.s1")
        df = df.Define("kinfit_s2",       "kinfit.s2")
        df = df.Define("kinfit_sl",       "kinfit.sl")
        df = df.Define("kinfit_sn",       "kinfit.sn")
        df = df.Define("kinfit_t1",       "kinfit.t1")
        df = df.Define("kinfit_t2",       "kinfit.t2")
        df = df.Define("kinfit_tn",       "kinfit.tn")
        df = df.Define("kinfit_p1",       "kinfit.p1")
        df = df.Define("kinfit_p2",       "kinfit.p2")
        df = df.Define("kinfit_pn",       "kinfit.pn")
        df = df.Define("kinfit_chi2",     "kinfit.chi2")
        df = df.Define("kinfit_chi2_ndof","kinfit.chi2_ndof")
        df = df.Define("kinfit_valid",    "kinfit.valid")
        # Postfit scalars are projected from TLVs stored in KinFitResult.
        df = df.Define("kinfit_jet1_p",     "kinfit.j1.P()")
        df = df.Define("kinfit_jet1_pt",    "kinfit.j1.Pt()")
        df = df.Define("kinfit_jet1_theta", "kinfit.j1.Theta()")
        df = df.Define("kinfit_jet1_phi",   "kinfit.j1.Phi()")
        df = df.Define("kinfit_jet2_p",     "kinfit.j2.P()")
        df = df.Define("kinfit_jet2_pt",    "kinfit.j2.Pt()")
        df = df.Define("kinfit_jet2_theta", "kinfit.j2.Theta()")
        df = df.Define("kinfit_jet2_phi",   "kinfit.j2.Phi()")
        df = df.Define("kinfit_lep_p",      "kinfit.lep.P()")
        df = df.Define("kinfit_lep_pt",     "kinfit.lep.Pt()")
        df = df.Define("kinfit_lep_theta",  "kinfit.lep.Theta()")
        df = df.Define("kinfit_lep_phi",    "kinfit.lep.Phi()")
        df = df.Define("kinfit_nu_p",       "kinfit.nu.P()")
        df = df.Define("kinfit_nu_pt",      "kinfit.nu.Pt()")
        df = df.Define("kinfit_nu_theta",   "kinfit.nu.Theta()")
        df = df.Define("kinfit_nu_phi",     "kinfit.nu.Phi()")
        df = df.Define("kinfit_tl",         "kinfit.tl")
        df = df.Define("kinfit_pl",         "kinfit.pl")
        df = df.Define("Wlep_kinfit", "kinfit.lep + kinfit.nu")
        df = df.Define("Whad_kinfit", "kinfit.j1  + kinfit.j2")
        df = df.Define("WW_kinfit",   "Wlep_kinfit + Whad_kinfit")
        df = df.Define("kinfit_Wlep_m",     "Wlep_kinfit.M()")
        df = df.Define("kinfit_Wlep_p",     "Wlep_kinfit.P()")
        df = df.Define("kinfit_Wlep_pt",    "Wlep_kinfit.Pt()")
        df = df.Define("kinfit_Wlep_px",    "Wlep_kinfit.Px()")
        df = df.Define("kinfit_Wlep_py",    "Wlep_kinfit.Py()")
        df = df.Define("kinfit_Wlep_pz",    "Wlep_kinfit.Pz()")
        df = df.Define("kinfit_Whad_m",     "Whad_kinfit.M()")
        df = df.Define("kinfit_Whad_p",     "Whad_kinfit.P()")
        df = df.Define("kinfit_Whad_pt",    "Whad_kinfit.Pt()")
        df = df.Define("kinfit_Whad_px",    "Whad_kinfit.Px()")
        df = df.Define("kinfit_Whad_py",    "Whad_kinfit.Py()")
        df = df.Define("kinfit_Whad_pz",    "Whad_kinfit.Pz()")
        df = df.Define("kinfit_WW_m",           "WW_kinfit.M()")
        df = df.Define("kinfit_WW_m_minus_ecm", "kinfit_WW_m - FCCAnalyses::WWFunctions::ECM")
        df = df.Define("kinfit_WW_px",          "WW_kinfit.Px()")
        df = df.Define("kinfit_WW_py",          "WW_kinfit.Py()")
        df = df.Define("kinfit_WW_pz",          "WW_kinfit.Pz()")
        df = df.Define("kinfit_WW_p_imbalance_tot", "WW_kinfit.P()")

        return df

    def output():
        return all_branches
