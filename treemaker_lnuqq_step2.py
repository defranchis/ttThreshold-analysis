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

all_branches = ["lep_p_resp","reco_moff","reco_mon","truth_lnuqq_mon","truth_lnuqq_moff","m_gen_lnuqq","p_lnu_fromele","p_excljj","p_iso_lnu",
                "p_qq_fromele",    "nIsolep", "Isolep_p", 'Isolep_theta',"m_iso_lnu",'Isolep_phi',"Isolep_pt","Isolep_eta",
                "missing_p", "missing_p_theta", "missing_p_phi", "met_p_resp","missing_p_eta","missing_pt",
                "m_lnu_resol",
                "deltaM","sumP","sumPx","sumPy","sumPz","sumP_gen","sumPx_gen","sumPy_gen","sumPz_gen","sumPt","sumPt_gen",
                "p_lnu_resol",
                "m_lnuqq_resol",
                "m_qq_resol",
                "p_qq_resol","Whad_gen_old","sumP_gen_new","ngen_partons_fromele",
                "p_iso_lnuexcljj","e_iso_lnuexcljj","Whad_gen_pt","Wlep_gen_pt", "Whad_reco_pt", "Whad_reco_mass", "Wlep_reco_pt", "Wlep_reco_mass", "lep_eta_resol","lep_phi_resol","lep_dR","lep_theta_resol", "jet1_eta_resol","jet1_phi_resol","jet1_theta_resol",
                "jet2_eta_resol","jet2_phi_resol","jet2_theta_resol", "met_eta_resol","met_phi_resol","met_dR","met_theta_resol","lep_gen_costheta","lep_costheta", "met_gen_costheta","met_costheta",
                "jet1_gen_costheta","jet1_costheta", "jet2_gen_costheta","jet2_costheta", "met_costheta_resol","lep_costheta_resol","jet1_gen_theta","jet2_gen_theta","jet2_costheta_resol","jet1_costheta_resol"
]
all_branches+=["m_qq_fromele","m_lnu_fromele","jet2_p_resp","jet1_p_resp","mlnu_plus_mjj_reco","mlnu_plus_mqq_fromele_truth"]
all_branches+=[ "nRecoJets", "jet1_p", "jet2_p", "d_12","d_32","m_iso_lnuexcljj","jet1_pt","jet2_pt","jet1_eta","jet2_eta","jet1_phi","jet2_phi","jet1_mass","jet2_mass"]
all_branches+=["kinfit_mW","kinfit_gW","kinfit_s1","kinfit_s2","kinfit_sl","kinfit_sn",
               "kinfit_t1","kinfit_t2","kinfit_tn","kinfit_tl",
               "kinfit_p1","kinfit_p2","kinfit_pn","kinfit_pl",
               "kinfit_chi2","kinfit_chi2_ndof","kinfit_valid",
               "kinfit_mWlep","kinfit_mWhad",
               "kinfit_pt_j1","kinfit_pt_j2","kinfit_pt_lep","kinfit_pt_nu",
               "kinfit_p_j1","kinfit_p_j2","kinfit_p_lep","kinfit_p_nu",
               "kinfit_Wlep_px","kinfit_Wlep_py","kinfit_Wlep_pz",
               "kinfit_Whad_px","kinfit_Whad_py","kinfit_Whad_pz",
               "kinfit_theta_j1","kinfit_theta_j2","kinfit_theta_nu",
               "kinfit_phi_j1","kinfit_phi_j2","kinfit_phi_nu",
               "kinfit_deltaP",
               "kinfit_WW_px","kinfit_WW_py","kinfit_WW_pz",
               "kinfit_WW_m","kinfit_WW_m_minus_ecm"]
all_branches+=["pf_qq_mass","pf_qq_p","pf_qq_costheta","pf_qq_phi",
               "Whad_gen_mass","Whad_gen_p","Whad_gen_costheta","Whad_gen_phi",
               "pf_qq_m_resol","pf_qq_p_resol","pf_qq_costheta_resol","pf_qq_phi_resol"]
all_branches+=["px_tot_gen","py_tot_gen","pz_tot_gen",
               "px_tot_reco","py_tot_reco","pz_tot_reco",
               "px_tot_resol","py_tot_resol","pz_tot_resol","m_gen_lnuqq_minus_ecm"]
all_branches+=["jet1_theta","jet2_theta",
               "jet1_gen_pt","jet2_gen_pt","jet1_gen_p","jet2_gen_p","jet1_gen_phi","jet2_gen_phi",
               "lep_gen_pt","lep_gen_p","lep_gen_theta","lep_gen_phi",
               "nu_gen_pt","nu_gen_p","nu_gen_theta","nu_gen_phi",
               "m_reco_WW_minus_ecm"]

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

        df = df.Define("pf_qq_p4",
            "TLorentzVector("
            "ROOT::VecOps::Sum(FCCAnalyses::ReconstructedParticle::get_px(ReconstructedParticlesNoMuNoEl)),"
            "ROOT::VecOps::Sum(FCCAnalyses::ReconstructedParticle::get_py(ReconstructedParticlesNoMuNoEl)),"
            "ROOT::VecOps::Sum(FCCAnalyses::ReconstructedParticle::get_pz(ReconstructedParticlesNoMuNoEl)),"
            "ROOT::VecOps::Sum(FCCAnalyses::ReconstructedParticle::get_e (ReconstructedParticlesNoMuNoEl)))")
        df = df.Define("pf_qq_mass",     "pf_qq_p4.M()")
        df = df.Define("pf_qq_p",        "pf_qq_p4.P()")
        df = df.Define("pf_qq_costheta", "pf_qq_p4.CosTheta()")
        df = df.Define("pf_qq_phi",      "pf_qq_p4.Phi()")

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
        df = df.Define("Isolep_p",     "Isoleps_p4_reco.P()")
        df = df.Define("Isolep_e",     "Isoleps_p4_reco.E()")
        df = df.Define("Isolep_pt",    "Isoleps_p4_reco.Pt()")
        df = df.Define("Isolep_eta",   "Isoleps_p4_reco.Eta()")
        df = df.Define("Isolep_theta", "Isoleps_p4_reco.Theta()")
        df = df.Define("Isolep_phi",   "Isoleps_p4_reco.Phi()")
        df = df.Define("nIsolep",      "(int)Isoleps.size()")

        df = df.Define("missing_p_p4",    "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET, 0)")
        df = df.Define("missing_p",       "missing_p_p4.P()")
        df = df.Define("missing_pt",      "missing_p_p4.Pt()")
        df = df.Define("missing_p_theta", "missing_p_p4.Theta()")
        df = df.Define("missing_p_phi",   "missing_p_p4.Phi()")
        df = df.Define("missing_p_eta",   "missing_p_p4.Eta()")
        df = df.Define("Wlep_reco",
            "FCCAnalyses::WWFunctions::sum_p4({Isoleps_p4_reco, missing_p_p4})")

        df = df.Define("m_iso_lnu", "Wlep_reco.M()");
        df = df.Define("p_iso_lnu", "Wlep_reco.P()");

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

        df = df.Define("ngen_partons_fromele", "FCCAnalyses::MCParticle::get_n(gen_lightquarks_fromele)")

        # ── gen-level per-particle TLVs ───────────────────────────────────────
        df = df.Define("lep_p4_gen", "gen_leps_fromele_tlv[0]")
        df = df.Define("nu_p4_gen",  "gen_neutrinos_fromele_tlv[0]")
        df = df.Define("gen_q1_p4",  "gen_lightquarks_fromele_tlv[0]")
        df = df.Define("gen_q2_p4",  "gen_lightquarks_fromele_tlv[1]")
        # Massless-quark TLVs (same 3-momentum, E rescaled to |p|) used by Whad_gen for the W mass.
        df = df.Define("gen_q1_p4_massless",
            "TLorentzVector(gen_q1_p4.Px(), gen_q1_p4.Py(), gen_q1_p4.Pz(), gen_q1_p4.P())")
        df = df.Define("gen_q2_p4_massless",
            "TLorentzVector(gen_q2_p4.Px(), gen_q2_p4.Py(), gen_q2_p4.Pz(), gen_q2_p4.P())")

        # ── gen W candidates ──────────────────────────────────────────────────
        df = df.Define("Wlep_gen",            "FCCAnalyses::WWFunctions::sum_p4({lep_p4_gen, nu_p4_gen})")
        df = df.Define("m_lnu_fromele",       "Wlep_gen.M()")
        df = df.Define("p_lnu_fromele",       "Wlep_gen.P()")
        df = df.Define("Whad_gen_qq_fromele", "FCCAnalyses::WWFunctions::sum_p4({gen_q1_p4, gen_q2_p4})")  # mass-preserved
        df = df.Define("p_qq_fromele",        "Whad_gen_qq_fromele.P()")
        df = df.Define("m_qq_fromele",        "Whad_gen_qq_fromele.M()")

        df = df.Define("lep_eta_resol", "Isoleps_p4_reco.Eta() - lep_p4_gen.Eta()")
        df = df.Define("lep_phi_resol", "TVector2::Phi_mpi_pi(Isoleps_p4_reco.Phi() - lep_p4_gen.Phi())")
        df = df.Define("lep_dR",        "sqrt(lep_eta_resol*lep_eta_resol + lep_phi_resol*lep_phi_resol)")
        df = df.Define("met_phi_resol", "TVector2::Phi_mpi_pi(missing_p_p4.Phi() - nu_p4_gen.Phi())")
        df = df.Define("met_eta_resol", "missing_p_p4.Eta() - nu_p4_gen.Eta()")
        df = df.Define("met_dR",        "sqrt(met_eta_resol*met_eta_resol + met_phi_resol*met_phi_resol)")
        df = df.Define("lep_gen_costheta", "lep_p4_gen.CosTheta()");
        df = df.Define("lep_costheta",     "Isoleps_p4_reco.CosTheta()");
        df = df.Define("lep_costheta_resol", "lep_costheta - lep_gen_costheta")
        df = df.Define("lep_theta_resol",    "Isolep_theta - lep_p4_gen.Theta()")

        df = df.Define("met_gen_costheta", "nu_p4_gen.CosTheta()");
        df = df.Define("met_costheta",     "missing_p_p4.CosTheta()");
        df = df.Define("met_theta_resol",    "missing_p_theta - nu_p4_gen.Theta()")
        df = df.Define("met_costheta_resol", "met_costheta - met_gen_costheta")
        df = df.Define("Whad_gen",          "FCCAnalyses::WWFunctions::sum_p4({gen_q1_p4_massless, gen_q2_p4_massless})")
        df = df.Define("Whad_gen_old",      "Whad_gen_qq_fromele.M()")  # legacy: mass-preserved dijet mass
        df = df.Define("Whad_gen_mass",     "Whad_gen.M()")
        df = df.Define("Whad_gen_p",        "Whad_gen.P()")
        df = df.Define("Whad_gen_costheta", "Whad_gen.CosTheta()")
        df = df.Define("Whad_gen_phi",      "Whad_gen.Phi()")
        df = df.Define("pf_qq_m_resol",        "pf_qq_mass - Whad_gen_mass")
        df = df.Define("pf_qq_p_resol",        "pf_qq_p - Whad_gen_p")
        df = df.Define("pf_qq_costheta_resol", "pf_qq_costheta - Whad_gen_costheta")
        df = df.Define("pf_qq_phi_resol",      "TVector2::Phi_mpi_pi(pf_qq_phi - Whad_gen_phi)")
        df = df.Define("Wlnuqq_gen",
            "FCCAnalyses::WWFunctions::sum_p4({lep_p4_gen, nu_p4_gen, gen_q1_p4_massless, gen_q2_p4_massless})")
        df = df.Define("m_gen_lnuqq",          "Wlnuqq_gen.M()")
        df = df.Define("m_gen_lnuqq_minus_ecm", "m_gen_lnuqq - FCCAnalyses::WWFunctions::ECM")

        df = df.Define("jet1", "jets_p4[0]")
        df = df.Define("jet2", "jets_p4[1]")

        df = df.Define("jet1_pt","jet1.Pt()")
        df = df.Define("jet2_pt","jet2.Pt()")
        df = df.Define("jet1_eta","jet1.Eta()")
        df = df.Define("jet2_eta","jet2.Eta()")
        df = df.Define("jet1_mass","jet1.M()")
        df = df.Define("jet2_mass","jet2.M()")
        df = df.Define("jet1_p","jet1.P()")
        df = df.Define("jet2_p","jet2.P()")
        df = df.Define("jet1_theta", "jet1.Theta()")
        df = df.Define("jet2_theta", "jet2.Theta()")
        df = df.Define("jet1_phi",   "jet1.Phi()")
        df = df.Define("jet2_phi",   "jet2.Phi()")
        df = df.Define("nRecoJets", "jets_p4.size()")
        df = df.Filter("nRecoJets == 2")
        df = df.Define("d_12", "JetClusteringUtils::get_exclusive_dmerge(_jet, 1)")
        df = df.Define("d_32", "JetClusteringUtils::get_exclusive_dmerge(_jet, 2)")
        # Treat reco jets as massless when summing — matches kinfit (massless 4-vectors)
        # and gen Whad (gen_q*_p4_massless), so reco/gen/fit Whad and WW are on the same footing.
        df = df.Define("jet1_massless",
            "TLorentzVector(jet1.Px(), jet1.Py(), jet1.Pz(), jet1.P())")
        df = df.Define("jet2_massless",
            "TLorentzVector(jet2.Px(), jet2.Py(), jet2.Pz(), jet2.P())")
        df = df.Define("Whad_reco",
            "FCCAnalyses::WWFunctions::sum_p4({jet1_massless, jet2_massless})")
        df = df.Define("m_excl_jj", "Whad_reco.M()")

        df = df.Define("WW_iso_lnuexcljj","(Wlep_reco+Whad_reco)")
        df = df.Define("m_iso_lnuexcljj","WW_iso_lnuexcljj.M()")
        df = df.Define("p_iso_lnuexcljj","WW_iso_lnuexcljj.P()")
        df = df.Define("e_iso_lnuexcljj","WW_iso_lnuexcljj.E()")
        df = df.Define("m_reco_WW_minus_ecm", "m_iso_lnuexcljj - FCCAnalyses::WWFunctions::ECM")

        df = df.Define("WW_gen", "(Wlep_gen + Whad_gen)")
        df = df.Define("sumP_gen_new", "WW_gen.Px() + WW_gen.Py() + WW_gen.Pz()")
        df = df.Define("sumPt_gen",    "WW_gen.Px() + WW_gen.Py()")
        df = df.Define("sumPx_gen",    "WW_gen.Px()")
        df = df.Define("sumPy_gen",    "WW_gen.Py()")
        df = df.Define("sumPz_gen",    "WW_gen.Pz()")
        df = df.Define("sumP_gen",     "WW_gen.Px() + WW_gen.Py() + WW_gen.Pz()")

        df = df.Define("Wlep_gen_pt","Wlep_gen.Pt()")
        df = df.Define("Whad_gen_pt","Whad_gen.Pt()")
        df = df.Define("Wlep_reco_pt","Wlep_reco.Pt()")
        df = df.Define("Wlep_reco_mass","Wlep_reco.M()")
        df = df.Define("Whad_reco_pt","Whad_reco.Pt()")
        df = df.Define("Whad_reco_mass","Whad_reco.M()")

        df = df.Define("sumP",  "WW_iso_lnuexcljj.Px() + WW_iso_lnuexcljj.Py() + WW_iso_lnuexcljj.Pz()")
        df = df.Define("sumPt", "WW_iso_lnuexcljj.Px() + WW_iso_lnuexcljj.Py()")
        df = df.Define("sumPx", "WW_iso_lnuexcljj.Px()")
        df = df.Define("sumPy", "WW_iso_lnuexcljj.Py()")
        df = df.Define("sumPz", "WW_iso_lnuexcljj.Pz()")

        df = df.Define("px_tot_gen",  "Wlnuqq_gen.Px()")
        df = df.Define("py_tot_gen",  "Wlnuqq_gen.Py()")
        df = df.Define("pz_tot_gen",  "Wlnuqq_gen.Pz()")
        df = df.Define("px_tot_reco", "WW_iso_lnuexcljj.Px()")
        df = df.Define("py_tot_reco", "WW_iso_lnuexcljj.Py()")
        df = df.Define("pz_tot_reco", "WW_iso_lnuexcljj.Pz()")
        df = df.Define("px_tot_resol", "px_tot_reco - px_tot_gen")
        df = df.Define("py_tot_resol", "py_tot_reco - py_tot_gen")
        df = df.Define("pz_tot_resol", "pz_tot_reco - pz_tot_gen")
        df = df.Define(
            "deltaM",
            "FCCAnalyses::WWFunctions::deltaM(nIsolep, nRecoJets, Wlep_reco, Whad_reco)"
        )
        df = df.Define("p_excljj","Whad_reco.P()");

        df = df.Define("matched_gen_quarks","FCCAnalyses::WWFunctions::matchJets2(jet1, jet2, gen_q1_p4, gen_q2_p4)");
        df = df.Define("jet1_matched_q_p4", "matched_gen_quarks.first")
        df = df.Define("jet2_matched_q_p4", "matched_gen_quarks.second");
        df = df.Define("jet1_gen_pt",  "jet1_matched_q_p4.Pt()")
        df = df.Define("jet2_gen_pt",  "jet2_matched_q_p4.Pt()")
        df = df.Define("jet1_gen_p",   "jet1_matched_q_p4.P()")
        df = df.Define("jet2_gen_p",   "jet2_matched_q_p4.P()")
        df = df.Define("jet1_gen_phi", "jet1_matched_q_p4.Phi()")
        df = df.Define("jet2_gen_phi", "jet2_matched_q_p4.Phi()")
        df = df.Define("jet1_p_resp", "jet1_p / jet1_matched_q_p4.P()")
        df = df.Define("jet2_p_resp", "jet2_p / jet2_matched_q_p4.P()")
        df = df.Define("jet1_costheta",     "jet1.CosTheta()")
        df = df.Define("jet2_costheta",     "jet2.CosTheta()")
        df = df.Define("jet1_gen_costheta", "jet1_matched_q_p4.CosTheta()")
        df = df.Define("jet2_gen_costheta", "jet2_matched_q_p4.CosTheta()")
        df = df.Define("jet1_gen_theta", "jet1_matched_q_p4.Theta()")
        df = df.Define("jet2_gen_theta", "jet2_matched_q_p4.Theta()")
        df = df.Define("jet1_theta_resol", "jet1_theta - jet1_matched_q_p4.Theta()")
        df = df.Define("jet2_theta_resol", "jet2_theta - jet2_matched_q_p4.Theta()")
        df = df.Define("jet1_costheta_resol", "jet1_costheta - jet1_gen_costheta")
        df = df.Define("jet2_costheta_resol", "jet2_costheta - jet2_gen_costheta")
        df = df.Define("jet1_eta_resol", "jet1.Eta()-jet1_matched_q_p4.Eta()")
        df = df.Define("jet2_eta_resol", "jet2.Eta()-jet2_matched_q_p4.Eta()")
        df = df.Define("jet1_phi_resol", "TVector2::Phi_mpi_pi(jet1.Phi()-jet1_matched_q_p4.Phi())")
        df = df.Define("jet2_phi_resol", "TVector2::Phi_mpi_pi(jet2.Phi()-jet2_matched_q_p4.Phi())")

        df = df.Define("lep_p_resp",    "Isolep_p / lep_p4_gen.P()")
        df = df.Define("lep_gen_pt",    "lep_p4_gen.Pt()")
        df = df.Define("lep_gen_p",     "lep_p4_gen.P()")
        df = df.Define("lep_gen_theta", "lep_p4_gen.Theta()")
        df = df.Define("lep_gen_phi",   "lep_p4_gen.Phi()")
        df = df.Define("nu_gen_pt",     "nu_p4_gen.Pt()")
        df = df.Define("nu_gen_p",      "nu_p4_gen.P()")
        df = df.Define("nu_gen_theta",  "nu_p4_gen.Theta()")
        df = df.Define("nu_gen_phi",    "nu_p4_gen.Phi()")
        df = (df
                  .Define("truth_mlnu", "m_lnu_fromele")
                  .Define("truth_mqq",  "m_qq_fromele")
                  .Define("reco_mlnu",  "m_iso_lnu")
                  .Define("reco_mjj",   "m_excl_jj")
                  .Define("reco_mon",   "reco_mlnu >= reco_mjj ? reco_mlnu : reco_mjj")
                  .Define("reco_moff",  "reco_mlnu < reco_mjj ? reco_mlnu : reco_mjj")
                  .Define("truth_lnuqq_mon",  "truth_mlnu >= truth_mqq ? truth_mlnu : truth_mqq")
                  .Define("truth_lnuqq_moff", "truth_mlnu < truth_mqq ? truth_mlnu : truth_mqq")
                  .Define("mlnu_plus_mjj_reco",       "reco_mlnu + reco_mjj")
                  .Define("mlnu_plus_mqq_fromele_truth", "m_qq_fromele + m_lnu_fromele")
                  .Define("m_lnu_resol",  "reco_mlnu - m_lnu_fromele")
                  .Define("p_lnu_resol",  "p_iso_lnu - p_lnu_fromele")
                  .Define("m_lnuqq_resol","m_iso_lnuexcljj - m_gen_lnuqq")
                  .Define("m_qq_resol",   "reco_mjj - m_qq_fromele")
                  .Define("p_qq_resol",   "p_excljj - p_qq_fromele")
                  .Define("met_p_resp",   "missing_p / nu_p4_gen.P()")
                )

        # ── kinematic fit ──────────────────────────────────────────────────
        _kinfit_funcs = {
            "minuit": "FCCAnalyses::WWFunctions::kinFit",
            "bfgs":   "FCCAnalyses::WWFunctions::kinFitBFGS",
        }
        _kinfit_call   = _kinfit_funcs[KIN_FIT_METHOD]
        _kinfit_free_gw = "true" if KIN_FIT_FREE_GW else "false"
        df = df.Define("kinfit",
            _kinfit_call + "("
            "jet1_p, jet1_theta, jet1_phi,"
            "jet2_p, jet2_theta, jet2_phi,"
            "Isolep_p, Isolep_theta, Isolep_phi,"
            f"missing_p, missing_p_theta, missing_p_phi, {_kinfit_free_gw})"
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
        df = df.Define("kinfit_mWlep",    "kinfit.mWlep_postfit")
        df = df.Define("kinfit_mWhad",    "kinfit.mWhad_postfit")
        df = df.Define("kinfit_pt_j1",    "kinfit.pt_j1_postfit")
        df = df.Define("kinfit_pt_j2",    "kinfit.pt_j2_postfit")
        df = df.Define("kinfit_pt_lep",   "kinfit.pt_lep_postfit")
        df = df.Define("kinfit_pt_nu",    "kinfit.pt_nu_postfit")
        df = df.Define("kinfit_p_j1",     "kinfit.p_j1_postfit")
        df = df.Define("kinfit_p_j2",     "kinfit.p_j2_postfit")
        df = df.Define("kinfit_p_lep",    "kinfit.p_lep_postfit")
        df = df.Define("kinfit_p_nu",     "kinfit.p_nu_postfit")
        df = df.Define("kinfit_Wlep_px",  "kinfit.Wlep_px_postfit")
        df = df.Define("kinfit_Wlep_py",  "kinfit.Wlep_py_postfit")
        df = df.Define("kinfit_Wlep_pz",  "kinfit.Wlep_pz_postfit")
        df = df.Define("kinfit_Whad_px",  "kinfit.Whad_px_postfit")
        df = df.Define("kinfit_Whad_py",  "kinfit.Whad_py_postfit")
        df = df.Define("kinfit_Whad_pz",  "kinfit.Whad_pz_postfit")
        df = df.Define("kinfit_theta_j1", "kinfit.theta_j1_postfit")
        df = df.Define("kinfit_theta_j2", "kinfit.theta_j2_postfit")
        df = df.Define("kinfit_theta_nu", "kinfit.theta_nu_postfit")
        df = df.Define("kinfit_phi_j1",   "kinfit.phi_j1_postfit")
        df = df.Define("kinfit_phi_j2",   "kinfit.phi_j2_postfit")
        df = df.Define("kinfit_phi_nu",   "kinfit.phi_nu_postfit")
        df = df.Define("kinfit_tl",             "kinfit.tl")
        df = df.Define("kinfit_pl",             "kinfit.pl")
        df = df.Define("kinfit_deltaP",         "kinfit.deltaP_postfit")
        df = df.Define("kinfit_WW_px",          "kinfit.Wlep_px_postfit + kinfit.Whad_px_postfit")
        df = df.Define("kinfit_WW_py",          "kinfit.Wlep_py_postfit + kinfit.Whad_py_postfit")
        df = df.Define("kinfit_WW_pz",          "kinfit.Wlep_pz_postfit + kinfit.Whad_pz_postfit")
        df = df.Define("kinfit_WW_m",           "kinfit.mWW_postfit")
        df = df.Define("kinfit_WW_m_minus_ecm", "kinfit.mWW_postfit - FCCAnalyses::WWFunctions::ECM")

        return df

    def output():
        return all_branches
