# Step 1 of 2 — produce only the branches needed by fit_dcb_resolutions.py.
# Run fit_dcb_resolutions.py on the output before running treemaker_lnuqq_step2.py.
import re, ROOT
processList = {
    "wzp6_ee_munumuqq_noCut_ecm160": {"fraction": 1, "crossSection": 1},
    "wzp6_ee_munumuqq_noCut_ecm157": {"fraction": 1, "crossSection": 1},
    "wzp6_ee_munumuqq_noCut_ecm163": {"fraction": 1, "crossSection": 1},
}

available_ecm = ['157', '160', '163']

def _parse_ecm(name):
    m = re.search(r'_ecm(\d+)', name)
    if not m:
        raise ValueError(f"Cannot parse ecm from sample name: {name}")
    return int(m.group(1))

channel = "CHANNELNAMEHERE"
if channel not in ["lep", "semihad", "had"]:
    channel = "semihad"
print(channel)

prodTag   = "FCCee/winter2023/IDEA/"
outputDir = "outputs/treemaker/lnuqq/step1/{}".format(channel)

includePaths = ["examples/functions.h", "WWFunctions/WWFunctions.h"]

from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper

jetClusteringHelper = None

# Branches consumed by fit_dcb_resolutions.py --kinfit-only (= KINFIT_BRANCHES)
all_branches = [
    "jet1_p_resp", "jet2_p_resp", "lep_p_resp", "met_p_resp",
    "jet1_theta_resol", "jet2_theta_resol", "jet1_phi_resol", "jet2_phi_resol",
    "lep_theta_resol", "lep_phi_resol",
    "met_theta_resol", "met_phi_resol",
    "px_tot_gen", "py_tot_gen", "pz_tot_gen",
    "m_gen_lnuqq", "m_gen_lnuqq_minus_ecm",
]

_dataset_iter = iter(processList.keys())

class RDFanalysis:

    def analysers(df):

        _dataset = next(_dataset_iter)
        _ecm = _parse_ecm(_dataset)
        print(f"[treemaker] dataset={_dataset}  ecm={_ecm}")
        if str(_ecm) not in available_ecm:
            raise ValueError(f"ecm={_ecm} parsed from '{_dataset}' not in available_ecm={available_ecm}")
        ROOT.gInterpreter.ProcessLine(f"FCCAnalyses::WWFunctions::ECM = {_ecm};")
        ROOT.gInterpreter.ProcessLine('std::cout << "[DEBUG step1] ECM from WWFunctions = " << FCCAnalyses::WWFunctions::ECM << std::endl;')

        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Electron0", "Electron#0.index")

        df = df.Define("muons_all",
            "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
        df = df.Define("electrons_all",
            "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")

        df = df.Define("muons_sel",
            "FCCAnalyses::ReconstructedParticle::sel_p(12)(muons_all)")
        df = df.Define("electrons_sel",
            "FCCAnalyses::ReconstructedParticle::sel_p(12)(electrons_all)")

        df = df.Define("muons_iso",
            "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(muons_sel, ReconstructedParticles)")
        df = df.Define("electrons_iso",
            "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(electrons_sel, ReconstructedParticles)")

        df = df.Define("muons_sel_iso",
            "FCCAnalyses::ZHfunctions::sel_iso(0.7)(muons_sel, muons_iso)")
        df = df.Define("electrons_sel_iso",
            "FCCAnalyses::ZHfunctions::sel_iso(0.7)(electrons_sel, electrons_iso)")

        if channel == "had":
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 0")
        elif channel == "semihad":
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 1")
        else:
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 2")

        df = df.Define("Isolep_p",
            "muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_p(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_p(electrons_sel_iso)[0] : -999)")
        df = df.Define("Isolep_e",
            "muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_e(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_e(electrons_sel_iso)[0] : -999)")
        df = df.Define("Isolep_theta",
            "muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_theta(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_theta(electrons_sel_iso)[0] : -999)")
        df = df.Define("Isolep_phi",
            "muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_phi(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_phi(electrons_sel_iso)[0] : -999)")

        df = df.Define("Isoleps_p4_reco",
            "FCCAnalyses::WWFunctions::Isoleps_p4_reco(Isolep_p, Isolep_phi, Isolep_theta, Isolep_e)")

        df = df.Define("missing_p",     "FCCAnalyses::ReconstructedParticle::get_p(MissingET)[0]")
        df = df.Define("missing_p_theta", "ReconstructedParticle::get_theta(MissingET)[0]")
        df = df.Define("missing_p_phi",   "ReconstructedParticle::get_phi(MissingET)[0]")

        df = df.Define("missing_p_p4",
            "FCCAnalyses::WWFunctions::missing_p_p4(missing_p, missing_p_phi, missing_p_theta)")

        # ── jet clustering (no flavour tagging needed for kinfit inputs) ───────
        global jetClusteringHelper

        df = df.Define("ReconstructedParticlesNoMuons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, muons_sel_iso)")
        df = df.Define("ReconstructedParticlesNoMuNoEl",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoMuons, electrons_sel_iso)")

        nJets = 2 if channel == "semihad" else 4
        jetClusteringHelper = ExclusiveJetClusteringHelper("ReconstructedParticlesNoMuNoEl", nJets)
        df = jetClusteringHelper.define(df)

        df = df.Define("jets_p4",
            "JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets))

        df = df.Define("jet1", "jets_p4[0]")
        df = df.Define("jet2", "jets_p4[1]")
        df = df.Define("jet1_p", "jet1.P()")
        df = df.Define("jet2_p", "jet2.P()")
        df = df.Define("recoJet_theta", "JetClusteringUtils::get_theta(jet)")
        df = df.Define("jet1_theta", "recoJet_theta[0]")
        df = df.Define("jet2_theta", "recoJet_theta[1]")
        df = df.Define("recoJet_phi", "JetClusteringUtils::get_phi_std(jet)")
        df = df.Define("jet1_phi", "recoJet_phi[0]")
        df = df.Define("jet2_phi", "recoJet_phi[1]")
        df = df.Filter("jets_p4.size() == 2")

        # ── MC aliases & early gen filter ──────────────────────────────────────
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")

        df = df.Define("status1parts",
            "FCCAnalyses::MCParticle::sel_genStatus(1)(Particle)")
        df = df.Define("gen_leps_status1",
            "FCCAnalyses::MCParticle::sel_genleps(13,13,true)(status1parts)")
        df = df.Define("ngen_leps_status1",
            "FCCAnalyses::MCParticle::get_n(gen_leps_status1)")
        df = df.Define("gen_neutrinos_status1",
            "FCCAnalyses::MCParticle::sel_genleps(14,14, true)(status1parts)")
        df = df.Define("gen_neutrinos_status1_p",
            "FCCAnalyses::MCParticle::get_p(gen_neutrinos_status1)")
        df = df.Define("gen_lightquarks_fromele",
            "FCCAnalyses::MCParticle::sel_lightQuarks_fromele(true)(Particle,Particle0)")

        df = df.Filter("ngen_leps_status1 == 1 && gen_neutrinos_status1_p.size() == 1 && gen_lightquarks_fromele.size() > 1")

        # ── gen quantities needed for kinfit input branches ────────────────────
        df = df.Define("status2parts",
            "FCCAnalyses::MCParticle::sel_genStatus(2)(Particle)")

        df = df.Define("gen_leps_status2",
            "FCCAnalyses::MCParticle::sel_genleps(13,13,true)(status2parts)")
        df = df.Define("gen_leps_status2_p",
            "FCCAnalyses::MCParticle::get_p(gen_leps_status2)")

        df = df.Define("gen_leps_status1_p",   "FCCAnalyses::MCParticle::get_p(gen_leps_status1)")
        df = df.Define("gen_leps_status1_px",  "FCCAnalyses::MCParticle::get_px(gen_leps_status1)")
        df = df.Define("gen_leps_status1_py",  "FCCAnalyses::MCParticle::get_py(gen_leps_status1)")
        df = df.Define("gen_leps_status1_pz",  "FCCAnalyses::MCParticle::get_pz(gen_leps_status1)")
        df = df.Define("gen_leps_status1_pt",  "FCCAnalyses::MCParticle::get_pt(gen_leps_status1)")
        df = df.Define("gen_leps_status1_eta", "FCCAnalyses::MCParticle::get_eta(gen_leps_status1)")
        df = df.Define("gen_leps_status1_phi", "FCCAnalyses::MCParticle::get_phi(gen_leps_status1)")
        df = df.Define("gen_leps_status1_theta","FCCAnalyses::MCParticle::get_theta(gen_leps_status1)")

        df = df.Define("gen_neutrinos_status1_px",  "FCCAnalyses::MCParticle::get_px(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_py",  "FCCAnalyses::MCParticle::get_py(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_pz",  "FCCAnalyses::MCParticle::get_pz(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_pt",  "FCCAnalyses::MCParticle::get_pt(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_eta", "FCCAnalyses::MCParticle::get_eta(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_phi", "FCCAnalyses::MCParticle::get_phi(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_theta","FCCAnalyses::MCParticle::get_theta(gen_neutrinos_status1)")

        df = df.Define("gen_lightquarks_fromele_p",  "FCCAnalyses::MCParticle::get_p(gen_lightquarks_fromele)")
        df = df.Define("gen_lightquarks_fromele_px", "FCCAnalyses::MCParticle::get_px(gen_lightquarks_fromele)")
        df = df.Define("gen_lightquarks_fromele_py", "FCCAnalyses::MCParticle::get_py(gen_lightquarks_fromele)")
        df = df.Define("gen_lightquarks_fromele_pz", "FCCAnalyses::MCParticle::get_pz(gen_lightquarks_fromele)")
        df = df.Define("gen_lightquarks_fromele_e",  "FCCAnalyses::MCParticle::get_e(gen_lightquarks_fromele)")

        df = df.Define("gen_lightquarks",
            "FCCAnalyses::MCParticle::sel_lightQuarks(true)(status2parts)")
        df = df.Define("gen_lightquarks_px", "FCCAnalyses::MCParticle::get_px(gen_lightquarks)")
        df = df.Define("gen_lightquarks_py", "FCCAnalyses::MCParticle::get_py(gen_lightquarks)")
        df = df.Define("gen_lightquarks_pz", "FCCAnalyses::MCParticle::get_pz(gen_lightquarks)")
        df = df.Define("gen_lightquarks_e",  "FCCAnalyses::MCParticle::get_e(gen_lightquarks)")

        # ── kinfit input branches ──────────────────────────────────────────────
        df = df.Define("lep_p4_gen",
            "FCCAnalyses::WWFunctions::lep_p4_gen(gen_leps_status1_pt, gen_leps_status1_eta, gen_leps_status1_phi)")
        df = df.Define("nu_p4_gen",
            "FCCAnalyses::WWFunctions::nu_p4_gen(gen_neutrinos_status1_pt, gen_neutrinos_status1_eta, gen_neutrinos_status1_phi)")

        df = df.Define("lep_p_resp",      "Isolep_p / gen_leps_status1_p[0]")
        df = df.Define("met_p_resp",      "missing_p / gen_neutrinos_status1_p[0]")
        df = df.Define("lep_theta_resol", "Isolep_theta - gen_leps_status1_theta[0]")
        df = df.Define("lep_phi_resol",   "TVector2::Phi_mpi_pi(Isoleps_p4_reco.Phi() - lep_p4_gen.Phi())")
        df = df.Define("met_theta_resol", "missing_p_theta - gen_neutrinos_status1_theta[0]")
        df = df.Define("met_phi_resol",   "TVector2::Phi_mpi_pi(missing_p_p4.Phi() - nu_p4_gen.Phi())")

        df = df.Define("gen_lightquarks_p4",
            "FCCAnalyses::WWFunctions::build_p4(gen_lightquarks_px, gen_lightquarks_py, gen_lightquarks_pz, gen_lightquarks_e)")
        df = df.Define("matched_genjets_qq",
            "FCCAnalyses::WWFunctions::matchJets2(jet1, jet2, gen_lightquarks_p4[0], gen_lightquarks_p4[1])")
        df = df.Define("jet1_p_resp", "jet1_p / matched_genjets_qq.first.P()")
        df = df.Define("jet2_p_resp", "jet2_p / matched_genjets_qq.second.P()")

        df = df.Define("gen_lightquarks_fromele_p4",
            "FCCAnalyses::WWFunctions::build_p4(gen_lightquarks_fromele_px, gen_lightquarks_fromele_py, gen_lightquarks_fromele_pz, gen_lightquarks_fromele_e)")
        df = df.Define("matched_genjets",
            "FCCAnalyses::WWFunctions::matchJets2(jet1, jet2, gen_lightquarks_fromele_p4[0], gen_lightquarks_fromele_p4[1])")
        df = df.Define("jet1_matched_p4", "matched_genjets.first")
        df = df.Define("jet2_matched_p4", "matched_genjets.second")
        df = df.Define("jet1_theta_resol", "jet1_theta - jet1_matched_p4.Theta()")
        df = df.Define("jet2_theta_resol", "jet2_theta - jet2_matched_p4.Theta()")
        df = df.Define("jet1_phi_resol",   "TVector2::Phi_mpi_pi(jet1.Phi() - jet1_matched_p4.Phi())")
        df = df.Define("jet2_phi_resol",   "TVector2::Phi_mpi_pi(jet2.Phi() - jet2_matched_p4.Phi())")

        df = df.Define("px_tot_gen",
            "gen_leps_status1_px[0] + gen_neutrinos_status1_px[0] + ROOT::VecOps::Sum(gen_lightquarks_fromele_px)")
        df = df.Define("py_tot_gen",
            "gen_leps_status1_py[0] + gen_neutrinos_status1_py[0] + ROOT::VecOps::Sum(gen_lightquarks_fromele_py)")
        df = df.Define("pz_tot_gen",
            "gen_leps_status1_pz[0] + gen_neutrinos_status1_pz[0] + ROOT::VecOps::Sum(gen_lightquarks_fromele_pz)")

        df = df.Define("m_gen_lnuqq",
            "FCCAnalyses::WWFunctions::m_gen_lnuqq(gen_leps_status2_p, gen_lightquarks_fromele_p, gen_leps_status1_px, gen_leps_status1_py, gen_leps_status1_pz, gen_leps_status1_p, gen_neutrinos_status1_px, gen_neutrinos_status1_py, gen_neutrinos_status1_pz, gen_neutrinos_status1_p, gen_lightquarks_fromele_px, gen_lightquarks_fromele_py, gen_lightquarks_fromele_pz, gen_lightquarks_fromele_e)")
        df = df.Define("m_gen_lnuqq_minus_ecm", "m_gen_lnuqq - FCCAnalyses::WWFunctions::ECM")

        return df

    def output():
        return all_branches
