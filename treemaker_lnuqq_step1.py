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
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 0", "channel: 0 isolated leptons (had)")
        elif channel == "semihad":
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 1", "channel: 1 isolated lepton (semihad)")
        else:
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 2", "channel: 2 isolated leptons (lep)")

        df = df.Define("Isoleps", "ROOT::VecOps::Concatenate(muons_sel_iso, electrons_sel_iso)")
        df = df.Define("Isoleps_p4_reco", "FCCAnalyses::ReconstructedParticle::get_tlv(Isoleps, 0)")
        df = df.Define("missing_p_p4",    "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET, 0)")

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
        df = df.Filter("jets_p4.size() == 2", "exactly 2 reco jets")

        # ── gen-level filter (W-daughter selectors via parent==e±) ─────────────
        df = df.Alias("Particle0", "Particle#0.index")

        df = df.Define("gen_leps_fromele",
            "FCCAnalyses::WWFunctions::sel_genleps_fromele(13)(Particle, Particle0)")
        df = df.Define("gen_neutrinos_fromele",
            "FCCAnalyses::WWFunctions::sel_genleps_fromele(14)(Particle, Particle0)")
        df = df.Define("gen_lightquarks_fromele",
            "FCCAnalyses::WWFunctions::sel_lightQuarks_fromele()(Particle, Particle0)")

        df = df.Filter("gen_leps_fromele.size() == 1",                 "gen: exactly 1 muon fromele (W daughter)")
        df = df.Filter("gen_neutrinos_fromele.size() == 1",            "gen: exactly 1 nu_mu fromele (W daughter)")
        df = df.Filter("gen_lightquarks_fromele.size() == 2",          "gen: exactly 2 light quarks fromele (W daughters)")

        # ── gen kinematics (all from W-daughter selectors) ─────────────────────
        df = df.Define("gen_leps_fromele_tlv",       "FCCAnalyses::MCParticle::get_tlv(gen_leps_fromele)")
        df = df.Define("gen_neutrinos_fromele_tlv",  "FCCAnalyses::MCParticle::get_tlv(gen_neutrinos_fromele)")
        df = df.Define("gen_lightquarks_fromele_tlv","FCCAnalyses::MCParticle::get_tlv(gen_lightquarks_fromele)")

        df = df.Define("lep_p4_gen", "gen_leps_fromele_tlv[0]")
        df = df.Define("nu_p4_gen",  "gen_neutrinos_fromele_tlv[0]")
        df = df.Define("gen_q1_p4",  "gen_lightquarks_fromele_tlv[0]")
        df = df.Define("gen_q2_p4",  "gen_lightquarks_fromele_tlv[1]")
        # Massless quark TLVs (same 3-momentum, E = |p|) — used for m_gen_lnuqq so the
        # kinfit constraint PDF is fitted to a mass treatment matching the kinfit's
        # massless-jet WW.M() postfit.
        df = df.Define("gen_q1_p4_massless",
            "TLorentzVector(gen_q1_p4.Px(), gen_q1_p4.Py(), gen_q1_p4.Pz(), gen_q1_p4.P())")
        df = df.Define("gen_q2_p4_massless",
            "TLorentzVector(gen_q2_p4.Px(), gen_q2_p4.Py(), gen_q2_p4.Pz(), gen_q2_p4.P())")

        # ── kinfit input branches ──────────────────────────────────────────────
        df = df.Define("lep_p_resp",      "Isoleps_p4_reco.P() / lep_p4_gen.P()")
        df = df.Define("met_p_resp",      "missing_p_p4.P()    / nu_p4_gen.P()")
        df = df.Define("lep_theta_resol", "Isoleps_p4_reco.Theta() - lep_p4_gen.Theta()")
        df = df.Define("lep_phi_resol",   "TVector2::Phi_mpi_pi(Isoleps_p4_reco.Phi() - lep_p4_gen.Phi())")
        df = df.Define("met_theta_resol", "missing_p_p4.Theta() - nu_p4_gen.Theta()")
        df = df.Define("met_phi_resol",   "TVector2::Phi_mpi_pi(missing_p_p4.Phi()   - nu_p4_gen.Phi())")

        df = df.Define("matched_gen_quarks",
            "FCCAnalyses::WWFunctions::matchJets2(jet1, jet2, gen_q1_p4, gen_q2_p4)")
        df = df.Define("jet1_matched_q_p4", "matched_gen_quarks.first")
        df = df.Define("jet2_matched_q_p4", "matched_gen_quarks.second")
        df = df.Define("jet1_p_resp",      "jet1.P() / jet1_matched_q_p4.P()")
        df = df.Define("jet2_p_resp",      "jet2.P() / jet2_matched_q_p4.P()")
        df = df.Define("jet1_theta_resol", "jet1.Theta() - jet1_matched_q_p4.Theta()")
        df = df.Define("jet2_theta_resol", "jet2.Theta() - jet2_matched_q_p4.Theta()")
        df = df.Define("jet1_phi_resol",   "TVector2::Phi_mpi_pi(jet1.Phi() - jet1_matched_q_p4.Phi())")
        df = df.Define("jet2_phi_resol",   "TVector2::Phi_mpi_pi(jet2.Phi() - jet2_matched_q_p4.Phi())")

        df = df.Define("Wlnuqq_gen",
            "FCCAnalyses::WWFunctions::sum_p4({lep_p4_gen, nu_p4_gen, gen_q1_p4_massless, gen_q2_p4_massless})")
        df = df.Define("m_gen_lnuqq",           "Wlnuqq_gen.M()")
        df = df.Define("m_gen_lnuqq_minus_ecm", "m_gen_lnuqq - FCCAnalyses::WWFunctions::ECM")
        df = df.Define("px_tot_gen", "Wlnuqq_gen.Px()")
        df = df.Define("py_tot_gen", "Wlnuqq_gen.Py()")
        df = df.Define("pz_tot_gen", "Wlnuqq_gen.Pz()")

        # Cutflow diagnostic — triggers an extra pass over the data, but prints
        # pass/all and efficiency for every named Filter() above.
        print(f"\n[cutflow] dataset={_dataset}")
        df.Report().Print()
        print()

        return df

    def output():
        return all_branches
