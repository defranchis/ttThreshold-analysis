# Shared analyser building blocks for treemaker_lnuqq_step{1,2}.py.
# Each helper takes an RDataFrame and returns the extended frame, except
# `cluster_jets` which also returns the ExclusiveJetClusteringHelper instance
# (the entry script must keep it as a module-level global so the framework
# can register the jet collections).
import re
from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper

AVAILABLE_ECM = ['157', '160', '163']

# FSR dressing parameters (see project_lep_p_resp_fsr_dressing.md).
# Tuned 2026-05-05 from photon-investigation analysis: dR<0.1 around the iso
# lepton, E_γ>0.5 GeV. Recovers ~30% of deep-tail (lep_p_resp<0.88) events
# with negligible core impact and 0.1-0.2% overshoot rate.
FSR_DRESS_DR_MAX = 0.1
FSR_DRESS_E_MIN  = 0.5

def parse_ecm(name):
    m = re.search(r'_ecm(\d+)', name)
    if not m:
        raise ValueError(f"Cannot parse ecm from sample name: {name}")
    return int(m.group(1))


# ── selection ────────────────────────────────────────────────────────────────
def select_isoleps(df):
    df = df.Alias("Muon0",     "Muon#0.index")
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
    return df


def apply_channel_filter(df, channel):
    if channel == "had":
        df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 0",
                       "channel: 0 isolated leptons (had)")
    elif channel == "semihad":
        df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 1",
                       "channel: 1 isolated lepton (semihad)")
    else:
        df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 2",
                       "channel: 2 isolated leptons (lep)")

    df = df.Define("Isoleps_bare", "ROOT::VecOps::Concatenate(muons_sel_iso, electrons_sel_iso)")

    # FSR dressing: absorb nearby photons (RPs with type==22) into each iso
    # lepton; the same photons must be removed from the jet input.
    df = df.Define("Isoleps",
        f"FCCAnalyses::WWFunctions::dress_isoleps(Isoleps_bare, ReconstructedParticles, "
        f"{FSR_DRESS_DR_MAX}, {FSR_DRESS_E_MIN})")
    df = df.Define("FSR_photons",
        f"FCCAnalyses::WWFunctions::dressed_photons(Isoleps_bare, ReconstructedParticles, "
        f"{FSR_DRESS_DR_MAX}, {FSR_DRESS_E_MIN})")

    df = df.Define("ReconstructedParticlesNoMuons",
        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, muons_sel_iso)")
    df = df.Define("ReconstructedParticlesNoMuNoEl",
        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoMuons, electrons_sel_iso)")
    df = df.Define("ReconstructedParticlesNoMuNoElNoFSR",
        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoMuNoEl, FSR_photons)")
    return df


def cluster_jets(df, channel):
    nJets = 2 if channel == "semihad" else 4
    # Cluster on the FSR-dressed reduced collection so absorbed photons don't
    # double-count on the hadronic side.
    helper = ExclusiveJetClusteringHelper("ReconstructedParticlesNoMuNoElNoFSR", nJets)
    df = helper.define(df)
    df = df.Define("jets_p4",
        f"JetConstituentsUtils::compute_tlv_jets({helper.jets})")
    # ExclusiveJetClusteringHelper passes arg_sorted=1 to clustering_ee_kt → jets
    # are returned E-sorted (descending), not pT-sorted. jet1 = highest-E, jet2 =
    # second-highest-E. For WW at √s ≈ 2·mW, leading-E ≈ leading-pT in most
    # events, but they can disagree for forward/asymmetric configurations.
    df = df.Define("jet1", "jets_p4[0]")
    df = df.Define("jet2", "jets_p4[1]")
    df = df.Define("n_reco_jets", "(int)jets_p4.size()")
    df = df.Filter("n_reco_jets == 2", "exactly 2 reco jets")
    df = df.Define("d_12", "JetClusteringUtils::get_exclusive_dmerge(_jet, 1)")
    df = df.Define("d_32", "JetClusteringUtils::get_exclusive_dmerge(_jet, 2)")
    return df, helper


# ── reco kinematics ──────────────────────────────────────────────────────────
def define_reco_lep_met(df):
    df = df.Define("Isoleps_p4_reco", "FCCAnalyses::ReconstructedParticle::get_tlv(Isoleps, 0)")
    df = df.Define("missing_p_p4",    "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET, 0)")
    df = df.Define("n_lep_reco",      "(int)Isoleps.size()")

    df = df.Define("reco_lep_p",        "Isoleps_p4_reco.P()")
    df = df.Define("reco_lep_pt",       "Isoleps_p4_reco.Pt()")
    df = df.Define("reco_lep_eta",      "Isoleps_p4_reco.Eta()")
    df = df.Define("reco_lep_theta",    "Isoleps_p4_reco.Theta()")
    df = df.Define("reco_lep_phi",      "Isoleps_p4_reco.Phi()")
    df = df.Define("reco_lep_costheta", "Isoleps_p4_reco.CosTheta()")

    df = df.Define("reco_met_p",        "missing_p_p4.P()")
    df = df.Define("reco_met_pt",       "missing_p_p4.Pt()")
    df = df.Define("reco_met_theta",    "missing_p_p4.Theta()")
    df = df.Define("reco_met_phi",      "missing_p_p4.Phi()")
    df = df.Define("reco_met_eta",      "missing_p_p4.Eta()")
    df = df.Define("reco_met_costheta", "missing_p_p4.CosTheta()")

    df = df.Define("Wlep_reco",
        "FCCAnalyses::WWFunctions::sum_p4({Isoleps_p4_reco, missing_p_p4})")
    return df


def define_reco_jets_kinematics(df):
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

    # Reco jets treated as massless: jet mass is dominated by clustering /
    # detector, not the parton mass; matches kinfit's (p,theta,phi) jet param.
    df = df.Define("jet1_massless", "FCCAnalyses::WWFunctions::tlv_setmass(jet1, 0.)")
    df = df.Define("jet2_massless", "FCCAnalyses::WWFunctions::tlv_setmass(jet2, 0.)")
    df = df.Define("Whad_reco",
        "FCCAnalyses::WWFunctions::sum_p4({jet1_massless, jet2_massless})")
    return df


# ── gen-level (W decay products via parent==e±, since W is not in MC history) ─
def select_gen_fromele(df):
    df = df.Alias("Particle0", "Particle#0.index")
    df = df.Define("gen_leps_fromele",
        "FCCAnalyses::WWFunctions::sel_genleps_fromele(13)(Particle, Particle0)")
    df = df.Define("gen_neutrinos_fromele",
        "FCCAnalyses::WWFunctions::sel_genleps_fromele(14)(Particle, Particle0)")
    df = df.Define("gen_lightquarks_fromele",
        "FCCAnalyses::WWFunctions::sel_lightQuarks_fromele()(Particle, Particle0)")

    df = df.Filter("gen_leps_fromele.size() == 1",
                   "gen: exactly 1 muon fromele (W daughter)")
    df = df.Filter("gen_neutrinos_fromele.size() == 1",
                   "gen: exactly 1 nu_mu fromele (W daughter)")
    df = df.Filter("gen_lightquarks_fromele.size() == 2",
                   "gen: exactly 2 light quarks fromele (W daughters)")
    return df


def define_beam_kinematics(df):
    """Beam e± at the two relevant chain depths:
      depth=1 (post-BES, pre-ISR) → m(ee)−ECM gives BES;
      depth=2 (post-ISR, into hard process) → (depth1 − depth2) gives the
      total ISR 4-momentum."""
    df = df.Define("gen_beams",
        "FCCAnalyses::WWFunctions::sel_beam_electrons()(Particle, Particle0)")
    df = df.Filter("gen_beams.size() == 2",
                   "gen: exactly 2 post-BES beam e± (depth=1)")
    df = df.Define("gen_beams_tlv",
        "FCCAnalyses::MCParticle::get_tlv(gen_beams)")
    df = df.Define("gen_ee_p4",
        "FCCAnalyses::WWFunctions::sum_p4({gen_beams_tlv[0], gen_beams_tlv[1]})")
    df = df.Define("gen_ee_m_minus_ecm", "gen_ee_p4.M() - FCCAnalyses::WWFunctions::ECM")
    # Longitudinal momentum of the depth-1 e+e- system = E+ − E−, the BES
    # asymmetry between the two beams. Independent of m(ee)−ECM (sum vs diff of
    # independent Gaussians) but same width.
    df = df.Define("gen_ee_pz", "gen_ee_p4.Pz()")

    df = df.Define("gen_post_isr_e",
        "FCCAnalyses::WWFunctions::sel_post_isr_electrons()(Particle, Particle0)")
    df = df.Filter("gen_post_isr_e.size() == 2",
                   "gen: exactly 2 post-ISR e± (depth=2)")
    df = df.Define("gen_post_isr_e_tlv",
        "FCCAnalyses::MCParticle::get_tlv(gen_post_isr_e)")
    df = df.Define("gen_ee_postisr_p4",
        "FCCAnalyses::WWFunctions::sum_p4({gen_post_isr_e_tlv[0], gen_post_isr_e_tlv[1]})")
    df = df.Define("gen_isr_p4", "gen_ee_p4 - gen_ee_postisr_p4")
    df = df.Define("gen_isr_px", "gen_isr_p4.Px()")
    df = df.Define("gen_isr_py", "gen_isr_p4.Py()")
    df = df.Define("gen_isr_pz", "gen_isr_p4.Pz()")
    # m(WW) − m(ee) — pure ISR mass-loss, with BES variance subtracted off vs
    # the older m(WW) − ECM. In the no-ISR limit it is exactly 0; with ISR it
    # is < 0 (energy loss to the ISR photons).
    df = df.Define("gen_WW_m_minus_m_ee", "gen_WW_m - gen_ee_p4.M()")
    return df


def define_gen_kinematics(df):
    df = df.Define("gen_leps_fromele_tlv",
        "FCCAnalyses::MCParticle::get_tlv(gen_leps_fromele)")
    df = df.Define("gen_neutrinos_fromele_tlv",
        "FCCAnalyses::MCParticle::get_tlv(gen_neutrinos_fromele)")
    df = df.Define("gen_lightquarks_fromele_tlv",
        "FCCAnalyses::MCParticle::get_tlv(gen_lightquarks_fromele)")

    df = df.Define("lep_p4_gen", "gen_leps_fromele_tlv[0]")
    df = df.Define("nu_p4_gen",  "gen_neutrinos_fromele_tlv[0]")
    df = df.Define("gen_q1_p4",  "gen_lightquarks_fromele_tlv[0]")
    df = df.Define("gen_q2_p4",  "gen_lightquarks_fromele_tlv[1]")

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

    # Gen quarks keep their native (MC) masses; reco jets are massless.
    df = df.Define("Wlep_gen", "FCCAnalyses::WWFunctions::sum_p4({lep_p4_gen, nu_p4_gen})")
    df = df.Define("Whad_gen", "FCCAnalyses::WWFunctions::sum_p4({gen_q1_p4, gen_q2_p4})")
    df = df.Define("Wlnuqq_gen",
        "FCCAnalyses::WWFunctions::sum_p4({lep_p4_gen, nu_p4_gen, gen_q1_p4, gen_q2_p4})")

    df = df.Define("gen_Wlep_m",        "Wlep_gen.M()")
    df = df.Define("gen_Wlep_p",        "Wlep_gen.P()")
    df = df.Define("gen_Wlep_pt",       "Wlep_gen.Pt()")
    df = df.Define("gen_Wlep_px",       "Wlep_gen.Px()")
    df = df.Define("gen_Wlep_py",       "Wlep_gen.Py()")
    df = df.Define("gen_Wlep_pz",       "Wlep_gen.Pz()")
    df = df.Define("gen_Wlep_costheta", "Wlep_gen.CosTheta()")
    df = df.Define("gen_Wlep_phi",      "Wlep_gen.Phi()")

    df = df.Define("gen_Whad_m",        "Whad_gen.M()")
    df = df.Define("gen_Whad_p",        "Whad_gen.P()")
    df = df.Define("gen_Whad_pt",       "Whad_gen.Pt()")
    df = df.Define("gen_Whad_px",       "Whad_gen.Px()")
    df = df.Define("gen_Whad_py",       "Whad_gen.Py()")
    df = df.Define("gen_Whad_pz",       "Whad_gen.Pz()")
    df = df.Define("gen_Whad_costheta", "Whad_gen.CosTheta()")
    df = df.Define("gen_Whad_phi",      "Whad_gen.Phi()")

    df = df.Define("gen_WW_m",               "Wlnuqq_gen.M()")
    df = df.Define("gen_WW_m_minus_ecm",     "gen_WW_m - FCCAnalyses::WWFunctions::ECM")
    df = df.Define("gen_WW_px",              "Wlnuqq_gen.Px()")
    df = df.Define("gen_WW_py",              "Wlnuqq_gen.Py()")
    df = df.Define("gen_WW_pz",              "Wlnuqq_gen.Pz()")
    df = df.Define("gen_WW_p_imbalance_tot", "Wlnuqq_gen.P()")
    return df


def match_jets_to_quarks(df):
    df = df.Define("matched_gen_quarks",
        "FCCAnalyses::WWFunctions::matchJets2(jet1, jet2, gen_q1_p4, gen_q2_p4)")
    df = df.Define("jet1_matched_q_p4", "matched_gen_quarks.first")
    df = df.Define("jet2_matched_q_p4", "matched_gen_quarks.second")

    df = df.Define("jet1_matched_q_dR", "(double)jet1.DeltaR(jet1_matched_q_p4)")
    df = df.Define("jet2_matched_q_dR", "(double)jet2.DeltaR(jet2_matched_q_p4)")

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
    return df


# ── reco W / WW system (needs Wlep_reco AND Whad_reco) ───────────────────────
def define_reco_W_WW(df):
    df = df.Define("reco_Wlep_m",        "Wlep_reco.M()")
    df = df.Define("reco_Wlep_p",        "Wlep_reco.P()")
    df = df.Define("reco_Wlep_pt",       "Wlep_reco.Pt()")
    df = df.Define("reco_Wlep_px",       "Wlep_reco.Px()")
    df = df.Define("reco_Wlep_py",       "Wlep_reco.Py()")
    df = df.Define("reco_Wlep_pz",       "Wlep_reco.Pz()")
    df = df.Define("reco_Wlep_costheta", "Wlep_reco.CosTheta()")
    df = df.Define("reco_Wlep_phi",      "Wlep_reco.Phi()")

    df = df.Define("reco_Whad_m",        "Whad_reco.M()")
    df = df.Define("reco_Whad_p",        "Whad_reco.P()")
    df = df.Define("reco_Whad_pt",       "Whad_reco.Pt()")
    df = df.Define("reco_Whad_px",       "Whad_reco.Px()")
    df = df.Define("reco_Whad_py",       "Whad_reco.Py()")
    df = df.Define("reco_Whad_pz",       "Whad_reco.Pz()")
    df = df.Define("reco_Whad_costheta", "Whad_reco.CosTheta()")
    df = df.Define("reco_Whad_phi",      "Whad_reco.Phi()")

    df = df.Define("WW_reco", "(Wlep_reco + Whad_reco)")
    df = df.Define("reco_WW_m",               "WW_reco.M()")
    df = df.Define("reco_WW_m_minus_ecm",     "reco_WW_m - FCCAnalyses::WWFunctions::ECM")
    df = df.Define("reco_WW_px",              "WW_reco.Px()")
    df = df.Define("reco_WW_py",              "WW_reco.Py()")
    df = df.Define("reco_WW_pz",              "WW_reco.Pz()")
    df = df.Define("reco_WW_p_imbalance_tot", "WW_reco.P()")

    df = df.Define("deltaM",
        "FCCAnalyses::WWFunctions::deltaM(n_lep_reco, n_reco_jets, Wlep_reco, Whad_reco)")
    return df


# ── resolutions / responses (cross-level: reco − gen, reco / gen) ───────────
def define_resolutions(df):
    df = df.Define("lep_p_resp",         "reco_lep_p / gen_lep_p")
    df = df.Define("lep_theta_resol",    "reco_lep_theta - gen_lep_theta")
    df = df.Define("lep_phi_resol",      "TVector2::Phi_mpi_pi(reco_lep_phi - gen_lep_phi)")
    df = df.Define("lep_eta_resol",      "reco_lep_eta - gen_lep_eta")
    df = df.Define("lep_costheta_resol", "reco_lep_costheta - gen_lep_costheta")

    df = df.Define("met_p_resp",         "reco_met_p / gen_nu_p")
    df = df.Define("met_theta_resol",    "reco_met_theta - gen_nu_theta")
    df = df.Define("met_phi_resol",      "TVector2::Phi_mpi_pi(reco_met_phi - gen_nu_phi)")
    df = df.Define("met_eta_resol",      "reco_met_eta - gen_nu_eta")
    df = df.Define("met_costheta_resol", "reco_met_costheta - gen_nu_costheta")

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

    df = df.Define("Wlep_m_resol", "reco_Wlep_m - gen_Wlep_m")
    df = df.Define("Wlep_p_resol", "reco_Wlep_p - gen_Wlep_p")
    df = df.Define("Whad_m_resol", "reco_Whad_m - gen_Whad_m")
    df = df.Define("Whad_p_resol", "reco_Whad_p - gen_Whad_p")
    df = df.Define("WW_m_resol",   "reco_WW_m - gen_WW_m")
    df = df.Define("WW_px_resol",  "reco_WW_px - gen_WW_px")
    df = df.Define("WW_py_resol",  "reco_WW_py - gen_WW_py")
    df = df.Define("WW_pz_resol",  "reco_WW_pz - gen_WW_pz")
    return df


# ── kinematic fit (step2 only) ───────────────────────────────────────────────
def run_kinfit(df, free_gw=False):
    free_gw = "true" if free_gw else "false"
    df = df.Define("kinfit",
        "FCCAnalyses::WWFunctions::kinFit("
        "reco_jet1_p, reco_jet1_theta, reco_jet1_phi,"
        "reco_jet2_p, reco_jet2_theta, reco_jet2_phi,"
        "reco_lep_p,  reco_lep_theta,  reco_lep_phi,"
        f"reco_met_p, reco_met_theta, reco_met_phi, {free_gw})")

    for tag in ["mW","gW",
                "s1","s2","sl","sn",
                "t1","t2","tn","tl",
                "p1","p2","pn","pl",
                "bes_m_minus_ecm","bes_pz",
                "chi2","chi2_ndof","valid","valid_loose","status","edm",
                "winner_pass","n_passes_run","priors_swapped"]:
        df = df.Define(f"kinfit_{tag}", f"kinfit.{tag}")

    # Postfit scalars projected from TLVs in KinFitResult.
    for obj, src in [("jet1","j1"), ("jet2","j2"), ("lep","lep"), ("nu","nu")]:
        df = df.Define(f"kinfit_{obj}_p",     f"kinfit.{src}.P()")
        df = df.Define(f"kinfit_{obj}_pt",    f"kinfit.{src}.Pt()")
        df = df.Define(f"kinfit_{obj}_theta", f"kinfit.{src}.Theta()")
        df = df.Define(f"kinfit_{obj}_phi",   f"kinfit.{src}.Phi()")

    df = df.Define("Wlep_kinfit", "kinfit.lep + kinfit.nu")
    df = df.Define("Whad_kinfit", "kinfit.j1  + kinfit.j2")
    df = df.Define("WW_kinfit",   "Wlep_kinfit + Whad_kinfit")

    for W, src in [("Wlep","Wlep_kinfit"), ("Whad","Whad_kinfit")]:
        df = df.Define(f"kinfit_{W}_m",  f"{src}.M()")
        df = df.Define(f"kinfit_{W}_p",  f"{src}.P()")
        df = df.Define(f"kinfit_{W}_pt", f"{src}.Pt()")
        df = df.Define(f"kinfit_{W}_px", f"{src}.Px()")
        df = df.Define(f"kinfit_{W}_py", f"{src}.Py()")
        df = df.Define(f"kinfit_{W}_pz", f"{src}.Pz()")

    df = df.Define("kinfit_WW_m",               "WW_kinfit.M()")
    df = df.Define("kinfit_WW_m_minus_ecm",     "kinfit_WW_m - FCCAnalyses::WWFunctions::ECM")
    df = df.Define("kinfit_WW_px",              "WW_kinfit.Px()")
    df = df.Define("kinfit_WW_py",              "WW_kinfit.Py()")
    df = df.Define("kinfit_WW_pz",              "WW_kinfit.Pz()")
    df = df.Define("kinfit_WW_p_imbalance_tot", "WW_kinfit.P()")
    return df


