# Step 1 of 2 — produce only the branches needed by fit_dcb_resolutions.py.
# Run fit_dcb_resolutions.py on the output before running treemaker_lnuqq_step2.py.
# Does NOT include WWKinReco.h / outputs/response/functions/, so it builds
# without DCB params headers.
import ROOT
import treemaker_common as tc

processList = {
    "wzp6_ee_munumuqq_noCut_ecm160": {"fraction": 1, "crossSection": 1},
    "wzp6_ee_munumuqq_noCut_ecm157": {"fraction": 1, "crossSection": 1},
    "wzp6_ee_munumuqq_noCut_ecm163": {"fraction": 1, "crossSection": 1},
}

channel = "CHANNELNAMEHERE"
if channel not in ["lep", "semihad", "had"]:
    channel = "semihad"
print(channel)

prodTag      = "FCCee/winter2023/IDEA/"
outputDir    = "outputs/treemaker/lnuqq/step1/{}".format(channel)
includePaths = ["examples/functions.h", "WWFunctions/WWFunctions.h"]

# Branches consumed by fit_dcb_resolutions.py --kinfit-only.
all_branches = [
    "jet1_p_resp", "jet2_p_resp", "lep_p_resp", "met_p_resp",
    "jet1_theta_resol", "jet2_theta_resol", "jet1_phi_resol", "jet2_phi_resol",
    "lep_theta_resol", "lep_phi_resol",
    "met_theta_resol", "met_phi_resol",
    "gen_WW_px", "gen_WW_py", "gen_WW_pz",
    "gen_WW_m", "gen_WW_m_minus_ecm",
]

# Module-level helper, set inside analysers(); fccanalysis reads it from here
# to register the jet collections.
jetClusteringHelper = None

_dataset_iter = iter(processList.keys())

class RDFanalysis:

    def analysers(df):
        global jetClusteringHelper

        _dataset = next(_dataset_iter)
        _ecm = tc.parse_ecm(_dataset)
        print(f"[treemaker step1] dataset={_dataset}  ecm={_ecm}")
        if str(_ecm) not in tc.AVAILABLE_ECM:
            raise ValueError(f"ecm={_ecm} parsed from '{_dataset}' not in AVAILABLE_ECM={tc.AVAILABLE_ECM}")
        ROOT.gInterpreter.ProcessLine(f"FCCAnalyses::WWFunctions::ECM = {_ecm};")
        ROOT.gInterpreter.ProcessLine(
            'std::cout << "[DEBUG step1] ECM from WWFunctions = " << FCCAnalyses::WWFunctions::ECM << std::endl;')

        df = tc.select_isoleps(df)
        df = tc.apply_channel_filter(df, channel)
        df, jetClusteringHelper = tc.cluster_jets(df, channel)

        df = tc.define_reco_lep_met(df)
        df = tc.define_reco_jets_kinematics(df)

        df = tc.select_gen_fromele(df)
        df = tc.define_gen_kinematics(df)

        df = tc.define_reco_W_WW(df)
        df = tc.match_jets_to_quarks(df)
        df = tc.define_resolutions(df)

        print(f"\n[cutflow] dataset={_dataset}")
        df.Report().Print()
        print()

        return df

    def output():
        return all_branches
