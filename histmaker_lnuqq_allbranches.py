from treemaker_lnuqq import all_branches

processList = {
    "wzp6_ee_munumuqq_noCut_ecm160": {},
    "wzp6_ee_munumuqq_noCut_ecm163": {},
    "wzp6_ee_munumuqq_noCut_ecm157": {},
}

prodTag    = None
procDict   = "FCCee_procDict_winter2023_IDEA.json"
inputDir   = "outputs/treemaker/lnuqq/semihad/"
outputDir  = "outputs/histmaker/lnuqq/semihad/"
nCPUS      = -1
doScale    = False
intLumi    = 1


def _binning(var):
    if var.startswith("n") and ("jets" in var or "lep" in var or "parton" in var):
        return (10, -0.5, 9.5)
    if "kinfit_valid" == var:
        return (3, -0.5, 2.5)
    if "kinfit_chi2" == var:
        return (100, 0, 50)
    if var in ("kinfit_mW", "kinfit_mWlep", "kinfit_mWhad",
               "reco_moff", "reco_mon",
               "truth_lnuqq_mon", "truth_lnuqq_moff",
               "truth_lnuqq_qqfromele_mon", "truth_lnuqq_qqfromele_moff",
               "m_gen_lnuqq", "m_lnu_status1", "m_lnu_status2",
               "m_qq_status2", "m_qq_fromele", "m_iso_lnu",
               "m_iso_lnuexcljj", "mlnu_plus_mjj_reco",
               "mlnu_plus_mqq_status2_truth", "mlnu_plus_mqq_fromele_truth",
               "jet1_mass", "jet2_mass", "lep_res"):
        return (100, 0, 200)
    if var == "kinfit_gW":
        return (100, 0, 5)
    if var in ("kinfit_s1", "kinfit_s2", "kinfit_sl", "kinfit_sn"):
        return (100, 0.5, 2.0)
    if var in ("kinfit_t1", "kinfit_t2", "kinfit_tn",
               "kinfit_p1", "kinfit_p2", "kinfit_pn"):
        return (100, -5, 5)
    if "d_12" == var:
        return (100, 0, 5000)
    if var.endswith("_dcostheta") or "dcostheta" in var:
        return (100, -1, 1)
    if var.endswith("_costheta") or "costheta" in var:
        return (100, -1.0, 1.0)
    if var.endswith("_dtheta") or "dtheta" in var or "deta" in var or "dphi" in var:
        return (100, -1, 1)
    if var.endswith("_dR"):
        return (100, 0, 6)
    if var.endswith("_phi") or "phi" in var:
        return (100, -3.2, 3.2)
    if var.endswith("_theta") or "theta" in var:
        return (100, 0, 3.2)
    if var.endswith("_eta"):
        return (100, -5, 5)
    if "res_jet" in var or var.startswith("res_"):
        return (100, -0.5, 0.5)
    if var.startswith("diff_") or var.startswith("deltaM") or var.startswith("sumP"):
        return (100, -50, 50)
    if "sumP" in var or var in ("sumPx", "sumPy", "sumPz", "sumPx_gen", "sumPy_gen", "sumPz_gen"):
        return (100, -50, 50)
    if var.endswith("_pt") or "_pt_" in var or var in ("sumPt", "sumPt_gen", "missing_pt",
                                                         "Whad_gen_pt", "Wlep_gen_pt",
                                                         "Whad_reco_pt", "Wlep_reco_pt",
                                                         "Isolep_pt"):
        return (100, 0, 100)
    # momenta (p)
    return (100, 0, 200)


def build_graph(df, dataset):
    results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")

    cols = [str(c) for c in df.GetColumnNames()]

    for var in all_branches:
        if var not in cols:
            continue
        binning = _binning(var)
        results.append(df.Histo1D(("no_cut_" + var, "", *binning), var))

    return results, weightsum
