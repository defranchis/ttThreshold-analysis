import ROOT
from treemaker_lnuqq import all_branches

intLumi      = 1
intLumiLabel = "L = 10 ab^{-1}"
ana_tex      = "e^{+}e^{-} #rightarrow WW #rightarrow l#nu qq"
energy       = 160.0
collider     = "FCC-ee"
formats      = ["png", "pdf"]

inputDir  = "outputs/histmaker/lnuqq/"
outdir    = "outputs/plots/lnuqq/allbranches/"

plotStatUnc = True

colors = {}
colors["WW"] = ROOT.kBlue + 1

procs = {}
procs["signal"] = {"WW": ["wzp6_ee_munumuqq_noCut_ecm160"]}
procs["backgrounds"] = {}

legend = {}
legend["WW"] = "WW #rightarrow #mu#nu qq"

hists = {}

for var in all_branches:
    logy = False
    if var.startswith("n") and ("jets" in var or "lep" in var or "parton" in var):
        logy = True
    if "kinfit_valid" == var:
        logy = True

    hists["no_cut_" + var] = {
        "output":  var,
        "logy":    logy,
        "stack":   False,
        "rebin":   1,
        "xmin":    -1,
        "xmax":    -1,
        "ymin":    0 if not logy else 1,
        "ymax":    -1,
        "xtitle":  var,
        "ytitle":  "Events",
    }
