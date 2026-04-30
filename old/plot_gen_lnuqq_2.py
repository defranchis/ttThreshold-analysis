import ROOT
import os
ROOT.EnableImplicitMT()  
ROOT.gROOT.SetBatch(True)
import datetime
date = datetime.date.today().isoformat()

df = ROOT.RDataFrame("events", "outputs/treemaker/lnuqq/acceptance_atleast1_genlep_status1/wzp6_ee_lnuqq_ecm160.root")
outdir = f"/eos/user/a/anmehta/www/FCC_WW/plots_gen/{date}/" # el/"
os.makedirs(outdir, exist_ok=True)
os.system('cp ~/public/index.php %s/'%outdir)


#df_sel = df.Filter("nJet >= 2", "At least two jets")
numeric_types = (
    "Float_t", "Double_t",
    "Int_t", "UInt_t",
    "Long64_t", "ULong64_t",
    "Bool_t"
)
binning = {
    "m_lnu":(50, 0, 100),
    "m_qq":(50, 0, 100),
    "m_jj":(50, 0, 100),
    "GP_p":   (50, 0, 100),
    "gen_leps_status1_p":   (60, 0, 150),
    "theta":(16,-3.2,3.2),
    "phi":(16,-3.2,3.2),
    "ngen_taus_status2": (3,-0.5,2.5),
    "ngen_leps_status1": (3,-0.5,2.5),
    "Nlep": (9,-0.5,8.5),
#    "gen_leps_status1_phi":
    "Genjets_kt_e":   (50, 0, 100),
    "Genjets_kt_m":   (50, 0, 50),

}
default_binning = (50, 0, 100)  # auto-range
def is_numeric_scalar(col):
    t = df.GetColumnType(col)
    return t in numeric_types

def is_vector(col):
    return "RVec" in df.GetColumnType(col)

histos = {}
df2 = df  # will accumulate Defines
columns = df.GetColumnNames()

for col in columns:
    hname = f"h_{col}"
    title = f"{col};{col};Entries"
    ctype = df.GetColumnType(col)
    if "pdgId" in hname or "status" in hname or "charge" in hname: continue    
    if "RVec" in ctype or ctype in ["Float_t", "Double_t", "Int_t"]:
        if "phi" in hname:
            bins= binning.get("phi", default_binning)
        elif "theta"  in hname:
            bins= binning.get("theta", default_binning)
        elif col.starts_with("n"):
            bins= binning.get("Nlep", default_binning)
        else:
            bins = binning.get(col, default_binning)
        histos[col] = df.Histo1D(
            (f"h_{col}", f"{col};{col};Entries", *bins),
            col
        )
for col, h in histos.items():
    c = ROOT.TCanvas(f"c_{col}", "", 800, 600)
    h.Draw()
    c.SaveAs(f"{outdir}/{col}.png")
    c.SaveAs(f"{outdir}/{col}.pdf")
