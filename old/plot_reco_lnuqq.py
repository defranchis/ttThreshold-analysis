import ROOT
import os, sys
ROOT.EnableImplicitMT()  
ROOT.gROOT.SetBatch(True)
import datetime
date = datetime.date.today().isoformat()

ROOT.EnableImplicitMT()
lepflav=sys.argv[1]
filename = "outputs/treemaker/lnuqq/semihad/wzp6_ee_munumuqq_noCut_ecm160.root" 
treename = "events"


f = ROOT.TFile.Open(filename)
tree = f.Get(treename)

df = ROOT.RDataFrame(treename, filename)


outdir = f"/eos/user/a/anmehta/www/FCC_WW/plots_reco/{date}_{lepflav}/" #_inclSample/" # el/"
os.makedirs(outdir, exist_ok=True)
os.system('cp ~/public/index.php %s/'%outdir)
#cols = df.GetColumnNames();
#for c in cols:
#    print(c)
#print("done")

numeric_types = (
    "Float_t", "Double_t",
    "Int_t", "UInt_t",
    "Long64_t", "ULong64_t",
    "Bool_t"
)
binning = {
    "m_lnu_status2":(50, 0, 150),
    "m_qq_fromele":(50, 0, 150),
    "m_jj":(50, 0, 100),
    "GP_p":   (50, 0, 100),
    "gen_leps_status1_p":   (60, 0, 150),
    "theta":(16,-3.2,3.2),
    "phi":(16,-3.2,3.2),
    "ngen_taus_status2": (3,-0.5,2.5),
    "ngen_leps_status1": (5,-0.5,4.5),
    "Nlep": (5,-0.5,4.5),
#    "gen_leps_status1_phi":
    "Genjets_kt_e":   (50, 0, 100),
    "Genjets_kt_m":   (50, 0, 50),

}


def get_binning(name, leaf):
    if name in binning:
        return binning[name]

    minV = leaf.GetMinimum()
    maxV = leaf.GetMaximum()

    if "jj" in name:
        minV, maxV = 0, 200
    elif "res_jet" in name:
        minV, maxV = -0.5, 0.5
    elif "res" in name:
        minV, maxV = -0.1, 0.1
    elif "sumP" in name:
        minV, maxV = -10, 10
    elif "_costheta" in name:
        minV, maxV = -1.0, 1.0
    elif "_theta" in name:
        minV, maxV = -0, 3.2
    elif "_dcostheta" in name:
        minV, maxV = -1,1
    elif "_d" in name:
        minV, maxV = -1,1
    elif "phi" in name:
        minV, maxV = -3.2,3.2
    # fallback if ROOT gives nonsense
#    if minV == maxV:
 #       minV, maxV = 0, 1

    return (100, minV, maxV)


hists = []

for br in tree.GetListOfBranches():
    name = br.GetName()
    leaf = br.GetLeaf(name)
    nbins, minV, maxV = get_binning(name, leaf)
    if not leaf:
        print(f"Skipping {name} (no leaf)")
        continue
    if leaf.GetLen() > 1 or "[" in leaf.GetTypeName():
        print(f"Vector branch: {name}")

        # Flatten vector branch
        df_vec = df.Define(f"{name}_flat", name)
        
        h = df_vec.Histo1D(
            (f"h_{name}", name, 100, minV, maxV), #leaf.GetMinimum(), leaf.GetMaximum()),
            f"{name}_flat"
        )
        hists.append(h)
    else:
        h = df.Histo1D(
            (f"h_{name}", name, 100, minV, maxV), #leaf.GetMinimum(), leaf.GetMaximum()),
            name
        )
        #print(name,h.Integral())
        hists.append(h)


for h in hists:
    c = ROOT.TCanvas("c"+h.GetName(),h.GetName(), 800, 600)
    c.Clear()
    #if "sum" in h.GetName(): c.SetLogy();
    h.Draw()
    c.SaveAs(f"{outdir}/{h.GetName()}.png")
