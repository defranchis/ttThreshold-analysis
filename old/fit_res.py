import ROOT
import os, sys
ROOT.EnableImplicitMT()  
ROOT.gROOT.SetBatch(True)
import datetime
date = datetime.date.today().isoformat()
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(1111) 
ROOT.EnableImplicitMT()
lepflav=sys.argv[1]
filename = "outputs/treemaker/lnuqq/semihad/wzp6_ee_munumuqq_noCut_ecm160.root" 
treename = "events"


f = ROOT.TFile.Open(filename)
tree = f.Get(treename)

df = ROOT.RDataFrame(treename, filename)


outdir = f"/eos/user/a/anmehta/www/FCC_WW/plots_reco/{date}_{lepflav}_fits/" #_inclSample/" # el/"
os.makedirs(outdir, exist_ok=True)
os.system('cp ~/public/index.php %s/'%outdir)
cols = df.GetColumnNames();
#for c in cols:
#    print(c)
#print("done")
#df_sel = df.Filter("nJet >= 2", "At least two jets")
numeric_types = (
    "Float_t", "Double_t",
    "Int_t", "UInt_t",
    "Long64_t", "ULong64_t",
    "Bool_t"
)

toPlot=["lep_res",
"missing_p_res",
"res_jet2_qq_fromele",
"res_jet1_qq_fromele",
"res_jet2_qq",
"res_jet1_qq"]

#toPlot=[ "diff_RG_m_lnu",        "diff_RG_p_lnu",	 "diff_RG_m_lnuqq",         "diff_RG_m_qq"]

hists = []

for br in tree.GetListOfBranches():
    name = br.GetName()
    leaf = br.GetLeaf(name)

    #    if name not in toPlot: continue
    if not ("res" in name or  "_d" in name): continue
    
    if not leaf:
        print(f"Skipping {name} (no leaf)")
        continue
    if leaf.GetLen() > 1 or "[" in leaf.GetTypeName():
        print(f"Vector branch: {name}")
        
        # Flatten vector branch
        df_vec = df.Define(f"{name}_flat", name)
        print(name)
        maxV=leaf.GetMaximum() if "res" not in name else 0.5
        minV=leaf.GetMinimum() if "res" not in name else -0.5
        
        h = df_vec.Histo1D(
            (f"h_{name}", name, 100, minV, maxV), #leaf.GetMinimum(), leaf.GetMaximum()),
            f"{name}_flat"
        )
        hists.append(h)
    else:
        #print(f"Scalar branch: {name}")
        
        maxV=leaf.GetMaximum() if "res" not in name else 0.5
        minV=leaf.GetMinimum() if "res" not in name else -0.5
        h = df.Histo1D(
            (f"h_{name}", name, 100, minV, maxV), #leaf.GetMinimum(), leaf.GetMaximum()),
            name

        )
        #print(name,h.Integral())
        hists.append(h)

    c = ROOT.TCanvas("c", "", 800, 600)

for h in hists:
    c.Clear()
    h.Draw()
    fit = ROOT.TF1("fit","gaus",-0.5,0.5)
    h.Fit(fit,"R")
    c.Update()
    c.SaveAs(f"{outdir}/{h.GetName()}.png")
