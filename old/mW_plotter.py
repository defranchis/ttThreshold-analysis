import ROOT
ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
import uproot
fname="resolution_fits.root"
file = uproot.open(fname)
#tree = file["fitResults"]
df = ROOT.RDataFrame("fitResults", fname)
h1 = df.Histo1D(("h1", "Fit results;m_{W} [GeV];A.U.", 100,50,100), "mW")
h2 = df.Histo1D(("h2", "Fit results;m_{W} [GeV];A.U.", 100,50,100), "mWlep_postfit")
h3 = df.Histo1D(("h3", "Fit results;m_{W} [GeV];A.U.", 100,50,100), "mWhad_postfit")
h4 = df.Histo1D(("h4", "Fit results;m_{W} [GeV];A.U.", 100,50,100), "mWhad_prefit")
h5 = df.Histo1D(("h5", "Fit results;m_{W} [GeV];A.U.", 100,50,100), "mWlep_prefit")


h1 = h1.GetValue()
h2 = h2.GetValue()
h3 = h3.GetValue()
h4 = h4.GetValue()
h5 = h5.GetValue()
for h in [h1, h2, h3,h4,h5]:
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

h1.SetLineColor(ROOT.kRed)
h2.SetLineColor(ROOT.kBlue)
h5.SetLineColor(ROOT.kBlue)
h3.SetLineColor(ROOT.kGreen+2)
h4.SetLineColor(ROOT.kGreen+2)

h1.SetLineWidth(2)
h2.SetLineWidth(2)
h3.SetLineWidth(2)
h4.SetLineStyle(2)
h5.SetLineStyle(2)
c = ROOT.TCanvas("c", "c", 800, 600)
ymax=max(h4.GetMaximum(),h3.GetMaximum(),h2.GetMaximum(),h3.GetMaximum(),h5.GetMaximum())
h4.GetYaxis().SetRangeUser(0.,ymax+0.05)

h4.Draw("HIST")
h5.Draw("HIST SAME")
h2.Draw("HIST SAME")
h3.Draw("HIST SAME")
h1.Draw("HIST SAME")

mW = 80.419  # W boson mass in GeV
line = ROOT.TLine(mW, 0, mW, h1.GetMaximum()*1.05)  # from y=0 to slightly above max
line.SetLineColor(ROOT.kBlack)
line.SetLineStyle(ROOT.kDashed)  # dotted / dashed
line.SetLineWidth(2)
line.Draw("SAME")

leg = ROOT.TLegend(0.7, 0.6, 0.85, 0.85)
leg.AddEntry(h1, "Fitted", "l")
leg.AddEntry(h2, "Wlep_postfit", "l")
leg.AddEntry(h5, "Wlep_prefit", "l")
leg.AddEntry(h3, "Whad_postfit", "l")
leg.AddEntry(h4, "Whad_prefit", "l")
leg.AddEntry(line,"Input","l")

leg.Draw()
c.Draw()
c.SaveAs("/eos/user/a/anmehta/www/FCC_WW//mW.png")
c.SaveAs("/eos/user/a/anmehta/www/FCC_WW//mW.pdf")
