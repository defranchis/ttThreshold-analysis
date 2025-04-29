import ROOT,sys,optparse,os
import datetime
date = datetime.date.today().isoformat()
def if3(cond, iftrue, iffalse):
    return iftrue if cond else iffalse
from plotter_withsyst import *


#def drawSLatex(xpos,ypos,text,size):
#    latex = ROOT.TLatex()
#    latex.SetNDC()
#    latex.SetTextAlign(12)
#    latex.SetTextSize(size)
#    latex.SetTextFont(42)
#    latex.DrawLatex(xpos,ypos,text)

def stackPlot(fname,vname,lumi,channel,config,ecm,useLog,showInt,pstyle,sel):
    if ecm == "365":
        lumi_txt=f'{lumi/1e6:.1f}'
    else:     lumi_txt=f'{lumi/1e3:.1f}'
    Canv = ROOT.TCanvas(f'Canv_{channel}_{config}_{ecm}',"",600,600)
    Canv.Range(0,0,1,1);   Canv.SetFillColor(0);   Canv.SetBorderMode(0);   Canv.SetBorderSize(2);
    Canv.SetTickx(1);   Canv.SetTicky(1);   Canv.SetLeftMargin(0.16);   Canv.SetRightMargin(0.08);
    Canv.SetBottomMargin(0.13);   Canv.SetFrameFillStyle(0);   Canv.SetFrameBorderMode(0);
    pf="_"+pstyle
    morey=1.5
    
    #legend = ROOT.TLegend(0.5 if showInt else 0.635,0.67,0.8 if showInt else  0.875,0.85);
    legend = ROOT.TLegend(0.65,0.635,0.9,0.8);
    legend.SetNColumns(1);legend.SetFillColor(0);legend.SetFillStyle(0); legend.SetShadowColor(0);   legend.SetLineColor(0);
    legend.SetTextFont(42);        legend.SetBorderSize(0);   legend.SetTextSize(0.035);
    if "stack" in pstyle:
        hs=ROOT.THStack(f'hs_{channel}_{config}_{ecm}',"");
    if useLog:
        Canv.SetLogy();        pf+='_log'  ;      morey=200
    else:
        if "norm" not in pstyle:
            print(type(hs),'prob')
            #hs.GetYaxis().SetTitleOffset(1.25);
    f_in=ROOT.TFile.Open(fname)
    h_sig=f_in.Get('x_sig');
    h_bkg=f_in.Get('x_bkg_%s'%channel); 
    h_bkg.SetDirectory(0);    h_sig.SetDirectory(0);   
    if ecm in ["340","345","365"]:
        h_bkg1=f_in.Get('x_bkg1_%s'%channel);        h_bkg1.SetDirectory(0);
        h_bkg2=f_in.Get('x_bkg2_%s'%channel);        h_bkg2.SetDirectory(0);
        

    if pstyle != "stack":
        h_sig.SetLineStyle(1);        h_bkg.SetLineStyle(1);
        h_sig.SetLineWidth(2);        h_bkg.SetLineWidth(2);
        h_sig.SetLineColor(ROOT.kAzure+1);         h_bkg.SetLineColor(ROOT.kOrange+1);
        if ecm in ["340","345","365"]: h_bkg1.SetLineColor(ROOT.kViolet+5);         h_bkg1.SetLineWidth(2); h_bkg2.SetLineColor(ROOT.kGreen+2);         h_bkg2.SetLineWidth(2);
    else:        
        h_sig.SetFillColor(ROOT.kAzure+1);  h_sig.SetLineColor(ROOT.kBlack)
        h_bkg.SetFillColor(ROOT.kOrange+1); h_bkg.SetLineColor(ROOT.kBlack)
        hs.Add(h_bkg);    hs.Add(h_sig);
        if ecm in ["340","345","365"]:
            h_bkg1.SetFillColor(ROOT.kViolet+5);
            h_bkg2.SetFillColor(ROOT.kGreen+2);
            h_bkg1.SetLineColor(ROOT.kBlack);  hs.Add(h_bkg1);
            h_bkg2.SetLineColor(ROOT.kBlack);  hs.Add(h_bkg2);
            
    f_in.Close();
    #legend.AddEntry('NULL',f'{channel}; {sel}','')#e^{#plus}e^{#minus} #rightarrow WbWb/WW  #rightarrow %s'%channel,'')
    legend.AddEntry(h_sig,  'WbWb'+ (f"({h_sig.Integral():.2e})"  if showInt else ''),"F" if pstyle == "stack" else 'l');
    legend.AddEntry(h_bkg,  'WW ' + (f"({h_bkg.Integral():.2e})"  if showInt else ''),"F" if pstyle == "stack" else 'l');
    if ecm in ["340","345","365"]:
        legend.AddEntry(h_bkg1, 'WWZ '+ (f"({h_bkg1.Integral():.2e})" if showInt else ''),"F" if pstyle == "stack" else 'l');
        legend.AddEntry(h_bkg2, 'qq ' + (f"({h_bkg2.Integral():.2e})" if showInt else ''),"F" if pstyle == "stack" else 'l');
        
    if  "norm" not in pstyle:
        hs.Draw("HIST" if pstyle ==  "stack" else 'nostackhist');
        hs.GetXaxis().SetTitle(xtitle)
        hs.GetYaxis().SetTitle("Arbitrary units");
        hs.GetYaxis().SetLabelSize(0.04);    hs.GetYaxis().SetTitleSize(0.045);    hs.GetYaxis().SetTitleOffset(1.22);
        hs.GetXaxis().SetTitleSize(0.045);    hs.GetXaxis().SetTitleOffset(1.0); hs.GetXaxis().SetLabelSize(0.04);
        hs.GetYaxis().SetMaxDigits(3);
        hs.GetXaxis().SetTitleFont(42);        hs.GetYaxis().SetTitleFont(42);
        hs.SetMinimum(0);
        hs.SetMaximum(morey*hs.GetHistogram().GetMaximum())
        #if ecm in ["340","345","365"]: hs.SetMaximum(morey*(h_sig.Integral()+h_bkg.Integral()+h_bkg1.Integral()));

    else:
        h_sig.Scale(1.0/h_sig.Integral()); h_bkg.Scale(1.0/h_bkg.Integral());
        if ecm in ["340","345","365"]:
            h_bkg1.Scale(1.0/h_bkg1.Integral());
            h_bkg2.Scale(1.0/h_bkg2.Integral());
        h_sig.GetYaxis().SetRangeUser(0,1.25*max(h_sig.GetMaximum(),h_bkg.GetMaximum(),h_bkg2.GetMaximum(),h_bkg1.GetMaximum()))        
        h_sig.Draw("hist"); 
        h_sig.GetXaxis().SetTitle(xtitle)
        h_sig.GetYaxis().SetTitle("a.u."); h_sig.SetMaximum(1.0);         h_sig.SetMinimum(0);
        h_sig.GetYaxis().SetLabelSize(0.04);    h_sig.GetYaxis().SetTitleSize(0.045);    h_sig.GetYaxis().SetTitleOffset(1.22);
        h_sig.GetXaxis().SetTitleSize(0.045);    h_sig.GetXaxis().SetTitleOffset(1.0); h_sig.GetXaxis().SetLabelSize(0.04);
        h_sig.GetYaxis().SetMaxDigits(3);        h_sig.GetXaxis().SetTitleFont(42);        h_sig.GetYaxis().SetTitleFont(42);

        h_bkg.Draw("histsame");

        if ecm in ["340","345","365"]:
            h_bkg1.Draw("histsame");             h_bkg2.Draw("histsame");


    t2a = drawSLatex(0.2,0.85,"#bf{FCC-ee} #it{Simulation (Delphes)}",0.04);
    if ecm == "365":
        t3a = drawSLatex(0.64,0.915,"%s ab^{#minus1} (%s GeV)"%(lumi_txt,ecm),0.035);
    else:    
        t3a = drawSLatex(0.64,0.915,"%s fb^{#minus1} (%s GeV)"%(lumi_txt,ecm),0.035);

    t4a = drawSLatex(0.2,0.8,nice_names[channel]+" "+nice_names[sel],0.035);
    legend.Draw("same");
    #hs.SetMinimum(0.01);
    Canv.Update();
    plotsdir=f"/eos/user/a/anmehta/www/FCC_top/{date}_variables"
    if not os.path.isdir(plotsdir):        os.system("mkdir %s"%plotsdir);  os.system('cp ~/public/index.php %s/'%plotsdir)

    if "norm" in pstyle : pf="_norm"
    Canv.Print(f"{plotsdir}/{vname}_{channel}_{config}_{ecm}{pf}.pdf")
    Canv.Print(f"{plotsdir}/{vname}_{channel}_{config}_{ecm}{pf}.png")
    return True

def getHist(isSig,proc,vname,h_name,xsec_sig,channel,config,ecm,lumi):
    sf=1.0;sumW=1.0;xsec=1.0;
    f_in=ROOT.TFile.Open(f'/eos/cms/store/cmst3/group/top//FCC_tt_threshold/output_condor_20250212_1138/WbWb/outputs/histmaker/{channel}/{config}/{proc}.root')
    print("looking for ",vname, "in \t",f_in.GetName())
    h_in=f_in.Get(vname).Clone(h_name);
    xsec=f_in.Get('crossSection').GetVal();
    sumW=f_in.Get('sumOfWeights').GetVal()
    if isSig:
        xsec=xsec_sig;
    N_tot=f_in.Get('eventsProcessed').GetVal()
    print('input ylds',h_in.Integral())
    sf=xsec*lumi/N_tot #sumW
    print('xsec\t',xsec,'\t n_tot\t',N_tot,"\t lumi\t",lumi,"\t sf \t",sf)
    h_in.Scale(sf);    h_in.SetDirectory(0);    f_in.Close();
    #print('integral after scaling',h_in.Integral())
    return h_in

def cards(mkplots,lumi,xsec_sig,channel,sel,config,bWP,ecm,logy,vname,xtitle,showInt,pstyle):
    h_sig=getHist(True,f'wzp6_ee_WbWb_ecm{ecm}',vname,"x_sig",xsec_sig,channel,config,ecm,lumi)
    print('sig',h_sig.Integral())
    h_obs  = h_sig.Clone("x_data_obs")
    h_bkg  = getHist(False,f'p8_ee_WW_ecm{ecm}',vname,"x_bkg_%s"%channel,1.0,channel,config,ecm,lumi)
    if ecm in ["340","345","365"]:
        h_bkg1 = getHist(False,f'wzp6_ee_WWZ_Zbb_ecm{ecm}',vname,"x_bkg1_%s"%channel,1.0,channel,config,ecm,lumi)
        h_bkg2 = getHist(False,f'wzp6_ee_qq_ecm{ecm}',vname,"x_bkg2_%s"%channel,1.0,channel,config,ecm,lumi)
        h_obs.Add(h_bkg1);h_obs.Add(h_bkg2)
        
    h_obs.Add(h_bkg); 
    print('bkg',h_bkg.Integral())
    fout_name=f"{channel}_{sel}_{config}_bWP{bWP}_{ecm}.root"
    f_out=ROOT.TFile(fout_name,"RECREATE");
    f_out.cd();
    h_sig.Write();    h_bkg.Write();
    if ecm in ["340","345","365"]:
        h_bkg1.Write();h_bkg2.Write();
    h_obs.Write();    f_out.Close();
    if mkplots:
        stackPlot(fout_name,vname,lumi,channel,config,ecm,logy,showInt,pstyle,sel)


if __name__ == '__main__':
    ROOT.gROOT.SetBatch()
    ROOT.gStyle.SetOptStat(0)

    parser = optparse.OptionParser(usage='usage: %prog [opts] ', version='%prog 1.0')
    parser.add_option('-c',  '--ch',       dest='channel',   type='string',         default='semihad',    	help='had/semihad')
    parser.add_option('-f',  '--fconf',    dest='config',    type='string',         default='sig_vs_wwz',      	help='withflav/noflav/withbtaggedJet/noBDT')
    parser.add_option('-s',  '--sel',      dest='sel' ,      type='string',         default='no_cut',       	help='no_cut/effp9_twob/effp9_oneb/effp9_zerob')
    parser.add_option('-e',  '--ecm',      dest='ecm' ,      type='string',         default='345',        	help='ecm')
    parser.add_option('-w',  '--bwp',      dest='btagWP',    type='string',         default='9',      	        help='btagWP:nom(9)/up(91)/dn(89)')
    parser.add_option('-v',  '--vname',    dest='vname',     type='string',         default='no_cut_njets_R5',    help='plot this variable')
    parser.add_option('-l',  '--lum',      dest='lum' ,      type=float,            default=41.0,         	help='lumi in fb')
    parser.add_option('-p',  '--plots',    dest='mkplots',   action='store_true',   default=False,        	help='make plots too')
    parser.add_option('-i',  '--sInt',     dest='showInt' ,  action='store_true',   default=False,        	help='show integral in legends')
    parser.add_option('--logy',            dest='logy' ,     action='store_true',   default=False,        	help='use log scale for y-axis')
    #    parser.add_option('--nostack',         dest='nostack' ,  action='store_true',   default=False,        	help='draw non-stacked plots with transparent fill style')
    parser.add_option('--style',           dest='style' ,    type='string',         default="stack",        	help='plotting style stack/nostack/norm')
    (opts, args) = parser.parse_args()
#    fancyname={'singlebin': "N_{events}",
#               'BDT_score': "BDT",
#               "njets_R5":"N_{jets}"}
    
    if opts.ecm == "365":
        lumi=2650*1000
    else:
        lumi=opts.lum*1000
    
    br_semihad=0.438
    br_had=0.457
    xsec_tt=0.1 if opts.ecm =="340" else 0.5
    xsec_sig=xsec_tt #*(br_semihad if "semihad" in opts.channel else br_had)
    hname=opts.sel+"_"+opts.vname
    if "no_cut" not in opts.sel:
        hname="effp"+opts.btagWP+"_"+opts.sel+"_"+opts.vname

    print(opts.vname,hname)
    xtitle= fancyname[opts.vname] #"N_{bjets}" if "nbjets" in opts.vname else "N_{jets}"    


    cards(opts.mkplots,lumi,xsec_sig,opts.channel,opts.sel,opts.config,opts.btagWP,opts.ecm,opts.logy,hname,xtitle,opts.showInt,opts.style)

