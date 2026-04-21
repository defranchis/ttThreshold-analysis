import ROOT,uproot,datetime

date = datetime.date.today().isoformat()
# global parameters

from treemaker_WbWb_reco import available_ecm, all_branches as branchList
#branchList=[]
branchList.append('BDT_score')
#branchList.append('nbjets')
#branchList.append('nbjets_sig')
#branchList.append('nbjets_cr')
#branchList.append('ntau_h')
#branchList.append('nbjets_semihad')
#branchList.append('nbjets_had')
#branchList.append('jet1_btagged')
#branchList.append('jet2_btagged')
#branchList.append('jet3_btagged')
#branchList.append('jet4_btagged')
#branchList.append('jet5_btagged')
#branchList.append('jet6_btagged')
#branchList.append('jet1_Qtagged')
#branchList.append('jet2_Qtagged')
#branchList.append('jet3_Qtagged')
#branchList.append('jet4_Qtagged')
#branchList.append('jet5_Qtagged')
#branchList.append('jet6_Qtagged')

def if3(cond, iftrue, iffalse):
    return iftrue if cond else iffalse


channel = 'CHANNELHERE' #'had' if hadronic else 'semihad'
ecm = ECMHERE
useflav=useflavHERE  #here it means including all flav-related info
usebtagged=usebtaggedHERE


#channel = 'semihad'
#ecm = '345'
#useflav=True
#usebtagged=True

##amhadronic = False
##amchannel = 'had' if hadronic else 'semihad'
##amuseflav=True
##amusebtagged=False
##amecm = 355

pf="%s"%if3(usebtagged,'withbtaggedJet',if3(useflav,'withflav','noflav'))
pf=pf+"WPpt8"



intLumi = 1
intLumiLabel = "L = 0.036 ab^{-1}"
ana_tex = "e^{+}e^{-} #rightarrow WbWb #rightarrow %s"%(channel)
delphesVersion = "3.4.2"
energy = ecm
collider = "FCC-ee"
formats = ["png","pdf","root"]
inputDir  = '/eos/cms/store/cmst3/group/top/anmehta/FCC//output_condor_06092024/WbWb/outputs/histmaker/{}/{}/'.format(channel,pf)
print('this is the inputDir',inputDir)
outdir    = '/eos/user/a/anmehta/www/FCC_top/{}/{}/{}/{}/'.format(date,channel,pf,ecm)
print('saving plots here',outdir)

plotStatUnc = True
colors = {}
colors["WbWb"] = ROOT.kRed
colors["WW"] = ROOT.kBlue

procs = {}
procs["signal"] = {"WbWb": ["wzp6_ee_WbWb_{}_ecm{}".format(channel,ecm)]}
procs["backgrounds"] = {"WW": ["p8_ee_WW_ecm{}".format(ecm)]}

print(procs["signal"],procs["backgrounds"])

legend = {}
legend["WbWb"] = "WbWb"
legend["WW"] = "WW"

hists = {}


for var in branchList:
    #print("plotting this variable %s"%var)
    logy = False
    if "tagged" in var:
        logy = True
    if var.endswith("_isB") or var.endswith("_isG") or var.endswith("_isQ") or var.endswith("_isS") or var.endswith("_isC") or var.endswith("_isTau") :
        continue 
    if var.endswith('d_12') or var.endswith('d_23') or var.endswith('d_34') or var.endswith('d_45') or var.endswith('d_56'):
        logy = True
    if 'BDT_score' in var:
        logy = True


    hists['no_cut_'+var] = {
        "output": var+"_stack",
        "logy": logy,
        "stack": True,
        "rebin": 1,
        "xmin": -1,
        "xmax": -1,
        "ymin": 0 if not logy else 1,
        "ymax": -1,
        "xtitle": var,
        "ytitle": "Events",
    }

    hists['no_cut_'+var] = {
        "output": var,
        "logy": logy,
        "stack": False,
        "rebin": 1,
        "xmin": -1,
        "xmax": -1,
        "ymin": 0 if not logy else 1,
        "ymax": -1,
        "xtitle": var,
        "ytitle": "Events",
    }


    hists['BDT_cut_'+var] = {
        "output": 'BDT_cut_'+var+"_nostack",
        "logy": logy,
        "stack": True,
        "rebin": 1,
        "xmin": -1,
        "xmax": -1,
        "ymin": 0 if not logy else 1,
        "ymax": -1,
        "xtitle": var,
        "ytitle": "Events",
    }

    
    hists['BDT_cut_'+var] = {
        "output": 'BDT_cut_'+var,
        "logy": logy,
        "stack": False,
        "rebin": 1,
        "xmin": -1,
        "xmax": -1,
        "ymin": 0 if not logy else 1,
        "ymax": -1,
        "xtitle": var,
        "ytitle": "Events",
    }



    
    hists['BDT_cut_zerobtag_'+var] = {
        "output": 'BDT_cut_zerobtag_'+var+"_nostack",
        "logy": logy,
        "stack": True,
        "rebin": 1,
        "xmin": -1,
        "xmax": -1,
        "ymin": 0 if not logy else 1,
        "ymax": -1,
        "xtitle": var,
        "ytitle": "Events",
    }
    hists['BDT_cut_zerobtag_'+var] = {
        "output": 'BDT_cut_zerobtag_'+var+"_nostack",
        "logy": logy,
        "stack": False,
        "rebin": 1,
        "xmin": -1,
        "xmax": -1,
        "ymin": 0 if not logy else 1,
        "ymax": -1,
        "xtitle": var,
        "ytitle": "Events",
    }

    
    hists['BDT_cut_onebtag_'+var] = {
        "output": 'BDT_cut_onebtag_'+var,
        "logy": logy,
        "stack": True,
        "rebin": 1,
        "xmin": -1,
        "xmax": -1,
        "ymin": 0 if not logy else 1,
        "ymax": -1,
        "xtitle": var,
        "ytitle": "Events",
    }


    
    hists['BDT_cut_onebtag_'+var] = {
        "output": 'BDT_cut_onebtag_'+var,
        "logy": logy,
        "stack": False,
        "rebin": 1,
        "xmin": -1,
        "xmax": -1,
        "ymin": 0 if not logy else 1,
        "ymax": -1,
        "xtitle": var,
        "ytitle": "Events",
    }


