
import uproot
import pandas as pd
import numpy as np

import os,sys,re
import ROOT
from xgboost import XGBClassifier
def if3(cond, iftrue, iffalse):
    return iftrue if cond else iffalse

sys.path.insert(0, "/afs/cern.ch/work/a/anmehta/public/FCC/ttThreshold-analysis/") #os.getcwd()+'../')
from treemaker_WbWb_reco import all_branches

outdir = '/eos/user/a/anmehta/www/FCC_top/BDT_model'



ecm_model='345'
branches_toTrain=[]
branches_toTrain=all_branches

bjet_wp=0.8

channel = 'lep' #'had' if hadronic else 'semihad'
base_dir="/eos/cms/store/cmst3/group/top/anmehta/FCC//output_condor_06092024/WbWb/" #{}/".format(channel)


def calcsumW(proc,channel):
    sumW=0
    nfiles=[os.path.join(base_dir,channel,proc,f) for f in os.listdir(os.path.join(base_dir,channel,proc)) if re.search('^events_\d*.root',f)]
    for i in nfiles:
        if re.search('events_\d*.root',i):
            fIn=ROOT.TFile.Open(i,"read")
            #print(i,fIn.Get("eventsProcessed").GetVal())
            sumW+=fIn.Get("eventsProcessed").GetVal();
            fIn.Close();
    return sumW

def evaluate(proc,ecm,channel,useflav,usebtagged):
    print('ineval',os.path.join(base_dir,channel,proc))
    infiles =  uproot.concatenate([os.path.join(base_dir,channel,proc,f)+':events' for f in os.listdir(os.path.join(base_dir,channel,proc)) if re.search('^events_\d*.root',f)],branches_toTrain ,library="pd")
    flavor_branches=[];
    if "lep" not in channel:
        flavor_branches=[v for v in all_branches if re.match(".*_is[a-z]$",v,re.IGNORECASE)] ##using taus in the training
        nbjets= np.where(infiles["jet1_isB"] > bjet_wp,1,0) + np.where(infiles["jet2_isB"] > bjet_wp ,1,0) + np.where(infiles["jet3_isB"] > bjet_wp,1,0) + np.where(infiles["jet4_isB"] > bjet_wp,1,0) + np.where(infiles["jet5_isB"] > bjet_wp,1,0)+np.where(infiles["jet6_isB"] > bjet_wp,1,0)
        ntau_h= np.where(infiles["jet1_isTau"] > 0.95,1,0) + np.where(infiles["jet2_isTau"] > 0.95 ,1,0) + np.where(infiles["jet3_isTau"] > 0.95,1,0) + np.where(infiles["jet4_isTau"] > 0.95,1,0) + np.where(infiles["jet5_isTau"] > 0.95,1,0)
        jet1_btagged =np.where(infiles["jet1_isB"] > bjet_wp,1, 0)
        jet2_btagged =np.where(infiles["jet2_isB"] > bjet_wp,1, 0)
        jet3_btagged =np.where(infiles["jet3_isB"] > bjet_wp,1, 0)
        jet4_btagged =np.where(infiles["jet4_isB"] > bjet_wp,1, 0)
        jet5_btagged =np.where(infiles["jet5_isB"] > bjet_wp,1, 0)
        jet6_btagged =np.where(infiles["jet6_isB"] > bjet_wp,1, 0)
        jet1_Qtagged =np.where(infiles["jet1_isQ"] > 0.5,1, 0)
        jet2_Qtagged =np.where(infiles["jet2_isQ"] > 0.5,1, 0)
        jet3_Qtagged =np.where(infiles["jet3_isQ"] > 0.5,1, 0)
        jet4_Qtagged =np.where(infiles["jet4_isQ"] > 0.5,1, 0)
        jet5_Qtagged =np.where(infiles["jet5_isQ"] > 0.5,1, 0)
        jet6_Qtagged =np.where(infiles["jet6_isQ"] > 0.5,1, 0)
        
    else:
        nbjets= np.where(infiles["jet1_isB"] > bjet_wp,1,0) + np.where(infiles["jet2_isB"] > bjet_wp ,1,0) + np.where(infiles["jet3_isB"] > bjet_wp,1,0) + np.where(infiles["jet4_isB"] > bjet_wp,1,0) + np.where(infiles["jet5_isB"] > bjet_wp,1,0) + np.where(infiles["jet6_isB"] > bjet_wp,1,0)
        ntau_h=[]
        
    if usebtagged and 'lep' not in 'channel':
        infiles["jet1_btagged"] =np.where(infiles["jet1_isB"] > bjet_wp,1, 0)
        infiles["jet2_btagged"] =np.where(infiles["jet2_isB"] > bjet_wp,1, 0)
        infiles["jet3_btagged"] =np.where(infiles["jet3_isB"] > bjet_wp,1, 0)
        infiles["jet4_btagged"] =np.where(infiles["jet4_isB"] > bjet_wp,1, 0)
        infiles["jet5_btagged"] =np.where(infiles["jet5_isB"] > bjet_wp,1, 0)
        infiles["jet6_btagged"] =np.where(infiles["jet6_isB"] > bjet_wp,1, 0)
        infiles=infiles.drop(columns=flavor_branches, axis=1)
    if not usebtagged and not useflav :
        #if not 'lep' in 'channel':
        #      print("dropping columns")
        infiles=infiles.drop(columns=flavor_branches, axis=1)
    
        
    ##load model 
    pf="%s"%if3(usebtagged,'withbtaggedJet',if3(useflav,'withflav','noflav'))

    pf=pf+"WPpt%s"%(str(bjet_wp).split('.')[-1])
    print('runnning for %s %s channel'%(pf,channel))
    bst = XGBClassifier()
    bst.load_model('{}/model_{}_{}_{}.json'.format(outdir,channel,ecm_model,pf))
    preds = bst.predict_proba(infiles)
    pd = (infiles)
    pd['BDT_score'] = preds[:,1]
    pd['BDT_score'] = pd['BDT_score'].astype('float32')
    pd["nbjets"] = nbjets 
    pd["ntau_h"] = ntau_h
    pd["jet1_btagged"] = jet1_btagged 
    pd["jet2_btagged"] = jet2_btagged
    pd["jet3_btagged"] = jet3_btagged
    pd["jet4_btagged"] = jet4_btagged
    pd["jet5_btagged"] = jet5_btagged
    pd["jet6_btagged"] = jet6_btagged
    pd["jet1_Qtagged"] = jet1_Qtagged
    pd["jet2_Qtagged"] = jet2_Qtagged
    pd["jet3_Qtagged"] = jet3_Qtagged
    pd["jet4_Qtagged"] = jet4_Qtagged
    pd["jet5_Qtagged"] = jet5_Qtagged
    pd["jet6_Qtagged"] = jet6_Qtagged
    
    #print((infiles["jet1_isB"]).shape)

    data_dict = {name: pd[name].values for name in pd.columns}
    odir=os.path.join(base_dir,channel,pf,proc)
    print('this is output dir name',odir,pf)
    if not os.path.exists(odir):
        os.makedirs(odir)
    fname_new = os.path.join(odir,'events_combined.root')
    if os.path.exists(fname_new):
        os.remove(fname_new)
    #print ('this is the name of the output file',fname_new)
    rdf = ROOT.RDF.MakeNumpyDataFrame(data_dict)
    rdf.Snapshot('events', fname_new,  pd.columns)
    print("now trying to update the existing root file")
    fname=ROOT.TFile.Open(fname_new,"update")
    sumW=calcsumW(proc,channel)
    fname.cd();
    p = ROOT.TParameter(int)("eventsProcessed", sumW)
    #print(sumW,p.GetVal())
    p.Write();
    fname.Write();fname.Close();
for ecm in ['365','340','345','350','355']:
    for ch in ['semihad','had']:
        #if ecm == '340' and ch == 'lep': continue
        sig="wzp6_ee_WbWb_{0}_ecm{1}".format(ch,ecm)
        bkg="p8_ee_WW_ecm{}".format(ecm)
        print(sig,bkg,ecm,ch)
        evaluate(sig,ecm,ch,True,False)
        evaluate(bkg,ecm,ch,True,False)
        evaluate(sig,ecm,ch,False,True)
        evaluate(bkg,ecm,ch,False,True)
        evaluate(sig,ecm,ch,False,False)
        evaluate(bkg,ecm,ch,False,False)






#eos/cms/store/cmst3/group/top/anmehta/FCC//output_condor_06092024/WbWb/lep/wzp6_ee_WbWb_had_ecm345
