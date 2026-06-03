import uproot
import pandas as pd
import numpy as np

import os,sys,re
import ROOT
from xgboost import XGBClassifier
import argparse

def if3(cond, iftrue, iffalse):
    return iftrue if cond else iffalse

sys.path.insert(0, "/afs/cern.ch/work/a/anmehta/public/FCC/ttThreshold-analysis/") #os.getcwd()+'../')
from treemaker_WbWb_reco import all_branches
from multiprocessing import Pool

outdir = '/eos/cms/store/cmst3/group/top/anmehta/FCC/BDT_model/'


ecm_model='345'
branches_toTrain=all_branches
base_dir="/eos/cms/store/cmst3/group/top/anmehta/FCC//output_condor_20241005_1231/WbWb/" #{}/".format(channel)


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

def evaluate(proc,ecm,channel):
    print('ineval',os.path.join(base_dir,channel,proc))
    infiles =  uproot.concatenate([os.path.join(base_dir,channel,proc,f)+':events' for f in os.listdir(os.path.join(base_dir,channel,proc)) if re.search('^events_\d*.root',f)],branches_toTrain ,library="pd")

    flavor_branches=[v for v in all_branches if '_is' in v or 'nbjets' in v]
    #bjet_wpL=0.5;
    #bjet_wpT=0.8;
    pd_out=(infiles)
    #nbjets_loose= np.where(infiles["jet1_isB"] > bjet_wpL,1,0) + np.where(infiles["jet2_isB"] > bjet_wpL ,1,0) + np.where(infiles["jet3_isB"] > bjet_wpL,1,0) + np.where(infiles["jet4_isB"] > bjet_wpL,1,0) + np.where(infiles["jet5_isB"] > bjet_wpL,1,0)+np.where(infiles["jet6_isB"] > bjet_wpL,1,0)
    #nbjets_tight= np.where(infiles["jet1_isB"] > bjet_wpT,1,0) + np.where(infiles["jet2_isB"] > bjet_wpT ,1,0) + np.where(infiles["jet3_isB"] > bjet_wpT,1,0) + np.where(infiles["jet4_isB"] > bjet_wpT,1,0) + np.where(infiles["jet5_isB"] > bjet_wpT,1,0)+np.where(infiles["jet6_isB"] > bjet_wpT,1,0)
    infiles=infiles.drop(columns=flavor_branches, axis=1)
    
        
    ##load model 
    pf="noflav"
    print('runnning for %s %s channel'%(pf,channel))
    bst = XGBClassifier()
    print('model used is','{}/model_{}_{}_{}.json'.format(outdir,channel,ecm_model,pf))
    bst.load_model(f'{outdir}/model_{channel}_345_noflav.json') #model_{}_{}_{}.json'.format(outdir,channel,ecm_model,pf))
    preds = bst.predict_proba(infiles)
    pd = (infiles)
    pd['BDT_score'] = preds[:,1]
    pd['BDT_score'] = pd['BDT_score'].astype('float32')
    pd_out['BDT_score']=pd['BDT_score']
    #pd["nbjets_tight"] = nbjets_tight
    #pd["nbjets_loose"] = nbjets_loose
    data_dict = {name: pd_out[name].values for name in pd_out.columns}
    odir=os.path.join(base_dir,channel,pf,proc)
    print('this is output dir name',odir,pf)
    if not os.path.exists(odir):
        os.makedirs(odir)
    fname_new = os.path.join(odir,'events_combined.root')
    if os.path.exists(fname_new):
        os.remove(fname_new)
    #print ('this is the name of the output file',fname_new)
    rdf = ROOT.RDF.FromNumpy(data_dict)
    rdf.Snapshot('events', fname_new,  pd_out.columns)
    print("now trying to update the existing root file")
    fname=ROOT.TFile.Open(fname_new,"update")
    sumW=calcsumW(proc,channel)
    fname.cd();
    p = ROOT.TParameter(int)("eventsProcessed", sumW)
    #print(sumW,p.GetVal())
    p.Write();
    fname.Write();fname.Close();

def parallel_evaluate(args):
    proc, ecm, ch = args
    evaluate(proc, ecm, ch)

def main(ncores=4):
    tasks = []
    for ecm in ['340', '345', '350', '355', '365']:
        for ch in ['semihad', 'had']:
            sig = "wzp6_ee_WbWb_ecm{}".format(ecm)
            bkg = "p8_ee_WW_ecm{}".format(ecm)
            tasks.append((sig, ecm, ch))
            tasks.append((bkg, ecm, ch))

    with Pool(ncores) as pool:
        pool.map(parallel_evaluate, tasks)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ncores", type=int, default=4)
    args = args.parse_args()
    main(ncores=args.ncores)