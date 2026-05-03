import uproot
import pandas as pd
import numpy as np

import os,sys,re
import ROOT
from xgboost import XGBClassifier
import argparse

def if3(cond, iftrue, iffalse):
    return iftrue if cond else iffalse

sys.path.insert(0, "/afs/cern.ch/work/a/anmehta/public/FCC_ver2/FCCAnalyses/ttThreshold-analysis/") #os.getcwd()+'../')
from treemaker_WbWb_reco import all_branches
from multiprocessing import Pool

outdir = '/eos/cms/store/cmst3/group/top//FCC_tt_threshold/BDT_model/'


ecm_model='345'

base_dir="/eos/cms/store/cmst3/group/top//FCC_tt_threshold/output_condor_20250212_1138/WbWb/"
#base_dir="/eos/cms/store/cmst3/group/top//FCC_tt_threshold/output_condor_20250130_1158/WbWb/"


def calcsumW(proc,channel):
    sumW=0;
    nfiles=[os.path.join(base_dir,channel,proc,f) for f in os.listdir(os.path.join(base_dir,channel,proc)) if re.search('^events_\d*.root',f)]
    for i in nfiles:
        if re.search('events_\d*.root',i):
            fIn=ROOT.TFile.Open(i,"read")
            sumW+=fIn.Get("eventsProcessed").GetVal();
           
            fIn.Close();
    return sumW

def evaluate(proc,ecm,channel,variation):
    print("this is the proc in eval",proc)
    branches_toTrain=['jet1_R5_p','jet2_R5_p','jet1_R5_theta','jet2_R5_theta','mbbar']
    col_name=''
    #    if channel == "semihad":
    #        branches_toTrain+=['missing_p','lep_p','lep_theta']
    if variation == "up":
        col_name='mbbar_p91'
    elif variation == "down":
        col_name='mbbar_p89'
    else:
        col_name='mbbar_p9'

    print("mbbar col name to be used",col_name)
    branches_toTrain.append(col_name)
    dataframes = []
    root_files=[os.path.join(base_dir,channel,proc,f)  for f in os.listdir(os.path.join(base_dir,channel,proc)) if f.endswith(".root")]

    for file in root_files:
        with uproot.open(file) as root_file:
            tree = root_file["events"]
            df = tree.arrays(library="pd")
            dataframes.append(df)

    infiles = pd.concat(dataframes, ignore_index=True)
    #print(infiles.head())
    
    infiles=infiles.rename(columns={col_name: "mbbar"})
    pd_out=(infiles)
    infiles = infiles.drop(columns=[f for f in infiles.columns if f not in branches_toTrain])
    print("columns to be used are \t",infiles.columns)
    pf_model="sig_vs_wwz"
    pf=pf_model#+"_njcut"
    if len(variation) > 0:
        pf=pf+"_btag"+variation
        
        
    print('runnning for %s %s channel'%(pf,channel))
    bst = XGBClassifier()

    #bst.load_model(f'{outdir}/model_{channel}_345{pf_model}_nolepvars.json')
    bst.load_model(f'{outdir}/model_{channel}_345{pf_model}_nolepvars_Feb2025.json')
    #print('model used is',f'{outdir}/model_{channel}_345{pf_model}_nolepvars_njcut_Feb2025.json')
    #bst.load_model(f'{outdir}/model_{channel}_345{pf_model}_nolepvars_njcut_Feb2025.json')


    preds = bst.predict_proba(infiles)
    pd_s = (infiles) 
       
    pd_s['BDT_score'] = preds[:,1]
    pd_s['BDT_score'] = pd_s['BDT_score'].astype('float32')
    pd_out['BDT_score']=pd_s['BDT_score']
    data_dict = {name: pd_out[name].values for name in pd_out.columns}

    odir=os.path.join(base_dir,channel,pf,proc)
    print('this is output dir name',odir,pf)
    if not os.path.exists(odir):        os.makedirs(odir)
    fname_new = os.path.join(odir,'events_combined.root')
    if os.path.exists(fname_new):
        os.remove(fname_new)
    print ('this is the name of the output file',fname_new)
    rdf = ROOT.RDF.FromNumpy(data_dict)
    rdf.Snapshot('events', fname_new,  pd_out.columns)
    #print("now trying to update the existing root file")
    fname=ROOT.TFile.Open(fname_new,"update")
    sumW=calcsumW(proc,channel)
    fname.cd();
    p = ROOT.TParameter(int)("eventsProcessed", sumW)
    print(p.GetVal())
    p.Write();
    fname.Write();fname.Close();

def parallel_evaluate(args):
    proc, ecm, ch = args
    evaluate(proc, ecm, ch)

def main(ncores=4):
    tasks = []
    for ecm in ['340', '345', '365']:
        for ch in ["had",'semihad']:
            for var in ["","down","up"]: 
#                sig=f"wzp6_ee_WbWb_ecm{ecm}"
#                bkg=f"wzp6_ee_WWZ_Zbb_ecm{ecm}"
#                bkg1=f"p8_ee_WW_ecm{ecm}"
#                bkg2=f"wzp6_ee_qq_ecm{ecm}"
#                bkg3=f"wzp6_ee_qq_PSdown_ecm{ecm}"
#                bkg4=f"wzp6_ee_qq_PSup_ecm{ecm}"
#                
#                evaluate(sig, ecm, ch,var)
#                evaluate(bkg1, ecm, ch,var)
#                evaluate(bkg2, ecm, ch,var)
#                evaluate(bkg3, ecm, ch,var)
#                evaluate(bkg4, ecm, ch,var)
                #evaluate(bkg, ecm, ch,var)
                procs=[];
#                procs.append(f"wzp6_ee_WbWb_ecm{ecm}")
#                procs.append(f"wzp6_ee_WWZ_Zbb_ecm{ecm}")
#                procs.append(f"p8_ee_WW_ecm{ecm}")
#                procs.append(f"p8_ee_WW_PSup_ecm{ecm}")
#                procs.append(f"p8_ee_WW_PSdown_ecm{ecm}")
#                procs.append(f"wzp6_ee_WbWb_PSup_ecm{ecm}")
#                procs.append(f"wzp6_ee_WbWb_PSdown_ecm{ecm}")
#                procs.append(f"wzp6_ee_WbWb_mtop173p5_ecm{ecm}")
#                procs.append(f"wzp6_ee_WbWb_mtop171p5_ecm{ecm}")
#                procs.append(f"wzp6_ee_qq_ecm{ecm}")
#                procs.append(f"wzp6_ee_qq_PSdown_ecm{ecm}")
#                procs.append(f"wzp6_ee_qq_PSdown_ecm{ecm}")
                procs.append(f"p8_ee_ZZ_ecm{ecm}")
                for proc in procs:
                    print("this is what i will run on ",proc)
                    evaluate(proc, ecm, ch,var)
                                
                    #tasks.append((proc, ecm, ch,var))
    #with Pool(ncores) as pool:
        #pool.map(parallel_evaluate, tasks)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ncores", type=int, default=4)
    args = args.parse_args()
    main(ncores=args.ncores)
