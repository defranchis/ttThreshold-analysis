import os, sys
import re

debug = False
### to be customized by user

## analysis script
script = "../treemaker_WbWb_reco.py"

analysis_name = "WbWb"

## run nev_per_job = -1 to run on all event in input root files
queue = "workday"

nev_per_job = -1
ncpus = 4

## list of samples to run on training jets
samples = [
#    "wzp6_ee_WbWb_semihad_ecm340",
 #   "wzp6_ee_WbWb_had_ecm340",
    #"wzp6_ee_WbWb_lep_ecm340",
  #  "p8_ee_WW_ecm340",
#    "wzp6_ee_WbWb_semihad_ecm345",
#    "wzp6_ee_WbWb_had_ecm345",
#    "wzp6_ee_WbWb_semihad_ecm350",
#    "wzp6_ee_WbWb_had_ecm350",
#    "wzp6_ee_WbWb_semihad_ecm355",
#    "wzp6_ee_WbWb_had_ecm355",
#    "wzp6_ee_WbWb_lep_ecm340",
#    "wzp6_ee_WbWb_lep_ecm345",
#    "wzp6_ee_WbWb_lep_ecm350",
#    "wzp6_ee_WbWb_lep_ecm355",
    "p8_ee_WW_ecm365",
#    "wzp6_ee_WbWb_lep_ecm365",
#    "wzp6_ee_WbWb_semihad_ecm365",
#    "wzp6_ee_WbWb_had_ecm365",
#    "wzp6_ee_WbWb_semihad_mtop171p5_ecm365",
#    "wzp6_ee_WbWb_semihad_mtop173p5_ecm365",
#    "p8_ee_WW_ecm345",
#    "wzp6_ee_qq_ecm345",
#    "p8_ee_WW_ecm350",
#    "p8_ee_WW_ecm355",
]

indir = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"

channels = ['had','semihad','lep']

for channel in channels:

    outdir = "/eos/cms/store/cmst3/group/top/anmehta/FCC/output_condor_06092024/{}/{}".format(analysis_name,channel)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(script, 'r') as file:
        content = file.read()
        content = re.sub(r'CHANNELNAMEHERE',channel,content)
        
#    if channel == 'had':
#        content = re.sub(r'hadronic\s*=\s*False', 'hadronic = True', content)
#    elif channel == 'semihad':
#        content = re.sub(r'hadronic\s*=\s*True', 'hadronic = False', content)
#    elif  channel == 'lep':
#        content = re.sub(r'hadronic\s*=\s*True', 'hadronic = False', content)
#    else :
#        raise ValueError("channel not recognized")
    
    content = re.sub(r'includePaths\s*=\s*\["examples/functions.h"\]', 'includePaths = ["functions.h"]', content)
    
    base_name, ext = os.path.splitext(script)
    new_script = "{}_{}.py".format(base_name, channel)

    outdir_script = 'scripts'
    if not os.path.exists(outdir_script):
        os.makedirs(outdir_script)
    os.system("cp ../examples/functions.h {}".format(outdir_script))
    new_script = '{}/{}'.format(outdir_script, new_script.split('/')[-1])

    lines = content.splitlines()
    content = [line for line in lines if not 'outputDir' in line]
        
    
    with open(new_script, 'w') as file:
        for line in content:
            file.write(line)
            file.write('\n')
        
    ### run condor jobs
    if not os.path.exists("std"):
        os.makedirs("std")
    os.system("rm -rf std/*")
    for s in samples:
        cmd = "python submitAnalysisJobs.py --indir {}/{} ".format(indir, s)
        cmd += "--outdir {}/{} ".format(outdir, s)
        cmd += "--queue {} --script {} --nev {} ".format(queue, new_script, nev_per_job)
        cmd += "--ncpus {} ".format(ncpus)
        if debug:
            cmd += "--dry"
            print(cmd)
        os.system(cmd)
