import uproot
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import date

ecm_scan = [340,345, 350, 355] # List of ecm values to scan


hadronic = False
channel = 'had' if hadronic else 'semihad'
def if3(cond, iftrue, iffalse):
    return iftrue if cond else iffalse


def plot_signal_background(sig,bkg,ecm,channel,pf):


    tree_s = uproot.open(sig)["events"]
    tree_b = uproot.open(bkg)["events"]
    
    sig_deno =  uproot.open(sig)["eventsProcessed"].value
    bkg_deno =  uproot.open(bkg)["eventsProcessed"].value
    eosdir="/eos/user/a/anmehta/www/FCC_top/"
    odir = '{od}/{when}_{ecm}_{FS}{pf}_BDTcut'.format(od=eosdir,when=date.today(),FS=channel,ecm=ecm,pf=pf)
    if not os.path.exists(odir):
        os.makedirs(odir)
        os.system('cp ~anmehta/public/index.php {od}'.format(od=odir))
    selection="BDT_score > 0.5"
    data_sig = tree_s.arrays(
        filter_name=[tree_s.keys()],  # Replace with your branch names
        cut=selection,
        library="np"  # Load data as numpy arrays
    )
    data_bkg = tree_b.arrays(
        filter_name=[tree_b.keys()],  # Replace with your branch names
        cut=selection,
        library="np"  # Load data as numpy arrays
    )

    
    for var in tree_b.keys():
        if var == 'nlep': continue
        
        n, bins, _ = plt.hist(data_sig[var], bins=50, alpha=0.5, label='WbWb', color='red',density=False)
        bin_width = bins[1] - bins[0]
        sig_integral = bin_width * sum(n)        
        n_bkg, bins_b, _ =plt.hist(data_bkg[var], bins=50, alpha=0.5, label='WW', color='blue',density=False)
        bkg_integral = bin_width * sum(n_bkg)
        plt.xlabel(var)
        plt.ylabel('Normalised distribution')
        plt.title('{} {} GeV'.format(var, ecm))
        plt.legend()
        #print(sig_integral,sig_deno,bkg_integral,bkg_deno,sig_integral/sig_deno,bkg_integral/bkg_deno)
        #plt.text(x=0.5,y=0.5,s='Acceptances : Sig: %.3f/%.3f; Bkg: %.3f/%.3f'%(sig_integral,sig_deno,bkg_integral,bkg_deno),transform = plt.gca().transAxes)
        plt.savefig('{}/{}_mplotlib.png'.format(odir, var))
        plt.savefig('{}/{}_mplotlib.pdf'.format(odir, var))
        plt.yscale("log")
        plt.savefig('{}/{}_mplotlib_log.png'.format(odir, var))
        plt.savefig('{}/{}_mplotlib_log.pdf'.format(odir, var))
        plt.close()


for ecm in ecm_scan:
    for channel in ['semihad']: #,'had']:
        for useflav in [True, False]:
            for usebtagged in [True, False]:
                if usebtagged and useflav:
                    continue
                else:
                    pf="%s"%if3(usebtagged,'withbtaggedJet',if3(useflav,'withflav','noflav'))
                    base_dir="/eos/cms/store/cmst3/group/dpsww/condor_WbWb/{}/".format(channel)
                    fname='%s/events_combined.root'%pf
                    sig=os.path.join(base_dir,"wzp6_ee_WbWb_{0}_ecm{1}/".format(channel,ecm),fname)
                    bkg=os.path.join(base_dir,"p8_ee_WW_ecm{}".format(ecm),fname)
                    plot_signal_background(sig,bkg,ecm,channel,pf)
