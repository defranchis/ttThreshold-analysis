import uproot
import pandas as pd
import numpy as np
import os,sys
import re 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
sys.path.insert(0, "/afs/cern.ch/work/a/anmehta/public/FCC/ttThreshold-analysis/") #os.getcwd()+'../')
from treemaker_WbWb_reco import all_branches
#from sklearn.feature_selection import chi2
#from scipy.stats import chisquare

def if3(cond, iftrue, iffalse):
    return iftrue if cond else iffalse



ecm = '345'

nJ='1' #sys.argv[3]

    


pf="noflav"
print('this is the config',pf)

branches_toTrain=[]
branches_toTrain=all_branches


channel =  if3(nJ == '0','lep',if3(nJ=='1', 'semihad','had')) #'hadronic','semihad','lep'
base_dir="/eos/cms/store/cmst3/group/top/anmehta/FCC//output_condor_06092024/WbWb/{}/".format(channel)
sig="wzp6_ee_WbWb_{0}_ecm{1}".format(channel,ecm)
bkg="p8_ee_WW_ecm{}".format(ecm)



infile_s =  uproot.concatenate([os.path.join(base_dir,sig,f)+':events' for f in os.listdir(os.path.join(base_dir,sig)) if f.endswith(".root")],branches_toTrain ,library="pd")
infile_b =  uproot.concatenate([os.path.join(base_dir,bkg,f)+':events' for f in os.listdir(os.path.join(base_dir,bkg)) if f.endswith(".root")],branches_toTrain ,library="pd")

flavor_branches=[v for v in all_branches if '_is' in v or 'nbjets' in v]


infile_s=infile_s.drop(columns=flavor_branches, axis=1)
infile_b=infile_b.drop(columns=flavor_branches, axis=1)



print(infile_s)
print(pf)

pd_s = (infile_s)
pd_b = (infile_b)


pd_s['target'] = 1.
pd_b['target'] = 0.

pd_all = pd.concat([pd_s, pd_b])


X_train, X_test, y_train, y_test = train_test_split(pd_all.drop(columns=['target']), pd_all['target'], test_size=0.5) #train_test_split
print('starting to train with features',X_train.columns,len(X_train.columns))
bst = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, verbosity=0, objective='binary:logistic')
print('created model instance ')
bst.fit(X_train, y_train)
print('fitted the model')
preds = bst.predict_proba(X_test)
print('predictions')
pd_preds = pd.DataFrame()
pd_preds['target'] = y_test
pd_preds['preds'] = preds[:,1]


train_preds = bst.predict_proba(X_train)
pd_train_preds = pd.DataFrame()
pd_train_preds['target']  = y_train
pd_train_preds['preds'] = train_preds[:,1]


import matplotlib.pyplot as plt

# Plot signal and background predictions

bins=np.linspace(0,1,51)
n,bins,_=plt.hist(pd_preds[pd_preds['target'] == 1]['preds'], bins=bins, label='Signal-test', histtype='step',color='red', log=True)
mid = 0.5*(bins[1:] + bins[:-1])
plt.errorbar(mid, n, yerr=np.sqrt(n), fmt='none',ecolor='red')

n1,bins1,_=plt.hist(pd_preds[pd_preds['target'] == 0]['preds'], bins=bins, label='Background-test', histtype='step',color='blue', log=True)
mid1 = 0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(mid1, n1, yerr=np.sqrt(n1), fmt='none',ecolor='blue')
#ax.errorbar(bin_mids, n_SM, yerr = calculate_stat_bin_ucts(Y_pred[Y_test==0], w_SM, bins), fmt='none', ecolor='black', capsize=0)


n2,bins2,_=plt.hist(pd_train_preds[pd_train_preds['target'] == 1]['preds'], bins=bins, label='Signal-train', histtype='step',color='green', log=True)
mid2 = 0.5*(bins2[1:] + bins2[:-1])
plt.errorbar(mid2, n2, yerr=np.sqrt(n2), fmt='none',ecolor='green')

n3,bins3,_=plt.hist(pd_train_preds[pd_train_preds['target'] == 0]['preds'], bins=bins, label='Background-train',  histtype='step',color='orange', log=True)
mid3 = 0.5*(bins3[1:] + bins3[:-1])
plt.errorbar(mid3, n3, yerr=np.sqrt(n3), fmt='none',ecolor='orange')

#hist1=np.array(n)
#hist2=np.array(n2)
#chi2_scores, p_values = chi2(hist1,hist2)
#dist = chisquare(hist1, hist2)
#print("Chi-square distance:", dist.statistic)
#print("P-value:", dist.pvalue)

#chi = np.sum([((a - b) ** 2) / (a + b)  for (a, b) in zip(n, n2)])
#print('this',chi)

#print('Chi-square score:', chi2_scores,bins)
#print('p-value:', p_values)


plt.xlabel('Prediction')
plt.ylabel('Count')
plt.legend()
print('Saving model')
outdir = '/eos/user/a/anmehta/www/FCC_top/BDT_model/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

bst.save_model('{}/model_{}_{}_{}.json'.format(outdir,channel,ecm,pf))
plt.savefig('{}/preds_{}_{}_{}.png'.format(outdir,channel,ecm,pf))
plt.savefig('{}/preds_{}_{}_{}.pdf'.format(outdir,channel,ecm,pf))
plt.close()
