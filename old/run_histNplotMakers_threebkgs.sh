#!/bin/bash
for ecm in  340 345 365 
do
    for channel in semihad had
    do
	for conf in  sig_vs_wwz_btagdown sig_vs_wwz  sig_vs_wwz_btagup
	do
	    fnameH=histmaker_WbWb_reco_${ecm}_${channel}_{conf}.py
	    cat histmaker_WbWb_reco.py > dummy.py
	    sed -e 's/CHANNELHERE/'$channel'/g;s/ECMHERE/'$ecm'/g;s/CONFHERE/'$conf'/g' dummy.py > $fnameH
	    fccanalysis run $fnameH
	    mv $fnameH jobs/
	done
    done
done
