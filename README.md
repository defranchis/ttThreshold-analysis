# ttThreshold-analysis

Install FCCAnalyses framework:

```
git clone --branch pre-edm4hep1 https://github.com/HEP-FCC/FCCAnalyses.git
cd FCCAnalyses
source ./setup.sh
fccanalyses build -j 6
cd ../..
```

Then clone this repository:
```
git clone git@github.com:ttThreshold-FCC/ttThreshold-analysis.git
cd ttThreshold-analysis
```

Before running, don't forget to setup the environment (every time!):
```
source ../FCCAnalyses/setup.sh
```

Download some example input files:

```
mkdir localSamples && cd localSamples
mkdir  p8_ee_WW_mumu_ecm240 && cd p8_ee_WW_mumu_ecm240
wget https://fccsw.web.cern.ch/tutorials/apr2023/tutorial2/p8_ee_WW_mumu_ecm240_edm4hep.root
cd ..
mkdir p8_ee_ZZ_mumubb_ecm240 && cd p8_ee_ZZ_mumubb_ecm240
wget https://fccsw.web.cern.ch/tutorials/apr2023/tutorial2/p8_ee_ZZ_mumubb_ecm240_edm4hep.root
cd ..
mkdir p8_ee_ZH_Zmumu_ecm240  && cd p8_ee_ZH_Zmumu_ecm240
wget https://fccsw.web.cern.ch//tutorials/apr2023/tutorial2/p8_ee_ZH_Zmumu_ecm240_edm4hep.root
cd ../..
```


You can also check out some examples:

- Analyse events with histmaker

```
fccanalysis run examples/histmaker_recoil.py
fccanalysis plots examples/plots_recoil.py
```

- Create flat ntuples and analyse events

```
fccanalysis run examples/treemaker_flavor.py
fccanalysis run examples/histmaker_flavor.py
fccanalysis plots examples/plots_flavor.py
```

For more info, check out the [FCCAnalyses tutorial](https://hep-fcc.github.io/fcc-tutorials/main/fast-sim-and-analysis/fccanalyses/doc/starterkit/FccFastSimAnalysis/Readme.html)
