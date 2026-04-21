import os, copy, ROOT
import urllib
processList = {
#    "wzp6_ee_mumuqq_noCut_ecm160": {
#        "fraction": 1,
#        "crossSection": 1,
#    },
#    "wzp6_ee_munumuqq_noCut_ecm157":{
#        "fraction": 1,
#        "crossSection": 1,
#    },
    "wzp6_ee_munumuqq_noCut_ecm160":{
        "fraction": 1,
        "crossSection": 1,
    },
}


ROOT.gInterpreter.Declare('#include "TMath.h"')
available_ecm = ['160']#'340','345', '350', '355','365']

hadronic  = False
#semihad  = False
#lep      = False
ecm       = 160
#print(ecm)

saveExclJets = True
saveMCTruth = True
if not str(ecm) in available_ecm:
    raise ValueError("ecm value not in available_ecm")

channel = "CHANNELNAMEHERE"

if  channel not in ["lep","semihad","had"]:
    channel="semihad"
print(channel)    

lepFlav="mu"
#processList={key: value for key, value in all_processes.items() } #if str(ecm) in available_ecm and str(ecm) in key } # (True if str('p8_ee_WW_ecm'+ecm) in key else str('wzp6_ee_WbWb_ecm'+ecm) in key)}

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
prodTag     = "FCCee/winter2023/IDEA/"

#Optional: output directory, default is local running directoryp
outputDir   = "outputs/treemaker/lnuqq/{}".format(channel)


# additional/costom C++ functions, defined in header files (optional)
includePaths = ["examples/functions.h"]

## latest particle transformer model, trained on 9M jets in winter2023 samples
model_name = "fccee_flavtagging_edm4hep_wc" #"fccee_flavtagging_edm4hep_wc_v1"

## model files locally stored on /eos
eos_dir ="/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"
model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"    # "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)

url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

## get local file, else download from url
def get_file_path(url, filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)

def get_files(eos_dir, proc):
    files=[]
    basepath=os.path.join(eos_dir,proc)
    if os.path.exists(basepath):
        files =  [os.path.join(basepath,x) for x in os.listdir(basepath) if os.path.isfile(os.path.join(basepath, x)) ]
    return files

weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)

#inputDir = eos_dir

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
    InclusiveJetClusteringHelper,
)

jetFlavourHelper = None
jetFlavourHelper_R5 = None
jetClusteringHelper = None
jetClusteringHelper_R5 = None

all_branches = ["jet_res","res_jet1","res_jet2","lep_res","reco_moff","reco_mon","truth_moff","truth_mon","truth_lnuqq_mon","truth_lnuqq_moff","m_gen_lnuqq","p_lnu_status2","p_excljj","p_iso_lnu",
                "p_qq_fromele",    "nIsolep", "Isolep_p", 'Isolep_theta',"m_iso_lnu",'Isolep_phi',"Isolep_pt","Isolep_eta",
                "missing_p", "missing_p_theta", "missing_p_phi", "missing_p_res","missing_p_eta","missing_pt",
                #"muons_iso",
                "diff_RG_m_lnu",#"resP",
                "deltaM","deltaP","deltaPx","deltaPy","deltaPz","deltaP_gen","deltaPx_gen","deltaPy_gen","deltaPz_gen","deltaPt","deltaPt_gen",
                "diff_RG_p_lnu",
                "diff_RG_m_lnuqq",
                "diff_RG_m_qq",
                "diff_RG_p_qq",
                "p_iso_lnuexcljj","e_iso_lnuexcljj",
                "njets_R5"]
if saveMCTruth: all_branches+=["gen_leps_status1_p","ngen_leps_status2","gen_leps_status2_p","m_lnu_status1","m_genkt5_jj","m_qq_status2","m_qq_fromele","m_lnu_status2","ngen_leps_status1", "gen_lightquarks_p","Genjets_kt_e","m_genkt_ee_jj","Genjets_kt_ee_e","Genjets_kt_ee_p","Genjets_kt_ee_p1","Genjets_kt_ee_p2","res_jet2_qq_fromele","res_jet1_qq_fromele","res_jet2_qq","res_jet1_qq","truth_lnuqq_qqfromele_mon","truth_lnuqq_qqfromele_moff","mlnu_plus_mjj_reco","mlnu_plus_mqq_status2_truth","mlnu_plus_mqq_fromele_truth","p_qq_status2"]

if saveExclJets: all_branches+=[ "nRecoJets", "jet1_p", "jet2_p", "d_12","m_iso_lnuexcljj","jet1_pt","jet2_pt","jet1_eta","jet2_eta","jet1_phi","jet2_phi","jet1_mass","jet2_mass"]
#print('saving these branches',all_branches)
# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:

    # __________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):

        # __________________________________________________________
        # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2

        # define some aliases to be used later on
        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Electron0","Electron#0.index")

        # get all the leptons from the collection
        df = df.Define(
            "muons_all",
            "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)",
        )
        df = df.Define(
            "electrons_all",
            "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)",
        )

        # select leptons with momentum > 12 GeV
        df = df.Define(
            "muons_sel",
            "FCCAnalyses::ReconstructedParticle::sel_p(12)(muons_all)",
        )

        df = df.Define(
            "electrons_sel",
            "FCCAnalyses::ReconstructedParticle::sel_p(12)(electrons_all)",
        )

        
        # compute the muon isolation and store muons with an isolation cut of 0df = df.25 in a separate column muons_sel_iso
        df = df.Define(
            "muons_iso",
            "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(muons_sel, ReconstructedParticles)",
        )
        # compute the electron isolation and store electrons with an isolation cut of 0df = df.25 in a separate column electrons_sel_iso
        df = df.Define(
            "electrons_iso",
            "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(electrons_sel, ReconstructedParticles)",
        )

        df = df.Define(
            "muons_sel_iso",
            "FCCAnalyses::ZHfunctions::sel_iso(0.7)(muons_sel, muons_iso)",
        )

        df = df.Define(
            "electrons_sel_iso",
            "FCCAnalyses::ZHfunctions::sel_iso(0.7)(electrons_sel, electrons_iso)",
        )

        if channel == "had":
            #hadronic=True
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 0")
        elif  channel == "semihad":
            #semihad=True
            #df = df.Filter("muons_all.size() + electrons_all.size() > 0")
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 1")
        else:
            #lep=True
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 2")


            df = df.Define(
                "muons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons_all)"
            )

            df = df.Define(
                "muons_theta",
                "FCCAnalyses::ReconstructedParticle::get_theta(muons_all)",
            )
            df = df.Define(
                "muons_phi",
                "FCCAnalyses::ReconstructedParticle::get_phi(muons_all)",
            )
            df = df.Define(
                "muons_q",
                "FCCAnalyses::ReconstructedParticle::get_charge(muons_all)",
            )
            df = df.Define(
                "muons_n", "FCCAnalyses::ReconstructedParticle::get_n(muons_all)",
            )


            df = df.Define(
                "Isomuons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons_sel_iso)"
            )

            df = df.Define(
                "Isomuons_theta",
                "FCCAnalyses::ReconstructedParticle::get_theta(muons_sel_iso)",
            )
            df = df.Define(
                "Isomuons_phi",
                "FCCAnalyses::ReconstructedParticle::get_phi(muons_sel_iso)",
            )
            df = df.Define(
                "Isomuons_q",
                "FCCAnalyses::ReconstructedParticle::get_charge(muons_sel_iso)",
            )
            df = df.Define(
                "Isomuons_n", "FCCAnalyses::ReconstructedParticle::get_n(muons_sel_iso)",
            )

            

            df = df.Define(
                "Isoelectrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons_sel_iso)"
            )

            df = df.Define(
                "Isoelectrons_theta",
                "FCCAnalyses::ReconstructedParticle::get_theta(electrons_sel_iso)",
            )
            df = df.Define(
                "Isoelectrons_phi",
                "FCCAnalyses::ReconstructedParticle::get_phi(electrons_sel_iso)",
                )
            df = df.Define(
                "Isoelectrons_q",
                "FCCAnalyses::ReconstructedParticle::get_charge(electrons_sel_iso)",
                )
            df = df.Define(
                "Isoelectrons_n", "FCCAnalyses::ReconstructedParticle::get_n(electrons_sel_iso)",
            )
        print(df.GetColumnType("muons_sel_iso"))
        df = df.Define("Isoleps", "ROOT::VecOps::Concatenate(electrons_sel_iso,muons_sel_iso)") #####FIXME            )
        df = df.Define("Isoleps_p", "FCCAnalyses::ReconstructedParticle::get_p(Isoleps)")
        df = df.Define("Isoleps_px", "FCCAnalyses::ReconstructedParticle::get_px(Isoleps)")
        df = df.Define("Isoleps_py", "FCCAnalyses::ReconstructedParticle::get_py(Isoleps)")
        df = df.Define("Isoleps_pz", "FCCAnalyses::ReconstructedParticle::get_pz(Isoleps)")
        df = df.Define("Isoleps_e", "FCCAnalyses::ReconstructedParticle::get_e(Isoleps)")
        
        ## here cluster jets in the events but first remove muons from the list of
        ## reconstructed particles
        
        ## create a new collection of reconstructed particles removing muons with p>12
        df = df.Define(
            "ReconstructedParticlesNoMuons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles,muons_sel_iso)",
        )
        df = df.Define(
            "ReconstructedParticlesNoMuNoEl",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoMuons,electrons_sel_iso)",
        )


        ## perform exclusive jet clustering
        global jetClusteringHelper
        global jetFlavourHelper
        global jetFlavourHelper_R5
        global jetClusteringHelper_R5
        
        ## define jet and run clustering parameters
        ## name of collections in EDM root files
        collections = {
            "GenParticles": "Particle",
            "PFParticles": "ReconstructedParticles",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "EFlowTrack_1",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "dNdx": "EFlowTrack_2",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }

        nJets = 2 if  channel == "semihad" else 4

        collections_noleps = copy.deepcopy(collections)
        collections_noleps["PFParticles"] = "ReconstructedParticlesNoMuNoEl"
        if saveExclJets:
            jetClusteringHelper = ExclusiveJetClusteringHelper(
                collections_noleps["PFParticles"], nJets
            )
        
        jetClusteringHelper_R5  = InclusiveJetClusteringHelper(
            collections_noleps["PFParticles"], 0.5, 10, "R5"
        )
        if  saveExclJets:
            df = jetClusteringHelper.define(df)
        df = jetClusteringHelper_R5.define(df)
        ## define jet flavour tagging parameters
        if saveExclJets:
            jetFlavourHelper = JetFlavourHelper(
                collections_noleps,
                jetClusteringHelper.jets,
                jetClusteringHelper.constituents,
            )
        jetFlavourHelper_R5 = JetFlavourHelper(
            collections_noleps,
            jetClusteringHelper_R5.jets,
            jetClusteringHelper_R5.constituents,
            "R5",
        )
        if saveExclJets:    df = jetFlavourHelper.define(df)
        
        df = jetFlavourHelper_R5.define(df)
        ## tagger inference
        if  saveExclJets: df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)
        df = jetFlavourHelper_R5.inference(weaver_preproc, weaver_model,df)

        df = df.Define(
            "Isolep_p", "muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_p(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_p(electrons_sel_iso)[0] : -999) "
        )
        df = df.Define(
            "Isolep_pt", "muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_pt(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_pt(electrons_sel_iso)[0] : -999) "
        )
        df = df.Define(
            "Isolep_eta", "muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_eta(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_eta(electrons_sel_iso)[0] : -999) "
        )

        df = df.Define(
            'Isolep_theta', 'muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_theta(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_theta(electrons_sel_iso)[0] : -999) '
        )
        df = df.Define(
            'Isolep_phi', 'muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_phi(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_phi(electrons_sel_iso)[0] : -999) '
        )
        df = df.Define("nIsolep","electrons_sel_iso.size()+muons_sel_iso.size()")


        df = df.Define("missing_p","FCCAnalyses::ReconstructedParticle::get_p(MissingET)[0]",)
        df = df.Define("missing_pt","FCCAnalyses::ReconstructedParticle::get_pt(MissingET)[0]",)
        df = df.Define('missing_p_theta', 'ReconstructedParticle::get_theta(MissingET)[0]',)
        df = df.Define('missing_p_phi', 'ReconstructedParticle::get_phi(MissingET)[0]',)
        df = df.Define('missing_p_eta', 'ReconstructedParticle::get_eta(MissingET)[0]',)
        #df = df.Define("lep_p4", "ROOT::Math::PtEtaPhiMVector(Isolep_pt, Isolep_eta,Isolep_phi,0.0)"),

        df = df.Define(
            "Wlep_reco",
            """
            if (nIsolep < 1) return -1.0;
            TLorentzVector Isolep, nu;
            Isolep.SetPxPyPzE(
            Isolep_p * cos(Isolep_phi) *sin(Isolep_theta),
            Isolep_p * sin(Isolep_phi) * sin(Isolep_theta),
            Isolep_p * cos(Isolep_theta),
            Isolep_p
            );
            nu.SetPxPyPzE(
            missing_p * cos(missing_p_phi) * sin(missing_p_theta),
            missing_p * sin(missing_p_phi) * sin(missing_p_theta),
            missing_p * cos(missing_p_theta),
            missing_p
            );
            return (Isolep + nu);
            """
            )
        
        df = df.Define("m_iso_lnu", "Wlep_reco.M()");
        df = df.Define("p_iso_lnu", "Wlep_reco.P()");
        
        if    saveExclJets:
            df = df.Define(
                "jets_p4",
                "JetConstituentsUtils::compute_tlv_jets({})".format(
                    jetClusteringHelper.jets
                ),
            )
        
        df = df.Define(
            "jets_R5_p4",
            "JetConstituentsUtils::compute_tlv_jets({})".format(
                jetClusteringHelper_R5.jets
            ),
        )

        df = df.Define("jets_R5_p",           "JetClusteringUtils::get_p({})".format(jetClusteringHelper_R5.jets))
        df = df.Define("jets_R5_theta",       "JetClusteringUtils::get_theta({})".format(jetClusteringHelper_R5.jets))
        df = df.Define("jet1_R5_p","jets_R5_p[0]")
        df = df.Define("jet2_R5_p","jets_R5_p[1]")
        df = df.Define("njets_R5","return int(jets_R5_p.size())")

        df = df.Define("jet1_R5_theta","jets_R5_theta[0]")
        df = df.Define("jet2_R5_theta","jets_R5_theta[1]")
        df = df.Define("m_R5_jj",  "jets_R5_p4.size() >   1 ? JetConstituentsUtils::InvariantMass(jets_R5_p4[0],  jets_R5_p4[1])  : -999")

        ROOT.gInterpreter.Declare("""
        #include "ROOT/RVec.hxx"
        using namespace ROOT::VecOps;

        float jetAngle(
        float px1,float py1,float pz1,
        float px2,float py2,float pz2){
        
        float dot = px1*px2 + py1*py2 + pz1*pz2;
        
        float mag1 = sqrt(px1*px1 + py1*py1 + pz1*pz1);
        float mag2 = sqrt(px2*px2 + py2*py2 + pz2*pz2);
        
        return acos(dot/(mag1*mag2));
        }

RVec<float> matchJetsAndComputeResolution(

    const RVec<float>& reco_px,
    const RVec<float>& reco_py,
    const RVec<float>& reco_pz,
    const RVec<float>& reco_E,
    const RVec<float>& truth_px,
    const RVec<float>& truth_py,
    const RVec<float>& truth_pz,
    const RVec<float>& truth_E
){

    RVec<float> resolution;
    for(size_t i=0;i<reco_E.size();i++){
        float bestAngle = 999.;
        int bestMatch = -1;
        for(size_t j=0;j<truth_E.size();j++){
            float ang = jetAngle(reco_px[i], reco_py[i], reco_pz[i],truth_px[j], truth_py[j], truth_pz[j]);
            if(ang < bestAngle){
                bestAngle = ang;
                bestMatch = j;
            }
        }
        if(bestMatch >= 0){
//        std::cout<<"found one match"<<std::endl;
            float resp =(reco_E[i] - truth_E[bestMatch])/truth_E[bestMatch];
            resolution.push_back(resp);
        }
    }

    return resolution;
}
""")


        # get all the leptons from the collectio
        if  saveMCTruth:
            df = df.Alias("Particle0", "Particle#0.index") #parents
            df = df.Alias("Particle1", "Particle#1.index") #daughters
            df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
            df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
            df = df.Define("status1parts",           "FCCAnalyses::MCParticle::sel_genStatus(1)(Particle)")
            df = df.Define("status2parts",           "FCCAnalyses::MCParticle::sel_genStatus(2)(Particle)")
            df = df.Define("nstatus1parts",          "FCCAnalyses::MCParticle::get_n(status1parts)")
            if lepFlav=="mu":
                df = df.Define("gen_leps_status1",                "FCCAnalyses::MCParticle::sel_genleps(13,14,true)(status1parts)") #11
                df = df.Define("gen_leps_status2",                "FCCAnalyses::MCParticle::sel_genleps(13,14,true)(status2parts)") #11
                df = df.Define("neutrinos",                       "FCCAnalyses::MCParticle::sel_genleps(14,14, true)(status1parts)")
                df = df.Define("neutrinos_2",                     "FCCAnalyses::MCParticle::sel_genleps(14,14, true)(status2parts)")
            elif lepFlav=="el":
                df = df.Define("gen_leps_status1",                "FCCAnalyses::MCParticle::sel_genleps(11,12,true)(status1parts)") #11
                df = df.Define("gen_leps_status2",                "FCCAnalyses::MCParticle::sel_genleps(11,12,true)(status2parts)") #11
                df = df.Define("neutrinos",                       "FCCAnalyses::MCParticle::sel_genleps(12,12, true)(status1parts)")
                df = df.Define("neutrinos_2",                     "FCCAnalyses::MCParticle::sel_genleps(12,12, true)(status2parts)")
            else:
                df = df.Define("gen_leps_status1",                "FCCAnalyses::MCParticle::sel_genleps(11,13,true)(status1parts)") #11
                df = df.Define("gen_leps_status2",                "FCCAnalyses::MCParticle::sel_genleps(11,13,true)(status2parts)") #11
                df = df.Define("neutrinos",                       "FCCAnalyses::MCParticle::sel_genleps(12,13, true)(status1parts)")
                df = df.Define("neutrinos_2",                      "FCCAnalyses::MCParticle::sel_genleps(12,13, true)(status2parts)")
            
            df = df.Define("ngen_leps_status1",               "FCCAnalyses::MCParticle::get_n(gen_leps_status1)")
            df = df.Define("ngen_leps_status2",               "FCCAnalyses::MCParticle::get_n(gen_leps_status2)")
            df = df.Define("gen_leps_status2_p",              "FCCAnalyses::MCParticle::get_p(gen_leps_status2)")
            df = df.Define("gen_leps_status1_p",              "FCCAnalyses::MCParticle::get_p(gen_leps_status1)")
            df = df.Define("gen_leps_status1_px",             "FCCAnalyses::MCParticle::get_px(gen_leps_status1)")
            df = df.Define("gen_leps_status1_py",             "FCCAnalyses::MCParticle::get_py(gen_leps_status1)")
            df = df.Define("gen_leps_status1_pz",             "FCCAnalyses::MCParticle::get_pz(gen_leps_status1)")

            df = df.Define("gen_leps_status2_px",             "FCCAnalyses::MCParticle::get_px(gen_leps_status2)")
            df = df.Define("gen_leps_status2_py",             "FCCAnalyses::MCParticle::get_py(gen_leps_status2)")
            df = df.Define("gen_leps_status2_pz",             "FCCAnalyses::MCParticle::get_pz(gen_leps_status2)")

            
            df = df.Define("gen_leps_status1_theta",  "FCCAnalyses::MCParticle::get_theta(gen_leps_status1)")
            df = df.Define("gen_leps_status1_phi",    "FCCAnalyses::MCParticle::get_phi(gen_leps_status1)")
            df = df.Define("gen_leps_status1_charge", "FCCAnalyses::MCParticle::get_charge(gen_leps_status1)")
            df = df.Define("gen_leps_status1_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_leps_status1)")
            df = df.Define("gen_leps_status1_e",      "FCCAnalyses::MCParticle::get_e(gen_leps_status1)")
            df = df.Define("gen_leps_status2_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_leps_status2)")
            df = df.Define("gen_leps_status2_theta",  "FCCAnalyses::MCParticle::get_theta(gen_leps_status2)")
            df = df.Define("gen_leps_status2_phi",    "FCCAnalyses::MCParticle::get_phi(gen_leps_status2)")
            df = df.Define("gen_leps_status2_e",      "FCCAnalyses::MCParticle::get_e(gen_leps_status2)")
            df = df.Define("gen_neutrinos_status1",  "FCCAnalyses::MCParticle::sel_genleps(14,16, true)(neutrinos)")
            df = df.Define("ngen_neutrinos_status1", "FCCAnalyses::MCParticle::get_n(gen_neutrinos_status1)")

            df = df.Define("gen_neutrinos_status1_p",      "FCCAnalyses::MCParticle::get_p(gen_neutrinos_status1)")
            df = df.Define("gen_neutrinos_status1_theta",  "FCCAnalyses::MCParticle::get_theta(gen_neutrinos_status1)")
            df = df.Define("gen_neutrinos_status1_phi",    "FCCAnalyses::MCParticle::get_phi(gen_neutrinos_status1)")
            df = df.Define("gen_neutrinos_status1_charge", "FCCAnalyses::MCParticle::get_charge(gen_neutrinos_status1)")
            df = df.Define("gen_neutrinos_status1_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_neutrinos_status1)")
            df = df.Define("gen_neutrinos_status1_e",      "FCCAnalyses::MCParticle::get_e(gen_neutrinos_status1)")
            df = df.Define("gen_neutrinos_status2",        "FCCAnalyses::MCParticle::sel_genleps(14,16, true)(neutrinos_2)")
            df = df.Define("ngen_neutrinos_status2",       "FCCAnalyses::MCParticle::get_n(gen_neutrinos_status2)")
            df = df.Define("gen_neutrinos_status2_p",      "FCCAnalyses::MCParticle::get_p(gen_neutrinos_status2)")
            df = df.Define("gen_neutrinos_status2_theta",  "FCCAnalyses::MCParticle::get_theta(gen_neutrinos_status2)")
            df = df.Define("gen_neutrinos_status2_phi",    "FCCAnalyses::MCParticle::get_phi(gen_neutrinos_status2)")
            df = df.Define("gen_neutrinos_status2_charge", "FCCAnalyses::MCParticle::get_charge(gen_neutrinos_status2)")
            df = df.Define("gen_neutrinos_status2_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_neutrinos_status2)")
            df = df.Define("gen_neutrinos_status2_e",      "FCCAnalyses::MCParticle::get_e(gen_neutrinos_status2)")
            df = df.Define('gen_lightquarks_fromele',     'FCCAnalyses::MCParticle::sel_lightQuarks_fromele(true)(Particle,Particle0)')
            df = df.Define("ngen_partons_fromele",         "FCCAnalyses::MCParticle::get_n(gen_lightquarks_fromele)");
            df = df.Define("gen_lightquarks_fromele_p",    "FCCAnalyses::MCParticle::get_p(gen_lightquarks_fromele)")
            df = df.Define("gen_lightquarks_fromele_px",    "FCCAnalyses::MCParticle::get_px(gen_lightquarks_fromele)")
            df = df.Define("gen_lightquarks_fromele_py",    "FCCAnalyses::MCParticle::get_py(gen_lightquarks_fromele)")
            df = df.Define("gen_lightquarks_fromele_pz",    "FCCAnalyses::MCParticle::get_pz(gen_lightquarks_fromele)")

            df = df.Define("gen_lightquarks_fromele_theta","FCCAnalyses::MCParticle::get_theta(gen_lightquarks_fromele)")
            df = df.Define("gen_lightquarks_fromele_phi",  "FCCAnalyses::MCParticle::get_phi(gen_lightquarks_fromele)")
            df = df.Define("gen_lightquarks_fromele_e",    "FCCAnalyses::MCParticle::get_e(gen_lightquarks_fromele)")
            df = df.Define('gen_lightquarks',              'FCCAnalyses::MCParticle::sel_lightQuarks(true)(status2parts)')
            df = df.Define("ngen_partons",                 "FCCAnalyses::MCParticle::get_n(gen_lightquarks)");

            df = df.Define("gen_lightquarks_p",      "FCCAnalyses::MCParticle::get_p(gen_lightquarks)")
            df = df.Define("gen_lightquarks_px",      "FCCAnalyses::MCParticle::get_px(gen_lightquarks)")
            df = df.Define("gen_lightquarks_py",      "FCCAnalyses::MCParticle::get_py(gen_lightquarks)")
            df = df.Define("gen_lightquarks_pz",      "FCCAnalyses::MCParticle::get_pz(gen_lightquarks)")

                    
            df = df.Define("gen_lightquarks_theta",  "FCCAnalyses::MCParticle::get_theta(gen_lightquarks)")
            df = df.Define("gen_lightquarks_phi",    "FCCAnalyses::MCParticle::get_phi(gen_lightquarks)")
            df = df.Define("gen_lightquarks_charge", "FCCAnalyses::MCParticle::get_charge(gen_lightquarks)")
            df = df.Define("gen_lightquarks_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_lightquarks)")
            df = df.Define("gen_lightquarks_e",      "FCCAnalyses::MCParticle::get_e(gen_lightquarks)")
            df = df.Define("gen_lightquarks_mother_pdgId", "FCCAnalyses::MCParticle::get_leptons_origin(gen_lightquarks,Particle,Particle0)")

            df = df.Filter("ngen_leps_status1 > 1 && ngen_partons_fromele > 1")
            df = df.Define("Whad_gen_status2",
                """
                if (gen_lightquarks_p.size() < 2) return -1.0;
                TLorentzVector q1,q2;
                q1.SetPxPyPzE(
                gen_lightquarks_p[1] * cos(gen_lightquarks_phi[1]) *sin(gen_lightquarks_theta[1]),
                gen_lightquarks_p[1] * sin(gen_lightquarks_phi[1]) * sin(gen_lightquarks_theta[1]),
                gen_lightquarks_p[1] * cos(gen_lightquarks_theta[1]),
                gen_lightquarks_e[1]
                );
                q2.SetPxPyPzE(
                gen_lightquarks_p[0] * cos(gen_lightquarks_phi[0]) * sin(gen_lightquarks_theta[0]),
                gen_lightquarks_p[0] * sin(gen_lightquarks_phi[0]) * sin(gen_lightquarks_theta[0]),
                gen_lightquarks_p[0] * cos(gen_lightquarks_theta[0]),
                gen_lightquarks_e[0]
                );
                return (q1 + q2);
                """
            )
            df = df.Define("m_qq_status2","Whad_gen_status2.M()")
            df = df.Define("p_qq_status2","Whad_gen_status2.P()")
            df = df.Define("Whad_gen_qq_fromele",
                """
                if (gen_lightquarks_fromele_p.size() < 2) return -1.0;
                TLorentzVector lep, nu;
                lep.SetPxPyPzE(
                gen_lightquarks_fromele_p[1] * cos(gen_lightquarks_fromele_phi[1]) *sin(gen_lightquarks_fromele_theta[1]),
                gen_lightquarks_fromele_p[1] * sin(gen_lightquarks_fromele_phi[1]) * sin(gen_lightquarks_fromele_theta[1]),
                gen_lightquarks_fromele_p[1] * cos(gen_lightquarks_fromele_theta[1]),
                gen_lightquarks_fromele_e[1]
                );
                nu.SetPxPyPzE(
                gen_lightquarks_fromele_p[0] * cos(gen_lightquarks_fromele_phi[0]) * sin(gen_lightquarks_fromele_theta[0]),
                gen_lightquarks_fromele_p[0] * sin(gen_lightquarks_fromele_phi[0]) * sin(gen_lightquarks_fromele_theta[0]),
                gen_lightquarks_fromele_p[0] * cos(gen_lightquarks_fromele_theta[0]),
                gen_lightquarks_fromele_e[0]
                );
                return (lep + nu);
                """
            )
            df = df.Define("p_qq_fromele","Whad_gen_qq_fromele.P()");
            df = df.Define("m_qq_fromele","Whad_gen_qq_fromele.M()");

            
            
            #print('this',df["gen_lep_status2_p"].to_string(index=False))        
            df = df.Define(
                "GenParticlesNoEMuNoNu_status1",
                "FCCAnalyses::MCParticle::remove(status2parts,gen_leps_status1)",
            )
            

            df = df.Define("GP_px",         "MCParticle::get_px(GenParticlesNoEMuNoNu_status1)")
            df = df.Define("GP_py" ,        "MCParticle::get_py(GenParticlesNoEMuNoNu_status1)")
            df = df.Define("GP_pz" ,        "MCParticle::get_pz(GenParticlesNoEMuNoNu_status1)")
            df = df.Define("GP_e"  ,        "MCParticle::get_e(GenParticlesNoEMuNoNu_status1)")
            df = df.Define("GP_m"  ,        "MCParticle::get_mass(GenParticlesNoEMuNoNu_status1)")
            df = df.Define("GP_q"  ,        "MCParticle::get_charge(GenParticlesNoEMuNoNu_status1)")
            df = df.Define("GP_p"  ,        "MCParticle::get_p(GenParticlesNoEMuNoNu_status1)")
            df = df.Define("GP_pdgId"  ,    "MCParticle::get_pdg(GenParticlesNoEMuNoNu_status1)")
            df = df.Define("GP_status"  ,   "MCParticle::get_genStatus(GenParticlesNoEMuNoNu_status1)")
            
            df = df.Define("pseudo_jets",         "JetClusteringUtils::set_pseudoJets_xyzm(GP_px, GP_py, GP_pz, GP_m)")
            df = df.Define("FCCAnalysesJets_kt",  "JetClustering::clustering_kt(0.5, 2, 4, 0, 20)(pseudo_jets)")
            df = df.Define("Genjets_kt",          "JetClusteringUtils::get_pseudoJets(FCCAnalysesJets_kt)")
            df = df.Define("Genjets_kt_e",        "JetClusteringUtils::get_e(Genjets_kt)")
            df = df.Define("Genjets_kt_px",       "JetClusteringUtils::get_px(Genjets_kt)")
            df = df.Define("Genjets_kt_py",       "JetClusteringUtils::get_py(Genjets_kt)")
            df = df.Define("Genjets_kt_pz",       "JetClusteringUtils::get_pz(Genjets_kt)")
            df = df.Define("Genjets_kt_m",        "JetClusteringUtils::get_m(Genjets_kt)")
            df = df.Define("FCCAnalysesJets_kt_ee",  "JetClustering::clustering_ee_kt(2, 5, 0, 0)(pseudo_jets)")
            df = df.Define("Genjets_kt_ee",          "JetClusteringUtils::get_pseudoJets(FCCAnalysesJets_kt_ee)")
            df = df.Define("Genjets_kt_ee_p",        "JetClusteringUtils::get_p(Genjets_kt_ee)")
            df = df.Define("Genjets_kt_ee_p1",       "Genjets_kt_ee_p[0]");
            df = df.Define("Genjets_kt_ee_p2",       "Genjets_kt_ee_p[1]");
            df = df.Define("Genjets_kt_ee_e",        "JetClusteringUtils::get_e(Genjets_kt_ee)")
            df = df.Define("Genjets_kt_ee_px",       "JetClusteringUtils::get_px(Genjets_kt_ee)")
            df = df.Define("Genjets_kt_ee_py",       "JetClusteringUtils::get_py(Genjets_kt_ee)")
            df = df.Define("Genjets_kt_ee_pz",       "JetClusteringUtils::get_pz(Genjets_kt_ee)")
            df = df.Define("Genjets_kt_ee_m",        "JetClusteringUtils::get_m(Genjets_kt_ee)")

            
            df = df.Define(
                "m_genkt5_jj",
                """
                if (Genjets_kt_e.size() < 2) return -1.0;
                TLorentzVector q1, q2;
                q1.SetPxPyPzE(Genjets_kt_px[0], Genjets_kt_py[0], Genjets_kt_pz[0], Genjets_kt_e[0]);
                q2.SetPxPyPzE(Genjets_kt_px[1], Genjets_kt_py[1], Genjets_kt_pz[1], Genjets_kt_e[1]);
                return (q1 + q2).M();
                """
            )
            df = df.Define(
                "m_genkt_ee_jj",
                """
                if (Genjets_kt_ee_e.size() < 2) return -1.0;
                TLorentzVector q1, q2;
                q1.SetPxPyPzE(Genjets_kt_ee_px[0], Genjets_kt_ee_py[0], Genjets_kt_ee_pz[0], Genjets_kt_ee_e[0]);
                q2.SetPxPyPzE(Genjets_kt_ee_px[1], Genjets_kt_ee_py[1], Genjets_kt_ee_pz[1], Genjets_kt_ee_e[1]);
                return (q1 + q2).M();
                """
            )

            
            df = df.Define(
                "m_lnu_status1",
                """
                if (gen_neutrinos_status1.size() + gen_leps_status1.size() < 2) return -1.0;
                TLorentzVector lep, nu;
                lep.SetPxPyPzE(
                gen_leps_status1_p[0] * cos(gen_leps_status1_phi[0]) *sin(gen_leps_status1_theta[0]),
                gen_leps_status1_p[0] * sin(gen_leps_status1_phi[0]) * sin(gen_leps_status1_theta[0]),
                gen_leps_status1_p[0] * cos(gen_leps_status1_theta[0]),
                gen_leps_status1_e[0]
                );
                nu.SetPxPyPzE(
                gen_neutrinos_status1_p[0] * cos(gen_neutrinos_status1_phi[0]) * sin(gen_neutrinos_status1_theta[0]),
                gen_neutrinos_status1_p[0] * sin(gen_neutrinos_status1_phi[0]) * sin(gen_neutrinos_status1_theta[0]),
                gen_neutrinos_status1_p[0] * cos(gen_neutrinos_status1_theta[0]),
                gen_neutrinos_status1_e[0]
                );
                return (lep + nu).M();
                """
            )

            df = df.Define(
                "m_lnu_status2",
                """
                if (gen_leps_status2.size() < 2) return -1.0;
                TLorentzVector lep, nu;
                lep.SetPxPyPzE(
                gen_leps_status2_p[0] * cos(gen_leps_status2_phi[0]) *sin(gen_leps_status2_theta[0]),
                gen_leps_status2_p[0] * sin(gen_leps_status2_phi[0]) * sin(gen_leps_status2_theta[0]),
                gen_leps_status2_p[0] * cos(gen_leps_status2_theta[0]),
                gen_leps_status2_e[0]
                );
                nu.SetPxPyPzE(
                gen_leps_status2_p[1] * cos(gen_leps_status2_phi[1]) * sin(gen_leps_status2_theta[1]),
                gen_leps_status2_p[1] * sin(gen_leps_status2_phi[1]) * sin(gen_leps_status2_theta[1]),
                gen_leps_status2_p[1] * cos(gen_leps_status2_theta[1]),
                gen_leps_status2_e[1]
                );
                return (lep + nu).M();
                """
            )
            
            df = df.Define(
                "p_lnu_status2",
                """
                if (gen_leps_status2.size() < 2) return -1.0;
                TLorentzVector lep, nu;
                lep.SetPxPyPzE(
                gen_leps_status2_p[0] * cos(gen_leps_status2_phi[0]) *sin(gen_leps_status2_theta[0]),
                gen_leps_status2_p[0] * sin(gen_leps_status2_phi[0]) * sin(gen_leps_status2_theta[0]),
                gen_leps_status2_p[0] * cos(gen_leps_status2_theta[0]),
                gen_leps_status2_e[0]
                );
                nu.SetPxPyPzE(
                gen_leps_status2_p[1] * cos(gen_leps_status2_phi[1]) * sin(gen_leps_status2_theta[1]),
                gen_leps_status2_p[1] * sin(gen_leps_status2_phi[1]) * sin(gen_leps_status2_theta[1]),
                gen_leps_status2_p[1] * cos(gen_leps_status2_theta[1]),
                gen_leps_status2_e[1]
                );
                return (lep + nu).P();
                """
            )



            df = df.Define(
            "m_gen_lnuqq",
            """
            if (gen_leps_status2.size() < 2 || gen_lightquarks_fromele_p.size() < 2) return -1.0;
            TLorentzVector Isolep, nu,j1,j2;
            Isolep.SetPxPyPzE(gen_leps_status2_px[0],gen_leps_status2_py[0],gen_leps_status2_pz[0],gen_leps_status2_p[0]);
            nu.SetPxPyPzE(gen_leps_status2_px[1],gen_leps_status2_py[1],gen_leps_status2_pz[1],gen_leps_status2_p[1]);
            j1.SetPxPyPzE(gen_lightquarks_fromele_px[0],gen_lightquarks_fromele_py[0],gen_lightquarks_fromele_pz[0],gen_lightquarks_fromele_e[0]);
            j2.SetPxPyPzE(gen_lightquarks_fromele_px[1],gen_lightquarks_fromele_py[1],gen_lightquarks_fromele_pz[1],gen_lightquarks_fromele_e[1]);
            return (Isolep + nu + j1 + j2).M();
            """
	    )


            
        print("moving to jet")    
        if  saveExclJets:
            df = df.Define("jet1", "jets_p4[0]")
            df = df.Define("jet2", "jets_p4[1]")

            df = df.Define("jet1_pt","jet1.Pt()")
            df = df.Define("jet2_pt","jet2.Pt()")
            df = df.Define("jet1_eta","jet1.Eta()")
            df = df.Define("jet2_eta","jet2.Eta()")
            df = df.Define("jet1_mass","jet1.M()")
            df = df.Define("jet2_mass","jet2.M()")

            
            df = df.Define("jet1_p","jet1.P()")
            df = df.Define("jet2_p","jet2.P()")
            df = df.Define("recoJet_theta", "JetClusteringUtils::get_theta(jet)")
            df = df.Define("jet1_theta","recoJet_theta[0]")
            df = df.Define("jet2_theta","recoJet_theta[1]")
            df = df.Define("recoJet_e", "JetClusteringUtils::get_e(jet)")
            df = df.Define("recoJet_phi", "JetClusteringUtils::get_phi_std(jet)")
            df = df.Define("recoJet_px", "JetClusteringUtils::get_px(jet)")
            df = df.Define("recoJet_py", "JetClusteringUtils::get_py(jet)")
            df = df.Define("recoJet_pz", "JetClusteringUtils::get_pz(jet)")
            df = df.Define("jet1_phi","recoJet_phi[0]")
            df = df.Define("jet2_phi","recoJet_phi[1]")
            df = df.Define("nRecoJets", "jets_p4.size()")
            df = df.Filter("nRecoJets == 2")
            df = df.Define("d_12", "JetClusteringUtils::get_exclusive_dmerge(_jet, 1)")
            df = df.Define("m_excl_jj",  "JetConstituentsUtils::InvariantMass(jets_p4[0],  jets_p4[1])")
            df = df.Define("Whad_reco",
            """
            if (nIsolep < 1 || nRecoJets < 2) return -1.0;
            TLorentzVector j1,j2,Whad;
            j1.SetPxPyPzE(jets_p4[0].Px(),jets_p4[0].Py(),jets_p4[0].Pz(),jets_p4[0].E());
            j2.SetPxPyPzE(jets_p4[1].Px(),jets_p4[1].Py(),jets_p4[1].Pz(),jets_p4[1].E())
            Whad=j1+j2;
            return Whad;
            """
	    )

            df = df.Define("WW_iso_lnuexcljj","(Wlep_reco+Whad_reco)")
#            "m_iso_lnuexcljj",
#            """
#            if (nIsolep < 1 || nRecoJets < 2) return -1.0;
#            TLorentzVector Isolep, nu,j1,j2;
#            Isolep.SetPxPyPzE(Isolep_p * cos(Isolep_phi) *sin(Isolep_theta),Isolep_p * sin(Isolep_phi) * sin(Isolep_theta),Isolep_p * cos(Isolep_theta),Isolep_p);
#            nu.SetPxPyPzE(missing_p * cos(missing_p_phi) * sin(missing_p_theta),missing_p * sin(missing_p_phi) * sin(missing_p_theta),missing_p * cos(missing_p_theta),missing_p);
#            j1.SetPxPyPzE(jets_p4[0].Px(),jets_p4[0].Py(),jets_p4[0].Pz(),jets_p4[0].E());
#            j2.SetPxPyPzE(jets_p4[1].Px(),jets_p4[1].Py(),jets_p4[1].Pz(),jets_p4[1].E());
#            return (Isolep + nu + j1 + j2);
#            """
#	    )
            df = df.Define("m_iso_lnuexcljj","WW_iso_lnuexcljj.M()")
            df = df.Define("p_iso_lnuexcljj","WW_iso_lnuexcljj.P()")
            df = df.Define("e_iso_lnuexcljj","WW_iso_lnuexcljj.E()")
            
            df = df.Define("deltaP","sqrt(pow((Wlep_reco.Px() + Whad_reco.Px()),2) + pow((Wlep_reco.Py() + Whad_reco.Py()),2) + pow((Wlep_reco.Pz() +Whad_reco.Pz()),2))");
            df = df.Define("deltaPt",,"sqrt(pow((Wlep_reco.Px() + Whad_reco.Px()),2) + pow((Wlep_reco.Py() + Whad_reco.Py()),2))");
            df = df.Define("deltaPx",,"sqrt(pow((Wlep_reco.Px() + Whad_reco.Px()),2))");
            df = df.Define("deltaPy",,"sqrt(pow((Wlep_reco.Py() + Whad_reco.Py()),2))");
            df = df.Define("deltaPz",,"sqrt(pow((Wlep_reco.Pz() + Whad_reco.Pz()),2))");
            
            print("going now")

            df = df.Define(
                "deltaM",
            """
            if (nIsolep < 1 || nRecoJets < 2) return -1.0;
            TLorentzVector Isolep, nu,j1,j2, FS,Delta,P_initial;
            P_initial.SetPxPyPzE(0,0,0,160);
            Isolep.SetPxPyPzE(Isolep_p * cos(Isolep_phi) *sin(Isolep_theta),Isolep_p * sin(Isolep_phi) * sin(Isolep_theta),Isolep_p * cos(Isolep_theta),Isolep_p);
            nu.SetPxPyPzE(missing_p * cos(missing_p_phi) * sin(missing_p_theta),missing_p * sin(missing_p_phi) * sin(missing_p_theta),missing_p * cos(missing_p_theta),missing_p);
            j1.SetPxPyPzE(jets_p4[0].Px(),jets_p4[0].Py(),jets_p4[0].Pz(),jets_p4[0].E());
            j2.SetPxPyPzE(jets_p4[1].Px(),jets_p4[1].Py(),jets_p4[1].Pz(),jets_p4[1].E());
            FS= Isolep + nu + j1 + j2;
            Delta=P_initial-FS;

            return Delta.M();
            """
	    )

            
            df = df.Define("p_excljj","Whad_reco.P()");    
            df = df.Define("jet_res","matchJetsAndComputeResolution(""recoJet_px, recoJet_py, recoJet_pz, recoJet_e,""Genjets_kt_ee_px, Genjets_kt_ee_py, Genjets_kt_ee_pz, Genjets_kt_ee_e)" )
            df = df.Define("res_jet1", "jet_res[0]")
            df = df.Define("res_jet2", "jet_res[1]")
            
            df = df.Define("jet_res_qq_fromele","matchJetsAndComputeResolution(""recoJet_px, recoJet_py, recoJet_pz, recoJet_e,""gen_lightquarks_fromele_px, gen_lightquarks_fromele_py, gen_lightquarks_fromele_pz, gen_lightquarks_fromele_e)" )
            df = df.Define("res_jet1_qq_fromele", "jet_res_qq_fromele[0]")
            df = df.Define("res_jet2_qq_fromele", "jet_res_qq_fromele[1]")
            
            df = df.Define("jet_res_qq","matchJetsAndComputeResolution(""recoJet_px, recoJet_py, recoJet_pz, recoJet_e,""gen_lightquarks_px, gen_lightquarks_py, gen_lightquarks_pz, gen_lightquarks_e)" )
            df = df.Define("res_jet1_qq", "jet_res_qq[0]")
            df = df.Define("res_jet2_qq", "jet_res_qq[1]")

            
            df = df.Define("lep_res","matchJetsAndComputeResolution(""Isoleps_px, Isoleps_py, Isoleps_pz, Isoleps_e,""gen_leps_status1_px, gen_leps_status1_py,gen_leps_status1_pz, gen_leps_status1_e)" )
            df = (df
                  .Define("truth_mlnu", "m_lnu_status2")
                  .Define("truth_mjj",  "m_genkt_ee_jj")
                  .Define("truth_mqq",  "m_qq_status2")
                  .Define("truth_mqq_fromele",  "m_qq_fromele")
                  .Define("reco_mlnu",  "m_iso_lnu")
                  .Define("reco_mjj",   "m_excl_jj")
                  #.Filter("truth_mlnu>0 && truth_mjj>0 && reco_mlnu>0 && reco_mjj>0")
                  .Define("truth_mon",  "truth_mlnu >= truth_mjj ? truth_mlnu : truth_mjj")
                  .Define("truth_moff", "truth_mlnu < truth_mjj ? truth_mlnu : truth_mjj")
                  .Define("reco_mon",   "reco_mlnu >= reco_mjj ? reco_mlnu : reco_mjj")
                  .Define("reco_moff",  "reco_mlnu < reco_mjj ? reco_mlnu : reco_mjj")
                  .Define("truth_lnuqq_mon",  "truth_mlnu >= truth_mqq ? truth_mlnu : truth_mqq")
                  .Define("truth_lnuqq_moff", "truth_mlnu < truth_mqq ? truth_mlnu : truth_mqq")
                  .Define("truth_lnuqq_qqfromele_mon",  "truth_mlnu >= truth_mqq_fromele ? truth_mlnu : truth_mqq_fromele")
                  .Define("truth_lnuqq_qqfromele_moff", "truth_mlnu < truth_mqq_fromele ? truth_mlnu : truth_mqq_fromele")
                  .Define("mlnu_plus_mjj_reco", "reco_mlnu + reco_mjj")
                  .Define("mlnu_plus_mqq_status2_truth",("m_qq_status2 + m_lnu_status2"))
                  .Define("mlnu_plus_mqq_fromele_truth",("m_qq_fromele + m_lnu_status2"))
                  .Define("diff_RG_m_lnu",("reco_mlnu- m_lnu_status2"))
                  .Define("diff_RG_p_lnu",("p_iso_lnu - p_lnu_status2"))
                  .Define("diff_RG_m_lnuqq",("m_iso_lnuexcljj-m_gen_lnuqq"))
                  .Define("diff_RG_m_qq",("reco_mjj-truth_mqq_fromele"))
                  .Define("diff_RG_p_qq",("p_excljj - p_qq_status2"))
                  .Define("missing_p_res",("(missing_p - gen_neutrinos_status1_p[0])/gen_neutrinos_status1_p[0]"))
                  .Define("eta_miss", "-log(tan(missing_p_theta/2.))")

                )


            df = df.Define("Wlep_gen",
            """
            if (ngen_leps_status1 < 1 || nRecoJets < 2) return -1.0;
            TLorentzVector lep, nu,Wlep;
            lep.SetPxPyPzE(gen_leps_status1_p * cos(gen_leps_status1_phi) *sin(gen_leps_status1_theta),gen_leps_status1_p * sin(gen_leps_status1_phi) * sin(gen_leps_status1_theta),gen_leps_status1_p * cos(gen_leps_status1_theta),gen_leps_status1_p);
            nu.SetPxPyPzE(gen_neutrinos_status1_p* cos(gen_neutrinos_status1_phi) * sin(gen_neutrinos_status1_theta),gen_neutrinos_status1_p* sin(gen_neutrinos_status1_phi) * sin(gen_neutrinos_status1_theta),gen_neutrinos_status1_p* cos(gen_neutrinos_status1_theta),gen_neutrinos_status1_e);
            Wlep=lep+nu;
            return Wlep;
            """
	    )
            df = df.Define("Whad_gen",
            """
            if (ngen_leps_status1 < 1 || nRecoJets < 2) return -1.0;
            TLorentzVector j1,j2,Whad;
            j1.SetPxPyPzE(gen_lightquarks_fromele_px[0],gen_lightquarks_fromele_py[0],gen_lightquarks_fromele_pz[0],gen_lightquarks_fromele_e[0]);
            j2.SetPxPyPzE(gen_lightquarks_fromele_px[1],gen_lightquarks_fromele_py[1],gen_lightquarks_fromele_pz[1],gen_lightquarks_fromele_e[1]);

            Whad=j1+j2;
            return Whad;
            """
	    )
            df = df.Define("deltaPt_gen","sqrt(pow((Wlep_gen.Px() + Whad_gen.Px()),2) + pow((Wlep_gen.Py() + Whad_gen.Py()),2))");
            df = df.Define("deltaPx_gen","sqrt(pow((Wlep_gen.Px() + Whad_gen.Px()),2))");
            df = df.Define("deltaPy_gen","sqrt(pow((Wlep_gen.Py() + Whad_gen.Py()),2))");
            df = df.Define("deltaPz_gen","sqrt(pow((Wlep_gen.Pz() + Whad_gen.Pz()),2))");
            df = df.Define("deltaP_gen","sqrt(pow((Wlep_gen.Pz() + Whad_gen.Pz()),2) +pow((Wlep_gen.Px() + Whad_gen.Px()),2) + pow((Wlep_gen.Py() + Whad_gen.Py()),2))");
            
            #            m_on  = max(m_lnu , m_jj)
            #           m_off = min(m_lnu , m_jj)

#            ROOT.gInterpreter.Declare("""
#            #include <cmath>
#            #include <vector>
#            
#            struct FitResult {
#            float mW;
#            float gW;
#            float scale_lep;
#            float scale_jet1;
#            float scale_jet2;
#            float scale_mp;
#            float chi2;
#            };
#            TLorentzVector lep,jet1,jet2,mp, Wlep,Whad,WW;
#            lep.SetPxPyPzE(
#            Isolep_p * cos(Isolep_phi) *sin(Isolep_theta),
#            Isolep_p * sin(Isolep_phi) * sin(Isolep_theta),
#            Isolep_p * cos(Isolep_theta),
#            Isolep_p
#            );
#            mp.SetPxPyPzE(
#            missing_p * cos(missing_p_phi) * sin(missing_p_theta),
#            missing_p * sin(missing_p_phi) * sin(missing_p_theta),
#            missing_p * cos(missing_p_theta),
#            missing_p
#            );
#            jet1.SetPxPyPzE(jets_p4[0].Px(),jets_p4[0].Py(),jets_p4[0].Pz(),jets_p4[0].E());
#            jet2.SetPxPyPzE(jets_p4[1].Px(),jets_p4[1].Py(),jets_p4[1].Pz(),jets_p4[1].E());
#TODODODODODO
#            
#            }
#            """)
            
        return df

    # __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        #print('incl jets',jetFlavourHelper_R5.outputBranches())
        #print('excl jets',jetFlavourHelper.outputBranches())
        #all_branches += jetFlavourHelper_R5.outputBranches()
        return all_branches

        
        #all_branches+= jetClusteringHelper.outputBranches()

        ## outputs jet scores and constituent breakdown
        #branchList += jetFlavourHelper.outputBranches()
    
        #return all_branches #branchList




        ##test command fccanalysis run --nevents=10 treemaker_lnuqq_reco.py
