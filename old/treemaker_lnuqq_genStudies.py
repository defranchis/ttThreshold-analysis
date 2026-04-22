import os, copy, ROOT

# list of processes
processList = {
<<<<<<< HEAD
#    "wzp6_ee_lnuqq_ecm160": {
#        "fraction": 1,
#        "crossSection": 1,
 #   },

    "wzp6_ee_munumuqq_noCut_ecm160":{
=======
    "wzp6_ee_lnuqq_ecm160": {
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
        "fraction": 1,
        "crossSection": 1,
    },
}

acceptance_study = True

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
#prodTag     = "FCCee/winter2023/IDEA/"
<<<<<<< HEAD
lepFlav="mu"
#Optional: output directory, default is local running directory
outputDir   = f"./outputs/treemaker/lnuqq/acceptance_atleast1_genlep_status1_{lepFlav}_nocuts"
=======

#Optional: output directory, default is local running directory
outputDir   = "./outputs/treemaker/lnuqq/acceptance_atleast1_genlep_status1_mu"
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f


# additional/costom C++ functions, defined in header files (optional)
includePaths = ["examples/functions.h"]

## latest particle transformer model, trained on 9M jets in winter2023 samples
model_name = "fccee_flavtagging_edm4hep_wc_v1"
model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"    # "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)

url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)
## model files needed for unit testing in CI

## model files locally stored on /eos
eos_dir ="/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"


<<<<<<< HEAD


=======
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
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

#files_In=get_files(eos_dir,samples[0])
weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)
inputDir    = eos_dir#get_files(eos_dir,samples[0])
from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
)

<<<<<<< HEAD
=======
#jetFlavourHelper = None
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
)

>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
jetClusteringHelper = None
jetFlavourHelper = None
# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:

    # __________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):

        # __________________________________________________________
        # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2

        # define some aliases to be used later on
        df = df.Alias("Particle0", "Particle#0.index") #parents
        df = df.Alias("Particle1", "Particle#1.index") #daughters
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        # get all the leptons from the collectio
        df = df.Define("status1parts",           "FCCAnalyses::MCParticle::sel_genStatus(1)(Particle)")
        df = df.Define("status2parts",           "FCCAnalyses::MCParticle::sel_genStatus(2)(Particle)")
<<<<<<< HEAD
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
            df = df.Define("neutrinos_2",                       "FCCAnalyses::MCParticle::sel_genleps(12,12, true)(status2parts)")
        else:
            df = df.Define("gen_leps_status1",                "FCCAnalyses::MCParticle::sel_genleps(11,13,true)(status1parts)") #11
            df = df.Define("gen_leps_status2",                "FCCAnalyses::MCParticle::sel_genleps(11,13,true)(status2parts)") #11
            df = df.Define("neutrinos",                       "FCCAnalyses::MCParticle::sel_genleps(12,13, true)(status1parts)")
            df = df.Define("neutrinos_2",                       "FCCAnalyses::MCParticle::sel_genleps(12,13, true)(status2parts)")
            
        df = df.Define("ngen_leps_status1",               "FCCAnalyses::MCParticle::get_n(gen_leps_status1)")
        df = df.Define("ngen_leps_status2",               "FCCAnalyses::MCParticle::get_n(gen_leps_status2)")
        df = df.Define("gen_leps_status2_p",              "FCCAnalyses::MCParticle::get_pt(gen_leps_status2)")
        df = df.Define("gen_leps_status1_mother_pdgId",   "FCCAnalyses::MCParticle::get_leptons_origin(gen_leps_status1,Particle,Particle0)")
        df = df.Define("gen_leps_status1_fromW",          "FCCAnalyses::MCParticle::sel_genlepsfromW()(gen_leps_status1,gen_leps_status1_mother_pdgId)")
        df = df.Define("ngen_leps_status1_fromW",         "FCCAnalyses::MCParticle::get_n(gen_leps_status1_fromW)")
        df = df.Define("gen_leps_status2_mother_pdgId",   "FCCAnalyses::MCParticle::get_leptons_origin(gen_leps_status2,Particle,Particle0)")

        df = df.Define("gen_leps_status1_p",      "FCCAnalyses::MCParticle::get_pt(gen_leps_status1)")
        df = df.Define("gen_leps_status1_theta",  "FCCAnalyses::MCParticle::get_theta(gen_leps_status1)")
        df = df.Define("gen_leps_status1_phi",    "FCCAnalyses::MCParticle::get_phi(gen_leps_status1)")
        df = df.Define("gen_leps_status1_charge", "FCCAnalyses::MCParticle::get_charge(gen_leps_status1)")
        df = df.Define("gen_leps_status1_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_leps_status1)")
        df = df.Define("gen_leps_status1_e",      "FCCAnalyses::MCParticle::get_e(gen_leps_status1)")
        df = df.Define("gen_leps_status2_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_leps_status2)")
        df = df.Define("gen_leps_status2_theta",  "FCCAnalyses::MCParticle::get_theta(gen_leps_status2)")
        df = df.Define("gen_leps_status2_phi",    "FCCAnalyses::MCParticle::get_phi(gen_leps_status2)")
        df = df.Define("gen_leps_status2_e",      "FCCAnalyses::MCParticle::get_e(gen_leps_status2)")

        
        #df = df.Define("gen_taus_status2",       "FCCAnalyses::MCParticle::sel_gentausfromele()(gen_leps_status2,gen_leps_status2_mother_pdgId)")
        df = df.Define("gen_taus_status2",        "FCCAnalyses::MCParticle::sel_genleps(15,16,true)(status2parts)")
        df = df.Define("ngen_taus_status2",       "FCCAnalyses::MCParticle::get_n(gen_taus_status2)")
=======
        df = df.Define("gen_leps_status1",       "FCCAnalyses::MCParticle::sel_genleps(13,13, true)(status1parts)") #11
        df = df.Define("ngen_leps_status1",      "FCCAnalyses::MCParticle::get_n(gen_leps_status1)")
        df = df.Define("gen_leps_status1_mother_pdgId", "FCCAnalyses::MCParticle::get_leptons_origin(gen_leps_status1,Particle,Particle0)")
        df = df.Define("gen_leps_status1_fromW",       "FCCAnalyses::MCParticle::sel_genlepsfromW()(gen_leps_status1,gen_leps_status1_mother_pdgId)")
        df = df.Define("ngen_leps_status1_fromW",      "FCCAnalyses::MCParticle::get_n(gen_leps_status1_fromW)")
        df = df.Define("gen_leps_status2",       "FCCAnalyses::MCParticle::sel_pdgID(15, true)(status2parts)")
        df = df.Define("gen_leps_status2_mother_pdgId", "FCCAnalyses::MCParticle::get_leptons_origin(gen_leps_status2,Particle,Particle0)")
    
        df = df.Define("gen_taus_status2",       "FCCAnalyses::MCParticle::sel_gentausfromele()(gen_leps_status2,gen_leps_status2_mother_pdgId)")
        df = df.Define("ngen_taus_status2",      "FCCAnalyses::MCParticle::get_n(gen_taus_status2)")
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
        df = df.Define("gen_taus_status2_p",      "FCCAnalyses::MCParticle::get_p(gen_taus_status2)")
        df = df.Define("gen_taus_status2_theta",  "FCCAnalyses::MCParticle::get_theta(gen_taus_status2)")
        df = df.Define("gen_taus_status2_phi",    "FCCAnalyses::MCParticle::get_phi(gen_taus_status2)")
        df = df.Define("gen_taus_status2_charge", "FCCAnalyses::MCParticle::get_charge(gen_taus_status2)")
        df = df.Define("gen_taus_status2_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_taus_status2)")
        df = df.Define("gen_taus_status2_e",      "FCCAnalyses::MCParticle::get_e(gen_taus_status2)")

        
        
<<<<<<< HEAD
        df = df.Filter("ngen_leps_status1 > 1 && ngen_taus_status2 == 0")
=======
        df = df.Filter("ngen_taus_status2 == 0 && ngen_leps_status1 == 1")
        
        df = df.Define("gen_leps_status1_p",      "FCCAnalyses::MCParticle::get_p(gen_leps_status1)")
        df = df.Define("gen_leps_status1_theta",  "FCCAnalyses::MCParticle::get_theta(gen_leps_status1)")
        df = df.Define("gen_leps_status1_phi",    "FCCAnalyses::MCParticle::get_phi(gen_leps_status1)")
        df = df.Define("gen_leps_status1_charge", "FCCAnalyses::MCParticle::get_charge(gen_leps_status1)")
        df = df.Define("gen_leps_status1_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_leps_status1)")
        df = df.Define("gen_leps_status1_e",  "FCCAnalyses::MCParticle::get_e(gen_leps_status1)")

        #        if acceptance_study:
        #           return df
        
        df = df.Define("neutrinos",  "FCCAnalyses::MCParticle::sel_genleps(14,12, true)(status1parts)")
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
        df = df.Define("gen_neutrinos_status1",  "FCCAnalyses::MCParticle::sel_genleps(14,16, true)(neutrinos)")
        df = df.Define("ngen_neutrinos_status1", "FCCAnalyses::MCParticle::get_n(gen_neutrinos_status1)")

        df = df.Define("gen_neutrinos_status1_p",      "FCCAnalyses::MCParticle::get_p(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_theta",  "FCCAnalyses::MCParticle::get_theta(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_phi",    "FCCAnalyses::MCParticle::get_phi(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_charge", "FCCAnalyses::MCParticle::get_charge(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_e",  "FCCAnalyses::MCParticle::get_e(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_mother_pdgId", "FCCAnalyses::MCParticle::get_leptons_origin(gen_neutrinos_status1,Particle,Particle0)")
<<<<<<< HEAD
        #df=df.Define("isQQ",      "MCParticle::get_tree(11)(Particle,Particle0)")

        df = df.Define("gen_neutrinos_status2",  "FCCAnalyses::MCParticle::sel_genleps(14,16, true)(neutrinos_2)")
        df = df.Define("ngen_neutrinos_status2", "FCCAnalyses::MCParticle::get_n(gen_neutrinos_status2)")

        df = df.Define("gen_neutrinos_status2_p",      "FCCAnalyses::MCParticle::get_p(gen_neutrinos_status2)")
        df = df.Define("gen_neutrinos_status2_theta",  "FCCAnalyses::MCParticle::get_theta(gen_neutrinos_status2)")
        df = df.Define("gen_neutrinos_status2_phi",    "FCCAnalyses::MCParticle::get_phi(gen_neutrinos_status2)")
        df = df.Define("gen_neutrinos_status2_charge", "FCCAnalyses::MCParticle::get_charge(gen_neutrinos_status2)")
        df = df.Define("gen_neutrinos_status2_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_neutrinos_status2)")
        df = df.Define("gen_neutrinos_status2_e",      "FCCAnalyses::MCParticle::get_e(gen_neutrinos_status2)")

        
        #df=df.Define("isQQ",      "MCParticle::get_decay(11,5,false)(Particle,Particle1)")
        df = df.Define('gen_lightquarks_fromele',        'FCCAnalyses::MCParticle::sel_lightQuarks_fromele(true)(Particle,Particle0)')
        df = df.Define("ngen_partons_fromele",           "FCCAnalyses::MCParticle::get_n(gen_lightquarks_fromele)");
=======
        df=df.Define("isQQ",      "MCParticle::get_tree(11)(Particle,Particle0)")

        #df=df.Define("isQQ",      "MCParticle::get_decay(11,5,false)(Particle,Particle1)")
        df = df.Define('gen_lightquarks_fromele',    'FCCAnalyses::MCParticle::sel_lightQuarks_fromele(true)(Particle,Particle0)')
        df = df.Define("ngen_partons_fromele","FCCAnalyses::MCParticle::get_n(gen_lightquarks_fromele)");
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
        df = df.Define("gen_lightquarks_fromele_p",      "FCCAnalyses::MCParticle::get_p(gen_lightquarks_fromele)")
        df = df.Define("gen_lightquarks_fromele_theta",  "FCCAnalyses::MCParticle::get_theta(gen_lightquarks_fromele)")
        df = df.Define("gen_lightquarks_fromele_phi",    "FCCAnalyses::MCParticle::get_phi(gen_lightquarks_fromele)")
        df = df.Define("gen_lightquarks_fromele_charge", "FCCAnalyses::MCParticle::get_charge(gen_lightquarks_fromele)")
        df = df.Define("gen_lightquarks_fromele_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_lightquarks_fromele)")
<<<<<<< HEAD
        df = df.Define("gen_lightquarks_fromele_e",      "FCCAnalyses::MCParticle::get_e(gen_lightquarks_fromele)")
        
        df = df.Define('gen_lightquarks',    'FCCAnalyses::MCParticle::sel_lightQuarks(true)(status2parts)')
        df = df.Define("ngen_partons",        "FCCAnalyses::MCParticle::get_n(gen_lightquarks)");
=======
        df = df.Define("gen_lightquarks_fromele_e",  "FCCAnalyses::MCParticle::get_e(gen_lightquarks_fromele)")
        
        df = df.Define('gen_lightquarks',    'FCCAnalyses::MCParticle::sel_lightQuarks(true)(Particle)')
        df = df.Define("ngen_partons","FCCAnalyses::MCParticle::get_n(gen_lightquarks)");
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f

        df = df.Define("gen_lightquarks_p",      "FCCAnalyses::MCParticle::get_p(gen_lightquarks)")
        df = df.Define("gen_lightquarks_theta",  "FCCAnalyses::MCParticle::get_theta(gen_lightquarks)")
        df = df.Define("gen_lightquarks_phi",    "FCCAnalyses::MCParticle::get_phi(gen_lightquarks)")
        df = df.Define("gen_lightquarks_charge", "FCCAnalyses::MCParticle::get_charge(gen_lightquarks)")
        df = df.Define("gen_lightquarks_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_lightquarks)")
        df = df.Define("gen_lightquarks_e",  "FCCAnalyses::MCParticle::get_e(gen_lightquarks)")
        df = df.Define("gen_lightquarks_mother_pdgId", "FCCAnalyses::MCParticle::get_leptons_origin(gen_lightquarks,Particle,Particle0)")


        df = df.Define(
<<<<<<< HEAD
            "m_qq_status2",
=======
            "m_qq",
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
            """
            if (gen_lightquarks_p.size() < 2) return -1.0;
            TLorentzVector q1,q2;
            q1.SetPxPyPzE(
<<<<<<< HEAD
            gen_lightquarks_p[1] * cos(gen_lightquarks_phi[1]) *sin(gen_lightquarks_theta[1]),
            gen_lightquarks_p[1] * sin(gen_lightquarks_phi[1]) * sin(gen_lightquarks_theta[1]),
            gen_lightquarks_p[1] * cos(gen_lightquarks_theta[1]),
=======
            gen_lightquarks_p[1] * cos(gen_leps_status1_phi[1]) *sin(gen_leps_status1_theta[1]),
            gen_lightquarks_p[1] * sin(gen_leps_status1_phi[1]) * sin(gen_leps_status1_theta[1]),
            gen_lightquarks_p[1] * cos(gen_leps_status1_theta[1]),
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
            gen_lightquarks_e[1]
            );
            q2.SetPxPyPzE(
            gen_lightquarks_p[0] * cos(gen_lightquarks_phi[0]) * sin(gen_lightquarks_theta[0]),
            gen_lightquarks_p[0] * sin(gen_lightquarks_phi[0]) * sin(gen_lightquarks_theta[0]),
            gen_lightquarks_p[0] * cos(gen_lightquarks_theta[0]),
            gen_lightquarks_e[0]
            );
            return (q1 + q2).M();
            """
        )


        df = df.Define(
            "m_qq_fromele",
            """
            if (gen_lightquarks_fromele_p.size() < 2) return -1.0;
            TLorentzVector lep, nu;
            lep.SetPxPyPzE(
<<<<<<< HEAD
            gen_lightquarks_fromele_p[1] * cos(gen_lightquarks_fromele_phi[1]) *sin(gen_lightquarks_fromele_theta[1]),
            gen_lightquarks_fromele_p[1] * sin(gen_lightquarks_fromele_phi[1]) * sin(gen_lightquarks_fromele_theta[1]),
            gen_lightquarks_fromele_p[1] * cos(gen_lightquarks_fromele_theta[1]),
=======
            gen_lightquarks_fromele_p[1] * cos(gen_leps_status1_phi[1]) *sin(gen_leps_status1_theta[1]),
            gen_lightquarks_fromele_p[1] * sin(gen_leps_status1_phi[1]) * sin(gen_leps_status1_theta[1]),
            gen_lightquarks_fromele_p[1] * cos(gen_leps_status1_theta[1]),
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
            gen_lightquarks_fromele_e[1]
            );
            nu.SetPxPyPzE(
            gen_lightquarks_fromele_p[0] * cos(gen_lightquarks_fromele_phi[0]) * sin(gen_lightquarks_fromele_theta[0]),
            gen_lightquarks_fromele_p[0] * sin(gen_lightquarks_fromele_phi[0]) * sin(gen_lightquarks_fromele_theta[0]),
            gen_lightquarks_fromele_p[0] * cos(gen_lightquarks_fromele_theta[0]),
            gen_lightquarks_fromele_e[0]
            );
            return (lep + nu).M();
            """
        )

        
<<<<<<< HEAD
        #print('this',df["gen_lep_status2_p"].to_string(index=False))        
        df = df.Define(
            "GenParticlesNoEMu",
            "FCCAnalyses::MCParticle::remove(status2parts,gen_leps_status2)",
=======
        #print('this',df["gen_lep_status1_p"].to_string(index=False))        
        df = df.Define(
            "GenParticlesNoEMu",
            "FCCAnalyses::MCParticle::remove(status1parts,gen_leps_status1)",
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
        )
        df = df.Define(
            "GenParticlesNoLeps",
            "FCCAnalyses::MCParticle::remove(GenParticlesNoEMu,gen_taus_status2)",
        )
        df = df.Define(
            "GenParticlesNoLepsNoNeu",
            "FCCAnalyses::MCParticle::remove(GenParticlesNoLeps,gen_neutrinos_status1)",
        )
        global jetClusteringHelper
        global jetFlavourHelper
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
#        collections_noleps = copy.deepcopy(collections)
#        collections_noleps["GenParticles"] = "GenParticlesNoLepsNoNeu"
 #       nJets=2
  #      jetClusteringHelper = ExclusiveJetClusteringHelper(collections_noleps["GenParticles"], nJets)
        #        jetFlavourHelper = JetFlavourHelper(
        #            collections_noleps,
        #            jetClusteringHelper.jets,
        #            jetClusteringHelper.constituents,
        #        )



#        df = df.Define("jets_p4", "JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets))
        #        df = df.Define("all_invariant_masses", "JetConstituentsUtils::all_invariant_masses(jets_p4)")
 #       df = df.Define(
  #          "jj_m",
   #         "JetConstituentsUtils::InvariantMass(jets_p4[0], jets_p4[1])",
    #    )

     #   df = df.Define("jet1", "genjets_excl_p4[0]")
       # df = df.Define("jet2", "genjets_excl_p4[1]")
      #  df = df.Define("jet1_p","jet1.P()")
        #df = df.Define("jet2_p","jet2.P()")
#        df = df.Define("jet_phi", "JetClusteringUtils::get_phi_std(jet)")
#        df = df.Define("jet1_phi","jet_phi[0]")
#        df = df.Define("jet2_phi","jet_phi[1]")
#        df = df.Define("jet_theta", "JetClusteringUtils::get_theta(jet)")
#        df = df.Define("jet1_theta","jet_theta[0]")
#        df = df.Define("jet2_theta","jet_theta[1]")
#        df = df.Define(
#            "mjj_excl",
#            """
#            TLorentzVector lep, nu;
#            lep.SetPxPyPzE(
#            jet1.P() * cos(jet_phi[0]) *sin(jet_theta[0]),
#            jet1.P() * sin(jet_phi[0]) * sin(jet_theta[0]),
#            jet1.P() * cos(jet_theta[0]),
#            jet1.E()
#            );
#            nu.SetPxPyPzE(
#            jet2.P() * cos(gen_lightquarks_fromele_phi[1]) * sin(gen_lightquarks_fromele_theta[1]),
#            jet2.P() * sin(gen_lightquarks_fromele_phi[1]) * sin(gen_lightquarks_fromele_theta[1]),
#            jet2.P() * cos(gen_lightquarks_fromele_theta[1]),
#            jet2.E()
#            );
#            return (lep + nu).M();
#            """
#        )

        
        df = df.Define("GP_px",   "MCParticle::get_px(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_py" ,  "MCParticle::get_py(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_pz" ,  "MCParticle::get_pz(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_e"  ,  "MCParticle::get_e(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_m"  ,  "MCParticle::get_mass(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_q"  ,  "MCParticle::get_charge(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_p"  ,  "MCParticle::get_p(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_pdgId"  ,  "MCParticle::get_pdg(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_status"  ,  "MCParticle::get_genStatus(GenParticlesNoLepsNoNeu)")
        
        df = df.Define("pseudo_jets",         "JetClusteringUtils::set_pseudoJets_xyzm(GP_px, GP_py, GP_pz, GP_m)")

   #     df = jetClusteringHelper.define(df)
    #    df = df.Define(
     #       "genjets_excl_p4",
      #      "JetConstituentsUtils::compute_tlv_jets({})".format(
       #     "pseudo_jets"
	#    ),
        #)
        
        df = df.Define("FCCAnalysesJets_kt",  "JetClustering::clustering_kt(0.5, 2, 4, 0, 20)(pseudo_jets)")
        df = df.Define("Genjets_kt",          "JetClusteringUtils::get_pseudoJets(FCCAnalysesJets_kt)")
        df = df.Define("Genjets_kt_e",        "JetClusteringUtils::get_e(Genjets_kt)")
        df = df.Define("Genjets_kt_px",       "JetClusteringUtils::get_px(Genjets_kt)")
        df = df.Define("Genjets_kt_py",       "JetClusteringUtils::get_py(Genjets_kt)")
        df = df.Define("Genjets_kt_pz",       "JetClusteringUtils::get_pz(Genjets_kt)")
        df = df.Define("Genjets_kt_m",        "JetClusteringUtils::get_m(Genjets_kt)")

        df = df.Define(
            "m_jj",
            """
            if (Genjets_kt_e.size() < 2) return -1.0;
            TLorentzVector q1, q2;
            q1.SetPxPyPzE(Genjets_kt_px[0], Genjets_kt_py[0], Genjets_kt_pz[0], Genjets_kt_e[0]);
            q2.SetPxPyPzE(Genjets_kt_px[1], Genjets_kt_py[1], Genjets_kt_pz[1], Genjets_kt_e[1]);
            return (q1 + q2).M();
            """
        )

        df = df.Define(
            "m_lnu",
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
<<<<<<< HEAD

        df = df.Define(
            "m_lnu_status2",
            """
            if (gen_neutrinos_status2.size() + gen_leps_status2.size() < 2) return -1.0;
            TLorentzVector lep, nu;
            lep.SetPxPyPzE(
            gen_leps_status2_p[0] * cos(gen_leps_status2_phi[0]) *sin(gen_leps_status2_theta[0]),
            gen_leps_status2_p[0] * sin(gen_leps_status2_phi[0]) * sin(gen_leps_status2_theta[0]),
            gen_leps_status2_p[0] * cos(gen_leps_status2_theta[0]),
            gen_leps_status2_e[0]
            );
            nu.SetPxPyPzE(
            gen_neutrinos_status2_p[0] * cos(gen_neutrinos_status2_phi[0]) * sin(gen_neutrinos_status2_theta[0]),
            gen_neutrinos_status2_p[0] * sin(gen_neutrinos_status2_phi[0]) * sin(gen_neutrinos_status2_theta[0]),
            gen_neutrinos_status2_p[0] * cos(gen_neutrinos_status2_theta[0]),
            gen_neutrinos_status2_e[0]
            );
            return (lep + nu).M();
            """
        )

=======
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
        


        
        return df
    
    # __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
<<<<<<< HEAD
        branchList = ["gen_leps_status1_p","ngen_leps_status2","gen_leps_status2_p","gen_leps_status1_pdgId","gen_leps_status2_pdgId",
            #            "isQQ",

                      "gen_leps_status1_theta",  
                      "gen_leps_status1_phi",    
#            "gen_leps_status1_charge", 
 #           "gen_leps_status1_pdgId",
  #          "ngen_leps_status1_fromW",
                      "gen_leps_status1_e",
                      "ngen_leps_status1",
            #"ngen_taus_status2",
            #"gen_taus_status2_p",      
            #"gen_taus_status2_theta",  
            #"gen_taus_status2_phi",    
            #"gen_taus_status2_charge", 
            #"gen_taus_status2_pdgId",
            #"gen_taus_status2_e",
                      "gen_leps_status1_mother_pdgId",
                      "ngen_partons", "ngen_partons_fromele",
                      "GP_p","GP_pdgId","GP_status",
                      "Genjets_kt_e",
                      "Genjets_kt_px",
                      "Genjets_kt_py",
                      "Genjets_kt_pz",
                      "Genjets_kt_m",
                      "nstatus1parts",
                      "m_lnu","m_qq_status2","m_qq_fromele",#"mjj_excl",
                      "gen_neutrinos_status1_mother_pdgId","m_lnu_status2",
                      "m_jj",
                      "gen_lightquarks_e","gen_lightquarks_p","gen_lightquarks_theta","gen_lightquarks_pdgId","gen_lightquarks_charge","gen_lightquarks_phi","gen_lightquarks_mother_pdgId",
                      "gen_lightquarks_fromele_e","gen_lightquarks_fromele_p","gen_lightquarks_fromele_theta","gen_lightquarks_fromele_pdgId","gen_lightquarks_fromele_charge","gen_lightquarks_fromele_phi",
=======
        branchList = [
            "isQQ",
            "gen_leps_status1_p",      
            "gen_leps_status1_theta",  
            "gen_leps_status1_phi",    
            "gen_leps_status1_charge", 
            "gen_leps_status1_pdgId",
            "ngen_leps_status1_fromW",
            "gen_leps_status1_e",
            "ngen_leps_status1","ngen_taus_status2",
            "gen_taus_status2_p",      
            "gen_taus_status2_theta",  
            "gen_taus_status2_phi",    
            "gen_taus_status2_charge", 
            "gen_taus_status2_pdgId",
            "gen_taus_status2_e",
            "gen_leps_status2_mother_pdgId",
            "ngen_partons", "ngen_partons_fromele",
            "GP_p","GP_pdgId","GP_status",
            "Genjets_kt_e",
            "Genjets_kt_px",
            "Genjets_kt_py",
            "Genjets_kt_pz",
            "Genjets_kt_m",
            "m_lnu","m_qq","m_qq_fromele",#"mjj_excl",
            "gen_neutrinos_status1_mother_pdgId","m_jj",
            "gen_lightquarks_e","gen_lightquarks_p","gen_lightquarks_theta","gen_lightquarks_pdgId","gen_lightquarks_charge","gen_lightquarks_phi","gen_lightquarks_mother_pdgId",
            "gen_lightquarks_fromele_e","gen_lightquarks_fromele_p","gen_lightquarks_fromele_theta","gen_lightquarks_fromele_pdgId","gen_lightquarks_fromele_charge","gen_lightquarks_fromele_phi",
>>>>>>> 0e892b7b349a784dc02ceb8b4d42a3e5aac2957f
        ]
         #   branchList += add


        ## outputs jet scores and constituent breakdown
        #branchList += jetFlavourHelper.outputBranches()

        return branchList
