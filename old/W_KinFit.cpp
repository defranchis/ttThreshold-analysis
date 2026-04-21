#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <ROOT/Minuit2/Minuit2Minimizer.h>
#include <Math/Functor.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>

// --------------------------
// Detector resolution / scale
// --------------------------
const double sigma_j = 0.03;
const double sigma_l = 0.001;
const double sigma_E = 0.5;
const double sigma_p = 0.5;
double sigma_mjj, sigma_mlnu, sigma_mlnujj, sigma_pjj, sigma_plnu, sigma_dM;

// --------------------------
// Event structure
// --------------------------
struct Event {
    TLorentzVector j1, j2, lep, mp;
};
struct ScaleResult {
  double s1, s2, sl, sm;
  TLorentzVector j1f, j2f, lepf, mpf;
  double mjj, mlnu;
};

ScaleResult FitEventScales(const Event &ev, double mW=80.4, double gW=2.1)
{
  auto chi2 = [&ev, mW, gW](const double *x) {
    double s1 = x[0], s2 = x[1], sl = x[2], sm = x[3];

    TLorentzVector j1f = ev.j1; j1f *= s1;
    TLorentzVector j2f = ev.j2; j2f *= s2;
    TLorentzVector lepf = ev.lep; lepf *= sl;
    TLorentzVector mpf  = ev.mp;  mpf  *= sm;

    TLorentzVector W_had = j1f + j2f;
    TLorentzVector W_lep = lepf + mpf;
    TLorentzVector lnuqq_sys = W_had + W_lep;

    double mjj = W_had.M();
    double mlnu = W_lep.M();

    // Breit-Wigner
    double bw = 2*log(pow(mjj*mjj - mW*mW,2) + (mW*gW)*(mW*gW));
    bw += 2*log(pow(mlnu*mlnu - mW*mW,2) + (mW*gW)*(mW*gW));

    // Detector mass constraints
    double mass = pow((mjj - mW)/sigma_mjj,2) + pow((mlnu - mW)/sigma_mlnu,2);

    // Energy-momentum conservation
    double cons = pow((lnuqq_sys.E() - 160.0)/(sigma_E),2);
    cons += pow(lnuqq_sys.Px()/sigma_p,2) + pow(lnuqq_sys.Py()/sigma_p,2) + pow(lnuqq_sys.Pz()/sigma_p,2);
    cons += pow((W_lep.P() - W_had.P())/sqrt(sigma_plnu*sigma_plnu + sigma_pjj*sigma_pjj),2);
    cons += pow((mlnu - mjj)/sqrt(sigma_mlnu*sigma_mlnu + sigma_mjj*sigma_mjj),2);

    // Scale penalties
    double res = pow((s1-1)/sigma_j,2) + pow((s2-1)/sigma_j,2) + pow((sl-1)/sigma_l,2) + pow((sm-1)/sigma_l,2);

    return bw + mass + cons + res;
  };

  ROOT::Minuit2::Minuit2Minimizer min(ROOT::Minuit2::kMigrad);
  min.SetMaxFunctionCalls(10000);
  min.SetTolerance(1e-6);
  min.SetStrategy(1);

  ROOT::Math::Functor f(chi2, 4);
  min.SetFunction(f);

  min.SetVariable(0, "s1", 1.0, 0.01);
  min.SetVariable(1, "s2", 1.0, 0.01);
  min.SetVariable(2, "sl", 1.0, 0.01);
  min.SetVariable(3, "sm", 1.0, 0.01);

  min.Minimize();
  double s1 = min.X()[0];
  double s2 = min.X()[1];
  double sl = min.X()[2];
  double sm = min.X()[3];

  TLorentzVector j1f = ev.j1; j1f *= s1;
  TLorentzVector j2f = ev.j2; j2f *= s2;
  TLorentzVector lepf = ev.lep; lepf *= sl;
  TLorentzVector mpf  = ev.mp;  mpf  *= sm;

  double mjj  = (j1f + j2f).M();
  double mlnu = (lepf + mpf).M();

  ScaleResult result {s1,s2,sl,sm,j1f,j2f,lepf,mpf,mjj,mlnu};
  return result;
}

// --------------------------
double GlobalChi2(const double *x, const std::vector<Event> &events, const std::vector<ScaleResult> &scales){
  
  double mW = x[0];
  double gW = x[1];
  double total = 0.0;
  for (size_t i=0; i<events.size(); ++i) {
    const ScaleResult &res = scales[i];
    double mjj = res.mjj;
    double mlnu = res.mlnu;

        // Breit-Wigner
    double bw = 2*log(pow(mjj*mjj - mW*mW,2) + (mW*gW)*(mW*gW));
    bw += 2*log(pow(mlnu*mlnu - mW*mW,2) + (mW*gW)*(mW*gW));
    
    // Detector mass constraints
    double mass = pow((mjj - mW)/sigma_mjj,2) + pow((mlnu - mW)/sigma_mlnu,2);
    
    // Energy-momentum constraints
    double cons = pow((res.j1f + res.j2f + res.lepf + res.mpf).E() - 160.0,2)/sigma_E/sigma_E;
    cons += pow((mlnu - mjj)/sqrt(sigma_mlnu*sigma_mlnu + sigma_mjj*sigma_mjj),2);
    
    total += bw + mass + cons;
  }
  return total;
}
int main() {

    std::string fileName = "/afs/cern.ch/work/a/anmehta/public/FCC_ver2/FCCAnalyses/ttThreshold-analysis/outputs/treemaker/lnuqq/semihad/wzp6_ee_munumuqq_noCut_ecm160.root";
    ROOT::RDataFrame df("events", fileName);

    // Compute resolutions
    auto h_mjj    = df.Histo1D({"h_mjj","m_{jj} resolution",200,-20,20},"diff_RG_m_qq");
    auto h_mlnu   = df.Histo1D({"h_mlnu","m_{lnu} resolution",200,-20,20},"diff_RG_m_lnu");
    auto h_mlnujj = df.Histo1D({"h_mlnujj","m_{lnuqq} resolution",200,-20,20},"diff_RG_m_lnuqq");
    auto h_pjj    = df.Histo1D({"h_pjj","p_{qq} resolution",200,-20,20},"diff_RG_p_qq");
    auto h_plnu   = df.Histo1D({"h_plnu","p_{lnu} resolution",200,-20,20},"diff_RG_p_lnu");
    auto h_dM     = df.Histo1D({"h_dM","delta M",200,-20,20},"deltaM");
    std::cout << "Resolutions: " 
              << sigma_mjj << " " << sigma_mlnu << " " << sigma_mlnujj << std::endl;

    // Build events vector
    std::vector<Event> events;
    auto arrays = df.AsNumpy({"jet1_pt","jet1_eta","jet1_phi","jet1_mass",
                              "jet2_pt","jet2_eta","jet2_phi","jet2_mass",
                              "Isolep_pt","Isolep_eta","Isolep_phi",
                              "missing_p","missing_p_phi"});

    size_t N = arrays["jet1_pt"].size();
    for (size_t i=0;i<N;i++){
        Event ev;
        ev.j1.SetPtEtaPhiM(arrays["jet1_pt"][i], arrays["jet1_eta"][i], arrays["jet1_phi"][i], arrays["jet1_mass"][i]);
        ev.j2.SetPtEtaPhiM(arrays["jet2_pt"][i], arrays["jet2_eta"][i], arrays["jet2_phi"][i], arrays["jet2_mass"][i]);
        ev.lep.SetPtEtaPhiM(arrays["Isolep_pt"][i], arrays["Isolep_eta"][i], arrays["Isolep_phi"][i], 0.0);
        ev.mp.SetPtEtaPhiM(arrays["missing_p"][i], 0.0, arrays["missing_p_phi"][i], 0.0);
        events.push_back(ev);
    }

    std::cout << "Loaded " << events.size() << " events\n";
    std::vector<ScaleResult> scales;
    for (const auto &ev : events) {
      scales.push_back(FitEventScales(ev));
    }
    auto globalChi2 = [&events,&scales](const double *x) {
      return GlobalChi2(x, events, scales);
    };

    ROOT::Minuit2::Minuit2Minimizer min(ROOT::Minuit2::kMigrad);
    min.SetFunction(ROOT::Math::Functor(globalChi2,2));
    min.SetVariable(0, "mW", 80.4, 0.01);
    min.SetVariable(1, "gW", 2.1, 0.01);
    min.SetLimitedVariable(1,"gW",2.1,0.01,0.5,5.0);
    min.Minimize();

    double mW_fit = min.X()[0];
    double gW_fit = min.X()[1];
    double err_mW = min.Errors()[0];
    double err_gW = min.Errors()[1];

    std::cout << "Fit results:\n";
    std::cout << "mW = " << mW_fit << " +/- " << err_mW << "\n";
    std::cout << "GammaW = " << gW_fit << " +/- " << err_gW << "\n";

    TFile outFile("fit_output.root","RECREATE");
    TTree outTree("fit","fit results");

    float mjj_fit, mlnu_fit, chi2_val, s1_out, s2_out, sl_out, sm_out;
    outTree.Branch("mjj_fit",&mjj_fit,"mjj_fit/F");
    outTree.Branch("mlnu_fit",&mlnu_fit,"mlnu_fit/F");
    outTree.Branch("chi2",&chi2_val,"chi2/F");
    outTree.Branch("s1",&s1_out,"s1/F");
    outTree.Branch("s2",&s2_out,"s2/F");
    outTree.Branch("sl",&sl_out,"sl/F");
    outTree.Branch("sm",&sm_out,"sm/F");
    for (size_t i=0;i<events.size();++i){
      const auto &ev = events[i];
      const auto &res = scales[i];
      mjj_fit = res.mjj;
      mlnu_fit = res.mlnu;
      s1_out = res.s1;
      s2_out = res.s2;
      sl_out = res.sl;
      sm_out = res.sm;
      chi2_val = 0; // optional: store chi2 from per-event fit
      outTree.Fill();
    }
    outFile.Write();
    outFile.Close();
    
    std::cout << "Saved per-event fit output to fit_output.root\n";
    
    return 0;
}




 
