# Breit-Wigner jet → W pairing (standalone)

A small, self-contained tool for fully-hadronic **WW → 4 jets** events. Given the
4 jets, it decides **which two came from the first W and which two from the second**,
using only how well the two di-jet masses match the W resonance. There is **no
kinematic fit and no Minuit** — it is plain arithmetic on the jet 4-vectors and runs
in microseconds per event.

The same number it uses to choose the pairing (the "goodness-of-fit", `gof`) also
works as a **WW-vs-background discriminant**: a ZZ → 4 jets event has di-jet masses
near the Z, not the W, so it scores a worse `gof`.

---

## How to run

You need a key4hep / FCCAnalyses environment that provides the `fccanalysis`
command. From the **repository root**:

```bash
source <your FCCAnalyses setup.sh>     # provides `fccanalysis`
bash bw_pairing_standalone/run.sh
```

`run.sh` does three things: process the WW signal, process a ZZ control sample, then
run the analysis. To do it by hand:

```bash
# 1) WW signal (also stores the gen-truth pairing, so efficiency can be measured)
BW_BOSON=W BW_SAMPLE=p8_ee_WW_ecm160 fccanalysis run bw_pairing_standalone/treemaker_bw_pairing.py
# 2) ZZ control (the same W-hypothesis pairing applied to ZZ, no truth)
BW_BOSON=Z BW_SAMPLE=p8_ee_ZZ_ecm160 fccanalysis run bw_pairing_standalone/treemaker_bw_pairing.py
# 3) efficiency + WW-vs-ZZ plot
python3 bw_pairing_standalone/analyze_bw_pairing.py
```

Ntuples land in `outputs/bw_pairing/{W,Z}/`, the plot in `bw_pairing_plots/`.

Knobs (environment variables, all optional):

| variable | meaning | default |
|----------|---------|---------|
| `BW_SAMPLE`      | dataset name | `p8_ee_WW_ecm160` |
| `BW_BOSON`       | `W` (WW→4q, with truth) or `Z` (ZZ→4q, no truth) | `W` |
| `BW_SQRTD45_MAX` | genuine-4-jet cut, `sqrt(d_45) <` this [GeV]; `0` disables | `7.0` |
| `BW_FRACTION`    | fraction of the sample to process | `0.001` |
| `BW_OUTDIR`      | output directory | `outputs/bw_pairing/<BOSON>` |

---

## What you should reproduce

`analyze_bw_pairing.py` prints the correct-pairing fraction:

```
all events       ~ 0.70
matched (dR<0.1) ~ 0.91     <- jets that cleanly match the 4 quarks
unmatched        ~ 0.63
```

and produces this plot — the winner's `gof` for WW (split by whether the jets match
the quarks) and for ZZ. The three populations separate; WW-matched (lowest gof) is
cleanly distinct from ZZ:

![Expected WW vs ZZ gof](reference_gof_WW_vs_ZZ.png)

(Reference: `p8_ee_{WW,ZZ}_ecm160`, genuine-4-jet cut `sqrt(d_45) < 7`. Your medians
should match to within statistics.)

---

## The physics it implements

A W decays to two quarks, each of which becomes a jet, so a WW → 4q event is two
di-jets, each with an invariant mass near `m_W ≈ 80.4 GeV`. The 4 jets can be split
into two di-jets in **three ways**, and the job is to pick the one where both di-jet
masses look most like a W.

"Looks like a W" is scored with the relativistic Breit-Wigner — the W resonance
lineshape, peaked at `m_W` with width `Γ_W ≈ 2.1 GeV`. For each of the three splits
`k`, with di-jet masses `m_a, m_b`:

```
gof[k]  = -2 * ( log BW(m_a) + log BW(m_b) )      lower = more W-like; 0 = both on the pole
prob[k] = BW(m_a)·BW(m_b) / sum_j (...)           posterior over the 3 splits; sums to 1
pairing = argmin_k gof[k]                          the chosen split
```

`gof` is referenced to the pole (both masses exactly `m_W`) so it is ≥ 0. The winner's
`gof` is also the WW-vs-ZZ discriminant. `prob` ranks the three splits but is
**over-confident** as a literal probability — the natural width `Γ_W` is narrower than
the jet mass resolution — so use it as a ranking, not a calibrated P(correct).

Two project-specific points worth knowing:

- **Why no kinematic fit.** This tool is the discriminating core of the full WW→4q
  kinematic fit. In that fit, a per-term study showed the di-jet-mass Breit-Wigner
  term is the *only* part that actually separates the three pairings — the jet
  detector resolution priors carry no pairing information. So the fit is dropped and
  only that term is kept; it reproduces the fit's pairing power at a tiny fraction of
  the cost.

- **The ceiling is QCD radiation, not the discriminant.** On *matched* events (the 4
  jets faithfully represent the 4 quarks) the tool is ~91% correct. The lower "all
  events" number is not a failure of the Breit-Wigner: it is events where a hard gluon
  was radiated, so the four jets no longer correspond to the four quarks and there is
  no well-defined correct pairing. The `sqrt(d_45)` cut (below) removes those.

---

## A guided tour of the code

Read the files in this order. Each is documented at the top; this is the map.

### 1. `treemaker_bw_pairing.py` — the pipeline (start here)

One FCCAnalyses processing step that turns raw events into a flat TTree. Inside
`RDFanalysis.analysers`, in order:

1. **Signal definition first.** Select the gen-level 4 quarks from the two bosons
   (`sel_quarks_fromBoson`) and require exactly 4. Doing this *up front* means every
   reco cut afterwards is an efficiency measured on true signal — so "4 reco jets"
   comes out ~100%, not artificially low.
2. **Cluster into exactly 4 jets** with the exclusive ee-kt (Durham) algorithm. For a
   4-quark final state at an e⁺e⁻ collider this is the natural choice: it forces the
   event into 4 jets with no jet-radius parameter and no leftover particles.
3. **Genuine-4-jet cut.** `d_45` is the Durham distance at which a *fifth* jet would
   appear; a large value means a hard gluon was radiated and the "4 jets" no longer
   map onto the 4 quarks. Keep `sqrt(d_45) < 7 GeV`. This is a standard ee 4-jet
   selection and uses **no truth**, so it works on data.
4. **Run the tool:** `bwPairing(jet1..jet4)` → the chosen pairing and the per-split
   `gof` / `prob` / masses, all written to branches.
5. **Gen-truth (WW only):** match the 4 jets to the 4 quarks (`matchJets4`), label
   each jet by which boson its quark came from, and from those labels build the *true*
   pairing (`pairing_index_from_groups`). `bwpair_correct` is then simply "did the tool
   pick the true pairing?" — the efficiency numerator.

The `BW_BOSON` switch makes the same script run on ZZ (`pdg 23`, no truth) so the
W-hypothesis pairing can be applied to a background sample.

### 2. `BWPairing.h` — the tool itself

`bwPairing(j1, j2, j3, j4)` returns a `BWPairingResult` with `pairing`, `gof[3]`,
`prob[3]`, the di-jet masses, and the winner's values. The body is short: loop over the
3 splits, form the two di-jet 4-vectors, evaluate the Breit-Wigner at each mass, combine
into `gof` and `prob` (numerically-stable softmax), pick the lowest `gof`. The header
comment explains every field. This is the only file you need to reuse the tool elsewhere.

Calling it directly from C++:

```cpp
#include "BWPairing.h"
using namespace FCCAnalyses::WWFunctions;          // jet1..jet4 are TLorentzVector
BWPairingResult r = bwPairing(jet1, jet2, jet3, jet4);
int   best = r.pairing;     // 0,1,2 -> chosen split
float gof  = r.gof_best;    // discriminant (low = W-like)
float prob = r.prob_best;   // posterior of the chosen split
// convention: 0:(j1 j2)(j3 j4)  1:(j1 j3)(j2 j4)  2:(j1 j4)(j2 j3)
```

### 3. `JetQuarkMatching.h` — gen-truth helpers (for measuring efficiency only)

Pure truth matching, **no Breit-Wigner**. Three helpers used only to define the true
pairing so the efficiency can be measured — the tool itself needs none of them:
`sel_quarks_fromBoson` (the 4 quarks grouped by parent boson), `matchJets4` (global
min-total-ΔR assignment of the 4 jets to the 4 quarks), and `pairing_index_from_groups`
(jet W-labels → pairing index).

### 4. `analyze_bw_pairing.py` — the analysis

`pairing_efficiency()` reads the WW output and returns the chosen pairing plus the
correct-pairing fraction split into all / matched / unmatched ("matched" = all 4 jets
within `dR < 0.1` of their quark, i.e. where the truth is reliable). `main()` prints
those numbers and overlays the winner's `gof` for WW-matched, WW-unmatched and ZZ to
make the plot above.

### 5. `run.sh`

Glues the three steps together: WW, then ZZ, then the analysis.
