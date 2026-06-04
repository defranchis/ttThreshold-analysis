# Breit-Wigner jet → W pairing (standalone)

A small, self-contained tool that decides, for a fully-hadronic **WW → 4 jets**
event, **which two jets came from the first W and which two from the second** —
using only the Breit-Wigner compatibility of the two di-jet masses with the W
resonance. **No kinematic fit, no Minuit** — it is pure arithmetic and runs in
microseconds per event.

It also works as a **WW-vs-background discriminant** (e.g. ZZ → 4 jets): the
goodness-of-fit of the best pairing under the W hypothesis is larger for ZZ
(di-jet masses near the Z, not the W).

## The idea in one line

The 4 jets can be split into 2+2 in **three ways**. For each split we compute the
two di-jet masses `m_a, m_b` and score them with a relativistic Breit-Wigner at
the W pole. The most W-like split wins.

For pairing *k*:
```
gof[k]  = -2 * ( log BW(m_a) + log BW(m_b) )   (referenced to the W pole, so >= 0)
prob[k] = BW(m_a)·BW(m_b) / sum_j ( ... )       (posterior over the 3 splits, sums to 1)
```
The chosen pairing is the one with the smallest `gof` (largest `prob`).
`gof` of the winner is also the **WW-vs-ZZ discriminant**.

## Files

| file | what it is |
|------|------------|
| `BWPairing.h`            | the tool: `bwPairing(jet1..jet4) -> {pairing, gof[3], prob[3], masses}` |
| `BWMatching.h`           | gen-truth helpers (only used to *measure* the efficiency) |
| `treemaker_bw_pairing.py`| the **single** processing step (cluster 4 jets → cut → pair → truth) |
| `analyze_bw_pairing.py`  | pairing efficiency + WW-vs-ZZ gof plot |
| `run.sh`                 | runs WW, ZZ, then the analysis |

## How to run

You need a key4hep / FCCAnalyses environment that provides the `fccanalysis`
command (the same one used for the rest of the analysis). From the **repository
root**:

```bash
source <your FCCAnalyses setup.sh>     # provides `fccanalysis`
bash bw_pairing_standalone/run.sh
```

or step by step:

```bash
# WW signal (also writes the gen-truth pairing so we can measure efficiency)
BW_BOSON=W BW_SAMPLE=p8_ee_WW_ecm160 fccanalysis run bw_pairing_standalone/treemaker_bw_pairing.py
# ZZ control (WW hypothesis applied to ZZ->4q)
BW_BOSON=Z BW_SAMPLE=p8_ee_ZZ_ecm160 fccanalysis run bw_pairing_standalone/treemaker_bw_pairing.py
# analysis
python3 bw_pairing_standalone/analyze_bw_pairing.py
```

Outputs:
- ntuples in `outputs/bw_pairing/W/` and `outputs/bw_pairing/Z/` (TTree `events`),
- plots in `bw_pairing_plots/`.

### Knobs (environment variables, all optional)

| variable | default | meaning |
|----------|---------|---------|
| `BW_SAMPLE`       | `p8_ee_WW_ecm160` | dataset name |
| `BW_BOSON`        | `W`               | `W` (WW→4q, with truth) or `Z` (ZZ→4q) |
| `BW_SQRTD45_MAX`  | `7.0`             | genuine-4-jet cut: keep `sqrt(d_45) < this` [GeV]; `0` disables |
| `BW_FRACTION`     | `0.001`           | fraction of the dataset to process |
| `BW_OUTDIR`       | `outputs/bw_pairing/<BOSON>` | output directory |

## What the pieces mean

**The genuine-4-jet cut (`sqrt(d_45)`)** — `d_45` is the Durham scale at which the
event would resolve a 5th jet. A large `d_45` means a hard gluon was radiated, so
forcing the event into exactly 4 jets no longer corresponds to the 4 quarks. The
cut `sqrt(d_45) < 7 GeV` keeps the genuinely 4-jet events; below it the system is
well defined. This is a standard e+e- 4-jet selection and is **data-applicable**
(no truth needed).

**"matched" vs "unmatched"** — purely a *truth* label for measuring performance.
An event is *matched* if all 4 reco jets sit within `dR < 0.1` of their gen quark,
i.e. the reconstruction faithfully represents the 4 quarks and the true pairing is
reliable. On these events the BW pairing is correct **~90%** of the time. On
*unmatched* events there often is no well-defined correct pairing (the parton→jet
correspondence is broken by radiation), so a lower number there is mostly truth
ambiguity, not a failure of the tool.

**The gof for WW vs ZZ** — apply the W-hypothesis pairing to both samples. WW-matched
events peak at low gof, ZZ at high gof (its di-jet masses prefer the Z, ~91 GeV,
which is far from the W pole). `analyze_bw_pairing.py` makes this plot.

## Using `bwPairing` directly in C++

```cpp
#include "BWPairing.h"
using namespace FCCAnalyses::WWFunctions;
// jet1..jet4 are TLorentzVector
BWPairingResult r = bwPairing(jet1, jet2, jet3, jet4);   // mW=80.385, Gamma=2.085 by default
int    best   = r.pairing;     // 0,1,2  -> the chosen 2+2 split
float  gof    = r.gof_best;    // discriminant (low = W-like)
float  p      = r.prob_best;   // probability the chosen split is correct
// pairing convention: 0:(j1 j2)(j3 j4)  1:(j1 j3)(j2 j4)  2:(j1 j4)(j2 j3)
```

## A note on the probabilities

`prob[k]` is the **pure W-lineshape** posterior. It is *not* calibrated as a
literal P(correct), because the natural BW width (Γ_W ≈ 2 GeV) is narrower than the
detector di-jet mass resolution (~a few GeV) — so the probabilities are
over-confident. Detector resolution is deliberately **not** folded into the BW
width (that would mis-state the physical width). If a calibrated probability is
needed, the proper way is a Breit-Wigner ⊗ Gaussian (Voigtian) resolution model,
or the full kinematic fit. For *choosing the pairing*, the bare BW is already
optimal.
