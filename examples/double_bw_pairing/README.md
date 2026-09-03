# The correlated double-BW term for a WW hadronic kinfit

A minimal, self-contained example of the term used in this repo's WW->4q /
WW->lnuqq kinematic fit to constrain both hadronic W masses to the W
lineshape jointly, how to use it (on its own, no fit needed) to pick the
correct jet pairing, and how to normalize it for use in a likelihood fit
(and why a plain chi2 fit that floats mW needs the same normalization).

Files:

- [`double_bw_term.py`](double_bw_term.py) -- the physics, standalone
  (numpy only). Read this first.
- [`demo_pairing_and_normalization.py`](demo_pairing_and_normalization.py) --
  runs it on real WW->4q reconstructed jets from this repo's ntuples and
  makes the 3 plots below.

The production, performance-optimized version of this term (Minuit2-fitted,
table-interpolated normalization) lives in
[`WWFunctions/WWKinReco.h`](../../WWFunctions/WWKinReco.h):
`bw_phasespace_neg2ll` (~L684) and `log_Z_bw_phasespace` /
`log_Z_bw_phasespace_ontf` (~L474-666). `log_z_bw_phasespace` here is a
readability-first 1:1 port of `log_Z_bw_phasespace_ontf`; `neg2ll_bw_phasespace`
matches `bw_phasespace_neg2ll` except for one deliberate change in how the
kinematically-forbidden region is floored -- see its docstring for why.

## 1. The term, and why it's "correlated"

A hadronic W-pair event gives two W-candidate dijet masses `m_h`, `m_l` (the
higher- and lower-mass dijet) from some jet pairing. The natural first guess
is to treat them as two independent Breit-Wigners and add their `-2 ln L`:

```
-2 ln L(m_h) - 2 ln L(m_l) = -2 ln BW(m_h) - 2 ln BW(m_l)
```

That's wrong (or at least incomplete): `m_h` and `m_l` are not independent.
Both come from the same parent system of invariant mass `m_WW = sqrt(s_WW)`
(the reconstructed di-W / 4-jet system), and energy-momentum conservation
couples them through the 2-body-decay phase-space factor -- the Kallen
function:

```
lambda(s_WW, m_h^2, m_l^2) = s_WW^2 + m_h^4 + m_l^4 - 2 s_WW m_h^2 - 2 s_WW m_l^2 - 2 m_h^2 m_l^2
```

The joint density is `BW(m_h) * BW(m_l) * sqrt(lambda) / s_WW`, and the full
term (`double_bw_term.neg2ll_bw_phasespace`, matching
`WWKinReco.h::bw_phasespace_neg2ll`) is:

```
-2 ln L = -2[ln BW(m_h) + ln BW(m_l)] + 4 ln(pi) - ln(lambda) + 2 ln(s_WW)
```

`sqrt(lambda)` is a function of **both** masses at once -- it does not
factorize into `f(m_h) * g(m_l)` -- and it vanishes outside the kinematic
triangle `m_h + m_l < m_WW`, which caps how far the two masses can jointly
wander from the two peaks. That coupling through the parent system is the
"correlation": you cannot compute a proper -2 ln L for `m_h` and `m_l`
separately and add them, you need the joint term above.

## 2. Using it to pick the jet pairing

WW->4q gives 3 ways to split the 4 jets into 2 dijets. For each pairing,
evaluate `neg2ll_bw_phasespace(m_h, m_l, m_WW, mW, gW)` at a fixed reference
`(mW, gW)` and take the `argmin` over the 3 pairings
(`double_bw_term.choose_pairing`).

Note the **normalization is not needed for this step**: `log_Z(mW, gW,
m_WW)` depends only on `(mW, gW, m_WW)`, which are identical for all 3
pairings of the same event (same reconstructed parent mass), so it cancels
in the `argmin` and can be dropped entirely -- normalization only matters
once you compare across *different* `mW` hypotheses (Section 3).

Measured on a real ecm240 WW->4q sample (`demo_pairing_and_normalization.py`,
269728 events, well above the `2*mW` threshold so the kinematic triangle
isn't tight):

| pick | pairing efficiency |
|---|---|
| correlated double-BW x phase-space (`choose_pairing`) | **87.6%** |
| naive `argmin(\|mA-mW\|+\|mB-mW\|)` (no lineshape/phase-space) | 87.6% |
| BW-only, no phase-space (drop the `sqrt(lambda)` factor) | 87.4% |

![pairing dijet masses](https://mdefranc.web.cern.ch/mW/example_double_bw/pairing_dijet_masses.png)

One honest caveat, directly checkable from the table above: at this energy
the `sqrt(lambda)` phase-space factor is a wash for *pairing* -- the full
term ties the naive mass-only pick, and even edges out the BW-only variant
by less than the sample's statistical noise. It clearly matters far more for
*normalization* (see the `logZ` shape in Section 3, which is steep right at
threshold), so keep it for physical correctness, but don't expect a large
pairing-efficiency win from it alone at this energy.
- This lineshape-only discriminant is not guaranteed to be the best
  possible pairing pick -- it's a simple, principled starting point (a
  correct likelihood-ratio between the 3 kinematic hypotheses), not
  necessarily the ceiling. A more powerful discriminant would need a
  calibrated reconstructed-response model (e.g. dijet-mass and
  dijet-opening-angle templates fit to your own detector simulation)
  rather than the generator-level lineshape alone -- out of scope for
  this minimal example.

## 3. Normalizing it (needed even for a floating-mW chi2 fit)

If `mW` (or `gW`) is a **floated fit parameter** -- not just used to rank
pairings at a fixed reference value -- the un-normalized term above is not a
genuine `-2 ln L` in `mW`: the *peak height* of `BW(m_h)*BW(m_l)` itself
depends on `mW`, so minimizing the un-normalized term biases the fit toward
whichever `mW` inflates that peak height the most, independent of whether it
actually describes the data. This is exactly as true for a chi2-style kinfit
that adds this term and floats `mW` as it is for an explicit likelihood fit
-- "chi2" vs "likelihood" is a labeling choice, the term is a `-2 ln L`
either way and needs the same fix.

The fix is `log_Z(mW, gW, m_WW)`: the normalization of the joint density
`BW(m_h)*BW(m_l)*sqrt(lambda)/s_WW` over the full 2-D kinematic triangle
`{m_h, m_l > 0, m_h+m_l < m_WW}`. It is a genuine 2-D integral -- it does
**not** factorize into the product of two independent 1-D BW normalizations,
for the same phase-space/triangle-coupling reason as Section 1.
`double_bw_term.log_z_bw_phasespace` computes it via 24-point Gauss-Legendre
quadrature after a `t = atan((m^2-mW^2)/(mW*gW))` substitution that flattens
each BW peak into a uniform measure (see the docstring and
`WWKinReco.h::log_Z_bw_phasespace_ontf` for the derivation); the production
kinfit interpolates a precomputed table of it instead, purely for speed.

`log_Z` is not flat in `mW` -- measured at the median reconstructed parent
mass of a real ecm160 sample (`m_WW` = 153 GeV, i.e. **at** the WW threshold
scan energies this repo targets, where the effect is largest):

![log Z vs mW](https://mdefranc.web.cern.ch/mW/example_double_bw/logz_vs_mw.png)

And the practical consequence, from a toy `-2 ln L(mW)` scan over 3000
MC-truth-paired ecm160 events (so there's no pairing confusion, isolating
the normalization effect):

| curve | -2lnL minimum |
|---|---|
| un-normalized (`neg2ll_bw_phasespace`) | 75.48 GeV |
| normalized (`neg2ll_bw_phasespace_normalized`) | 78.35 GeV |
| generator `mW` | 80.419 GeV |

![mW scan, normalized vs unnormalized](https://mdefranc.web.cern.ch/mW/example_double_bw/mw_scan_norm_vs_unnorm.png)

Normalizing shifts the minimum by **+2.9 GeV**, roughly a third of the way
back to the generator value. Both curves still sit visibly below the
generator `mW` in absolute terms -- that residual offset is the (unrelated)
uncalibrated jet energy scale of these raw reco jets, not a normalization
effect; a real measurement calibrates jet response first (see this repo's
`WWFunctions/` prior-fitting machinery) and then measures a residual bias at
the few-to-few-tens-of-MeV level, not GeV. The point of this demo is the
**shift** between the two curves, not their absolute location.

## Running it yourself

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh   # uproot, numpy, matplotlib
cd examples/double_bw_pairing
python3 demo_pairing_and_normalization.py
```

`ROOT_FILE_PAIRING` / `ROOT_FILE_NORM` / `OUTDIR` / `NTOY` env vars override
the input ntuples, output directory, and toy-scan sample size -- see the
script's docstring.
