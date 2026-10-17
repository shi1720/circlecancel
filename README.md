# CircleCancel

**Gain-Aware Pole-Zero Cancellation on Protected Circles**  
Shivam Gupta · 17 October 2026

Directly remove selected zero/pole factors from a scalar complex rational model while bounding its relative change on one or more entire circles. The implementation supplies gain proposals, exact budgeted matching, and a separate verifier with no scientific dependencies.

[Paper](Gain_Aware_Pole_Zero_Cancellation_Shivam_Gupta.pdf) · [LaTeX](manuscript/main.tex) · [Recorded results](results/summary.json)

## What the guarantee covers

The declared factored rational function, with coordinates interpreted as exact dyadic rationals. The verifier proves continuous-circle error inequalities and the total multiplicative budget. It does not validate root extraction from a barycentric fit, identify physical poles, enforce passivity or state-space stability, or guarantee improved error to an underlying analytic function. Matching optimality is relative to the supplied edge certificates and product composition rule.

The manuscript proves the single-circle formula, common-gain convex-hull obstruction, multiplicative error bound, and its matching specialization. Mӧbius circle geometry, pseudohyperbolic distance, SOCP, and minimum-cost flow are established mathematics; the contribution is the cancellation contract and its auditable realization.

## Reproduce

Python 3.11+ recommended. The recorded run used Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, CVXPY 1.9.3, and Matplotlib 3.11.2 (see requirements-lock.txt for exact installed packages).

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python experiments.py
python verify_certificate.py results/certificates.json
```

The independent checker alone needs only Python:

```bash
python3 verify_certificate.py results/certificates.json
```

Expected replay: **180 cases and 1,379 selected cancellations**. The seven test groups include exhaustive comparison with all matchings on 150 small graphs and deliberate certificate tampering.

`experiments.py` regenerates the raw CSV data, JSON certificates, and the three figures. Root calculations and optimization may vary by platform; certificates are accepted only after exact checks. Runtime measurements are machine dependent. Conic-solver accuracy warnings may occur for nearly degenerate pairs; no solver status substitutes for exact edge verification.

Compile the manuscript from its directory using Tectonic or a standard LaTeX installation:

```bash
cd manuscript
tectonic main.tex
```

The manuscript contains local copies of the figure PDFs so it is a self-contained LaTeX project. `experiments.py` regenerates the primary copies in `figures/`.

## Small example

```python
from fractions import Fraction
from circlecancel import Circle, certified_edge, select
from circlecancel.core import reduced_factors

zeros, poles = [0.2 + 0.1j], [0.201 + 0.101j]
circles = [Circle(0, 1), Circle(0, 1.1)]
gain, error = certified_edge(zeros[0], poles[0], circles)
edges = {(0, 0): error}
gains = {(0, 0): gain}
product, matching = select(1, 1, edges, Fraction(1, 100))
new_zeros, new_poles, new_gain = reduced_factors(
    zeros, poles, matching, gains
)
assert product - 1 <= Fraction(1, 100)
```

For full portable proof objects, use the JSON schema exemplified in `results/certificates.json`; each root, gain, radius, and tolerance is serialized as a rational string. `verify_certificate.py` uses a separate trigonometric nonnegativity calculation rather than the optimizer's image-circle calculation.

## Evidence and limits

- 144 manufactured bank/budget cases: four regimes, 12 seeds, three budgets.
- 36 noisy AAA-generated nominal factorizations: three functions, three noise levels, four seeds.
- 20–23 of 27 poles removed from the AAA-generated models under a declared 1% relative compression tolerance.
- Unprotected nearest-pair removal violates the requested tolerance in 37/144 manufactured cases at matched deletion counts.
- Only 3/36 AAA reductions improve sampled error to the original analytic function. This is compression, not a denoising claim.
- 120 two-circle gain comparisons illustrate why separate normalization can be insufficient.
- 150 exact exhaustive matching audits plus independent exact replay of every final result.

The synthetic experiments do not establish production superiority, market demand, patentability, or universal novelty. Closest prior work and searched contribution boundaries are recorded in `literature_review.csv`. An overlap audit against the author's supplied earlier manuscripts is in `prior_work_audit.csv`.

## Files

- `circlecancel/core.py`: gain proposals, exact edge checks, matching, retained factors.
- `verify_certificate.py`: independent standard-library verifier.
- `experiments.py`: deterministic experiment and figure generator.
- `tests/test_core.py`: mathematical and implementation checks.
- `results/`: full measurements and certificates.
- `figures/`: vector PDF figures and PNG previews.
- `manuscript/`: self-contained paper source.

The MIT license applies to software. The manuscript and figures are supplied as research artifacts; no acceptance or peer-review status is asserted.
