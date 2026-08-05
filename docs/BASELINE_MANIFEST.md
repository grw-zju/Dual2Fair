# Baseline Manifest

Implementations under `baseline/` are local compatible implementations unless explicitly linked to a pinned upstream commit. They must not be described as exact official reproductions.

| Method | Local file | Status |
|---|---|---|
| UFR | `baseline/ufr.py` | local implementation; optional Gurobi path |
| HyperUOF | `baseline/hyperuof.py` | local implementation |
| DPR | `baseline/dpr.py` | local approximation |
| FairDual | `baseline/fairdual.py` | local approximation |
| CPFair | `baseline/cpfair.py` | local implementation |
| MultiFR | `baseline/multifr.py` | local implementation |
| Ada2Fair | `baseline/ada2fair.py` | local approximation |
| FAIR | `baseline/fair_method.py` | local implementation |
| FairSort | `baseline/fairsort.py` | local implementation |

Any future official adapter must record repository URL, commit hash, local modifications, input conversion, and hyperparameter search space before being included in comparison tables.
