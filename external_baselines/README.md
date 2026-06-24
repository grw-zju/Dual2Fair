# External baseline sources

This directory stores upstream repositories used to audit and adapt baseline
implementations in this project.

## Cloned official repositories

- `user-fairness`: official code for UFR, `rutgerswiselab/user-fairness`.
  The upstream implementation depends on Gurobi and ranking-file inputs. The
  local `baseline/ufr.py` adapter uses a Gurobi optimizer when `gurobipy` is
  installed and falls back to a deterministic greedy optimizer otherwise.
- `Item-Underrecommendation-Bias`: official code for DPR,
  `Zziwei/Item-Underrecommendation-Bias`. The upstream implementation targets
  Python 2 and TensorFlow 1. The local `baseline/dpr.py` ports the relevant
  score-distribution and adversarial group-regularization ideas to PyTorch.
- `Ada2Fair`: official code for Ada2Fair, `Sherry-XLL/Ada2Fair`. The upstream
  implementation is RecBole-based. The local `baseline/ada2fair.py` ports the
  dynamic provider/user fairness weight generation into this project's model
  interface.
- `FairDual`: official code for FairDual, `XuChen0427/FairDual`. The upstream
  implementation is built for BigRec/Llama-style recommendation. The local
  `baseline/fairdual.py` keeps the learnable shadow-price idea and adapts it to
  item exposure disparity in this project.

## No official repository found in this pass

No authoritative official repository was found for CPFair, MultiFR, FAIR, or
FairSort through paper pages, GitHub repository search, and keyword search.
Their local adapters remain self-contained PyTorch/numpy implementations.
