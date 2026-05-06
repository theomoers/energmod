# Pre-Integration Shortlist Note

## Included Models

- `Correlated geometric random walk`: primary rank 1, LR CRPS=0.101699, beta CRPS=0.20106, log score=-13.7501, fit availability=1.000, fit success=1.000.
- `Way-style fixed-rho benchmark`: primary rank 2, LR CRPS=0.0967522, beta CRPS=0.175784, log score=-13.7418, fit availability=1.000, fit success=1.000.
- `Shared-state Bayesian no-intercept regime-learning Wright`: primary rank 3, LR CRPS=0.100785, beta CRPS=0.180575, log score=-13.6026, fit availability=0.901, fit success=1.000.

## Excluded Models

- `Random walk`: outperformed by shortlisted models.
- `Random walk with drift`: outperformed by shortlisted models.
- `Local linear trend`: warning rate too high.
- `Time-trend regression`: outperformed by shortlisted models.
- `FD Wright (parameter uncertainty)`: outperformed by shortlisted models.
- `Stochastic Wright iid`: outperformed by shortlisted models.
- `Autocorrelated Wright (estimated MA1)`: excluded by selection rule.
- `Way-style fixed-rho benchmark (selected rho = 0.40)`: outperformed by shortlisted models.
- `ECM Wright`: excluded by selection rule.
- `Distributed-lag Wright`: excluded by selection rule.
- `TVP Wright`: excluded by selection rule; warning rate too high; fragility too high.
- `Hierarchical Bayesian Wright`: excluded by selection rule.
- `Regime-switching Wright`: excluded by selection rule; fragility too high.

## Current Scope

- This package is still pre-energymod. No system-model coupling is done here.
- The shortlisted implementation set is the current champion plus the main challenger and non-learning benchmark.
- The artifact schema is intentionally thin and is meant to support a later handoff into energymod.