# Pre-Integration Shortlist Note

## Included Models

- `Correlated geometric random walk`: primary rank 1, LR CRPS=0.0872444, beta CRPS=0.16976, log score=-13.5394, fit availability=1.000, fit success=1.000.
- `Way-style fixed-rho benchmark (selected rho = 0.40)`: primary rank 2, LR CRPS=0.0941153, beta CRPS=0.170377, log score=-13.6323, fit availability=1.000, fit success=1.000.
- `Shared-state Bayesian residual-regime Wright`: primary rank 3, LR CRPS=0.0975061, beta CRPS=0.176391, log score=-13.6779, fit availability=0.901, fit success=1.000.

## Excluded Models

- `Local linear trend`: warning rate too high.
- `Way-style fixed-rho benchmark`: outperformed by shortlisted models.
- `Regime-switching Wright`: excluded by selection rule; fragility too high.
- `Distributed-lag Wright`: excluded by selection rule.
- `Random walk with drift`: outperformed by shortlisted models.
- `ECM Wright`: excluded by selection rule.
- `Stochastic Wright iid`: outperformed by shortlisted models.
- `Autocorrelated Wright (estimated MA1)`: excluded by selection rule.
- `FD Wright (parameter uncertainty)`: outperformed by shortlisted models.
- `Random walk`: outperformed by shortlisted models.
- `TVP Wright`: excluded by selection rule; warning rate too high; fragility too high.
- `Time-trend regression`: outperformed by shortlisted models.
- `Hierarchical Bayesian Wright`: excluded by selection rule.

## Current Scope

- This package is still pre-energymod. No system-model coupling is done here.
- The shortlisted implementation set is the current champion plus the main challenger and non-learning benchmark.
- The artifact schema is intentionally thin and is meant to support a later handoff into energymod.