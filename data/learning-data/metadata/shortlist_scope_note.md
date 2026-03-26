# Pre-Integration Shortlist Note

## Included Models

- `Shared-state Bayesian regime Wright`: primary rank 1, LR CRPS=0.0854507, beta CRPS=0.164741, log score=-13.4698, fit availability=0.901, fit success=1.000.
- `Correlated geometric random walk`: primary rank 2, LR CRPS=0.0872444, beta CRPS=0.16976, log score=-13.5394, fit availability=1.000, fit success=1.000.
- `Way-style fixed-rho benchmark (rho = 0.35)`: primary rank 3, LR CRPS=0.0926192, beta CRPS=0.177143, log score=-13.6686, fit availability=1.000, fit success=1.000.

## Excluded Models

- `Random walk`: outperformed by shortlisted models.
- `Random walk with drift`: outperformed by shortlisted models.
- `Local linear trend`: warning rate too high.
- `Time-trend regression`: outperformed by shortlisted models.
- `Stochastic Wright iid`: outperformed by shortlisted models.
- `Autocorrelated Wright (estimated MA1)`: outperformed by shortlisted models.
- `Way-style fixed-rho benchmark`: outperformed by shortlisted models.
- `ECM Wright`: outperformed by shortlisted models.
- `Distributed-lag Wright`: outperformed by shortlisted models.
- `TVP Wright`: warning rate too high; fragility too high.
- `Hierarchical Bayesian Wright`: outperformed by shortlisted models.
- `Regime-switching Wright`: fragility too high.

## Current Scope

- This package is still pre-energymod. No system-model coupling is done here.
- The shortlisted implementation set is the current champion plus the main challenger and non-learning benchmark.
- The artifact schema is intentionally thin and is meant to support a later handoff into energymod.