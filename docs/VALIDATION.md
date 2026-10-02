# Validation record

This file states which checks have actually been executed. It is not a third-party certification.

## Integrity of the released sources and inputs

- The 30 numerical modules in `src/code/` are byte-identical to the modules used for the reported
  experiments; their SHA-256 hashes are listed in `provenance/frozen_module_manifest.json`.
- `src/run_experiment.py` and `src/run_timing.py` are portable wrappers around those modules.
  `src/code/network33.py` differs from the original study code only in how the case data are loaded;
  its network calculations are unchanged.
- The arrays in `data/study_inputs.npz` are bitwise equal to the inputs of the original formal runs
  (`provenance/original_data_notice.json`). Original household identifiers are absent.

## Checks in the original experiment environment (1 October 2026)

Recorded in `provenance/original_portable_checks.json` (environment:
`provenance/original_tested_environment.json`, Python 3.12.14):

- data-free solver check (`run_smoke.py`);
- small-scale runs of the PJ, centralized, deterministic, common-future-action, residual-box and PJ
  timing entry points;
- a complete 10-home, 144-step sensitivity case, which matched the archived aggregate results with a
  maximum absolute difference of 4.55e-13;
- a two-step, 100-home network run.

The full 50/100-home formal cases were not rerun for packaging; `results/reference_results.json`
contains the aggregate results of the original audited runs.

## Reruns in a newly created environment (1–2 October 2026)

A new virtual environment was created with Python 3.12.10 and the versions in `requirements.txt`
(CVXPY 1.6.5, Clarabel 0.11.1, OSQP 1.1.3, SCS 3.3.1, NumPy 2.3.5, SciPy 1.15.3, pandas 3.0.1).
OSQP imported without problems. The following full runs were repeated with the unchanged numerical
modules, via `figures/capture_run.py` and `figures/capture_convergence.py`:

| Run | Quantity | Rerun | Reference |
| --- | --- | --- | --- |
| 50 homes, centralized scenario MPC, seed 202, 672 steps | evaluation bill (AUD) | 1427.1850258007 | 1427.1850258007 |
| | cold / hot degree-hours | 10857.660592 / 51.286366 | identical |
| | cooling electricity (kWh) | 1636.908289 | identical |
| 50 homes, centralized deterministic MPC, 672 steps | evaluation bill (AUD) | 1589.9560184490 | 1589.9560184490 |
| | cold / hot degree-hours, cooling (kWh) | 11062.958688 / 0 / 2494.112855 | identical |
| Network extension, 100 homes, background 0.5 | min. voltage; max. branch / transformer loading | 0.95185 p.u.; 63.92 %; 49.11 % | as reported in the manuscript |
| Network extension, 100 homes, background 1.0 | min. voltage; max. branch / transformer loading; overload steps | 0.90639 p.u.; 104.90 %; 96.76 %; 4 | as reported in the manuscript |
| Fixed state 15 Jan 2014 12:00, PJ 10/50/100 homes | iterations to acceptance | 137 / 159 / 245 | (not archived) |
| Same state, GS 10 homes | outcome | iteration cap (1000) reached | all reference GS trials failed at the cap |

These reruns were carried out by the authors' side on a single Windows 11 machine (build 10.0.26200,
the same OS build as the original environment); they are not an independent third-party reproduction and
do not demonstrate portability to other operating systems or hardware. An earlier attempt to import OSQP (1.1.3 and 0.6.7.post3) in a separate
fresh Windows environment stalled during the original packaging; that problem did not occur here.

## Not verified

- Distributed PJ formal runs (Table 5) in the new environment.
- Execution on Linux or macOS.
- Reconstruction of `data/study_inputs.npz` from the original SGSC archives: the selection rules are
  documented in `DATA_PREPARATION.md`, but the full-archive eligibility screen depends on upstream
  summaries that are not part of this repository.

Failures that are part of the reported results (GS nonconvergence within the iteration cap, soft comfort
violations, network overloads, infeasibility at background 1.05) are expected outcomes and must not be
removed by loosening limits or tolerances.
