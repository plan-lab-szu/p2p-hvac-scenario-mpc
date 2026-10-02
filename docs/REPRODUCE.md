# Reproducing the results

All commands are run from the repository root with the environment from `requirements.txt`.
Output paths must not exist yet; choose a new name for every run. `INPUT` below means
`data/study_inputs.npz`.

Common settings (defaults of `src/run_experiment.py`): scenario seed 202, `S = 30` scenarios,
horizon `H = 8`, four PJ worker processes (fewer for small populations), initial `rho = 0.02`,
`beta = 0.002`. The formal evaluation starts at row 3648 (16 January 2014) and runs 672 half-hour
steps, of which the first 48 are warm-up and 624 are evaluated. Coordination state and the adaptive
penalty persist across closed-loop steps. The post-solve centralized comparison is not a stopping
criterion for the distributed algorithm.

Indicative run times are from the test machine (Windows 11, 20 logical CPUs) and are not guarantees.

## 1. Minimal checks

```bash
python src/run_smoke.py
python src/run_experiment.py --inputs data/study_inputs.npz --output runs/check_pj --homes 2 --steps 4
```

`run_smoke.py` solves a two-home, three-scenario artificial problem in memory and compares the
vectorized centralized scenario QP with a reference solver; it uses no study data.

## 2. Paired distributed/centralized evaluation (Table 5)

```bash
python src/run_experiment.py --inputs data/study_inputs.npz --output runs/n50_pj      --method pj      --homes 50 --start 3648 --steps 672 --warmup 48
python src/run_experiment.py --inputs data/study_inputs.npz --output runs/n50_central --method central --homes 50 --start 3648 --steps 672 --warmup 48
```

Repeat with `--homes 100` and new output names. Expected evaluation-period bills (grid charges plus
balancing surcharges) are listed in `results/reference_results.json` (`main`): 1427.1683 / 1427.1850 AUD
(PJ / centralized, 50 homes) and 3364.5019 / 3364.5310 AUD (100 homes). The centralized 50-home run takes
about 25 minutes; distributed PJ runs take several hours per population. PJ runs also write one clearing
certificate per step.

## 3. Uncertainty-model comparison (Table 6, Figs. 7–8)

Use the same 50/100-home, 672-step, 48-warm-up settings with:

```bash
--method central --mode deterministic
--method central --mode scenario --seed 202      # also --seed 101 and --seed 303
--method mean_open_loop                          # common-future-action baseline
--method component_residual                      # residual-box envelope
```

The residual-box calibration is rebuilt from the training prefix of the inputs. The box is empirical;
it is neither a distribution-free guarantee nor a fully adaptive robust counterpart.

## 4. Configuration sensitivity (Fig. 6, Table C.1)

```bash
python src/run_experiment.py --inputs data/study_inputs.npz --output runs/sens_R08 --method central --homes 10 --start 3552 --steps 144 --warmup 48 --vary R_C_per_kw --factor 0.8
```

Run factors 0.8 and 1.2 for `R_C_per_kw`, `C_kwh_per_C`, `COP` and `cooling_max_kw_electric`, and 0.5
and 1.5 for `pv`. Controller and simulated-building parameters change together; this is configuration
sensitivity, not robustness to parameter-identification error.

## 5. Network extension (Fig. 9)

```bash
python src/run_experiment.py --inputs data/study_inputs.npz --output runs/net05 --method network --network-case data/network_case.npz --homes 100 --start 3552 --steps 144 --warmup 48 --background 0.5
```

Repeat with `--background 1.0` and `--background 1.05`. This is the nominal centralized
common-future-action extension, not distributed network MPC. Realized AC checks are saved per step and
overloads are not discarded. Background 1.05 is infeasible in the reference study; the run then saves
`run.json` and exits unsuccessfully. This is an expected, reported outcome, not a reason to loosen
limits. Each run takes about 6 minutes.

## 6. Fixed-state timing (Table 4)

```bash
python src/run_timing.py --inputs data/study_inputs.npz --states data/timing_states.json --output runs/timing_all
```

Run on an otherwise idle machine. `--job-index 1` runs a single PJ job; omitting it runs all 24 jobs.
PJ uses the adaptive penalty policy and GS a fixed penalty, so the comparison does not isolate
parallelization. All eight reference GS trials stop at the 1000-iteration cap; do not convert capped
failures into timing or speedup samples. Elapsed times are hardware-dependent.

## 7. Figures 4–9

The rendered figures are in `figures/output/` and can be redrawn from the included data:

```bash
python -m pip install -r requirements-figures.txt
python figures/make_figs.py
```

| Output file | Manuscript figure |
| --- | --- |
| `figD_convergence_timing.pdf` | Fig. 4 |
| `figA_closed_loop_trajectories.pdf` | Fig. 5 |
| `figE_sensitivity.pdf` | Fig. 6 |
| `figB_comfort_decomposition.pdf` | Fig. 7 |
| `figC_cost_comfort_tradeoff.pdf` | Fig. 8 |
| `figF_network_extension.pdf` | Fig. 9 |

Figures 6–8 use only `results/reference_results.json` and the inputs. The trajectory and trace data in
`results/figure_data/` can be regenerated with the capture scripts, which reuse the numerical modules
unchanged and replicate the closed loop of `src/run_experiment.py` while additionally storing per-home
arrays:

```bash
python figures/capture_run.py --output results/figure_data/sc50.npz  --homes 50                       # ~25 min
python figures/capture_run.py --output results/figure_data/det50.npz --homes 50 --mode deterministic  # ~1 min
python figures/capture_run.py --output results/figure_data/net05.npz --method network --homes 100 --start 3552 --steps 144 --background 0.5
python figures/capture_run.py --output results/figure_data/net10.npz --method network --homes 100 --start 3552 --steps 144 --background 1.0
python figures/capture_convergence.py        # PJ (10/50/100 homes) and GS (10 homes) residual traces
```

`capture_run.py` overwrites its output file. The Fig. 5 trajectory is the centralized scenario-MPC run;
the paired distributed trajectory differs from it by at most 0.000284 °C in indoor temperature (Table 5).

Two derived quantities in Figs. 6–8 are computed by `figures/make_figs.py`: the free-floating
(zero-cooling) cold-violation floor, which bounds the cold violation of any cooling-only controller from
below, and the avoidable discomfort (cold violation above that floor plus hot violation). Their definition
is Eq. (37) of the manuscript.
