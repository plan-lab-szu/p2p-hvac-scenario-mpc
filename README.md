# Distributed scenario-based MPC for P2P energy trading with residential thermal flexibility

Code, processed experimental inputs and results for the article

> Q. Yang, H. Wang, T. Wang, D. Li, F. Liu, "Distributed Scenario-Based Model Predictive Control for
> Peer-to-Peer Energy Trading with Residential Thermal Flexibility", *Sustainable Energy, Grids and
> Networks* (under revision, manuscript SEGAN-D-26-02133).

The repository reproduces the closed-loop experiments of the revised manuscript:

| Manuscript item | What it shows | How to reproduce |
| --- | --- | --- |
| Table 4, Fig. 4 | Fixed-state PJ-ADMM timing; PJ/GS convergence traces | `src/run_timing.py`; `figures/capture_convergence.py` |
| Table 5 | Paired 50/100-home distributed (PJ) vs. centralized results | `src/run_experiment.py --method pj` and `--method central` |
| Table 6, Figs. 7–8 | Deterministic, scenario (seeds 101/202/303), common-future-action and residual-box baselines | `src/run_experiment.py --method central/mean_open_loop/component_residual` |
| Fig. 6, Table C.1 | Single-factor configuration sensitivity | `src/run_experiment.py --vary ...` |
| Fig. 9 | Nominal-network extension with realized AC checks | `src/run_experiment.py --method network` |
| Fig. 5 | Closed-loop trajectories (50 homes, 17–19 Jan 2014) | `figures/capture_run.py` |

Exact commands, expected values and run times are in [docs/REPRODUCE.md](docs/REPRODUCE.md).
What has and has not been verified is stated in [docs/VALIDATION.md](docs/VALIDATION.md).

## Repository layout

```
src/                     entry points and fixed configuration
src/code/                numerical modules (frozen; hashes in docs/provenance/frozen_module_manifest.json)
data/                    processed experimental inputs (see data/README.md)
results/                 audited aggregate results; trajectory/trace data behind the figures
figures/                 figure scripts; figures/output/ holds the rendered PDFs
docs/                    reproduction guide, validation record, data preparation, provenance
```

Run everything through the entry points in `src/` (and `figures/`). The modules in `src/code/` keep
historical `main()` functions that refer to the original working directories; do not execute them
directly.

## Quick start

Tested with Python 3.12 on Windows 11 (see [docs/VALIDATION.md](docs/VALIDATION.md)).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt        # Linux/macOS: .venv/bin/python
.venv/Scripts/python src/run_smoke.py                           # data-free solver check
.venv/Scripts/python src/run_experiment.py --inputs data/study_inputs.npz --output runs/check_pj --homes 2 --steps 4
```

Each run writes `run.json` (configuration, input hash, per-step records, summary) into a **new** output
directory; existing outputs are never overwritten. Do not run Python with `-O`: solver validation uses
assertions.

To redraw the figures from the included trajectory data:

```bash
.venv/Scripts/python -m pip install -r requirements-figures.txt
.venv/Scripts/python figures/make_figs.py
```

## Data

`data/study_inputs.npz` contains the processed inputs actually used in the experiments: measured
Smart Grid Smart City (SGSC) household base demand for 100 homes, simulated PV output and regional
ERA5 ambient temperature, at 30-minute resolution from 2013-11-01 to 2014-01-29. Original household
identifiers are not included. HVAC actions, building parameters and indoor temperatures are simulation
products, not household measurements. Sources, attribution and terms are listed in
[DATA_SOURCES.md](DATA_SOURCES.md); the household-selection and preprocessing rules are documented in
[docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md).

## Scope

The results are simulation-based. The core controller represents an uncongested community market; the
network experiment is a separate centralized validation and does not establish feeder security of the
distributed controller. Cooling-only HVAC with soft comfort bounds does not guarantee thermal comfort.
Timing results are machine-dependent and do not establish a parallelization speedup. See Section 6.5.4
of the manuscript for the full list of limitations.

## License

Code: MIT License (see [LICENSE](LICENSE)).

Data in `data/` and the derived results in `results/`: they contain or are derived from third-party
material that must be attributed under the source licenses — SGSC demand data (CC BY 3.0 AU), ERA5 and
Open-Meteo weather data (CC-BY / CC BY 4.0) and the Baran–Wu 33-bus test feeder. The required attribution
statements and the changes made are listed in [DATA_SOURCES.md](DATA_SOURCES.md). The authors' own
contribution to these files (household selection, simulated PV, time alignment, configuration, timing
states and derived results) is released under CC BY 4.0; see [data/LICENSE.md](data/LICENSE.md). The MIT
license does not apply to these files.

Data attribution: household demand derived from the Smart-Grid Smart-City Customer Trial Data
(© Commonwealth of Australia, Department of Climate Change, Energy, the Environment and Water,
CC BY 3.0 AU; selected households, converted to kW, identifiers removed). Contains modified Copernicus
Climate Change Service information 2026. Weather data by [Open-Meteo.com](https://open-meteo.com/).

## Acknowledgments of third-party sources

See [DATA_SOURCES.md](DATA_SOURCES.md) for citations (Open-Meteo, ERA5, Baran and Wu).

## Citation

See [CITATION.cff](CITATION.cff).
