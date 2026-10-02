# Preparation of the experimental inputs

`data/study_inputs.npz` was produced from the sources in `../DATA_SOURCES.md` by the steps below.
The released file makes it unnecessary to repeat them; they are documented so that the processing can be
inspected and, with access to the original archives, reconstructed.

## Arrays

| Array | Shape | Unit | Meaning |
| --- | --- | --- | --- |
| `time` | 4320 | – | consecutive half-hour labels, 2013-11-01 00:00 to 2014-01-29 23:30 (fixed study clock) |
| `base_kw` | 4320 × 100 | kW | measured SGSC household base demand |
| `sydney_utc10_end_pv` | 4320 × 100 | kW | simulated PV output |
| `sydney_utc10_end_ambient` | 4320 | °C | regional ERA5 ambient temperature |

The 100 columns follow the order of the anonymous thermal and PV parameter arrays in
`src/configuration_without_identifiers.json`; the first 50 columns form the 50-home case. Do not reorder
them.

## Household selection (SGSC)

- Survey attributes: `HAS_AIRCON = N`, domestic service, warm-temperate climate, `GENERAL_SUPPLY_CNT = 1`,
  `OTHER_LOAD_CNT = 0`, missing `AIRCON_TYPE_CD`.
- Explicit no-generation metadata, zero net/gross solar counts and no positive net/gross generation;
  households with conflicting home-area-network records are excluded.
- Two 90-day windows are screened: 2013-07-01 to 2013-09-28 and 2013-11-01 to 2014-01-29. Quality
  screening therefore uses the full input windows, including the later evaluation period.
- Completeness: 48 valid, non-duplicate half-hour readings per day.
- The sorted eligible identifiers are permuted with NumPy seed 20260911 and the first 200 are captured.
- In captured order, households with a constant combined-load run of 24 half-hours or longer in either
  window are excluded (a conservative quality rule, not proof of a meter fault). The first 100 survivors
  are kept in that order.
- Base demand is `(GENERAL_SUPPLY_KWH + CONTROLLED_LOAD_KWH) / 0.5 h`; missing readings are not imputed.

The full-archive eligibility screen depends on upstream summaries that are not included here, so exact
reselection from the raw archives has not been validated end to end.

## Weather and PV

- Open-Meteo historical API, ERA5 model, requested location (−33.87, 151.21), `timezone=UTC`, hourly
  `temperature_2m` and `global_tilted_irradiance` with tilt 20° and azimuth 180°. The reference
  responses covered 2013-10-30 to 2014-01-16 and 2014-01-15 to 2014-01-29; later API revisions may change
  values.
- Data are aligned to a fixed UTC+10 clock with interval-end labels. This regional alignment is a study
  assumption, not verified household geolocation.
- Half-hour ambient temperature is the mean of interpolated interval endpoints; preceding-hour mean
  radiation is held over its two half-hours.
- PV output is simulated with a NOCT/PVWatts-inspired model and the fixed installation parameters in the
  configuration (Table 3 of the manuscript), not observed household generation.

`src/prepare_weather_inputs.py` regenerates the weather and PV arrays from a retained Open-Meteo response
and a demand array in the fixed column order. `src/replay_cohort.py` replays the archived cohort
refinement from trusted local audit files; those files contain original identifiers and are not
distributed. It uses pickle files, so it must not be run on untrusted downloads.

## Network case

`data/network_case.npz` holds the 33-bus radial feeder of Baran and Wu (1989), with values as published in
MATPOWER's `case33bw`. `src/code/network33.py` reads these arrays and does not embed them. The arrays
correspond to the MATPOWER case as follows (checked on 2 October 2026, all identical):

| Array | Content |
| --- | --- |
| `P_KW`, `Q_KVAR` | `mpc.bus` columns 3 and 4 (active/reactive load in kW and kVAr), 33 buses |
| `BRANCH` | first 32 rows of `mpc.branch`, columns 1–4: from bus, to bus, r (Ω), x (Ω); the five tie lines (rows 33–37) are not used |
| `BASE_MVA`, `BASE_KV` | 10 MVA; 12.66 kV (`mpc.baseMVA`, `mpc.bus` column 10) |
