# Experimental inputs

| File | Content |
| --- | --- |
| `study_inputs.npz` | Inputs of all closed-loop experiments: 100 homes × 4320 half-hour rows (2013-11-01 to 2014-01-29); arrays `time`, `base_kw` (measured SGSC demand, kW), `sydney_utc10_end_pv` (simulated PV, kW), `sydney_utc10_end_ambient` (ERA5 temperature, °C). The first 50 columns form the 50-home case. |
| `network_case.npz` | 33-bus radial feeder data used by the network extension (`--network-case`). |
| `timing_states.json` | The 24 fixed states (temperatures and previous controls) of the timing tests in Table 4. |

SHA-256:

| File | Hash |
| --- | --- |
| `study_inputs.npz` | `bcce2b5066e4a33cb42ba83e3916ca1e9a86964e20cdc94d8c08b6a3be93ced3` |
| `network_case.npz` | `86a0e4c828d8ce2ff6b7a0895e2543def6054e72a0011053f69f9b0e981459a4` |
| `timing_states.json` | `fa31743f66b5539cb61b527b5ed165b8cf88497f768d062cc3483b150c0573ac` |

Sources, attribution and terms: `../DATA_SOURCES.md`. Processing: `../docs/DATA_PREPARATION.md`.
These files are not covered by the code license.

Attribution (required by the source licenses): household demand derived from the Smart-Grid Smart-City
Customer Trial Data (© Commonwealth of Australia, Department of Climate Change, Energy, the Environment
and Water, licensed under CC BY 3.0 AU, https://creativecommons.org/licenses/by/3.0/au/; selected
households, summed and converted to kW, identifiers removed). Contains modified Copernicus Climate Change
Service information 2026. Weather data by Open-Meteo.com (https://open-meteo.com/, CC BY 4.0).
Feeder data: M. E. Baran and F. F. Wu, IEEE Trans. Power Delivery 4(2), 1989, DOI 10.1109/61.25627.
