# Data sources, attribution and terms

The MIT license in `LICENSE` covers the code written for this study. It does **not** cover the files in
`data/`, or the results and figure data derived from them. Those remain subject to the terms of their
sources below. License terms were checked against the source pages on 2 October 2026.

## Measured household demand — Smart Grid Smart City (SGSC) customer trial data

- Dataset: *Smart-Grid Smart-City Customer Trial Data*, Australian Government data catalogue,
  https://www.data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data
  (dataset ID `4e21dea3-9b87-4610-94c7-15a8a77907ef`; interval resource
  `b71eb954-196a-4901-82fd-69b17f88521e`; household resource `0404c872-8a83-40e6-9c04-88dfec125aee`).
- Publisher / contact point: Australian Government Department of Climate Change, Energy, the Environment
  and Water (DCCEEW). The Smart Grid Smart City project (2010–2014) was jointly funded by the Australian
  Government and an industry consortium led by Ausgrid.
- **License: Creative Commons Attribution 3.0 Australia (CC BY 3.0 AU)**,
  https://creativecommons.org/licenses/by/3.0/au/. It permits reproducing the work, creating derivative
  works and distributing them, including in collections. Conditions (clause 4): keep copyright notices,
  name the attributed parties and the work's title, include the license or its URI, and clearly indicate
  that changes were made. The dataset page does not prescribe a particular attribution statement.
- Used for: `base_kw` in `data/study_inputs.npz` and the trajectories in `results/figure_data/` that
  contain it.
- Changes made (to be stated with any redistribution): 100 households selected by the rules in
  `docs/DATA_PREPARATION.md`; general-supply and controlled-load energy summed and converted to mean kW;
  customer identifiers removed; restricted to 2013-11-01 to 2014-01-29.
- Suggested attribution: "Household demand derived from the Smart-Grid Smart-City Customer Trial Data,
  © Commonwealth of Australia (Department of Climate Change, Energy, the Environment and Water), licensed
  under CC BY 3.0 AU. Selected households, summed and converted to kW, identifiers removed."
- Removing identifiers does not by itself guarantee that individual load profiles cannot be re-identified.
  The source dataset is itself published under the license above.

## Ambient temperature and irradiance — ERA5 via Open-Meteo

- ERA5 hourly data on single levels, Copernicus Climate Change Service (C3S) Climate Data Store,
  DOI 10.24381/cds.adbb2d47. The dataset page lists the licence as CC-BY (checked 2 October 2026).
- Accessed through the Open-Meteo historical weather API, https://open-meteo.com/en/docs/historical-weather-api.
  Open-Meteo API data are offered under CC BY 4.0 (https://open-meteo.com/en/licence): sharing and adapting
  are permitted with attribution, a link to the licence and an indication of changes.
- Used for: `sydney_utc10_end_ambient`, and as the irradiance input of the simulated PV in
  `sydney_utc10_end_pv`.
- Required attribution (C3S): "Contains modified Copernicus Climate Change Service information 2026.
  Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus
  information or data it contains." For a pure redistribution, the wording is "Generated using Copernicus
  Climate Change Service information". The statement format is taken from the C3S guidance on
  acknowledging CDS data (https://confluence.ecmwf.int/display/CKB/How+to+acknowledge+and+cite+a+Climate+Data+Store+%28CDS%29+catalogue+entry+and+the+data+published+as+part+of+it);
  the year is the year of access or modification.
- Required attribution (Open-Meteo): "Weather data by Open-Meteo.com" with a link to https://open-meteo.com/.
  Suggested citation: Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo.
  https://doi.org/10.5281/zenodo.7970649.
- Dataset citation for ERA5 (from the DataCite record of DOI 10.24381/cds.adbb2d47, retrieved 2 October 2026):
  C3S (2018). *ERA5 hourly data on single levels from 1940 to present* [Dataset]. Copernicus Climate Change
  Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/cds.adbb2d47. The data were obtained
  through the Open-Meteo API (see above), not downloaded directly from the CDS.
- Changes made: temperature averaged from interpolated interval endpoints, preceding-hour mean radiation
  held over two half-hours, conversion to a fixed UTC+10 interval-end clock, and PV output computed with a
  simplified NOCT/PVWatts-inspired model (see `docs/DATA_PREPARATION.md`).

## Distribution network — 33-bus radial test system

- M. E. Baran and F. F. Wu, "Network reconfiguration in distribution systems for loss reduction and load
  balancing", IEEE Transactions on Power Delivery, 4(2):1401–1407, 1989, DOI 10.1109/61.25627.
- The values were checked against MATPOWER's `case33bw.m`
  (https://github.com/MATPOWER/matpower/blob/master/data/case33bw.m), which states that its data come from
  the paper above: the bus loads `P_KW` and `Q_KVAR` and the 32 radial branch resistances/reactances in
  `data/network_case.npz` are identical to the MATPOWER values; the five tie lines of the case file are not
  used. `BASE_MVA` = 10 and `BASE_KV` = 12.66.
- **Terms:** MATPOWER's `LICENSE` states that its code is under the 3-clause BSD license, **but that "the
  MATPOWER case files distributed with MATPOWER are not covered by the BSD license"**; the data were
  "included with permission or converted from data available from a public source". `case33bw.m` carries no
  separate copyright or license notice, only the citation of the paper. We therefore do not claim any
  license for these values: they are the numerical parameters of a standard published test feeder, credited
  to Baran and Wu and to MATPOWER. The authors decided to include the array so that the network experiment
  runs without extra downloads. Users who prefer can obtain `case33bw` directly from MATPOWER; the mapping
  to the arrays in `data/network_case.npz` (`P_KW`, `Q_KVAR`, the first 32 branch rows of `mpc.branch`
  columns 1–4, `BASE_MVA`, `BASE_KV`) is described in `docs/DATA_PREPARATION.md`.

## Simulated quantities

PV output, building thermal parameters, HVAC actions and indoor temperatures are produced by the study's
models and configuration (`src/configuration_without_identifiers.json`). They are not household
measurements.

## Software dependencies

The Python packages in `requirements.txt` and `requirements-figures.txt` are installed from PyPI and are
not redistributed in this repository; each is subject to its own license.
