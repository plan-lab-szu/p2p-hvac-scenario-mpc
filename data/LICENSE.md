# Terms for the files in this directory

The files in `data/` combine third-party material with the authors' own processing. Different parts
therefore carry different terms.

## Third-party material

Values derived from third-party sources remain under their original terms and **must be attributed**:

| Content | Source and terms |
| --- | --- |
| Household demand (`base_kw`) | Smart-Grid Smart-City Customer Trial Data, **CC BY 3.0 AU** (https://creativecommons.org/licenses/by/3.0/au/) |
| Ambient temperature and irradiance (inputs of `sydney_utc10_end_ambient`, `sydney_utc10_end_pv`) | ERA5 (Copernicus Climate Change Service, CC-BY) obtained through Open-Meteo (**CC BY 4.0**) |
| Feeder parameters (`network_case.npz`) | Baran and Wu (1989), values as published with MATPOWER's `case33bw`; no separate license is claimed for these standard test-system values |

Exact attribution statements and the changes made to each source are in `../DATA_SOURCES.md`. You must keep
these notices when you redistribute these files or works derived from them. The SGSC-derived values cannot
be re-licensed by the authors; they stay under CC BY 3.0 AU.

## The authors' own contribution

The household selection and its ordering, the simulated PV output, the time alignment, the experiment
configuration, the fixed timing states (`timing_states.json`) and the results derived from them are released
by the authors under the **Creative Commons Attribution 4.0 International license (CC BY 4.0)**,
https://creativecommons.org/licenses/by/4.0/.

## In short

You may copy, redistribute and adapt these data, including commercially, if you credit the sources listed in
`../DATA_SOURCES.md` and this repository, link the applicable licenses, and indicate what you changed. No
endorsement by the original data providers is implied.

The MIT license of the code (`../LICENSE`) does not apply to the files in this directory.
