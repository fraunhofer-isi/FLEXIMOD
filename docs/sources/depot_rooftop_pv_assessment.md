<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: CC-BY-4.0
-->

# Depot rooftop-PV desktop assessment

## Site selection

The selected site is Borommaratchachonnani Bus Depot, centred near
13.78585 degrees N, 100.39017 degrees E. The supplied route document identifies
this depot for routes 7ก, 79, 101, and 509. This is distinct from the
Phutthamonthon Sai 2 depot near Soi 12.

The location was cross-checked against OpenStreetMap node 9561902029 and public
route information. OpenStreetMap contains the depot point but no building
footprints at the time of assessment.

## Roof measurement

Seven clearly visible roof sections were digitised to their apparent outer roof
edges from Esri World Imagery at web-map zoom level 18. The polygons represent
the gross roof footprint; the later usable-area factor, rather than an artificial
inset, accounts for setbacks, access routes, equipment, and other exclusions.
A former central polygon over parked vehicles and a displaced southern polygon
were removed. The source geometry is stored in
`docs/sources/borom_depot_roof_polygons.geojson` in longitude/latitude order.

| Roof | Gross area (m2) |
| --- | ---: |
| North workshop | 1,057.4 |
| East workshop | 535.0 |
| Southeast workshop | 1,096.4 |
| Southwest north section | 191.8 |
| Southwest middle section | 344.7 |
| Southwest south section | 588.5 |
| East narrow roof | 109.2 |
| **Total** | **3,923.0** |

The polygons represent horizontal plan area. Roof slope, structural capacity,
and precise edge geometry cannot be established from the imagery. Esri
explains that World Imagery combines multiple sources and that image capture
dates must be checked through its imagery metadata tools:

- <https://support.esri.com/en-us/knowledge-base/what-is-the-correct-way-to-cite-an-arcgis-online-basema-000012040>;
- <https://support.esri.com/en-us/knowledge-base/faq-can-the-date-of-an-image-be-determined-from-the-wor-000012181>.

Imagery attribution: Sources: Esri, DigitalGlobe, GeoEye, i-cubed, USDA FSA,
USGS, AEX, Getmapping, Aerogrid, IGN, IGP, swisstopo, and the GIS User
Community.

## Usable area and capacity

The desktop assessment uses 50%, 65%, and 80% usable-area cases because the
imagery cannot resolve structural exclusions, fire access, maintenance paths,
or every rooftop obstruction. The base case uses 0.20 kWp of modules per usable
square metre, a 1.20 DC-to-AC ratio, and 96% inverter efficiency.

| Case | Usable fraction | Usable area (m2) | DC capacity (MWp) | AC limit (MW) |
| --- | ---: | ---: | ---: | ---: |
| Conservative | 50% | 1,961.5 | 0.392 | 0.327 |
| Base | 65% | 2,549.9 | 0.510 | 0.425 |
| Optimistic | 80% | 3,138.4 | 0.628 | 0.523 |

NREL describes 1.20 as a reasonable default DC-to-AC ratio and 96% as the
PVWatts inverter efficiency used in REopt:

<https://reopt.nrel.gov/tool/reopt-user-manual.pdf>

The model's base available-PV series is:

```text
min(0.509990 MWp * solar_capacity_factor * 0.96, 0.424992 MW)
```

The 2024 solar capacity factor already includes the transparent temperature
model and 10% non-inverter system loss documented in
`docs/sources/renewable_availability_2024.md`.

## Interpretation and required field check

This is a desktop technical-potential estimate, not a construction-ready
design. Before using 0.510 MWp as an investment or procurement value, a
qualified engineer must verify roof ownership, present roof layout, material
and age, corrosion, drainage, wind loading, fire access, electrical routing,
and allowable structural loads. Any rejected roof polygon must be removed and
the forecast regenerated.
