# D0 diesel-bus reference assumptions

The D0 reference is a bus-traction-only comparison for the 2024 annual
simulation. It includes diesel fuel and direct bus emissions. It excludes
depot building electricity, demand charges, vehicle purchase, maintenance,
upstream fuel emissions, battery production and any carbon-credit revenue.

| Input | Value used | Reason and source |
| --- | ---: | --- |
| Diesel fuel consumption | 0.3921569 L/km | BMTA reported 2.55 km/L for its diesel-bus operation. The model uses its reciprocal. [BMTA Annual Report 2016](https://www.bmta.co.th/sites/default/files/files/download/2559.pdf) |
| Diesel retail price | 31.89 THB/L | Thailand's reported 2024 average retail diesel price. This price already reflects retail taxes, Oil Fund measures and the 2024 price-control arrangements. [NESDC 2024 economic report](https://www.nesdc.go.th/?ddl=76729&p=76726) |
| Diesel B7 Scope 1 emissions | 2.5503727392 kgCO2e/L | Direct fossil CO2, CH4 and N2O for on-road Diesel B7. [TGO emission-factor database](https://thaicarbonlabel.tgo.or.th/index.php?lang=TH&mod=YjNKbllXNXBlbUYwYVc5dWUyVnRhWE56YVc5dQ) |
| Diesel B7 biogenic CO2 | 0.11507832 kgCO2/L | Reported separately, not added to the fossil Scope 1 comparison. [TGO emission-factor database](https://thaicarbonlabel.tgo.or.th/index.php?lang=TH&mod=YjNKbllXNXBlbUYwYVc5dWUyVnRhWE56YVc5dQ) |
| Separate vehicle carbon tax in 2024 | 0 THB/L | Thailand announced a 200 THB/tCO2 oil-product carbon-price mechanism for later implementation in 2025. It was designed to be incorporated within existing oil excise rather than applied as a separate 2024 operator charge. [Government PRD announcement](https://thailand.prd.go.th/en/content/category/detail/id/52/iid/294932) |

The retail diesel price is used directly. No extra conventional fuel tax,
excise tax, VAT or carbon-tax surcharge is added to it.

For transparency, the notebook reports the diesel fossil Scope 1 total and the
B7 biogenic CO2 quantity separately. The latter is not included in the
principal operational-emissions total.
