# Data Sources & Real-World Calibration

SupplyMind's simulation parameters are calibrated against published industry data, not synthetic estimates. This document lists every real-world data source used.

---

## Company & Financial Data

| Data Point | Value Used | Source |
|---|---|---|
| TSMC 2024 annual revenue | $87.1B | [TSMC Q4 2024 earnings report](https://investor.tsmc.com/english/quarterly-results) |
| TSMC N5 wafer revenue | $16,000-$17,000/wafer | [SemiAnalysis 2024 wafer cost estimates](https://semianalysis.com/) |
| TSMC N7 wafer revenue | $9,500-$10,000/wafer | IC Insights, SemiAnalysis |
| Apple share of TSMC revenue | ~25% (~$22B/yr) | [TSMC annual report](https://investor.tsmc.com/english/annual-reports), [TrendForce](https://www.trendforce.com/) |
| Samsung SDI 2023 revenue | ~$20B | [Samsung SDI annual report](https://www.samsungsdi.com/ir/financial-info/earning.html) |
| Denso FY2023 revenue | ~$45B | [Denso annual report](https://www.denso.com/global/en/about-us/investors/) |
| Bosch Automotive 2023 revenue | ~$55B | [Robert Bosch GmbH annual report](https://www.bosch.com/stories/annual-report/) |
| CATL 2023 revenue | ~$50B | [CATL annual report](https://www.catl.com/en/investor/) |
| Infineon FY2023 revenue | ~$16.3B | [Infineon annual report](https://www.infineon.com/cms/en/about-infineon/investor/) |
| NXP 2023 revenue | ~$13.3B | [NXP annual report](https://investors.nxp.com/) |
| Renesas 2023 revenue | ~$10.5B | [Renesas annual report](https://www.renesas.com/us/en/about/investors) |
| Intel 2024 revenue | ~$54B | [Intel 10-K filing (SEC EDGAR)](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000050863&type=10-K) |
| Micron FY2024 revenue | ~$25B | [Micron annual report](https://investors.micron.com/) |
| Foxconn Thailand operations | ~$8.5B | Foxconn annual report, Thailand BOI |
| Delta Electronics Thailand | ~$3.2B | Delta Electronics annual report |

## Semiconductor Lead Times

| Parameter | Value | Source |
|---|---|---|
| TSMC N5/N3 normal lead time | 16-20 weeks | TSMC investor conference, Susquehanna Financial Group |
| TSMC N7 lead time | 14-18 weeks | Susquehanna Financial Group monthly tracker |
| COVID peak lead time (2021) | 40-52+ weeks | Susquehanna Financial Group |
| 2024 normalized lead time | 14-16 weeks | Susquehanna Financial Group |
| TSMC hot lot premium | 2-3x standard wafer price | Industry reports |
| Automotive chip lead (2024) | 16-22 weeks | S&P Global Mobility, Gartner |

## Commodity Prices (2024 Q4)

| Commodity | Price | Source |
|---|---|---|
| Lithium carbonate (battery) | $13,000-$15,000/MT | Fastmarkets lithium index |
| Lithium (Nov 2022 peak) | $80,000/MT | Trading Economics |
| Copper | $8,800-$9,400/MT | [LME spot price](https://www.lme.com/en/metals/non-ferrous/lme-copper) |
| Aluminum | $2,500-$2,700/MT | [LME spot price](https://www.lme.com/en/metals/non-ferrous/lme-aluminium) |
| Steel HRC (US) | $700-$800/short ton | Platts, CRU |
| Crude oil (Brent) | $70-$80/barrel | ICE Brent futures |
| Neodymium oxide | $65-$75/kg | Asian Metal |
| Dysprosium oxide | $280-$320/kg | Asian Metal |
| 40ft container Shanghai-LA | $3,800-$4,500 | [Freightos Baltic Index (FBX)](https://fbx.freightos.com/) |
| EV battery pack | $100-$150/kWh | BloombergNEF Battery Price Survey 2024 |

## Shipping & Logistics

| Parameter | Value | Source |
|---|---|---|
| Air freight vs sea (cost) | 4-12x sea freight | Freightos, IATA |
| Air freight Shanghai-LA | ~$4.50/kg | TAC Index 2024 |
| Sea freight Shanghai-LA | ~$0.45/kg | Freightos 2024 |
| Rail China-Europe | 2-3x sea cost | China Railway Express, DB Cargo |
| Rail China-Europe transit | 14-18 days | China Railway Express |
| Sea Asia-US West Coast transit | 14-18 days | Maersk, CMA CGM |
| Kaohsiung port dwell | 24-48 hours | Kaohsiung Port Authority |
| Long Beach port dwell | 3-5 days | Port of Long Beach |

## Supply Chain Cost Parameters

| Parameter | Value | Source |
|---|---|---|
| Inventory carrying cost | 23-27% of value/year | [CSCMP State of Logistics Report](https://cscmp.org/store/detail.aspx?id=SOL-23) |
| Semiconductor carrying cost | 25-40% (higher obsolescence) | Gartner, McKinsey |
| Supplier qualification (general) | $50K-$250K | ISM (Institute for Supply Management) |
| Supplier qualification (semiconductor) | $500K-$2M | [McKinsey semiconductor report](https://www.mckinsey.com/industries/semiconductors) |
| Dual-sourcing premium | 10-30% over single-source | [McKinsey supply chain resilience](https://www.mckinsey.com/capabilities/operations/our-insights/risk-resilience-and-rebalancing-in-global-value-chains) |
| Commodity hedge premium | 5-8% of notional | Options market benchmarks |
| SLA penalty (electronics OEM) | 2-10% of PO value/week | Industry contract benchmarks, ISM |
| Auto line stoppage cost | ~$22,000/minute | [Center for Automotive Research](https://www.cargroup.org/) |

## Historical Disruption Events

### Typhoon Gaemi (July 2024)
| Data Point | Value | Source |
|---|---|---|
| Dates | July 24-25, 2024 | Central Weather Administration Taiwan |
| Category | 4 equivalent (185 km/h sustained) | JTWC advisories |
| Port closure | ~2 days (Kaohsiung) | Kaohsiung Port Authority |
| TSMC impact | "Minimal" — precautionary shutdowns | TSMC official statement July 25 |
| Shipping delays post-reopening | 3-5 days | Freightos maritime tracking |
| Total insured losses | $1-2 billion | AON, Swiss Re estimates |

### 2011 Thailand Floods
| Data Point | Value | Source |
|---|---|---|
| Duration | July-December 2011 (~5 months) | World Bank Thailand Flood Report 2012 |
| Total economic loss | $45.7 billion | [World Bank Thailand Flood Report 2012](https://www.worldbank.org/en/country/thailand/publication/thai-flood-2011) |
| Insured losses | $16 billion | Swiss Re Sigma |
| HDD production impact | 40-45% of global production | IHS iSuppli |
| Recovery timeline | 6-9 months | DRAMeXchange |
| Factories flooded | 14,500 | Thai Ministry of Industry |

### 2002 ILWU Lockout
| Data Point | Value | Source |
|---|---|---|
| Duration | 10 days (Oct 1-11) | Federal Mediation records |
| Economic cost | ~$1 billion/day | Anderson Economic Group |
| Ports affected | 29 West Coast ports | PMA records |
| Backlog clearance | 6-8 weeks | Port of LA/Long Beach |

### 2014-15 ILWU Slowdown
| Data Point | Value | Source |
|---|---|---|
| Duration | ~9 months | FMCS, media coverage |
| Vessel wait times | 2-3 weeks at peak | Marine Exchange of Southern California |
| Agricultural export losses | $1.75 billion | American Farm Bureau Federation |

### 2021 Suez Canal Blockage
| Data Point | Value | Source |
|---|---|---|
| Duration | 6 days (March 23-29) | Suez Canal Authority |
| Daily cost to global trade | $9.6 billion/day | Lloyd's List |
| Vessels delayed | ~400 | Suez Canal Authority |
| Backlog clearance | 6-10 days | Maritime industry reports |

### August 2022 Taiwan Strait Exercises
| Data Point | Value | Source |
|---|---|---|
| Duration | Aug 4-10, 2022 | PLA Eastern Theater Command |
| Insurance premium increase | 50-100 basis points | Lloyd's of London |
| Carrier rerouting | Evergreen, Yang Ming rerouted | Reuters, carrier advisories |
| Transit delay | +1-3 days | Maritime industry reports |

### 2021-2022 Auto Chip Shortage
| Data Point | Value | Source |
|---|---|---|
| Lost auto revenue (2021) | $210 billion | [AlixPartners Auto Chip Shortage Study](https://www.alixpartners.com/insights-impact/insights/2021-auto-semiconductor-shortage/) |
| Vehicles not produced (2021) | 7.7 million | [AlixPartners](https://www.alixpartners.com/insights-impact/insights/2021-auto-semiconductor-shortage/) |
| Peak lead time | 40-52+ weeks | Susquehanna Financial Group |
| Full normalization | ~Q1 2024 | Gartner, Susquehanna |

### Commodity Price Shocks (Russia-Ukraine 2022)
| Data Point | Value | Source |
|---|---|---|
| Palladium | +80% in 2 weeks | LME, Financial Times |
| Nickel | +250% in 2 days | LME (March 8, 2022) |
| Lithium peak (Nov 2022) | $80,000/MT (5.7x baseline) | Trading Economics, Fastmarkets |

### Port Ransomware Incidents
| Data Point | Value | Source |
|---|---|---|
| NotPetya → Maersk (2017) | $300M loss, weeks manual ops | Maersk annual report |
| Port dwell time increase | 50-300% | Maritime industry reports (2021-2023) |

---

## Taiwan Blockade Scenario (Analytical)

The hard task's cascading crisis scenario is based on published analytical estimates:

| Data Point | Value | Source |
|---|---|---|
| Global GDP impact of Taiwan blockade | $2.5 trillion+ Year 1 | [Bloomberg Economics 2024 study](https://www.bloomberg.com/graphics/2024-taiwan-invasion-economic-costs/) |
| Taiwan share of sub-7nm chips | 60%+ | Semiconductor Industry Association (SIA) |
| Apple product exposure to TSMC | 30-40% | Apple supply chain analysis |

---

*All figures are from publicly available sources: company filings, government reports, trade publications (Lloyd's List, Freightos, Drewry, Susquehanna, S&P Global Mobility, CSCMP, World Bank, Swiss Re, Asian Metal, Fastmarkets, BloombergNEF).*

---

## Simulation Adjustments

Real-world supply chain disruptions unfold over weeks to months (e.g., the 2011 Thailand floods lasted 5 months, the 2021 auto chip shortage persisted for ~3 years). SupplyMind compresses these timelines to fit 30-60 step episodes, which is standard practice for reinforcement learning environments.

Key adjustments:

| Adjustment | Real World | Simulation | Rationale |
|---|---|---|---|
| Typhoon Gaemi port closure | ~2 days | 12-day active disruption | Scaled to fill meaningful fraction of 30-step episode |
| Thailand flood duration | 5 months | ~15-day active phase | Compressed proportionally; severity curves preserve relative impact |
| Port strike duration | 10 days (2002 ILWU) | 10-14 days | Roughly preserved; already fits episode scale |
| Cascading crisis timeline | Months to years | 60-day episode | Multiple cascading events compressed; relative ordering preserved |
| Typhoon Gaemi severity | Category 3 equivalent | Category 4 parameters | Elevated to create meaningful agent challenge within episode budget |

These adjustments preserve the **relative** severity, cascading order, and decision trade-offs of real events while making the environment practical for agent training and evaluation. The absolute durations and magnitudes should not be interpreted as historical claims.
