# external_data — Provenance Manifest

Source, fetch method, date, size, SHA-256, and downstream consumers for **every** dataset
in `external_data/`. This directory's data files are gitignored (large, regeneratable); the
fetcher (`fetch_all.py`) and this manifest are tracked so a fresh clone can rebuild the corpus.

## Rebuild

```bash
export SEC_CONTACT_EMAIL="you@example.com"   # SEC fair-access policy requires a contact e-mail
python external_data/fetch_all.py
```

All sources except EM-DAT are fetched automatically. EM-DAT requires a free registered
download (see below). The World Bank / FRED / Wikipedia / NOAA sources are *live* — re-running
`fetch_all.py` may produce newer data than the checksums below (which are a 2026-07-02 snapshot);
this is expected for live series and is not a corruption.

- Checksums / sizes verified: **2026-07-02**.
- `fetch_all.py`, `PROVENANCE.md` — tracked. Everything else here — gitignored.
- `fetch_results.json`, `fetch.log` — generated run artifacts (gitignored).

---

## Dataset summary

| Dataset | Files | Total size | Auto-fetch | Consumer(s) |
|---|---|---|---|---|
| SEC EDGAR 10-Ks | 20 | 74.3 MB | yes (SEC EDGAR) | v3 RAG beast, v3 block-4 RAG |
| EM-DAT disasters | 1 | 5.8 MB | **no — registered download** | `scripts/crisis_library/cook_v2.py` |
| FRED daily Brent | 1 | 169 KB | yes (FRED) | `versions/v5_phoenix/counterfactual_v2/platinum.py` (method C) |
| FRED truck PPI | 1 | 5 KB | yes (FRED) | (none currently — real series retained) |
| Policy papers | 3 | 2.8 MB | yes (FRBNY/BIS/FRBSF) | v3 RAG beast, v3 block-4 RAG |
| Wikipedia crisis articles | 31 | 988 KB | yes (Wikipedia API) | v3 RAG beast, v3 manual BEIR, v3 r4_* |
| World Bank macro | 6 | 260 KB | yes (World Bank API) | v3 RAG beast (world_bank chunks) |
| NOAA active storms | 1 | 4.5 KB | yes (NOAA NHC, live) | (live-feed demo; no static consumer) |
| WGI governance | 1 | 9.9 MB | no — see `wgi/PROVENANCE.md` | `rl/analysis/trained_models.py`, `rl/data/build_unified_buffer_v2.py` |

---

## 1. SEC EDGAR 10-K filings — `sec_10k/`

- **Source:** SEC EDGAR. Submissions JSON `https://data.sec.gov/submissions/CIK{cik}.json`;
  primary 10-K document `https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}`.
- **Fetch method:** `fetch_all.py::sec_edgar` (requires `SEC_CONTACT_EMAIL` in the User-Agent).
- **Date:** 2026-04-16. **Count:** 20 companies. **Total size:** 77,900,203 bytes.
- **Consumers:** `versions/v3_arcadia/40_granite/r5_rag_beast.py` (SEC corpus chunks),
  `versions/v3_arcadia/train_v3_block4_rag.py`.
- **License:** U.S. Government work / public domain.

Per-file SHA-256:

| File | Bytes | SHA-256 |
|---|---|---|
| `AAPL_10k.html` | 1,520,208 | `548ae59778cf08ee0f2ee088e7ece20d947076c3c01f74d2d65db4c2777e436a` |
| `CAT_10k.html` | 6,100,469 | `090ce2bd6d7dd1e85d4397ab152d3c3c3fb059fd4af0eb9aa979c1eb2ff92f15` |
| `CVX_10k.html` | 6,061,316 | `eba2b2a9b395e555114d1aa5f7e0048b03d6d49665b10e48cb9f3c23da4e1f07` |
| `F_10k.html` | 5,281,502 | `3bbda349b5831cfb9a2686dbdb7d87614bcdbe2d195aa8ecd9b39215945361f9` |
| `GE_10k.html` | 2,454,323 | `49ab3c5b6a485c0bd604619f7dec77543450adbd493e7fb6cc64acec98c7669e` |
| `GOOG_10k.html` | 2,616,499 | `c2f6301004f35411a20611c14ff01d80a85c0bcbab6053c80d8cc7f6fc747161` |
| `IBM_10k.html` | 1,077,083 | `d4669e1c5c4536b8297cc3c8d3366129d3fc8acf39f968f1c68dd29ebe03b86a` |
| `INTC_10k.html` | 3,320,720 | `67af52d83cdc32a0c52d6007ba0b1031f25c50bc1cae5ee0d626721c9989e42d` |
| `JNJ_10k.html` | 3,701,587 | `082f4e489d4c19f585cae4c8b62acea92bd6727576f7b0e450b9dfe8ecd42694` |
| `LMT_10k.html` | 2,909,179 | `252829a67a8ce7821a595295a9a20353f8a05c18a2ff989a9b21983a1ea51b32` |
| `MCD_10k.html` | 2,182,265 | `64e680630d5fe4448db26ff695aaf50189ec23427566f01508d3bf27aa8d8b3e` |
| `MSFT_10k.html` | 8,158,067 | `99d693f6c1544144ebeee92954f151a85bc62111837530a42855953bc01d0bbe` |
| `PFE_10k.html` | 5,222,324 | `175e07c21ee258eddd9952e443d34df2a297d0c38e2d1312dff0a64a31c401ab` |
| `PG_10k.html` | 2,491,287 | `771cbdd953a63a343f4ddb4e09a90d81cca8b9209730ae27db63461e1562336f` |
| `TGT_10k.html` | 7,694,408 | `c8ebfd008d81bba5618805ba416fb2ac71f343195fb1e612952322175da01980` |
| `TSLA_10k.html` | 2,391,529 | `90d5d2bc4057df41c645f85dcebe575af9b0953a6a1902d189297bec17fd440a` |
| `T_10k.html` | 3,931,782 | `01c1c129368cf58b57adb4204adf81f862af8346191689a54ed75d199b3e5ca3` |
| `UNH_10k.html` | 2,870,606 | `ac66517766674908c3702dab75f52fc650729cef9dd81dfe04a580a8429c8543` |
| `WMT_10k.html` | 2,323,981 | `276a6accff6066f00342ba03b390c6c2bd51aa7246f0346df22df0b839a5bf5c` |
| `XOM_10k.html` | 5,591,068 | `3591db2246ab14c52465d32f69b459b000919c141a5656c81cf59acf0a805be8` |

---

## 2. EM-DAT disasters — `emdat/emdat_public_2000_2026.xlsx`

- **Source:** EM-DAT, the International Disaster Database, CRED / UCLouvain —
  `https://public.emdat.be`.
- **Fetch method:** **manual, registered download** (free account required; no public API).
  There is no auto-fetcher — EM-DAT's terms require an authenticated download. To rebuild:
  register at `https://public.emdat.be`, query 2000–2026 (all disaster types, all countries),
  export to XLSX, and save as `external_data/emdat/emdat_public_2000_2026.xlsx`.
- **Date:** export "Last Update" stamps = 2025-12-20; 16,811 data rows, 47-column schema.
- **Size:** 6,085,729 bytes.
- **SHA-256:** `842180e4507ccee92954315204b05a797e8c031ea1cb1a584477d0f2c9d1d54a`
- **Consumer:** `scripts/crisis_library/cook_v2.py` (`raw_url` cited there) → produces
  `versions/v4_arcadia_live/scenarios/crisis_library_v2.{json,faiss,_emb.npz}`, consumed by
  the live server and demo orchestrator.
- **License:** EM-DAT non-commercial academic use (attribution required).

---

## 3. FRED time series — `fred_brent_daily.csv`, `fred_truck_transport.csv`

- **Source:** FRED (Federal Reserve Bank of St. Louis) keyless CSV endpoint
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES}`.
- **Fetch method:** `fetch_all.py::fred_series`.
- **License:** FRED terms of use (most series public; Brent sourced from EIA).

| Series | File | Header | Range | Bytes | SHA-256 | Consumer |
|---|---|---|---|---|---|---|
| `DCOILBRENTEU` (daily Brent, USD/bbl) | `fred_brent_daily.csv` | `observation_date,DCOILBRENTEU` | 1987-05-20 → 2026-06-29 (9,922 usable obs) | 173,220 | `0e88d6dd07f200f5d60df8ea00cdf2a1c47acf7419e128c801a373f21b28d017` | `platinum.py` method C (real BSTS-lite counterfactual) |
| `PCU484121484121` (monthly general-freight truck-transport PPI, index) | `fred_truck_transport.csv` | `observation_date,PCU484121484121` | 2003-12 → present | 5,125 | `7921252f8667858fe509e102355314a04a82fd3b0f04faeecf352e1c490dbfbf` | none currently (real series retained; formerly mislabeled as "Brent" by method C — fixed 2026-07-02) |

> **Note:** the old `GLBLALSCMINDX` ("supply-chain pressure") pull was removed — that id does
> not exist on FRED (HTTP 404). The NY Fed GSCPI is not on FRED; its paper is under `policy_papers/`.

---

## 4. Policy papers — `policy_papers/`

All three are **real, on-topic** supply-chain economics PDFs, verified by page-1 title. They
replace three previously-committed mislabeled files removed on 2026-07-02 (see "Deleted" below).

- **Fetch method:** `fetch_all.py::policy_papers`.
- **Consumers:** `versions/v3_arcadia/40_granite/r5_rag_beast.py` (policy corpus chunks),
  `versions/v3_arcadia/train_v3_block4_rag.py::load_policy_papers`.

| File | Paper | Source URL | Bytes | SHA-256 |
|---|---|---|---|---|
| `frbny_gscpi_sr1017.pdf` | FRBNY Staff Report No. 1017 — "The GSCPI: A New Barometer of Global Supply Chain Pressures" (Benigno, di Giovanni, Groen, Noble; May 2022) | `https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr1017.pdf` | 1,398,153 | `b6d1ef03eab445afc4a234e47e6a9437b30b307dbe9a89f61fae65f1a4e93bb6` |
| `bis_bottlenecks_bull48.pdf` | BIS Bulletin No 48 — "Bottlenecks: causes and macroeconomic implications" (Rees & Rungcharoenkitkul; Nov 2021) | `https://www.bis.org/publ/bisbull48.pdf` | 761,857 | `b17c5e1ea94aede14a8511f5833791512d1e932df428538133bfdcd63cec88b4` |
| `frbsf_supply_demand_inflation_wp2022-18.pdf` | FRBSF Working Paper 2022-18 — "Decomposing Supply and Demand Driven Inflation" (Adam Hale Shapiro) | `https://www.frbsf.org/economic-research/wp-content/uploads/sites/4/wp2022-18.pdf` | 825,932 | `f18a63703cc8c10307415d493d2792bd66bd2f11463aee57e14ca4b5cae59a20` |

---

## 5. Wikipedia crisis / chokepoint articles — `wikipedia_crises/`

- **Source:** English Wikipedia via the `wikipedia-api` package (MediaWiki API; redirects
  resolved). Content licensed CC BY-SA 4.0.
- **Fetch method:** `fetch_all.py::wikipedia_crises` (title list `WIKIPEDIA_TITLES`).
- **Date:** 26 articles 2026-04-16; 5 added 2026-07-02 (retry of the 6 titles that previously
  404'd — see below). **Count:** 31. **Total size:** 1,012,001 bytes.
- **Consumers:** `versions/v3_arcadia/40_granite/r5_rag_beast.py`, `.../r5_manual_beir.py`,
  `versions/v3_arcadia/30_dangerous/r4_*.py`.

**2026-07-02 retry — the six formerly-failing titles were remapped to current canonical pages:**

| Original (404) title | Canonical article fetched | Filename |
|---|---|---|
| `COVID-19_supply_chain_crisis` | 2021–2023 global supply chain crisis | `2021–2023_global_supply_chain_crisis.txt` |
| `Global_supply_chain_issues_(2020–present)` | 2021–2023 global supply chain crisis *(same article — deduped)* | *(as above)* |
| `2024_Baltimore_bridge_collapse` | Francis Scott Key Bridge collapse | `Francis_Scott_Key_Bridge_collapse.txt` |
| `2022_Russian_invasion_of_Ukraine_economic_impact` | Economic consequences of the Russo-Ukrainian war (2022–present) *(redirect)* | `Economic_impact_of_the_Russian_invasion_of_Ukraine.txt` |
| `Panama_Canal_drought_2023` | Panama Canal *(covers 2023–24 drought / draft restrictions; no standalone drought article exists)* | `Panama_Canal.txt` |
| `North_Field_(Qatar)` | South Pars/North Dome Gas-Condensate field | `South_Pars_North_Dome_Gas-Condensate_field.txt` |

Per-file SHA-256:

| File | Bytes | SHA-256 |
|---|---|---|
| `2011_Tōhoku_earthquake_and_tsunami.txt` | 70,637 | `17bb70d1ffb00a32dba7c997f01854e5211e7935e9475f0b85132fe1abce3b87` |
| `2020–2023_global_chip_shortage.txt` | 18,211 | `cb3f431a40b47e6191678ca0819fef347c04dcdcf3a4765c3282747681ba3ef8` |
| `2021_Suez_Canal_obstruction.txt` | 25,372 | `dfc8fc477dfb99cc57753a722ed4cacc235799b4491a0001ca077c856ac09e63` |
| `2021–2023_global_supply_chain_crisis.txt` | 7,843 | `c405a971d87d9fb8360371487a1a92f930c7ac3f8df23a7cd65c4d3892651552` |
| `Bab-el-Mandeb.txt` | 11,286 | `7d9da18aa4b2391f27fa36aaafa05fcc956361af369f2e2367720d82f1a13d80` |
| `Baltic_Dry_Index.txt` | 6,980 | `b6bc482d57b085a75cb4e83c609b625bc6727128421e0e1ab5a6a625a3f04ff2` |
| `Bullwhip_effect.txt` | 17,750 | `cd3c6c16720b1f3d90fa006a56d1109f38b76addfcba8ff4ec2f519fede59813` |
| `CHIPS_and_Science_Act.txt` | 55,759 | `8da184ca35268601354f241a49f65ec098d789ddd6643eb32ee85c18fc6b6e37` |
| `Container_ship.txt` | 38,384 | `1bcd5f5d1d25106981338b332d28e8ff83cd5ef8f55c03a7163d31bd7adf945c` |
| `Economic_impact_of_the_Russian_invasion_of_Ukraine.txt` | 53,047 | `bfccd50f778687b296f42214ab2ad3f6069aa425813115ad85e47f51d1c80d6d` |
| `Enterprise_resource_planning.txt` | 26,869 | `f3c7f397b80628f814dcf174180caf333af08ef409567ab7545efcf9dfcad696` |
| `Ever_Given.txt` | 8,611 | `a52b750a352d9ed4cd66b3a199c50f55211ae3f9f2a66983cfddafb777ced0e0` |
| `Foxconn.txt` | 30,535 | `eaecb9d3fb7f62f238c4d28f87fdfa5a7dfc5ebaf7a2ed2aaf923a2c45fef3f7` |
| `Francis_Scott_Key_Bridge_collapse.txt` | 41,639 | `46eee645872adf12fce8558014cee35b44009abc75e054079a0b9d7989a82d82` |
| `Inventory.txt` | 30,431 | `4da6a282626575fbaf6857a8fbc558223f62f04a7d20ee12271ecf4c74845bd0` |
| `Just-in-time_manufacturing.txt` | 24,396 | `07823d014e7498c9004f2c51c064dde8decb20d44aa08ce413a59d280e51ad88` |
| `Logistics.txt` | 39,404 | `dab26d17d852c259173900c5670866589f38286008596e533f7cf0c771b0865b` |
| `Panama_Canal.txt` | 72,728 | `905b9e3107da5f2849f3518dfc25ccbc19ec72a472863eb2309d7c4e2c746145` |
| `Port_of_Los_Angeles.txt` | 12,372 | `f2750065f232a49060d9e963f3c41bfc24e40226192aab3d6ba426c39a90863d` |
| `Port_of_Singapore.txt` | 16,212 | `a04647ee12c19cdafe9fb1e4b1d63ee2f66ea1eeaf085a72f2e7105342cfe496` |
| `Red_Sea_crisis.txt` | 39,356 | `c2ad6539d98bf58d1c9ba9a8271246d827376c70806d766d6ad269587cdd385b` |
| `Samsung_Electronics.txt` | 57,089 | `3edfd059ed7f7b29c5c83974444b36ef6bf1bedd69e5c4712302277ddc8c6bda` |
| `Semiconductor_industry.txt` | 7,332 | `f2c4ac6044bf018ff49d79ab7d0f1602dda9b8628ab58c9eabcac2ab15ae28ce` |
| `South_Pars_North_Dome_Gas-Condensate_field.txt` | 28,354 | `d174fec2388f211cc1841b5982c85a0d14bbb0d5067244718e6103af8ada7d46` |
| `Strait_of_Hormuz.txt` | 36,733 | `6b422d6e640a236cd435cc029390bea48a1dc8e664e4d3c94a60ae82cf4780dc` |
| `Strait_of_Malacca.txt` | 15,183 | `64dd1a930b3432f2c18bcf5757668a5bf0e3289663658f0c5aad5017a25eadfe` |
| `Suez_Canal.txt` | 57,539 | `a86dbe89ab601a6f7b1f251a9ba7fbee317c55795f36613d684a5ce50b9a78d7` |
| `Supply_chain_attack.txt` | 36,064 | `2c9b5654d7e027b47d0c43922f5ba49d882111741a126181e92ae893cc31d75d` |
| `Supply_chain_management.txt` | 68,256 | `f7c86cdacf0865a75b34574b7247cab6a15e514d4352c035ddc0580353ecc93b` |
| `TSMC.txt` | 27,009 | `b916f9195465cd7a8ccfd636cb02ca1bc5bb728d0a4e625d0a38deb0a55bbd15` |
| `Warehouse.txt` | 30,620 | `c0819756214c5d7c48d14ebd927dc855dcb94ba9003c0b2b30366c0f1f1f7a8e` |

---

## 6. World Bank macro indicators — `world_bank_macro/`

- **Source:** World Bank Indicators API v2 (keyless)
  `https://api.worldbank.org/v2/country/CN;DE;IN;JP/indicator/{INDICATOR}?format=json&per_page=500`.
- **Fetch method:** `fetch_all.py::worldbank_macro`. Countries: China, Germany, India, Japan.
- **Date:** 2026-04-16 (API `lastupdated` 2026-04-08). **Count:** 6 indicators.
- **Consumer:** `versions/v3_arcadia/40_granite/r5_rag_beast.py` (world_bank corpus chunks).
- **License:** World Bank Open Data (CC BY 4.0).

| File | Indicator | Bytes | SHA-256 |
|---|---|---|---|
| `wb_GDP_USD.json` | `NY.GDP.MKTP.CD` | 41,137 | `c8dbc1edadb27b883b3242c6a84cd44a77d71e0614dab57ccedf83f10311b6fb` |
| `wb_GDP_growth.json` | `NY.GDP.MKTP.KD.ZG` | 42,560 | `c79639ba60af38aecf7d85e0766c6bfd379412896bf07518511bcc0e8d7163aa` |
| `wb_Inflation_CPI.json` | `FP.CPI.TOTL.ZG` | 44,856 | `adf6bc61e144d349d2a914a669f72af8817c5d34905b5edc3f7ef7e779807b67` |
| `wb_Exports_pct_GDP.json` | `NE.EXP.GNFS.ZS` | 45,635 | `a919438961b524486e6a97ff5169eac16dc7dfd5e16ede15b53f58061ad71268` |
| `wb_Imports_pct_GDP.json` | `NE.IMP.GNFS.ZS` | 45,644 | `5a3212b1b48390cd9e3b51cbad7ce5fa24703446d19ede78d877b8f9ae7789d4` |
| `wb_Container_throughput.json` | `IS.SHP.GOOD.TU` | 46,522 | `126dbe9bbb1d562851d27ecc48140606e31364b93d00d69a3050440f631c3a9a` |

---

## 7. NOAA active storms — `noaa/current_storms.json`

- **Source:** NOAA National Hurricane Center `https://www.nhc.noaa.gov/CurrentStorms.json`.
- **Fetch method:** `fetch_all.py::noaa_storms` (requires a browser-style User-Agent; the SEC
  contact UA gets connection-reset by NOAA's CDN). This is a **live** feed — contents change.
- **Date:** 2026-07-02 snapshot. **Size:** 4,629 bytes.
- **SHA-256 (snapshot):** `e4cb0d469eddd586a2f603e14a47aff7a65d3eabc08b9c83a8a3813e030226ad`
- **Consumer:** none static (freshness/live-source demonstration; the training-time storm data
  used by the RL pipeline is `rl/data/ibtracs_wp.csv`, a separate dataset).
- **License:** U.S. Government work / public domain.

---

## 8. WGI governance — `wgi/wgidataset_with_sourcedata-2025.xlsx`

- See `external_data/wgi/PROVENANCE.md` (kept as-is). Source: World Bank Worldwide Governance
  Indicators 2025 release. Size 10,423,344 bytes;
  SHA-256 `25a2f9eabb90b0092973392c0b31571aa58b691cc5786292e504b52f693e1eb8`.
- **Consumers:** `rl/analysis/trained_models.py`, `rl/data/build_unified_buffer_v2.py`.
- Not auto-fetched (World Bank WGI portal download).

---

## Deleted 2026-07-02 (for DATASET_CARD / receipt de-listing)

Removed as dead (zero code consumers) or fake/mislabeled. These must be **struck from**
`FINAL_SUBMIT/DATASET_CARD.md` and `versions/v3_arcadia/results/R1_VERIFIED.json`:

| Removed path | Reason |
|---|---|
| `openflights/` (5 `.dat`, 3.9 MB) | zero consumers in any tracked code |
| `wto/world_bank_mfn_tariff.zip` | zero consumers |
| `worldbank/pink_sheet_monthly.xlsx` | zero consumers |
| `noaa/sample_track.zip` | zero consumers (static 2024 archive; not "realtime") |
| `un_comtrade/`, `imf_ifs/` (empty dirs) | fetches always failed (need paid subscription) |
| `emdat/.gitkeep` | pointless (dir was wholesale-gitignored) |
| `policy_papers/frbny_supply_chain.pdf` | **fake** — an HTML error page saved as `.pdf` |
| `policy_papers/bis_supply_chain.pdf` | **mislabeled** — actually "BIS WP 1071: Financial access and labor market outcomes" (credit lotteries), unrelated to supply chains |
| `policy_papers/frbsf_supply_chain_pressure.pdf` | **mislabeled** — actually "Dale W. Jorgenson: An Intellectual Biography", unrelated to supply chains |

Also removed: the dead `GLBLALSCMINDX` FRED pull (404) and the stale `extra_data_results.json`
run artifact (its fetcher lived in `versions/v3_arcadia/00_emergence/fetch_extra_data.py`).

## Known upstream bug (out of this scope — owned by the fix-paths agent)

`versions/v3_arcadia/40_granite/r5_rag_beast.py:134` ingests `world_bank_macro/*.json` assuming
a dict, but World Bank API files are top-level **lists** (`[meta, rows]`); the `d.items()` branch
yields `[]` and the `except: pass` at :137 hides it, so World Bank chunks are effectively empty.
Recorded as a blocker.
