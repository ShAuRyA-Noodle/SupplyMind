# Dataset Card — SupplyMind

**Regenerated 2026-07-15**, aligned with the per-file provenance manifest
[`external_data/PROVENANCE.md`](../external_data/PROVENANCE.md) (source URL + date + size + SHA-256
+ downstream consumer for every dataset). Claim status in [`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md) §E.

`external_data/` data files are gitignored (large, regeneratable); the fetcher
(`external_data/fetch_all.py`) and the provenance manifest are tracked, so a fresh clone can rebuild
the corpus:

```bash
export SEC_CONTACT_EMAIL="you@example.com"   # SEC fair-access UA
python external_data/fetch_all.py            # all sources except EM-DAT (registered download)
```

---

## Static corpora (real, provenance-stamped)

| Dataset | Size / count | What | Consumer | SHA-256 |
|---|---|---|---|---|
| **DataCo Supply Chain** | 180,519 orders, 5 regions | real Latin-American / global order records; late-delivery labels | FedAvg + TabPFN judge ([`fedavg_REAL.json`](../tests/receipts/fedavg_REAL.json)) | in provenance |
| **EM-DAT disasters** | 16,811 rows, 2000–2026, Last-Update **2025-12-20** | international disaster database → 1,500-event crisis library v2 | `scripts/crisis_library/cook_v2.py` | `842180e4…` |
| **SEC EDGAR 10-Ks** | 20 filings, 74.3 MB | public-company risk-factor text | RAG corpus | per-file in provenance |
| **Wikipedia crisis articles** | 31 articles, 1.0 MB | chokepoint / crisis reference text | RAG corpus | per-file in provenance |
| **World Bank macro** | 6 indicators × CN/DE/IN/JP | GDP, growth, CPI, trade, container throughput | RAG corpus (**now ingested — see below**) | per-file in provenance |
| **Policy papers** | 3 PDFs | FRBNY GSCPI SR1017 · BIS Bulletin 48 · FRBSF WP2022-18 (all real, title-verified) | RAG corpus | per-file in provenance |
| **FRED daily Brent** | `DCOILBRENTEU`, 9,922 obs, 1987–2026 | real daily crude price | Brent ensemble + causal counterfactual | `0e88d6dd…` |
| **FRED truck-transport PPI** | `PCU484121484121`, monthly | real freight price index | retained (no current consumer) | `79212528…` |
| **NOAA IBTRACS** | `rl/data/ibtracs_wp.csv` | tropical-cyclone tracks | RL storm features | tracked |
| **WGI governance** | `wgi/…-2025.xlsx`, 9.9 MB | Worldwide Governance Indicators 2025 | RL analysis + buffer build | `25a2f9ea…` |

The RAG **retrieval metrics** (P@1 0.962 etc.) were measured on the **6,483-chunk** corpus
([`ghost_models_eval_REAL.json`](../tests/receipts/ghost_models_eval_REAL.json)). After the
World-Bank ingestion fix (R17) the re-cooked corpus is **6,602 chunks** (wiki_crisis 711, sec_10k
5,790, policy 81, **world_bank 0 → 20**); WB-query P@1 went 0 → 1.0
([`rag_recook_REAL.json`](../tests/receipts/rag_recook_REAL.json)).

---

## Live data sources (real APIs, keys in `.env`)

Keyless: USGS earthquakes, GDELT, World Bank, NOAA NDBC/NHC, NASA EONET. Keyed (free tier): EIA,
NASA FIRMS, GFW (vessel density), NewsAPI, FRED, NOAA CDO. These feed the war-room live layer
(freshness stamps + loud-fail wiring is WP6.4).

| Source | What | Auth | Status |
|---|---|---|---|
| OpenRouter | LLM analyst / judge panel | API key | **key revoked → RERUN-PENDING** (ledger C1); no live-panel number cited as current |
| EIA | crude / fuel spot prices | key | 200 OK in [`api_keys_live_proof.json`](../tests/receipts/api_keys_live_proof.json) |
| NASA FIRMS | active fire incidents (MODIS NRT) | key | 200 OK (same receipt) |
| GFW | vessel density (Hormuz / Red Sea) | Bearer | last probe **503** (key authenticated, service returned no data) — surfaced, not counted OK (ledger E6) |
| NewsAPI / GDELT / USGS / FRED / NOAA | headlines, events, quakes, macro, weather | key/none | live-feed layer, WP6.4 |

---

## Splits (no leakage)

- **TFT WTI**: train 2,254 / val 281 / test 283 windows, chronological.
- **RAG**: 53 evaluation queries against the 6,483-chunk corpus.
- **Conformal**: real harvested transitions split 60/20/20 train/calibration/test (40,000 total),
  shuffled across trajectories before splitting ([`conformal_REAL.json`](../tests/receipts/conformal_REAL.json)).
- **FedAvg / TabPFN**: DataCo 80/20 train/test; leakage columns (actual shipping days, delivery /
  order status) explicitly excluded ([`fedavg_REAL.json`](../tests/receipts/fedavg_REAL.json)).
- **Brent backtest**: per-event walk-forward on real FRED; TabPFN trained leave-one-out so the event
  under test never trains on its own outcome ([`ensemble_brent_REAL.json`](../tests/receipts/ensemble_brent_REAL.json)).

## Corrections applied (ledger §E)

- EM-DAT is a 2000–2026 export (16,811 rows, updated 2025-12-20), **not** a "2024-01 snapshot".
- FRED = real daily Brent + truck PPI; the dead `GLBLALSCMINDX` pull (HTTP 404) was removed.
- `frbny_supply_chain.pdf` was an HTML error page — deleted; replaced by the real GSCPI SR1017.
- De-listed (downloaded but consumed by no code): OpenFlights, WTO tariff zip, World Bank pink
  sheet, NOAA `sample_track.zip`, empty `un_comtrade/` and `imf_ifs/`.
- World-Bank RAG chunks: 0 → 20 after the list-vs-dict ingestion fix (R17).

## License

Live API data subject to each provider's TOS. EM-DAT: non-commercial academic (attribution). SEC
EDGAR / FRED / World Bank / USGS / NOAA: public domain / open data. Wikipedia: CC BY-SA 4.0.
Full per-source licensing in [`external_data/PROVENANCE.md`](../external_data/PROVENANCE.md).
Citations in [`CITATIONS.bib`](CITATIONS.bib).
