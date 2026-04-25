# Dataset Card — SupplyMind data sources

## Live data (real APIs, fetched on demand)
| Source | What | Endpoint | Auth | Receipt |
|--------|------|----------|------|---------|
| OpenRouter | LLM-judge ensemble (12 frontier + 3 local Ollama) | `openrouter.ai/api/v1/chat/completions` | API key | `openrouter_liveness.json` |
| EIA | Crude / fuel spot prices | `api.eia.gov/v2/petroleum/pri/spt` | API key | `api_keys_live_proof.json` |
| NASA FIRMS | Active fire incidents (24h, MODIS NRT) | `firms.modaps.eosdis.nasa.gov/api/area/csv` | MAP key | `chained_live_demo.json` |
| GFW | Vessel positions (Hormuz / Red Sea) | `gateway.api.globalfishingwatch.org/v3/4wings/stats` | Bearer token | `chained_live_demo.json` |
| NewsAPI | Recent geopolitical events | `newsapi.org/v2/everything` | API key | event store |
| GDELT | Global event database | `api.gdeltproject.org/api/v2` | none | event store |
| USGS | Earthquakes (real-time) | `earthquake.usgs.gov/fdsnws/event/1/query` | none | event store |
| FRED | Macro indicators | `api.stlouisfed.org/fred/series/observations` | API key | event store |
| NOAA NDBC | Maritime weather | `www.ndbc.noaa.gov/data/realtime2` | none | NOAA benchmark |
| NASA EONET | Natural events | `eonet.gsfc.nasa.gov/api/v3/events` | none | event store |
| MarineTraffic | AIS vessel data | per-fleet API | partial | fallback to GFW |
| WHO DON | Disease outbreaks | `www.who.int/emergencies/disease-outbreak-news` | RSS | event store |
| SEC EDGAR | Public-company filings | `data.sec.gov/submissions` | none | corpus |
| CISA | Cyber security advisories | `www.cisa.gov/news-events/cybersecurity-advisories` | none | event store |
| OFAC | Sanctions list | `www.treasury.gov/ofac/downloads` | none | event store |
| World Bank | Trade indicators | `api.worldbank.org/v2` | none | event store |

## Static datasets
| Name | Size | Description | Path |
|------|------|-------------|------|
| EMDAT crisis library v2 | ~1500 events | historical disaster impact records | `ShAuRyA_Supplymind/scenarios/` |
| Hand-curated 8 events | 8 events | Iran/Israel/Hormuz/Red-Sea/Suez/Taiwan/Thailand/Tohoku | `ShAuRyA_Supplymind/realtime/crisis_library.py` |
| WTI crude time-series | 2,818 windows | DCOILWTICO from FRED | TFT training |
| Real company nodes | 40 nodes | TSMC/Samsung/Toyota etc with real coords | `data/companies_real.json` |
| Wordle dictionary | 102 words | 5-letter common words (tier-0 baseline) | `ShAuRyA_Phoenix/wordle_env/env.py` |
| Wordle tier 1+ | +200/+150/+80 words | RLVE expansion tiers | `rlve_curriculum.py` |
| RAG corpus | 6,483 chunks | wiki_crisis 564 + sec_10k 5790 + policy 129 | `R5_GRANITE.json` |
| Conformal calibration NLLs | 5,696 (v2) / 16,000 (v3) | nonconformity scores | `conformal_*.json` |

## Splits
- TFT WTI: train 2254 / val 281 / test 283 windows (no leakage; chronological)
- RAG: 53 evaluation queries against 6,483 chunks
- Conformal: 80/20 calib/test (random within-window)
- Wordle REINFORCE: tier-0 (5) → tier-1 (10) → tier-2 (20) curriculum-expanded
- Bootstrap leaderboard: 100 episodes per agent per task

## Data freshness
- Live APIs: queried at request time (no caching beyond 1h TTL in `realtime/store.py`)
- EMDAT crisis library: snapshot 2024-01 (versioned)
- Real company coords: snapshot 2024-Q3

## Data quality / honest caveats
- NewsAPI free tier: 100 req/day cap
- OpenRouter free models: rate-limit 429 on Gemma occasionally
- GFW: query refinement needed for clean 200 (currently `key authenticated`)
- BGE-rerank fails on Windows due to paging file; fallback to FAISS top-K passthrough
- Some user-claimed numbers reconciled exact: TFT 513,534 ✓, TFT 90,602 ✓, NOAA 60.07% ✓, F1 1.0/0.987/0.964 ✓

## License
Live API data subject to each provider's TOS. Static EMDAT library: research use. Wordle dictionary: public domain. SEC EDGAR: public domain.

## Citations
See `CITATIONS.bib`.
