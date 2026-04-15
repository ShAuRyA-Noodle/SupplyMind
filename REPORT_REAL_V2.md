# SupplyMind — REAL Data Benchmark v2

Evaluated on 27,083 held-out real DataCo transitions (Phase A unified buffer).
All agents trained on 126,360 stratified real-data transitions with NOAA/USGS/FRED injection.

| Agent | Full Match Acc | Action Type Acc | Target Node Acc |
|---|---:|---:|---:|
| BC_real_v2 | 0.3405 | 0.8646 | 0.3559 |
| CQL_real_v2 | 0.3491 | 0.8667 | 0.3702 |
| IQL_real_v2 | 0.0001 | 0.1361 | 0.0258 |
| TD3BC_real_v2 | 0.0002 | 0.0054 | 0.0399 |
| Federated_real | 0.0363 | 0.4281 | 0.0599 |