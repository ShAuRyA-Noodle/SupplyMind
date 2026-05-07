# R4 Dangerous V2 — BEAST Mode Results

- **Scenarios**: 26 real Wikipedia crisis articles
- **Judges**: deepseek-r1-local-q4, qwen25-14b-local, mistral-nemo-local
- **Critic**: qwen25-coder-local
- **Extractor (for DeepSeek 2-pass)**: qwen25-14b-local
- **Total runtime**: 15.3 min

## Agreement

- Krippendorff α (ordinal): **0.210**
- Fleiss κ (nominal): **0.01601164483260553**
- Pairwise weighted κ:
  - deepseek-r1-local-q4_vs_qwen25-14b-local: 0.158
  - deepseek-r1-local-q4_vs_mistral-nemo-local: 0.095
  - qwen25-14b-local_vs_mistral-nemo-local: 0.747

## Accuracy vs Ground Truth

| Judge | Correct / Total | Accuracy |
|-------|-----------------|----------|
| deepseek-r1-local-q4 | 8 / 26 | 0.308 |
| qwen25-14b-local | 14 / 26 | 0.538 |
| mistral-nemo-local | 18 / 26 | 0.692 |
| majority_vote | 18 / 26 | 0.692 |

## Calibration (ECE)

- deepseek-r1-local-q4: ECE = **0.1923** (n=26)
- qwen25-14b-local: ECE = **0.3404** (n=26)
- mistral-nemo-local: ECE = **0.2962** (n=26)

## Semantic Agreement (mxbai-embed-large-v1 cosine > 0.65)

- Vulnerabilities: mean Jaccard = **0.376**
- Mitigations: mean Jaccard = **0.578**

## Parse Success + Latency

- deepseek-r1-local-q4: 100% parse OK, 14.6s avg
- qwen25-14b-local: 100% parse OK, 6.3s avg
- mistral-nemo-local: 100% parse OK, 7.7s avg
- Critic (qwen25-coder-local): 100% parse OK

## Escalation Distribution

- C_SUITE_IMMEDIATE: 1
- C_SUITE_REVIEW: 8
- OPS_DIRECTOR_4H: 3
- OPS_DIRECTOR_24H: 5
- FYI_DASHBOARD: 9

## Per-scenario detail

| Scenario | GT | Majority | α | Escal. |
|----------|----|----------|----|--------|
| 2011_Tōhoku_earthquake_and_tsunami | CRITICAL | CRITICAL | 0.00 | C_SUITE_IMMEDIATE |
| 2020–2023_global_chip_shortage | CRITICAL | HIGH | 0.00 | C_SUITE_REVIEW |
| 2021_Suez_Canal_obstruction | HIGH | HIGH | 1.00 | OPS_DIRECTOR_4H |
| Bab-el-Mandeb | HIGH | MEDIUM | 0.00 | OPS_DIRECTOR_24H |
| Baltic_Dry_Index | LOW | LOW | 0.00 | FYI_DASHBOARD |
| Bullwhip_effect | MEDIUM | LOW | 0.00 | FYI_DASHBOARD |
| CHIPS_and_Science_Act | MEDIUM | MEDIUM | 0.00 | OPS_DIRECTOR_24H |
| Container_ship | LOW | LOW | 0.00 | FYI_DASHBOARD |
| Enterprise_resource_planning | LOW | LOW | 0.00 | FYI_DASHBOARD |
| Ever_Given | HIGH | HIGH | 0.00 | C_SUITE_REVIEW |
| Foxconn | MEDIUM | HIGH | 0.00 | C_SUITE_REVIEW |
| Inventory | LOW | LOW | 0.00 | FYI_DASHBOARD |
| Just-in-time_manufacturing | MEDIUM | LOW | 0.00 | FYI_DASHBOARD |
| Logistics | LOW | LOW | 0.00 | FYI_DASHBOARD |
| Port_of_Los_Angeles | MEDIUM | MEDIUM | 0.00 | OPS_DIRECTOR_24H |
| Port_of_Singapore | MEDIUM | HIGH | 0.00 | C_SUITE_REVIEW |
| Red_Sea_crisis | CRITICAL | HIGH | 0.00 | C_SUITE_REVIEW |
| Samsung_Electronics | MEDIUM | MEDIUM | 0.00 | OPS_DIRECTOR_24H |
| Semiconductor_industry | HIGH | MEDIUM | 0.00 | OPS_DIRECTOR_24H |
| Strait_of_Hormuz | HIGH | HIGH | 0.00 | C_SUITE_REVIEW |
| Strait_of_Malacca | HIGH | HIGH | 1.00 | OPS_DIRECTOR_4H |
| Suez_Canal | HIGH | HIGH | 0.00 | C_SUITE_REVIEW |
| Supply_chain_attack | HIGH | HIGH | 0.00 | C_SUITE_REVIEW |
| Supply_chain_management | LOW | LOW | 0.00 | FYI_DASHBOARD |
| TSMC | HIGH | HIGH | 1.00 | OPS_DIRECTOR_4H |
| Warehouse | LOW | LOW | 0.00 | FYI_DASHBOARD |