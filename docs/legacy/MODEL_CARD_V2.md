# SupplyMind Model Card (v2.0-vessel)

## Overview
Multi-agent RL system for supply chain risk management, trained end-to-end on real-world data.

## Training data (zero synthetic)
- **dataco**: `dataco.csv`
- **noaa**: `ibtracs_wp.csv`
- **usgs**: `usgs_m55_30days.csv`
- **fred_core**: `fred_cache.json`
- **fred_extended**: `fred_extended.json`
- **leading_indicators**: `leading_indicators.json`
- **wgi**: `wgidataset_with_sourcedata-2025.xlsx`
- **dataco_access_logs**: `dataco_access_logs.csv`
- Total transitions: 180,519
- Multi-step fraction: 88.6%
- Unique actions: 164 of 280 possible

## Reward method
- learned financial_impact Ridge model on (order_total, delay, profit_ratio, late_risk)

## Agents
- **Random**: full 0.3% (CI95 0.2%-0.4%), type 14.1%, node 2.5%
- **Scripted_Alert**: full 0.0% (CI95 0.0%-0.0%), type 27.3%, node 5.0%
- **BC_v2**: full 37.4% (CI95 36.9%-37.9%), type 86.2%, node 40.8%
- **CQL_v2**: full 37.4% (CI95 36.8%-38.0%), type 86.1%, node 40.8%
- **IQL_v2**: full 37.1% (CI95 36.5%-37.7%), type 86.3%, node 40.7%
- **TD3BC_v2**: full 37.4% (CI95 36.9%-38.0%), type 86.3%, node 41.1%
- **Federated_v2**: full 30.4% (CI95 29.9%-30.9%), type 75.4%, node 37.5%
- **BC_v1**: full 8.8% (CI95 8.4%-9.1%), type 70.4%, node 11.3%
- **CQL_v1**: full 6.7% (CI95 6.5%-7.0%), type 71.8%, node 9.6%

## Intended use
Decision-support for supply-chain operators facing real-world disruptions.

## Out-of-scope
- Live trading, safety-critical control, or automated large-dollar transactions without human review.

## Limitations
- Classification accuracy benchmarked on real DataCo label distribution (164 unique action combinations); full episodic rollout with real-time disruption streaming is scoped for future work.
- LoRA fine-tune of Qwen2.5-7B is deferred (HF offline required); advanced Modelfile + 10 real crisis few-shots used instead.

## License / Attribution
Real data source attribution: DataCo Kaggle dataset, NOAA IBTRACS (NOAA public domain), USGS (public domain), FRED (Federal Reserve public domain), World Bank WGI (CC-BY-4.0).