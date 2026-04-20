# SupplyMind — Model Card (Real Data v1.0)

## Overview
Multi-agent RL system for supply chain risk management, trained on real-world data.

## Training Data (real, no synthetic rollouts)
- DataCo Kaggle: 180,519 orders with customer/product/market/delivery fields
- NOAA IBTRACS: 4,289 Pacific typhoons (140-year history)
- USGS: 9 earthquakes (live feed)
- FRED: 17,679 daily commodity/FX data points (WTI oil, copper, 5 FX pairs)
- Stratified 70/15/15 by customer_segment x late_delivery_risk

## Environment
OpenEnv-compliant supply chain env, state=408, action=MultiDiscrete([7,40]).
State fusion: NOAA signals state[350:380], USGS state[380:400], FRED state[400:407].

## Agents (all trained on real unified buffer)
- **BC_real_v2**: full_acc=0.340, type_acc=0.865, node_acc=0.356
- **CQL_real_v2**: full_acc=0.349, type_acc=0.867, node_acc=0.370
- **IQL_real_v2**: full_acc=0.000, type_acc=0.136, node_acc=0.026
- **TD3BC_real_v2**: full_acc=0.000, type_acc=0.005, node_acc=0.040
- **Federated_real**: full_acc=0.036, type_acc=0.428, node_acc=0.060

## Analysis Modules (trained, not formulas)
- political_risk: Gradient boosting on WGI 6 dims (R2=0.994, 214 countries)
- dependency_scoring: MLP on DataCo (97.45% acc)
- financial_impact: Ridge on DataCo (R2=0.736)
- confidence: Isotonic calibration (ECE=0.0017)
- safety_stock: Empirical multiplier from DataCo lead-time

## Forecasting
- TFT (pure PyTorch): WTI oil 14-day quantile forecast on real FRED, test MAE $7.83
- MC Dropout on BC: 99.76% acc on low-uncertainty / 55.92% on high-uncertainty quartile

## LLM
- Explainer: Ollama qwen2.5:14b, 4-section structured output, quality-gated, no fallback
- RAG: Ollama nomic-embed-text, 248 real documents (crisis + NOAA + USGS + DataCo)
- supplymind-analyst:v2: Modelfile on qwen2.5:7b-instruct with real crisis few-shots

## Limitations
- DataCo is single-step per order; multi-step trajectories constructed per episode
- Action space 164 unique of 280 possible (reflects real distribution)
