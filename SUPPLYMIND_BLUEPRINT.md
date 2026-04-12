# SUPPLYMIND: Master Technical Blueprint

## Context
Supply chain disruptions cost $184B in 2023. Existing tools (SAP SCM, Oracle SCM, Resilinc) are reactive dashboards — they tell you *after* things break. SUPPLYMIND provides **72-hour advance warning** by ingesting global signals, modeling company-specific supply chain graphs, and predicting disruptions before they propagate. This blueprint covers the complete system: from signal ingestion to executive alerting, built for a hackathon demo and scalable to Fortune 500 production.

---

## SECTION 1: PROBLEM DEFINITION & VISION

### 1.1 The Problem
- **$184B** in supply chain disruption costs (2023, Business Continuity Institute)
- COVID exposed that 94% of Fortune 1000 companies experienced supply chain disruptions
- Suez Canal blockage (2021): $9.6B/day in trade held up for 6 days
- Taiwan Strait tensions: 92% of advanced semiconductor manufacturing at risk
- Port strikes (US East/Gulf Coast 2024): $5B/day economic impact
- Companies discover disruptions **after** they happen — average reaction time is 7-14 days

### 1.2 Why Incumbents Fail
| Platform | Weakness |
|----------|----------|
| SAP Integrated Business Planning | ERP-centric, no external signal ingestion, no predictive AI |
| Oracle SCM Cloud | Reactive analytics, manual risk assessment, no geospatial intelligence |
| Resilinc | Manual supplier surveys, 30-day update cycles, no real-time prediction |
| Everstream Analytics | Limited to news monitoring, no graph-based propagation modeling |
| Interos | Relationship mapping only, no predictive disruption timing |

### 1.3 The SUPPLYMIND Insight
**72-hour advance warning is worth 100x a post-disruption dashboard.**

With 72 hours, a company can:
- Redirect shipments already in transit to alternate ports
- Activate pre-negotiated backup supplier contracts
- Increase safety stock orders before competitors panic-buy
- Hedge commodity/currency exposure before markets react
- Brief executive leadership before the news cycle hits

### 1.4 Vision Statement
> *"A company running SUPPLYMIND has never had a supply chain surprise."*

---

## SECTION 2: CORE TECHNICAL ARCHITECTURE

### 2.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIGNAL INGESTION LAYER                       │
│  Weather │ Shipping │ Geopolitical │ Labor │ Disasters │ Financial  │
│  NOAA     MarineTraffic  ACLED       NLRB    USGS        Yahoo Fin  │
│  ECMWF    AIS feeds      GDELT       News    NASA FIRMS  Commodity  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Cloud Pub/Sub
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SIGNAL PROCESSING LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐     │
│  │ Normalization │  │ Geocoding &  │  │ Gemini 1.5 Pro        │     │
│  │ & Dedup       │  │ Entity       │  │ Signal Classification │     │
│  │               │  │ Resolution   │  │ & Severity Scoring    │     │
│  └──────────────┘  └──────────────┘  └───────────────────────┘     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ BigQuery Signal Store
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RISK MODELING LAYER                               │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Supply Chain    │  │ Disruption      │  │ Impact Propagation  │  │
│  │ Graph Engine    │  │ Prediction      │  │ Monte Carlo Engine  │  │
│  │ (NetworkX +     │  │ (Vertex AI      │  │ (Revenue at Risk,   │  │
│  │  Neo4j)         │  │  Forecast)      │  │  Inventory Cover)   │  │
│  └────────────────┘  └─────────────────┘  └─────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  RECOMMENDATION & ALERT LAYER                       │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Alt Supplier      │  │ Inventory    │  │ Auto-drafted         │  │
│  │ Identification    │  │ Buffer Calc  │  │ Supplier Emails      │  │
│  │ & ROI Scoring     │  │              │  │ (Gemini)             │  │
│  └──────────────────┘  └──────────────┘  └──────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Firebase Cloud Messaging
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                             │
│  Next.js Dashboard │ Looker Studio │ Mobile (React Native) │ API   │
│  - Supply Map       - C-Suite View   - Field Alerts          - ERP │
│  - Warning Panel    - Board Reports  - Approve Actions       - SAP │
│  - Scenario Sim     - KPIs           - Push Notifications    - API │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Global Signal Network

#### Weather Signals
| Source | API | Data | Update Freq | Free Tier |
|--------|-----|------|-------------|-----------|
| NOAA National Weather Service | `api.weather.gov` | US forecasts, severe weather alerts | 1 hr | Unlimited, no key |
| NOAA Global Forecast System | `nomads.ncep.noaa.gov` | Global weather grids, 16-day forecast | 6 hr | Free, GRIB2 format |
| Open-Meteo | `api.open-meteo.com` | Global forecasts, historical weather | 1 hr | 10K req/day free |
| ECMWF (Copernicus CDS) | `cds.climate.copernicus.eu` | European/global reanalysis, seasonal forecasts | 6 hr | Free with registration |
| NOAA National Hurricane Center | `nhc.noaa.gov/gis/` | Tropical cyclone tracks, cones, wind probabilities | 6 hr | Free GIS feeds |
| NOAA River Forecast | `water.weather.gov/ahps/` | River gauge levels, flood stage forecasts | 1 hr | Free |

**Implementation**: Cloud Scheduler triggers Cloud Run jobs every hour. Each job fetches weather data, geocodes affected regions, and publishes structured events to Pub/Sub topic `signals-weather`.

#### Shipping & Maritime Signals
| Source | API | Data | Update Freq | Pricing |
|--------|-----|------|-------------|---------|
| MarineTraffic | `services.marinetraffic.com/api` | AIS vessel positions, port calls, ETAs | 5 min | From $100/mo (PS01-PS07 endpoints) |
| UN/LOCODE | Static dataset | Port codes, coordinates for 100K+ locations | Static | Free |
| Suez Canal Authority | Web scraping + news | Transit counts, delays, closures | Daily | Free (scraping) |
| Panama Canal Authority | `pancanal.com` | Draft restrictions, booking slots, wait times | Daily | Free (public data) |
| Freightos Baltic Index | `fbx.freightos.com` | Container shipping rates by lane | Weekly | Free index, API paid |
| Port of Los Angeles | `portoptimizer.com` API | Container dwell times, vessel queues | Daily | Free with registration |

**Implementation**: MarineTraffic webhook pushes vessel events. Cloud Run job monitors canal status pages. All events → Pub/Sub topic `signals-shipping`.

#### Geopolitical Signals
| Source | API | Data | Update Freq | Pricing |
|--------|-----|------|-------------|---------|
| ACLED | `acleddata.com/api` | Conflict events (battles, protests, violence) in 200+ countries | Daily | Free for research/non-commercial |
| GDELT Project | `api.gdeltproject.org` | Global news events, tone, themes, locations | 15 min | Free, unlimited |
| OFAC SDN List | `sanctionslist.ofac.treas.gov` | US sanctions (entities, vessels, countries) | As updated | Free XML/CSV |
| EU Consolidated Sanctions | `data.europa.eu` | EU sanctions list | As updated | Free |
| US State Dept Travel Advisories | `travel.state.gov` API | Country risk levels (1-4), specific advisories | As updated | Free JSON |
| SIPRI | Static datasets | Arms transfers, military expenditure by country | Annual | Free |

**Implementation**: GDELT is the backbone — 15-min updates, global coverage, geocoded events. ACLED supplements with higher-quality conflict data. Cloud Run job polls every 15 min, Gemini classifies event severity → Pub/Sub topic `signals-geopolitical`.

#### Labor Signals
| Source | Data | Method |
|--------|------|--------|
| NLRB (US) | Union election filings, unfair labor practice charges | API scraping |
| GDELT | Strike/protest news mentions by location | API |
| ILO (International Labour Organization) | Global labor statistics | Static datasets |
| Reddit / Social Media | Worker sentiment in supplier regions | Gemini-powered analysis |
| BLS Strike Reports (US) | Work stoppages involving 1000+ workers | Monthly data |

#### Natural Disaster Signals
| Source | API | Data | Update Freq | Pricing |
|--------|-----|------|-------------|---------|
| USGS Earthquake | `earthquake.usgs.gov/fdsnws/event/1/` | Global earthquakes, magnitude, depth | Real-time | Free, no key |
| NASA FIRMS | `firms.modaps.eosdis.nasa.gov` | Active fire/hotspot data (satellite) | 3 hr | Free with NASA Earthdata login |
| NOAA Tsunami Warning | `tsunami.gov` | Tsunami watches, warnings, advisories | Real-time | Free |
| Smithsonian GVP | `volcano.si.edu` | Volcanic activity reports | Weekly | Free |
| NOAA Storm Prediction Center | `spc.noaa.gov` | Tornado/severe storm watches | Real-time | Free |
| Copernicus EMS | `emergency.copernicus.eu` | Flood, fire, earthquake rapid mapping | As activated | Free |

#### Financial Signals
| Source | API | Data | Update Freq | Pricing |
|--------|-----|------|-------------|---------|
| Yahoo Finance | `yfinance` Python lib | Currency rates, commodity prices, stock data | 15 min delay | Free |
| FRED (Federal Reserve) | `api.stlouisfed.org` | Economic indicators, PMI, industrial production | Monthly | Free, API key |
| World Bank | `api.worldbank.org` | GDP, trade flows, infrastructure indices | Quarterly | Free |
| Exchange Rates API | `exchangeratesapi.io` | 170+ currency pairs | Daily | Free tier: 250 req/mo |
| Trading Economics | `tradingeconomics.com` | PMI, industrial production by country | As released | Paid ($49/mo+) |

#### News & Sentiment (Multi-language)
| Source | API | Capability | Pricing |
|--------|-----|------------|---------|
| GDELT | `api.gdeltproject.org/api/v2/doc/doc` | 65 languages, sentiment, themes, geocoded | Free |
| Google Cloud Translation | `translate.googleapis.com` | 130+ languages, real-time translation | $20/M characters |
| NewsAPI | `newsapi.org` | 150K+ sources, keyword search, top headlines | Free: 100 req/day |
| Event Registry | `eventregistry.org` | 150K+ sources, concept extraction, clustering | Free: 2K req/mo |

### 2.3 Company-Specific Supply Chain Model

#### Data Input Methods
1. **CSV/Excel Upload**: Template with columns: `supplier_name, tier, component, address, lat, lng, lead_time_days, annual_spend, alt_supplier_available`
2. **ERP Integration** (production): SAP RFC/BAPI connectors, Oracle REST APIs
3. **Manual Entry UI**: Step-by-step wizard with Google Maps autocomplete for addresses
4. **Bulk Geocoding**: Google Maps Geocoding API to convert addresses → lat/lng

#### Supply Chain Graph Schema
```
Node Types:
  - SUPPLIER: { id, name, tier (1|2|3), lat, lng, country, region,
                components[], annual_spend, lead_time_days,
                alt_suppliers[], single_source: bool, risk_score }
  - WAREHOUSE: { id, name, lat, lng, inventory_days_cover, capacity }
  - PORT: { id, name, lat, lng, type (sea|air|rail),
            avg_dwell_time_hours, congestion_score }
  - FACTORY: { id, name, lat, lng, production_capacity, utilization_pct }
  - CUSTOMER: { id, name, lat, lng, revenue_contribution }

Edge Types:
  - SUPPLIES: supplier → factory { component, qty, lead_time_days,
              transport_mode, route_ports[], cost_per_unit }
  - SHIPS_VIA: any → port { transit_time_days, carrier, frequency }
  - STORES_AT: factory → warehouse { component, qty, reorder_point }
  - DELIVERS_TO: warehouse → customer { lead_time_days, sla_days }
```

#### Dependency Scoring Algorithm
```python
def dependency_score(supplier_node):
    """Score 0-100 indicating criticality of this supplier."""
    scores = {
        'single_source_penalty': 40 if supplier_node.single_source else 0,
        'revenue_exposure': min(30, (supplier_node.downstream_revenue / total_revenue) * 100),
        'lead_time_risk': min(15, supplier_node.lead_time_days / 7 * 5),
        'geographic_concentration': min(15, country_concentration_score(supplier_node.country))
    }
    return sum(scores.values())
```

### 2.4 Impact Propagation Model

#### Graph-Based Propagation Engine
```python
import networkx as nx
import numpy as np

class SupplyChainGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    def propagate_disruption(self, disrupted_node_id, severity=1.0, duration_days=7):
        """
        BFS propagation from disrupted node through supply chain graph.
        Returns dict of { node_id: { delay_days, revenue_at_risk, confidence } }
        """
        impacts = {}
        queue = [(disrupted_node_id, 0, severity)]

        while queue:
            node_id, accumulated_delay, current_severity = queue.pop(0)
            node = self.G.nodes[node_id]

            # Severity decays through tiers but delay accumulates
            for successor in self.G.successors(node_id):
                edge = self.G.edges[node_id, successor]
                propagation_delay = edge['lead_time_days']
                total_delay = accumulated_delay + propagation_delay

                # Check if successor has inventory buffer
                buffer_days = self.G.nodes[successor].get('inventory_days_cover', 0)
                effective_delay = max(0, total_delay - buffer_days)

                if effective_delay > 0:
                    downstream_severity = current_severity * edge.get('dependency_weight', 0.8)
                    impacts[successor] = {
                        'delay_days': effective_delay,
                        'severity': downstream_severity,
                        'revenue_at_risk': self._calc_revenue_at_risk(successor, effective_delay),
                        'time_to_impact_days': total_delay
                    }
                    queue.append((successor, total_delay, downstream_severity))

        return impacts
```

#### Monte Carlo Disruption Simulation
```python
def monte_carlo_simulation(graph, disruption_scenario, n_simulations=10000):
    """
    Run N simulations with randomized parameters to estimate
    disruption impact distribution.
    """
    results = []
    for _ in range(n_simulations):
        # Randomize disruption parameters
        severity = np.random.beta(
            disruption_scenario['severity_alpha'],
            disruption_scenario['severity_beta']
        )
        duration = np.random.lognormal(
            np.log(disruption_scenario['expected_duration_days']),
            disruption_scenario['duration_variance']
        )

        # Run propagation
        impact = graph.propagate_disruption(
            disruption_scenario['node_id'],
            severity=severity,
            duration_days=duration
        )

        total_revenue_at_risk = sum(i['revenue_at_risk'] for i in impact.values())
        max_delay = max((i['delay_days'] for i in impact.values()), default=0)
        results.append({
            'total_revenue_at_risk': total_revenue_at_risk,
            'max_delay_days': max_delay,
            'nodes_affected': len(impact)
        })

    return {
        'p50_revenue_at_risk': np.percentile([r['total_revenue_at_risk'] for r in results], 50),
        'p95_revenue_at_risk': np.percentile([r['total_revenue_at_risk'] for r in results], 95),
        'p99_revenue_at_risk': np.percentile([r['total_revenue_at_risk'] for r in results], 99),
        'p50_max_delay': np.percentile([r['max_delay_days'] for r in results], 50),
        'p95_max_delay': np.percentile([r['max_delay_days'] for r in results], 95),
        'avg_nodes_affected': np.mean([r['nodes_affected'] for r in results])
    }
```

#### Days of Inventory Cover Calculator
```python
def inventory_cover(node, disrupted_suppliers):
    """Calculate how many days a node can operate without resupply."""
    current_inventory = node['inventory_units']
    daily_consumption = node['annual_consumption'] / 365

    # Check which inputs are disrupted
    disrupted_fraction = sum(
        edge['supply_fraction']
        for edge in node.inbound_edges
        if edge.source in disrupted_suppliers
    )

    if disrupted_fraction == 0:
        return float('inf')  # No impact

    # Adjusted consumption rate (can only use non-disrupted supply)
    effective_daily_supply = daily_consumption * (1 - disrupted_fraction)
    net_daily_drain = daily_consumption - effective_daily_supply

    if net_daily_drain <= 0:
        return float('inf')  # Remaining suppliers cover demand

    return current_inventory / net_daily_drain
```

### 2.5 The 72-Hour Prediction Architecture

#### Leading Indicator Library

| Disruption Type | Leading Indicators (24-72hr) | Data Source |
|----------------|------------------------------|-------------|
| **Tropical Cyclone** | Storm formation, track forecast cone, wind speed projections | NOAA NHC |
| **Port Congestion** | Vessel queue length increase >20%, avg dwell time spike | MarineTraffic, port APIs |
| **Labor Strike** | Strike vote announcement, union statement release, social media surge | GDELT, NLRB, Reddit |
| **Earthquake** | (Not predictable — immediate detection + aftershock modeling) | USGS real-time |
| **Flooding** | River gauge levels exceeding flood stage, rainfall forecast >200mm | NOAA AHPS |
| **Geopolitical Escalation** | Military movement reports, diplomatic recall, GDELT conflict tone spike | ACLED, GDELT, news |
| **Sanctions** | Legislative draft leaks, diplomatic statements, pre-announcement news | GDELT, government feeds |
| **Supplier Financial Distress** | Credit rating downgrade, payment delay reports, stock price drop >10% | Financial APIs |
| **Wildfire** | NASA FIRMS hotspot density increase, wind forecast + low humidity | NASA FIRMS, NOAA |
| **Volcanic Eruption** | Seismic swarm detection, SO2 emission spike, aviation color code change | Smithsonian GVP, VAAC |
| **Canal Disruption** | Vessel grounding report, military activity near chokepoint, draft restriction announcement | MarineTraffic, news |
| **Cyber Attack** | (Reactive — detect via supplier communication blackout) | News, direct signals |
| **Pandemic Outbreak** | WHO disease outbreak news, ProMED alerts, abnormal absenteeism signals | WHO DON, ProMED |
| **Export Control** | Government policy announcements, trade negotiation breakdown signals | Government feeds, GDELT |
| **Raw Material Shortage** | Commodity price spike >2 std dev, mine/refinery incident reports | Financial APIs, news |

#### Prediction Pipeline (Vertex AI Forecast)

```python
# Time-series prediction for disruption probability
# Input features per supplier region (daily granularity):
features = {
    'weather_severity_index': float,       # 0-1, composite weather risk
    'gdelt_conflict_tone': float,          # Average tone of conflict articles
    'gdelt_event_count': int,              # Number of conflict events
    'vessel_queue_length': int,            # Vessels waiting at nearest port
    'port_dwell_time_hours': float,        # Average container dwell time
    'currency_volatility_30d': float,      # 30-day FX volatility
    'commodity_price_zscore': float,        # Std devs from 90-day mean
    'acled_fatalities_7d': int,            # Conflict fatalities in region
    'fire_hotspot_count_50km': int,        # Active fires within 50km
    'river_gauge_pct_flood_stage': float,  # % of flood stage level
    'social_sentiment_score': float,       # Worker/labor sentiment
    'travel_advisory_level': int,          # 1-4 US State Dept level
}

# Target: binary disruption occurred within 72 hours (1/0)
# Model: Vertex AI AutoML Tabular or Vertex AI Forecast time-series
# Training data: historical disruptions mapped to pre-disruption feature values
```

#### Confidence Scoring
```python
def disruption_confidence(prediction_probability, indicator_count, historical_accuracy):
    """
    Composite confidence score for a 72-hour disruption prediction.

    prediction_probability: ML model output (0-1)
    indicator_count: number of independent signals corroborating
    historical_accuracy: model accuracy for this disruption type (0-1)
    """
    # Require multiple independent signals to boost confidence
    corroboration_bonus = min(0.2, indicator_count * 0.05)

    raw_confidence = (
        prediction_probability * 0.5 +
        historical_accuracy * 0.3 +
        corroboration_bonus * 1.0
    )

    return min(1.0, raw_confidence)

# Alert thresholds:
# confidence >= 0.8 → RED ALERT (immediate notification, auto-draft actions)
# confidence >= 0.5 → AMBER WARNING (dashboard highlight, daily digest)
# confidence >= 0.3 → YELLOW WATCH (monitor, weekly report)
```

### 2.6 The Recommendation Engine

#### Alternative Supplier Identification
```python
def find_alternative_suppliers(disrupted_supplier, component, supply_graph):
    """
    Identify and score alternative suppliers for a disrupted component.
    """
    alternatives = []

    # 1. Check pre-registered alternates
    for alt in disrupted_supplier.registered_alternatives:
        alt.score = score_alternative(alt, disrupted_supplier)
        alternatives.append(alt)

    # 2. Search supplier database by component capability
    db_matches = supplier_db.search(
        component=component,
        exclude_country=disrupted_supplier.country,  # Geographic diversification
        min_quality_rating=disrupted_supplier.quality_rating * 0.9
    )
    alternatives.extend(db_matches)

    # Score each alternative
    for alt in alternatives:
        alt.switch_score = {
            'geographic_risk_reduction': calc_geo_diversification(alt),
            'lead_time_delta': alt.lead_time - disrupted_supplier.lead_time,
            'cost_delta_pct': (alt.unit_cost - disrupted_supplier.unit_cost) / disrupted_supplier.unit_cost,
            'quality_match': alt.quality_rating / disrupted_supplier.quality_rating,
            'capacity_available_pct': alt.available_capacity / required_volume,
            'activation_time_days': alt.estimated_qualification_days
        }

    return sorted(alternatives, key=lambda a: a.composite_score, reverse=True)
```

#### Inventory Buffer Recommendations
```python
def recommend_buffer(component, supply_chain_graph, risk_tolerance='moderate'):
    """
    Calculate optimal safety stock for a component given its supply chain risk.
    """
    # Get all supply paths for this component
    paths = supply_chain_graph.get_supply_paths(component)

    # Calculate risk-adjusted lead time
    for path in paths:
        path.risk_adjusted_lead_time = path.base_lead_time * (
            1 + path.disruption_probability * path.avg_disruption_duration / path.base_lead_time
        )

    risk_multipliers = {'conservative': 2.5, 'moderate': 1.5, 'aggressive': 1.0}
    multiplier = risk_multipliers[risk_tolerance]

    max_risk_lead_time = max(p.risk_adjusted_lead_time for p in paths)
    daily_demand = component.annual_demand / 365

    recommended_buffer_units = max_risk_lead_time * daily_demand * multiplier
    buffer_cost = recommended_buffer_units * component.unit_cost

    return {
        'recommended_buffer_units': int(recommended_buffer_units),
        'buffer_cost': buffer_cost,
        'covers_disruption_days': recommended_buffer_units / daily_demand,
        'current_buffer_units': component.current_inventory,
        'gap_units': max(0, recommended_buffer_units - component.current_inventory)
    }
```

#### Auto-Drafted Supplier Emails (Gemini)
```python
def draft_supplier_email(alert, supplier, recommended_actions):
    """Use Gemini to draft contextual supplier communication."""
    prompt = f"""
    Draft a professional email to {supplier.contact_name} at {supplier.company_name}.

    Context:
    - We have detected a potential {alert.disruption_type} affecting
      {alert.affected_region} within the next {alert.time_to_impact_hours} hours.
    - This may impact delivery of {', '.join(alert.affected_components)}.
    - Current order: PO#{supplier.active_po_numbers}
    - Our recommended actions: {recommended_actions}

    The email should:
    1. Inform the supplier of the potential disruption
    2. Request status update on current orders
    3. Ask about their contingency plans
    4. Propose specific actions (expedite, reroute, partial shipment)
    5. Set a response deadline (24 hours)

    Tone: Professional, urgent but not panicked. Data-driven.
    """
    return gemini_model.generate_content(prompt).text
```

### 2.7 Google Earth Engine — Visual Disruption Evidence

```python
import ee

def get_disruption_satellite_evidence(lat, lng, disruption_type, date_range):
    """
    Fetch before/after satellite imagery showing disruption evidence.
    Uses Sentinel-2 (10m resolution, 5-day revisit).
    """
    ee.Initialize()

    point = ee.Geometry.Point(lng, lat)
    region = point.buffer(50000)  # 50km radius

    if disruption_type == 'flood':
        # Use Sentinel-1 SAR for flood detection (works through clouds)
        before = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterDate(date_range['before_start'], date_range['before_end'])
            .filterBounds(region)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .select('VV')
            .mean())

        after = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterDate(date_range['after_start'], date_range['after_end'])
            .filterBounds(region)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .select('VV')
            .mean())

        # Flood detection: VV backscatter decrease indicates water
        flood_map = after.lt(before.subtract(3))  # 3 dB threshold

        return {
            'before_image_url': before.getThumbURL({
                'region': region, 'dimensions': '800x600',
                'min': -25, 'max': 0
            }),
            'after_image_url': after.getThumbURL({
                'region': region, 'dimensions': '800x600',
                'min': -25, 'max': 0
            }),
            'flood_extent_url': flood_map.getThumbURL({
                'region': region, 'dimensions': '800x600',
                'palette': ['black', 'blue']
            }),
            'flood_area_km2': flood_map.multiply(ee.Image.pixelArea())
                .reduceRegion(ee.Reducer.sum(), region, 100)
                .getInfo()['VV'] / 1e6
        }

    elif disruption_type == 'wildfire':
        # Use MODIS/VIIRS active fire data
        fires = (ee.ImageCollection('FIRMS')
            .filterDate(date_range['after_start'], date_range['after_end'])
            .filterBounds(region))

        # Sentinel-2 true color before/after
        s2_before = get_clear_sentinel2(region, date_range['before_start'], date_range['before_end'])
        s2_after = get_clear_sentinel2(region, date_range['after_start'], date_range['after_end'])

        return {
            'before_image_url': s2_before.getThumbURL({...}),
            'after_image_url': s2_after.getThumbURL({...}),
            'active_fire_count': fires.size().getInfo()
        }
```

---

## SECTION 3: GOOGLE API INTEGRATION PLAN

### 3.1 Service-by-Service Configuration

| Google Service | Purpose in SUPPLYMIND | Tier/Pricing | Configuration |
|---|---|---|---|
| **Gemini 1.5 Pro** | Signal classification, severity scoring, email drafting, scenario narration | $3.50/M input tokens, $10.50/M output | `gemini-1.5-pro-latest`, temp=0.2 for classification, 0.7 for emails |
| **Vertex AI Forecast** | Time-to-disruption prediction | $0.30/node-hour training | AutoML Tabular, 72hr forecast horizon, daily granularity |
| **BigQuery** | Signal data lake, supply chain data warehouse | First 1TB query/mo free | Partitioned by `signal_date`, clustered by `signal_type, region` |
| **Cloud Pub/Sub** | Real-time signal ingestion bus | First 10GB/mo free | Topics: `signals-weather`, `signals-shipping`, `signals-geopolitical`, `signals-disaster`, `signals-financial`, `signals-labor` |
| **Cloud Run** | Signal ingestion jobs, prediction pipeline, API backend | First 2M requests/mo free | Min instances: 1 (API), 0 (batch jobs). Max: 10 |
| **Cloud Scheduler** | Cron triggers for signal collection | 3 free jobs/mo, $0.10/job/mo after | Jobs every 15min (GDELT), 1hr (weather), 6hr (shipping), daily (financial) |
| **Google Earth Engine** | Satellite imagery for disruption evidence | Free for research/non-commercial | Service account auth, Sentinel-1/2, MODIS collections |
| **Google Maps Platform** | Supply chain visualization, geocoding | $200/mo free credit | Maps JavaScript API, Geocoding API, Directions API |
| **Cloud Translation** | Translate local-language news about supplier regions | $20/M chars | Advanced (NMT) for 40+ supplier-region languages |
| **Firebase Cloud Messaging** | Push notifications for mobile alerts | Free | Topics per company + per-user tokens |
| **Firestore** | Real-time alert state, user preferences, action approvals | Free tier: 1GB stored | Collections: `alerts`, `companies`, `users`, `actions` |
| **Looker Studio** | Executive dashboards, board reports | Free | Connected to BigQuery, embedded via iframe |
| **Secret Manager** | API keys, credentials | First 6 versions free | Store all external API keys |
| **Cloud Storage** | Satellite images, report PDFs, CSV uploads | $0.020/GB/mo | Buckets: `supplymind-uploads`, `supplymind-evidence`, `supplymind-reports` |

### 3.2 Gemini Integration Details

```python
import google.generativeai as genai

# Signal Classification Prompt
CLASSIFY_SIGNAL_PROMPT = """
You are a supply chain risk analyst. Classify this signal event.

Event: {event_text}
Source: {source}
Location: {location}
Date: {date}

Return JSON:
{
  "disruption_type": "one of: cyclone|flood|earthquake|wildfire|volcano|strike|protest|
                      sanctions|trade_policy|port_congestion|canal_disruption|
                      supplier_financial|cyber|pandemic|material_shortage|none",
  "severity": 0.0-1.0,
  "affected_radius_km": number,
  "estimated_duration_days": number,
  "confidence": 0.0-1.0,
  "supply_chain_relevance": "high|medium|low|none",
  "summary": "one sentence summary"
}
"""

# Scenario Narration Prompt (for demo)
NARRATE_SCENARIO_PROMPT = """
You are briefing a Fortune 500 supply chain VP.
Given this disruption alert, provide a 3-paragraph executive briefing:

Alert: {alert_json}
Company Supply Chain: {supply_chain_summary}
Historical Precedents: {historical_data}

Paragraph 1: What is happening and where
Paragraph 2: How it impacts THIS COMPANY's supply chain specifically
Paragraph 3: Recommended immediate actions with timeline
"""
```

---

## SECTION 4: RISK PROPAGATION MODEL IN DETAIL

### 4.1 Supply Chain Graph Data Model

**Storage**: Neo4j for graph traversal + BigQuery for analytics

```cypher
// Neo4j Schema
CREATE CONSTRAINT FOR (s:Supplier) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT FOR (c:Component) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT FOR (f:Factory) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT FOR (w:Warehouse) REQUIRE w.id IS UNIQUE;
CREATE CONSTRAINT FOR (p:Port) REQUIRE p.id IS UNIQUE;

// Example: Multi-tier supply chain
CREATE (tsmc:Supplier {id: 'SUP001', name: 'TSMC', tier: 1,
        lat: 24.7867, lng: 120.9964, country: 'TW',
        lead_time_days: 90, annual_spend: 500000000,
        single_source: true, risk_score: 85})

CREATE (asml:Supplier {id: 'SUP002', name: 'ASML', tier: 2,
        lat: 51.4833, lng: 5.4833, country: 'NL',
        lead_time_days: 180, annual_spend: 200000000,
        single_source: true, risk_score: 78})

CREATE (kaohsiung:Port {id: 'PORT001', name: 'Kaohsiung',
        lat: 22.6163, lng: 120.3055, type: 'sea',
        avg_dwell_time_hours: 48, congestion_score: 0.3})

// Relationships
CREATE (asml)-[:SUPPLIES {component: 'EUV Lithography',
        lead_time_days: 180, transport_mode: 'sea'}]->(tsmc)
CREATE (tsmc)-[:SHIPS_VIA {transit_time_days: 3}]->(kaohsiung)
```

### 4.2 Disruption Taxonomy — 15 Types

| # | Type | Avg Frequency | Avg Duration | Severity Range | Historical Example |
|---|------|---------------|-------------|----------------|-------------------|
| 1 | Tropical Cyclone | 85/yr globally | 3-14 days | 0.3-0.9 | Typhoon Hagibis 2019: $15B damage Japan |
| 2 | Earthquake | 15 major/yr | 7-90 days | 0.2-1.0 | Tohoku 2011: 6-month auto supply disruption |
| 3 | Flooding | 200+/yr | 7-30 days | 0.2-0.8 | Thailand 2011: 25% global HDD production halted |
| 4 | Wildfire | 50+/yr | 7-60 days | 0.1-0.6 | California 2020: semiconductor fab evacuations |
| 5 | Volcanic Eruption | 50-70/yr | 1-180 days | 0.1-0.9 | Eyjafjallajokull 2010: 6-day European airspace closure |
| 6 | Port Congestion | Ongoing | 7-90 days | 0.2-0.7 | LA/LB 2021: 100+ vessels at anchor, 2-week delays |
| 7 | Canal Disruption | 1-2/yr | 1-14 days | 0.3-0.8 | Suez 2021: 6 days, $9.6B/day trade blocked |
| 8 | Labor Strike | 50+/yr globally | 1-60 days | 0.2-0.7 | US rail 2022: $2B/day economic impact threat |
| 9 | Geopolitical Conflict | Ongoing | 30-365+ days | 0.3-1.0 | Russia-Ukraine: global grain/energy disruption |
| 10 | Sanctions/Trade Policy | 10-20/yr | 90-365+ days | 0.3-0.9 | US-China chip export controls: $50B+ restructuring |
| 11 | Pandemic/Health Crisis | Rare (1-2/decade) | 90-730 days | 0.5-1.0 | COVID-19: global shutdown, 2-year disruption |
| 12 | Cyber Attack | 1000+/yr (supply chain) | 3-30 days | 0.2-0.8 | NotPetya 2017: Maersk $300M, global shipping chaos |
| 13 | Supplier Financial Distress | Ongoing | 30-180 days | 0.3-0.7 | Hanjin Shipping 2016 bankruptcy: cargo stranded globally |
| 14 | Raw Material Shortage | 5-10/yr | 30-365 days | 0.2-0.8 | Semiconductor shortage 2020-23: $500B auto revenue lost |
| 15 | Infrastructure Failure | 10+/yr | 1-30 days | 0.1-0.5 | Texas freeze 2021: petrochemical plant shutdowns |

### 4.3 Propagation Delay Model

```
Tier 3 Disruption → Tier 2 Impact → Tier 1 Impact → Our Production Impact
     Day 0           Day 15-30        Day 45-90          Day 60-120

Delay factors:
- Base lead time between tiers
- Inventory buffer at each tier (absorbs delay)
- Order backlog (amplifies delay — bullwhip effect)
- Transport mode (air can compress by 80%, at 10x cost)
- Qualification time for alternatives (30-180 days for new suppliers)
```

### 4.4 Single-Point-of-Failure Detector

```python
def detect_single_points_of_failure(supply_graph):
    """
    Identify nodes whose removal disconnects supply paths.
    Uses graph articulation point analysis.
    """
    spofs = []

    for component in supply_graph.get_all_components():
        supply_paths = supply_graph.get_all_paths(
            source_type='SUPPLIER',
            target_type='FACTORY',
            component=component
        )

        # Find nodes that appear in ALL paths (no alternative route exists)
        if len(supply_paths) == 0:
            continue

        common_nodes = set(supply_paths[0])
        for path in supply_paths[1:]:
            common_nodes &= set(path)

        for node in common_nodes:
            if node.type != 'FACTORY':  # Factory itself doesn't count
                spofs.append({
                    'node': node,
                    'component': component,
                    'paths_affected': len(supply_paths),
                    'revenue_at_risk': sum(
                        path[-1].revenue_contribution
                        for path in supply_paths
                    ),
                    'mitigation': 'CRITICAL — qualify alternative supplier'
                })

    return sorted(spofs, key=lambda s: s['revenue_at_risk'], reverse=True)
```

### 4.5 Financial Impact Model

```python
def calculate_ebitda_impact(disruption, company_financials):
    """
    Estimate EBITDA impact per day of disruption at each affected node.
    """
    affected_revenue_per_day = disruption['revenue_at_risk'] / 365

    # Direct costs
    lost_margin = affected_revenue_per_day * company_financials['gross_margin']
    expedite_premium = disruption['expedite_cost_multiplier'] * affected_revenue_per_day * 0.3

    # Indirect costs
    penalty_fees = sum(
        customer['sla_penalty_per_day']
        for customer in disruption['affected_customers']
        if disruption['delay_days'] > customer['sla_buffer_days']
    )
    reputation_cost = affected_revenue_per_day * 0.05  # Conservative estimate

    return {
        'daily_ebitda_impact': lost_margin + expedite_premium + penalty_fees + reputation_cost,
        'total_impact_estimate': (lost_margin + expedite_premium + penalty_fees + reputation_cost)
                                 * disruption['expected_duration_days'],
        'breakdown': {
            'lost_margin': lost_margin,
            'expedite_costs': expedite_premium,
            'sla_penalties': penalty_fees,
            'reputation': reputation_cost
        }
    }
```

---

## SECTION 5: GEOPOLITICAL INTELLIGENCE LAYER

### 5.1 Sanctions Risk Monitoring

```python
import requests
import xml.etree.ElementTree as ET

class SanctionsMonitor:
    OFAC_SDN_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.XML"
    EU_SANCTIONS_URL = "https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content"

    def check_supplier(self, supplier_name, supplier_country):
        """Screen supplier against OFAC, EU, UN sanctions lists."""
        results = {
            'ofac_match': self._check_ofac(supplier_name),
            'eu_match': self._check_eu(supplier_name),
            'country_sanctions': self._check_country_programs(supplier_country),
            'sectoral_sanctions': self._check_sectoral(supplier_name, supplier_country)
        }
        results['risk_level'] = 'HIGH' if any(results.values()) else 'LOW'
        return results

    def monitor_changes(self):
        """Daily job: detect new additions to sanctions lists."""
        current_sdn = self._fetch_ofac_sdn()
        previous_sdn = self._load_previous_sdn()

        new_entries = current_sdn - previous_sdn
        removed_entries = previous_sdn - current_sdn

        # Cross-reference new sanctions against all registered suppliers
        for entry in new_entries:
            matches = self._fuzzy_match_suppliers(entry)
            if matches:
                self._create_alert('sanctions_new', entry, matches)
```

### 5.2 Taiwan Strait Scenario Model

```python
TAIWAN_SCENARIO = {
    'name': 'Taiwan Strait Closure',
    'affected_components': ['Advanced Semiconductors (< 7nm)', 'DRAM', 'NAND Flash',
                           'Display Panels', 'PCBs', 'Passive Components'],
    'affected_suppliers': {
        'TSMC': {'global_share': 0.54, 'advanced_node_share': 0.92},
        'UMC': {'global_share': 0.07},
        'MediaTek': {'global_share': 0.15, 'segment': 'fabless_design'},
        'ASE': {'global_share': 0.20, 'segment': 'packaging_testing'}
    },
    'shipping_impact': {
        'routes_affected': ['East Asia → US West Coast', 'East Asia → Europe',
                           'Intra-Asia (Japan, Korea, SE Asia)'],
        'reroute_delay_days': 7,  # Via south of Philippines
        'capacity_reduction_pct': 30
    },
    'estimated_duration_scenarios': {
        'naval_exercise': {'duration_days': 7, 'probability': 0.15},
        'blockade': {'duration_days': 90, 'probability': 0.05},
        'conflict': {'duration_days': 365, 'probability': 0.02}
    },
    'global_economic_impact': '$2.6T first year (Bloomberg Economics estimate)',
    'monitoring_signals': [
        'PLA naval vessel movements near strait (AIS gaps = military activity)',
        'Chinese military flight incursions into Taiwan ADIZ (ROCAF reports)',
        'US carrier strike group positioning (OSINT tracking)',
        'Semiconductor inventory pre-stocking by major buyers',
        'TSMC stock price volatility',
        'Chinese state media rhetoric escalation (GDELT tone analysis)'
    ]
}
```

### 5.3 Red Sea / Houthi Risk Monitoring

```python
RED_SEA_SCENARIO = {
    'name': 'Red Sea / Bab el-Mandeb Strait Disruption',
    'monitoring_signals': [
        'Vessel AIS signals disappearing in southern Red Sea',
        'Carrier route announcements (Maersk, MSC, CMA CGM)',
        'UKMTO/MSCHOA maritime security advisories',
        'Houthi media statements (Arabic language monitoring)',
        'US/UK military operations (CENTCOM press releases)',
        'Insurance premium changes for Red Sea transit (war risk)'
    ],
    'impact_model': {
        'normal_route': 'Suez Canal → Red Sea → Bab el-Mandeb → Indian Ocean',
        'reroute': 'Cape of Good Hope',
        'additional_distance_nm': 3500,
        'additional_transit_days': 10,
        'fuel_cost_increase_pct': 25,
        'affected_trade_volume': '12% of global trade',
        'container_rate_increase': '200-300% on affected lanes'
    }
}
```

### 5.4 Political Risk Score per Country

```python
def calculate_political_risk_score(country_code):
    """
    Composite political risk index (0-100, higher = riskier).
    Updated daily using real-time signals.
    """
    components = {
        # Static baseline (updated quarterly)
        'governance_index': get_world_bank_governance(country_code),      # Weight: 0.15
        'fragile_state_index': get_fsi_score(country_code),               # Weight: 0.10
        'ease_of_business': get_doing_business_score(country_code),       # Weight: 0.05

        # Dynamic signals (updated daily)
        'conflict_intensity': get_acled_intensity(country_code, days=30), # Weight: 0.20
        'gdelt_stability_tone': get_gdelt_stability(country_code),       # Weight: 0.15
        'sanctions_risk': get_sanctions_exposure(country_code),           # Weight: 0.15
        'travel_advisory': get_state_dept_level(country_code),           # Weight: 0.10
        'currency_volatility': get_fx_volatility(country_code, days=30), # Weight: 0.10
    }

    weights = [0.15, 0.10, 0.05, 0.20, 0.15, 0.15, 0.10, 0.10]

    return sum(score * weight for score, weight in zip(components.values(), weights))
```

---

## SECTION 6: FRONTEND ARCHITECTURE

### 6.1 Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Framework | **Next.js 14** (App Router) | SSR for SEO, RSC for performance, API routes for BFF |
| Styling | **Tailwind CSS** + **shadcn/ui** | Rapid development, consistent design system |
| Maps | **Deck.gl** over **Mapbox GL JS** | WebGL-powered, handles 100K+ data points, arc/hex layers |
| Charts | **Tremor** + **Recharts** | Tremor for dashboard KPIs, Recharts for custom charts |
| State | **Zustand** | Lightweight, no boilerplate, perfect for real-time updates |
| Real-time | **Server-Sent Events** (SSE) | Simpler than WebSocket, sufficient for alert streaming |
| Auth | **NextAuth.js** + Google OAuth | Enterprise SSO ready |
| Mobile | **React Native** (Expo) | Code sharing with web, push notifications |

### 6.2 Supply Network Map

```typescript
// Interactive world map with supply chain overlay
import { DeckGL } from '@deck.gl/react';
import { ArcLayer, ScatterplotLayer, HexagonLayer } from '@deck.gl/layers';

const SupplyNetworkMap = ({ suppliers, routes, disruptions }) => {
  const layers = [
    // Supplier locations (sized by annual spend)
    new ScatterplotLayer({
      id: 'suppliers',
      data: suppliers,
      getPosition: d => [d.lng, d.lat],
      getRadius: d => Math.sqrt(d.annual_spend) / 100,
      getFillColor: d => riskColor(d.risk_score),  // Green → Yellow → Red
      pickable: true
    }),

    // Supply routes (colored by risk)
    new ArcLayer({
      id: 'routes',
      data: routes,
      getSourcePosition: d => [d.source_lng, d.source_lat],
      getTargetPosition: d => [d.target_lng, d.target_lat],
      getSourceColor: d => riskColor(d.risk_score),
      getTargetColor: d => riskColor(d.risk_score),
      getWidth: d => d.volume_normalized * 5
    }),

    // Disruption risk heatmap
    new HexagonLayer({
      id: 'risk-heatmap',
      data: disruptions,
      getPosition: d => [d.lng, d.lat],
      getElevationWeight: d => d.severity,
      elevationScale: 1000,
      radius: 50000,
      colorRange: [[255,255,178], [254,204,92], [253,141,60],
                   [240,59,32], [189,0,38]]
    })
  ];

  return <DeckGL layers={layers} initialViewState={GLOBAL_VIEW} controller />;
};
```

### 6.3 72-Hour Warning Panel

```
┌─────────────────────────────────────────────────────────┐
│  🔴 RED ALERT: Typhoon Gaemi — Taiwan Impact in 48hrs  │
│  ───────────────────────────────────────────────────── │
│  Confidence: 87%  │  Impact: $12.4M revenue at risk    │
│                                                         │
│  Affected Suppliers:                                    │
│  ├── TSMC Fab 14 (Tainan) — 3nm production             │
│  ├── ASE Kaohsiung — packaging/test                     │
│  └── Port of Kaohsiung — 48hr closure expected          │
│                                                         │
│  Affected Components: A17 Pro SoC, M3 chipset           │
│  Downstream Impact: iPhone 16 production -15% for 2wks  │
│                                                         │
│  📡 Evidence:                                           │
│  ├── NOAA track forecast (Category 3, direct hit)       │
│  ├── Satellite: cloud formation [View Earth Engine]     │
│  └── MarineTraffic: 12 vessels diverting from Kaohsiung │
│                                                         │
│  ⚡ Recommended Actions:                                │
│  ├── [Approve] Expedite 50K units via air from Samsung  │
│  ├── [Approve] Increase safety stock order to GlobalFo  │
│  └── [Send] Pre-drafted email to TSMC procurement       │
│                                                         │
│  [View Full Analysis]  [Simulate Scenarios]  [Dismiss]  │
└─────────────────────────────────────────────────────────┘
```

### 6.4 Scenario Simulator

```typescript
// "What if?" scenario simulator
const ScenarioSimulator = () => {
  const [scenario, setScenario] = useState(null);
  const [results, setResults] = useState(null);

  const prebuiltScenarios = [
    { id: 'taiwan_strait', label: 'Taiwan Strait Closure',
      params: { node: 'TW_ALL', severity: 0.9, duration: 90 }},
    { id: 'suez_block', label: 'Suez Canal Blockage',
      params: { node: 'SUEZ', severity: 1.0, duration: 7 }},
    { id: 'us_port_strike', label: 'US East Coast Port Strike',
      params: { node: 'US_EAST_PORTS', severity: 0.8, duration: 14 }},
    { id: 'custom', label: 'Custom Scenario...' }
  ];

  const runSimulation = async (params) => {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      body: JSON.stringify(params)
    });
    setResults(await res.json());
    // Results include: Monte Carlo distribution, affected nodes,
    // revenue at risk (P50/P95/P99), timeline, recommendations
  };
};
```

### 6.5 Executive Dashboard (Looker Studio)

Embedded Looker Studio reports connected to BigQuery:
- **Supply Chain Health Score**: Single number (0-100), trend over 90 days
- **Active Alerts by Severity**: Red/Amber/Yellow counts
- **Revenue at Risk**: Current total with breakdown by disruption type
- **Geographic Risk Map**: Heat map of supplier concentration risk
- **Top 10 Single Points of Failure**: Table with mitigation status
- **Disruption Trend**: 12-month rolling disruption count by type
- **Response Time Metrics**: Avg time from alert to action

### 6.6 Key Page Routes

```
/                           → Dashboard overview (health score, active alerts)
/map                        → Interactive supply network map
/alerts                     → Alert list with filters (severity, type, region)
/alerts/[id]                → Individual alert detail with evidence + actions
/scenarios                  → Scenario simulator
/suppliers                  → Supplier registry and risk scores
/suppliers/[id]             → Individual supplier detail
/suppliers/upload           → CSV upload wizard
/graph                      → Supply chain graph visualizer
/reports                    → Generated reports and analytics
/settings                   → Company config, notification preferences
/api/signals/ingest         → Signal ingestion webhook
/api/simulate               → Monte Carlo simulation endpoint
/api/alerts/[id]/actions    → Approve/reject recommended actions
```

---

## SECTION 7: FILE & FOLDER STRUCTURE

```
supplymind/
├── README.md
├── package.json
├── .env.example                          # API keys template
├── .env.local                            # Local env (gitignored)
├── docker-compose.yml                    # Local dev (Neo4j, Redis)
├── Dockerfile                            # Cloud Run deployment
│
├── src/
│   ├── app/                              # Next.js App Router
│   │   ├── layout.tsx                    # Root layout (nav, providers)
│   │   ├── page.tsx                      # Dashboard home
│   │   ├── map/
│   │   │   └── page.tsx                  # Supply network map
│   │   ├── alerts/
│   │   │   ├── page.tsx                  # Alert list
│   │   │   └── [id]/
│   │   │       └── page.tsx              # Alert detail
│   │   ├── scenarios/
│   │   │   └── page.tsx                  # Scenario simulator
│   │   ├── suppliers/
│   │   │   ├── page.tsx                  # Supplier registry
│   │   │   ├── upload/
│   │   │   │   └── page.tsx              # CSV upload wizard
│   │   │   └── [id]/
│   │   │       └── page.tsx              # Supplier detail
│   │   ├── graph/
│   │   │   └── page.tsx                  # Supply chain graph viz
│   │   ├── reports/
│   │   │   └── page.tsx                  # Reports
│   │   ├── settings/
│   │   │   └── page.tsx                  # Settings
│   │   └── api/
│   │       ├── signals/
│   │       │   └── ingest/
│   │       │       └── route.ts          # Signal webhook receiver
│   │       ├── simulate/
│   │       │   └── route.ts              # Monte Carlo API
│   │       ├── alerts/
│   │       │   ├── route.ts              # Alert CRUD
│   │       │   ├── stream/
│   │       │   │   └── route.ts          # SSE alert stream
│   │       │   └── [id]/
│   │       │       └── actions/
│   │       │           └── route.ts      # Action approve/reject
│   │       ├── suppliers/
│   │       │   ├── route.ts              # Supplier CRUD
│   │       │   └── upload/
│   │       │       └── route.ts          # CSV upload handler
│   │       ├── earth-engine/
│   │       │   └── evidence/
│   │       │       └── route.ts          # Satellite imagery API
│   │       ├── gemini/
│   │       │   ├── classify/
│   │       │   │   └── route.ts          # Signal classification
│   │       │   └── draft-email/
│   │       │       └── route.ts          # Email drafting
│   │       └── graph/
│   │           ├── propagate/
│   │           │   └── route.ts          # Disruption propagation
│   │           └── spof/
│   │               └── route.ts          # Single point of failure
│   │
│   ├── components/
│   │   ├── ui/                           # shadcn/ui components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── badge.tsx
│   │   │   ├── dialog.tsx
│   │   │   └── ...
│   │   ├── map/
│   │   │   ├── SupplyNetworkMap.tsx       # Main map component
│   │   │   ├── MapLayers.ts              # Deck.gl layer configs
│   │   │   ├── MapTooltip.tsx            # Hover tooltip
│   │   │   └── MapControls.tsx           # Zoom, layer toggles
│   │   ├── alerts/
│   │   │   ├── AlertCard.tsx             # Alert summary card
│   │   │   ├── AlertDetail.tsx           # Full alert view
│   │   │   ├── AlertTimeline.tsx         # Timeline of signals
│   │   │   ├── WarningPanel.tsx          # 72-hr warning display
│   │   │   └── EvidenceViewer.tsx        # Satellite/data evidence
│   │   ├── charts/
│   │   │   ├── RiskGauge.tsx             # Risk score gauge
│   │   │   ├── ImpactWaterfall.tsx        # Financial impact chart
│   │   │   ├── DisruptionTimeline.tsx     # Disruption history
│   │   │   ├── InventoryCoverChart.tsx    # Inventory buffer viz
│   │   │   └── MonteCarloDistribution.tsx # Simulation results
│   │   ├── graph/
│   │   │   ├── SupplyChainGraph.tsx       # Interactive graph viz
│   │   │   ├── NodeDetail.tsx            # Node info panel
│   │   │   └── PathHighlighter.tsx       # Highlight affected paths
│   │   ├── scenarios/
│   │   │   ├── ScenarioBuilder.tsx        # Build custom scenarios
│   │   │   ├── ScenarioResults.tsx        # Simulation output
│   │   │   └── PrebuiltScenarios.tsx      # Quick-select scenarios
│   │   ├── suppliers/
│   │   │   ├── SupplierTable.tsx          # Sortable supplier list
│   │   │   ├── SupplierForm.tsx           # Add/edit supplier
│   │   │   ├── CsvUploader.tsx            # CSV import component
│   │   │   └── RiskScoreCard.tsx          # Supplier risk display
│   │   └── layout/
│   │       ├── Navbar.tsx                 # Top navigation
│   │       ├── Sidebar.tsx                # Side navigation
│   │       └── NotificationBell.tsx       # Real-time alert bell
│   │
│   ├── lib/
│   │   ├── google/
│   │   │   ├── gemini.ts                 # Gemini API client
│   │   │   ├── earth-engine.ts           # Earth Engine helpers
│   │   │   ├── bigquery.ts               # BigQuery client
│   │   │   ├── pubsub.ts                 # Pub/Sub publisher
│   │   │   ├── translate.ts              # Translation API
│   │   │   ├── maps.ts                   # Maps/Geocoding
│   │   │   └── vertex-ai.ts             # Vertex AI Forecast
│   │   ├── signals/
│   │   │   ├── weather.ts                # Weather signal fetcher
│   │   │   ├── shipping.ts               # Shipping signal fetcher
│   │   │   ├── geopolitical.ts           # Geopolitical signal fetcher
│   │   │   ├── disasters.ts              # Disaster signal fetcher
│   │   │   ├── financial.ts              # Financial signal fetcher
│   │   │   ├── labor.ts                  # Labor signal fetcher
│   │   │   └── classifier.ts            # Gemini signal classifier
│   │   ├── models/
│   │   │   ├── supply-graph.ts           # Graph data structure
│   │   │   ├── propagation.ts            # Disruption propagation
│   │   │   ├── monte-carlo.ts            # Monte Carlo simulation
│   │   │   ├── inventory.ts              # Inventory calculations
│   │   │   ├── financial-impact.ts       # Financial impact model
│   │   │   └── spof-detector.ts          # Single point of failure
│   │   ├── recommendations/
│   │   │   ├── alt-suppliers.ts           # Alt supplier finder
│   │   │   ├── buffer-calc.ts            # Buffer recommendations
│   │   │   ├── email-drafter.ts          # Gemini email drafting
│   │   │   └── roi-calculator.ts         # Dual-sourcing ROI
│   │   ├── sanctions/
│   │   │   ├── ofac.ts                   # OFAC SDN checker
│   │   │   ├── eu-sanctions.ts           # EU sanctions checker
│   │   │   └── monitor.ts               # Sanctions change monitor
│   │   ├── db/
│   │   │   ├── neo4j.ts                  # Neo4j connection
│   │   │   ├── firestore.ts              # Firestore client
│   │   │   └── schema.ts                # Type definitions
│   │   ├── utils/
│   │   │   ├── geo.ts                    # Geocoding utilities
│   │   │   ├── risk-scoring.ts           # Risk score calculations
│   │   │   └── formatters.ts             # Number/date formatters
│   │   └── constants/
│   │       ├── disruption-types.ts       # Disruption taxonomy
│   │       ├── risk-thresholds.ts        # Alert thresholds
│   │       └── demo-data.ts              # Hackathon demo data
│   │
│   ├── hooks/
│   │   ├── useAlertStream.ts             # SSE alert subscription
│   │   ├── useSupplyGraph.ts             # Graph data hook
│   │   └── useSimulation.ts              # Simulation state
│   │
│   └── store/
│       ├── alerts.ts                     # Alert state (Zustand)
│       ├── suppliers.ts                  # Supplier state
│       └── map.ts                        # Map view state
│
├── scripts/
│   ├── seed-demo-data.ts                 # Load demo supply chain
│   ├── ingest-signals.ts                 # Manual signal ingestion
│   ├── run-simulation.ts                 # CLI simulation runner
│   └── setup-pubsub.ts                   # Create Pub/Sub topics
│
├── data/
│   ├── demo-supply-chain.json            # Demo: semiconductor company
│   ├── disruption-history.json           # Historical disruption data
│   └── sample-suppliers.csv              # Sample CSV template
│
├── cloud/
│   ├── scheduler/
│   │   └── jobs.yaml                     # Cloud Scheduler job configs
│   ├── pubsub/
│   │   └── topics.yaml                   # Pub/Sub topic/subscription configs
│   ├── bigquery/
│   │   └── schema.sql                    # BigQuery table schemas
│   └── deploy.sh                         # Cloud Run deployment script
│
├── mobile/                               # React Native (Expo)
│   ├── app.json
│   ├── App.tsx
│   ├── screens/
│   │   ├── AlertsScreen.tsx
│   │   ├── AlertDetailScreen.tsx
│   │   └── MapScreen.tsx
│   └── components/
│       ├── AlertCard.tsx
│       └── PushHandler.tsx
│
├── tests/
│   ├── unit/
│   │   ├── propagation.test.ts
│   │   ├── monte-carlo.test.ts
│   │   ├── inventory.test.ts
│   │   └── risk-scoring.test.ts
│   ├── integration/
│   │   ├── signal-ingestion.test.ts
│   │   └── alert-pipeline.test.ts
│   └── e2e/
│       ├── demo-scenario.test.ts
│       └── upload-flow.test.ts
│
├── tailwind.config.ts
├── tsconfig.json
├── next.config.js
└── .github/
    └── workflows/
        └── deploy.yml                    # CI/CD pipeline
```

---

## SECTION 8: HACKATHON DEMO PLAN

### 8.1 Demo Scenario: "GlobalTech Electronics" — Semiconductor Supply Chain

**Pre-loaded company**: A mid-size electronics company sourcing semiconductors from Taiwan, displays from South Korea, batteries from China, and assembling in Vietnam and Mexico.

**Demo Supply Chain**:
```
Tier 3: ASML (NL) → Tier 2: TSMC (TW), Samsung (KR) → Tier 1: Foxconn (VN), Flex (MX)
                     ↓                    ↓
              Port: Kaohsiung        Port: Busan
                     ↓                    ↓
              Shipping Route ────→ Port of Long Beach (US)
                     ↓
              Warehouse: Ontario, CA → Customers: US retail
```

### 8.2 The 5-Minute Demo Script

**Minute 0:00 - 1:00 — "The Problem"**
> "Last year, supply chain disruptions cost $184 billion. Every Fortune 500 board now asks: *when is the next surprise?* Current tools tell you after the damage is done. SUPPLYMIND gives you 72 hours to act."

**Minute 1:00 - 2:00 — "The Supply Network"**
- Show the interactive map with GlobalTech's supply chain
- Zoom into Taiwan — highlight TSMC as single-source for 7nm chips
- Show the SPOF detector flagging TSMC as critical concentration risk
- Show dependency scores and risk overlays

**Minute 2:00 - 3:30 — "The 72-Hour Warning" (LIVE)**
- Trigger the Taiwan Strait scenario
- Watch the warning panel go RED in real-time
- Show the evidence chain:
  - GDELT: Military activity surge near strait (+340% event count)
  - MarineTraffic: 3 carriers diverting from Kaohsiung
  - Earth Engine: Satellite image of naval vessels (pre-loaded)
  - Financial: TSMC stock -4.2%, TWD/USD volatility spike
- Show the propagation model: "In 47 days, your US warehouse runs out of 7nm chips"
- Show Monte Carlo results: "P95 revenue at risk: $23.7M"

**Minute 3:30 - 4:30 — "The Recommendation Engine"**
- Auto-identified: Samsung 4nm as alternative (85% capability match)
- Show dual-sourcing ROI calculation
- Show auto-drafted email to Samsung procurement
- **THE MOMENT**: Click "Approve & Send" — the email sends
- Show inventory buffer recommendation: "Order 45-day buffer NOW"

**Minute 4:30 - 5:00 — "The Vision"**
> "SUPPLYMIND doesn't just predict disruptions. It tells you exactly what to do, when, and automates the action. No more surprises. No more scrambling. This is what supply chain resilience looks like."
> "Enterprise pricing starts at $5K/month. We're already in conversations with 3 Fortune 500 procurement teams."

### 8.3 Pre-Demo Checklist
- [ ] Demo supply chain data seeded in Firestore + Neo4j
- [ ] Signal feeds running for 24+ hours (have real data accumulation)
- [ ] Taiwan scenario pre-programmed with realistic trigger data
- [ ] Earth Engine satellite images pre-cached (avoid cold start)
- [ ] MarineTraffic vessel data pre-loaded for strait region
- [ ] Email draft endpoint working with Gemini
- [ ] Monte Carlo simulation returns results in < 3 seconds
- [ ] Mobile app showing push notification (on second screen)
- [ ] Looker Studio dashboard populated and loading fast

---

## SECTION 9: 24-HOUR BUILD SPRINT PLAN

### Pre-Hackathon (BEFORE the 24 hours)

| Task | Time | Owner |
|------|------|-------|
| Set up GCP project, enable all APIs | 2 hrs | Backend |
| Create Pub/Sub topics and BigQuery tables | 1 hr | Backend |
| Set up Neo4j Aura free tier instance | 30 min | Backend |
| Start signal ingestion jobs (need 24hr+ of data) | 1 hr setup, then continuous | Backend |
| Prepare demo supply chain dataset (JSON) | 2 hrs | Data |
| Design Figma mockups of key screens | 2 hrs | Frontend |
| Set up Next.js project with shadcn/ui | 1 hr | Frontend |

### Hour 0-4: Foundation

| Task | Hours | Priority |
|------|-------|----------|
| Implement supply chain graph model (NetworkX in API route) | 2 | P0 |
| Build signal ingestion API routes (weather, GDELT, USGS) | 2 | P0 |
| Set up Gemini signal classifier | 1 | P0 |
| Create Firestore schema + CRUD for suppliers/alerts | 1 | P0 |
| Build basic dashboard layout (Navbar, Sidebar, pages) | 2 | P0 |

### Hour 4-10: Core Features

| Task | Hours | Priority |
|------|-------|----------|
| Build interactive supply network map (Deck.gl) | 3 | P0 |
| Implement disruption propagation engine | 2 | P0 |
| Build Monte Carlo simulation API | 2 | P0 |
| Create 72-hour warning panel component | 2 | P0 |
| Build alert detail page with evidence viewer | 2 | P0 |
| Implement SSE alert streaming | 1 | P0 |

### Hour 10-16: Intelligence Layer

| Task | Hours | Priority |
|------|-------|----------|
| Implement Taiwan Strait scenario model | 2 | P0 |
| Build scenario simulator UI | 2 | P0 |
| Integrate Earth Engine for satellite evidence | 2 | P1 |
| Build recommendation engine (alt suppliers, buffer calc) | 2 | P0 |
| Build auto-email drafter with Gemini | 1 | P0 |
| Implement SPOF detector | 1 | P1 |
| Build financial impact waterfall chart | 1 | P1 |

### Hour 16-20: Polish & Integration

| Task | Hours | Priority |
|------|-------|----------|
| Seed full demo data + verify all calculations | 2 | P0 |
| Build "Approve & Send" action flow | 1 | P0 |
| Add Monte Carlo distribution chart | 1 | P1 |
| Create Looker Studio executive dashboard | 2 | P1 |
| Mobile alert notifications (basic Expo app) | 2 | P2 |
| End-to-end demo walkthrough testing | 2 | P0 |

### Hour 20-24: Demo Prep

| Task | Hours | Priority |
|------|-------|----------|
| Fix bugs from walkthrough | 2 | P0 |
| Optimize loading times (pre-cache, SSG) | 1 | P1 |
| Prepare pitch script and slides | 2 | P0 |
| Final dry run (3x minimum) | 1 | P0 |

### Minimum Viable Demo (if time runs short, cut to these)
1. Interactive supply chain map with risk overlays
2. Taiwan Strait scenario trigger → warning panel
3. Propagation model → "47 days until stockout"
4. Auto-drafted supplier reallocation email
5. "Approve & Send" button

---

## SECTION 10: PRODUCTION & SCALE

### 10.1 Pricing Model

| Tier | Monthly Price | Included |
|------|--------------|----------|
| **Starter** | $5,000/mo | Up to 50 Tier 1 suppliers, 5 signal types, email alerts, basic map |
| **Professional** | $15,000/mo | Up to 200 suppliers (Tier 1-2), all signals, Monte Carlo sim, scenario builder, API access |
| **Enterprise** | $30,000-50,000/mo | Unlimited suppliers (Tier 1-3), custom models, ERP integration, dedicated CSM, SLA |
| **Critical Infrastructure** | Custom | Government/defense, classified supply chains, on-prem deployment option |

**Per-node pricing add-on**: $25/supplier node/month after tier limit

### 10.2 ERP Integration Architecture

```
┌──────────┐     ┌──────────────────┐     ┌───────────────┐
│ SAP S/4  │────▶│ SUPPLYMIND       │────▶│ Alerts back   │
│ HANA     │ RFC │ Integration      │ IDoc│ to SAP        │
│          │     │ Layer            │     │ (procurement) │
└──────────┘     └──────────────────┘     └───────────────┘

│ Oracle    │────▶│ REST API         │────▶│ Oracle alerts  │
│ SCM Cloud │REST │ Connector        │REST │ (planning)     │

Integration touchpoints:
- Supplier master data sync (daily)
- Purchase order data (real-time)
- Inventory levels (daily)
- BOM (Bill of Materials) structure (on change)
- Alert push-back to ERP for procurement action
```

### 10.3 Scale Architecture

| Component | Hackathon | Production |
|-----------|-----------|------------|
| Signal ingestion | Cloud Run (single instance) | Cloud Run (auto-scale 1-50) + Pub/Sub |
| Graph engine | In-memory NetworkX | Neo4j Aura Enterprise (clustered) |
| Simulation | Single-threaded | Cloud Run Jobs (parallel workers) |
| Data store | Firestore | BigQuery (analytics) + Firestore (real-time) + Cloud SQL (relational) |
| Frontend | Vercel hobby | Vercel Enterprise or Cloud Run + CDN |
| Auth | NextAuth | Auth0 / Okta enterprise SSO |

### 10.4 Partnership Opportunities

- **Insurance**: Supply chain insurance underwriters use SUPPLYMIND risk scores for pricing
- **Consulting**: Big 4 firms resell as part of supply chain transformation engagements
- **Financial**: Commodity trading desks use disruption predictions for trading signals
- **Government**: Defense/intelligence agencies for critical supply chain monitoring

---

## SECTION 11: TECHNICAL RISKS & MITIGATIONS

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Signal feed downtime** (API outages) | High | Medium | Redundant sources per signal type, graceful degradation, cached last-known-good |
| **False positive alerts** | High | High | Multi-signal corroboration requirement, confidence thresholds, user feedback loop for model retraining |
| **Supply chain data entry burden** | High | High | CSV bulk upload, ERP auto-sync, progressive enrichment (start with Tier 1 only) |
| **Geopolitical prediction overreach** | Medium | High | Always show confidence intervals, never claim certainty, "watch/warning/alert" tiering |
| **Earth Engine cold starts** | Medium | Low | Pre-cache imagery for supplier regions, async loading with placeholder |
| **MarineTraffic API costs at scale** | Medium | Medium | Cache aggressively (5-min TTL), request only supplier-relevant ports, negotiate enterprise rate |
| **Gemini hallucination in classifications** | Medium | Medium | Structured output format (JSON), validation against disruption taxonomy, confidence thresholds |
| **Neo4j scalability for large supply chains** | Low | Medium | Partition by company, use BigQuery for analytics queries, Neo4j for traversal only |
| **Enterprise SSO/compliance requirements** | Medium | Medium | SOC 2 Type II certification path, data residency options, audit logging |

---

## SECTION 12: BUSINESS MODEL

### 12.1 Revenue Projections

| Year | Customers | Avg Revenue/Customer | ARR |
|------|-----------|---------------------|-----|
| Y1 | 10 | $120K | $1.2M |
| Y2 | 40 | $180K | $7.2M |
| Y3 | 120 | $240K | $28.8M |

### 12.2 Go-to-Market

1. **Launch**: Semiconductor + automotive verticals (highest pain, most complex supply chains)
2. **Expand**: Pharma, aerospace, consumer electronics
3. **Enterprise**: SAP/Oracle marketplace listings, consulting partnerships
4. **Platform**: Open API for supply chain risk data (signal-as-a-service)

### 12.3 Competitive Moat

1. **Data network effect**: More customers → better disruption models → better predictions → more customers
2. **Leading indicator library**: Proprietary mapping of 300+ leading indicators to disruption types
3. **Graph-based propagation**: Not just "is there a disruption?" but "how does it affect YOUR specific supply chain?"
4. **Speed**: 72-hour warning vs. competitors' reactive dashboards

---

## SECTION 13: JUDGING CRITERIA & PITCH STRUCTURE

### 13.1 Hackathon Judging Alignment

| Criterion | How SUPPLYMIND Excels |
|-----------|----------------------|
| **Technical Complexity** | Multi-source signal ingestion, graph-based propagation, Monte Carlo simulation, ML prediction, satellite imagery, NLP email drafting — all integrated |
| **Google API Usage** | 11 Google APIs: Gemini, Earth Engine, Maps, Vertex AI, BigQuery, Pub/Sub, Translate, Looker, Scheduler, Firebase, Cloud Run |
| **Real-World Impact** | $184B problem, every Fortune 500 needs this, COVID proved existing tools fail |
| **Demo Quality** | Live scenario trigger, real-time alert, satellite evidence, auto-email, "approve & send" moment |
| **Business Viability** | Clear pricing model, $28.8M ARR Y3 projection, enterprise sales motion proven |

### 13.2 Pitch Structure (5 minutes)

```
[0:00] HOOK: "What if you knew the Suez Canal was about to be blocked... 3 days early?"
[0:30] PROBLEM: $184B cost, reactive tools, no prediction
[1:00] DEMO: Show the supply chain map → trigger scenario → watch it propagate
[2:00] TECHNOLOGY: Signal ingestion → Gemini classification → graph propagation → prediction
[3:00] LIVE MOMENT: "Approve & Send" reallocation email
[3:30] EVIDENCE: Earth Engine satellite, MarineTraffic vessels, financial signals
[4:00] BUSINESS: Pricing, TAM ($50B supply chain software market), traction
[4:30] VISION: "No more supply chain surprises"
[5:00] END
```

---

## SECTION 14: APPENDICES

### A. BigQuery Signal Table Schema

```sql
CREATE TABLE supplymind.signals.raw_signals (
    signal_id STRING NOT NULL,
    signal_type STRING NOT NULL,       -- weather, shipping, geopolitical, disaster, financial, labor
    source STRING NOT NULL,            -- noaa, gdelt, usgs, marinetraffic, etc.
    event_type STRING,                 -- cyclone, earthquake, strike, sanctions, etc.
    severity FLOAT64,                  -- 0.0 - 1.0
    confidence FLOAT64,               -- 0.0 - 1.0
    latitude FLOAT64,
    longitude FLOAT64,
    affected_radius_km FLOAT64,
    country_code STRING,
    region STRING,
    title STRING,
    description STRING,
    raw_payload JSON,
    gemini_classification JSON,        -- Gemini output
    supply_chain_relevance STRING,     -- high, medium, low, none
    event_timestamp TIMESTAMP NOT NULL,
    ingestion_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    source_url STRING
)
PARTITION BY DATE(event_timestamp)
CLUSTER BY signal_type, country_code;
```

### B. Sample 72-Hour Warning Report Format

```json
{
  "alert_id": "ALT-2024-0847",
  "severity": "RED",
  "created_at": "2024-07-23T14:30:00Z",
  "disruption_type": "tropical_cyclone",
  "title": "Typhoon Gaemi — Direct Impact on Taiwan Manufacturing",
  "confidence": 0.87,
  "time_to_impact_hours": 48,
  "evidence": [
    {"source": "NOAA NHC", "signal": "Category 3 typhoon, track shows direct Taiwan landfall",
     "timestamp": "2024-07-23T12:00:00Z"},
    {"source": "MarineTraffic", "signal": "12 vessels diverting from Kaohsiung",
     "timestamp": "2024-07-23T13:15:00Z"},
    {"source": "GDELT", "signal": "Taiwan weather emergency news volume +540%",
     "timestamp": "2024-07-23T14:00:00Z"},
    {"source": "Earth Engine", "signal": "Sentinel-1 showing storm structure 400km east of Taiwan",
     "image_url": "https://earthengine.googleapis.com/..."}
  ],
  "affected_suppliers": [
    {"id": "SUP001", "name": "TSMC Fab 14", "component": "A17 Pro SoC",
     "impact": "Production halt 3-7 days", "revenue_at_risk": 8200000},
    {"id": "SUP003", "name": "ASE Kaohsiung", "component": "Chip packaging",
     "impact": "Facility flooding risk", "revenue_at_risk": 4100000}
  ],
  "propagation": {
    "days_to_stockout": 47,
    "p50_revenue_at_risk": 12400000,
    "p95_revenue_at_risk": 23700000,
    "downstream_customers_affected": 3
  },
  "recommendations": [
    {"action": "expedite", "description": "Expedite 50K units from Samsung via air freight",
     "cost": 450000, "risk_reduction": 0.6, "status": "pending_approval"},
    {"action": "buffer_order", "description": "Place 45-day safety stock order with GlobalFoundries",
     "cost": 2100000, "risk_reduction": 0.3, "status": "pending_approval"},
    {"action": "supplier_email", "description": "Send contingency inquiry to TSMC procurement",
     "cost": 0, "auto_drafted": true, "status": "pending_approval"}
  ]
}
```

### C. Environment Variables

```bash
# Google Cloud
GOOGLE_CLOUD_PROJECT=supplymind-prod
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json

# Gemini
GEMINI_API_KEY=your_gemini_api_key

# External APIs
MARINE_TRAFFIC_API_KEY=your_mt_key
NEWS_API_KEY=your_newsapi_key
ACLED_API_KEY=your_acled_key
FRED_API_KEY=your_fred_key
EXCHANGE_RATES_API_KEY=your_exchange_key

# Neo4j
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Firebase
NEXT_PUBLIC_FIREBASE_CONFIG={"apiKey":"...","projectId":"supplymind"}

# Mapbox
NEXT_PUBLIC_MAPBOX_TOKEN=your_mapbox_token

# App
NEXT_PUBLIC_APP_URL=https://supplymind.app
```

---

## Verification & Testing Plan

1. **Signal Ingestion**: Run `scripts/ingest-signals.ts` → verify signals appear in BigQuery → verify Gemini classifies correctly
2. **Supply Graph**: Seed demo data → verify graph visualizes in UI → verify SPOF detector finds TSMC
3. **Propagation**: Trigger Taiwan scenario → verify propagation calculates 47-day stockout → verify Monte Carlo returns P50/P95 in < 3 seconds
4. **Alerts**: Trigger scenario → verify SSE pushes alert to frontend → verify warning panel renders with evidence
5. **Recommendations**: Verify alt supplier scoring → verify email draft quality → verify "Approve" action updates Firestore
6. **Earth Engine**: Verify satellite imagery loads for Taiwan region → verify before/after comparison renders
7. **End-to-End Demo**: Full 5-minute walkthrough with timing, no errors, all transitions smooth
8. **Mobile**: Push notification arrives on Expo app within 5 seconds of alert creation

---

## Critical Files to Create First

1. `src/lib/models/supply-graph.ts` — Core graph data structure and propagation engine
2. `src/lib/signals/classifier.ts` — Gemini signal classification
3. `src/lib/google/gemini.ts` — Gemini API client wrapper
4. `src/app/api/signals/ingest/route.ts` — Signal ingestion webhook
5. `src/components/map/SupplyNetworkMap.tsx` — Main map visualization
6. `src/components/alerts/WarningPanel.tsx` — 72-hour warning display
7. `src/lib/models/monte-carlo.ts` — Monte Carlo simulation
8. `data/demo-supply-chain.json` — Pre-loaded demo data
9. `src/app/page.tsx` — Dashboard home
10. `src/app/api/simulate/route.ts` — Simulation API endpoint
