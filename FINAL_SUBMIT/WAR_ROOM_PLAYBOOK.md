# War Room Playbook — Plain-Language Field Guide

For judges, analysts, and house-help uncle. Every jargon term explained.

---

## 1. INPUT FIELDS — what to type / what each means

### Severity (0.00 – 1.00)
> **What it controls**: how badly the chokepoint is hit.

| Value | Meaning | Real-world analog |
|-------|---------|-------------------|
| 0.00 | nothing happens, normal day | any day in 2022 |
| 0.30 | minor incident, insurance up but ships pass | June 2019 Iran tanker threats |
| 0.55 | partial closure, ~30% transit cut | April 2024 Iran "True Promise" strikes |
| 0.85 | serious closure, most ships avoid | hypothetical Iran-Israel-USA serious escalation |
| 0.92 | full Reliance war scenario | this template default |
| 1.00 | full closure, military action, no transit | hypothetical worst case |

### Brent $/bbl (USD per barrel)
> **What it controls**: assumed crude oil price during the disruption window.

| Value | Context |
|-------|---------|
| $80 | normal baseline (today's range) |
| $108 | Iran-only strait threat |
| $132 | full Hormuz partial closure (default) |
| $152 | Israel-Iran-USA war scenario (Reliance template) |
| $200+ | full extended closure, military strike on facilities |

### Duration days (1 – 1200)
> **What it controls**: how long the disruption lasts.

| Value | Real-world analog |
|-------|-------------------|
| 7 | Suez 2021 short blockage |
| 21 | typical Hormuz-threat scare (default) |
| 45 | Reliance war scenario template |
| 90 | Houthi Red Sea persistence (sustained) |
| 180+ | structural war state, multi-month |

### OpenRouter 6-judge panel (checkbox)
> **What it does**: cross-checks our 3-judge local verdict against 6 independent frontier LLMs (GPT-OSS-120B, Gemma-4-31B, GLM-4.5, MiniMax-M2.5, Nemotron-3-Super-120B, Gemma-4-26B). Adds ~60 seconds. Returns Krippendorff α (inter-judge agreement coefficient).

### Reliance toggle (checkbox)
> **What it does**: turns on the 10-subsidiary RIL impact table (Jamnagar refinery, RIL Petrochem, KG-D6 E&P, Reliance Retail, Jio Platforms, Recron polyester, RIIL pipelines, Reliance General Insurance, Reliance Power, Network18). FY24-anchored numbers from RIL Integrated Annual Report.

---

## 2. HEADLINE CARDS — what they mean

### Risk level
- **LOW** (verdict score 0.0–0.3) — monitor only
- **MEDIUM** (0.3–0.6) — prepare contingencies, dust off plans
- **HIGH** (0.6–0.85) — activate plans now, brief executives
- **CRITICAL** (>0.85) — full emergency response, lock in hedges, reroute in-transit cargo

### Composite confidence (%)
> *"How sure are we?"*
- **Built from 3 signals**:
  - 55% — agreement among the LLM judges (more agreement = more confidence)
  - 25% — sharpness of sector ranking (sharper top-3 spread = sharper signal)
  - 20% — corroboration from live signals last 24h
- **Plain-uncle**: "90% means we are very sure. 50% means we are guessing more than measuring."

### Brent projection P50 ($/bbl)
> **P50 = the median**. Half of plausible outcomes are below this number, half above.
- **P10 / P90** give the 80% confidence band — where 80 out of 100 outcomes are expected to fall.
- **Plain-uncle**: "If we ran this disruption 100 times, half the time Brent ends up below the P50 number, half above. The p10-p90 band covers 80 out of 100 outcomes."

### P50 loss saving (with plan)
> If management follows our recommended action plan, this is how much money the company saves (median estimate) versus doing nothing.
- **P50 = median**. Half the time savings will be larger, half smaller.
- **Plain-uncle**: "If we ran this disruption 100 times and management followed our plan, half the time they save more than this number. It is the do-or-don't-act dollar comparison."

---

## 3. RECOMMENDED ACTIONS — every action explained

For each action card the demo shows: **action_type**, **target node**, **cost USD**, **save USD**, **ROI ratio**, **reason**, **why-it-works**.

### activate_backup_supplier
> Switch part of the volume to a pre-qualified second-source supplier (typically held on standby contract). Costs more per unit but guarantees flow continuity.
- **When it wins**: stockout cost > extra unit cost.
- **Typical cost**: $150K activation fee per ISM cited rate.
- **Typical save**: $1-5M of stockout-prevented revenue.

### reroute_shipment
> Re-direct in-transit cargo via Cape of Good Hope or East-West pipeline instead of Hormuz / Bab-el-Mandeb.
- **When it wins**: chokepoint transit risk > 18-22 day Cape detour cost.
- **Typical cost**: $20-30M demurrage per VLCC.
- **Typical save**: $80-150M of crude-on-water risk avoided.

### increase_safety_stock
> Build inventory buffer ahead of the disruption peak.
- **When it wins**: disruption duration > buffer days AND carrying cost < stockout cost.
- **Typical cost**: 25% per year carrying cost (CSCMP standard).
- **Typical save**: full disruption-window stockout avoidance.

### expedite_shipment
> Pay airfreight (~10x sea cost per IATA) for the highest-margin SKUs only.
- **When it wins**: SKU margin > 10x freight differential.
- **Surgical use only** — never blanket-applied. Economics break above ~5% of SKU portfolio.
- **Typical cost**: $0.5-2M per shipment.
- **Typical save**: $5-20M for premium SKUs.

### hedge_commodity
> Buy upside protection on Brent / naphtha / LNG via futures or swaps.
- **When it wins**: implied vol < expected realized vol.
- **Typical cost**: $50K-500K premium per million-bbl notional.
- **Typical save**: $3-10M if commodity moves against the firm.
- **Caveat**: expires worthless if disruption resolves quickly.

### issue_supplier_alert
> Pre-emptive notification to downstream customers that supply allocation may shift.
- **When it wins**: pre-empts customer dispute claims, preserves long-term contracts.
- **Cost**: minimal (legal + admin time).
- **Save**: prevents 8-figure contract violation litigation.

### do_nothing
> Decline to act.
- **When it's the right call**: signal weak, action cost > expected loss, OR buffers sufficient for disruption duration.
- **Note**: explicitly recommended action when warranted; not a default.

---

## 4. CHOKEPOINT MAP — colors and lines

| Color | Meaning |
|-------|---------|
| 🔴 Red dot (pulsing) | The chokepoint itself (Strait of Hormuz). |
| 🟡 Yellow dot | Producer country (Saudi Arabia, UAE, Iraq, Qatar, Iran). |
| 🔵 Cyan dot | Consumer country (India, China, Japan, Korea, EU). |
| 🟣 Violet dot | Bypass route (Habshan-Fujairah pipeline, East-West pipeline, Cape of Good Hope). |
| 🔵 Cyan line | Crude flow route (line thickness scales with mb/d flow rate). |
| 🟣 Violet line | Bypass route (alternate path that avoids Hormuz). |

**Click any line** to see flow rate, capacity, citing agency, and notes.

---

## 5. INDIA SECTOR RANKING — read the columns

| Column | Meaning |
|--------|---------|
| # | Rank by composite score (1 = most exposed) |
| Sector | Display name + feedstock chain summary |
| Score | 0.0-1.0 composite (4-channel scoring) |
| Driver | Which channel dominates: structural / severity / brent / duration |
| 1st symptom (d) | Days from disruption to user-visible symptom |
| 30d ₹ Cr (point) | Point-estimate 30-day INR-crore impact within published band |

**Hover any sector name** to see citation source + plain-English first-symptom narrative.

---

## 6. RELIANCE INDUSTRIES IMPACT — read the dashboard

The Reliance card shows three top-line numbers:

- **30d at-risk · point**: sum of 30-day point-estimate impacts across all 10 RIL subsidiaries.
- **30d at-risk · band**: low-to-high range (sum of low estimates → sum of high estimates).
- **% of FY24 revenue**: what fraction of RIL FY24 consolidated revenue (₹11.36 lakh Cr) is at risk in this 30-day window.

| % range | Interpretation |
|---------|---------------|
| < 1% | Manageable. Routine BCM activation. |
| 1–3% | Concerning. Board-level briefing warranted. |
| 3–5% | Serious. Activate full crisis-management plan. |
| > 5% | Severe. Disclosure to capital markets within 24h. |

For our default Reliance war scenario (severity 0.92 / Brent $152 / 45 days): **2.143% of FY24 revenue at risk** — concerning, board-level briefing warranted.

---

## 7. HONEST CAVEATS

The system surfaces honest caveats automatically:

- "No Hormuz-tagged signals last 24h" — projection is conditional, not a current incident.
- "Local Ollama LLM panel not reachable" — running deterministic rubric fallback (still real, but rule-based).
- "Crisis-library analog match returned no high-similarity events" — projection interpolated from closest available analog.
- "Sector loss bands are point-estimate ranges from published agency data; NOT precise dollar forecasts."
- "This system does NOT predict whether Hormuz will actually be closed. It quantifies second-order industrial effects conditional on closure."

These caveats are NEVER omitted. They appear in every receipt.

---

## 8. RECEIPT — the sha256 stamp

Every war-room run produces a JSON receipt with `receipt_sha256` field. The sha256 is computed over the entire response payload (sorted-keys canonical JSON). Two reruns with identical inputs produce identical receipts. **Tamper-detect**: change any field, the sha changes.

Click `⊟ open receipt` to view + copy + download.
