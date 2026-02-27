# Walgreens Pharmacy Portfolio Optimizer — Deployment Guide

## What This Does

Scores every Walgreens location in the US on a 0–1 viability scale using census demographics, home prices, chronic disease prevalence, and competitive pharmacy data. Outputs strategic actions (PROTECT, CONSOLIDATE, CLOSE, etc.) for each of the ~6,664 Walgreens ZCTAs.

## Files

| File | Purpose |
|------|---------|
| `run_complete_system.py` | **Start here.** Orchestrator that runs Part 2 → Part 3 in sequence |
| `profit_model_v2.py` | Part 2: ZCTA-level profit scoring (census + health + cost data) |
| `walgreens_optimizer_v2.py` | Part 3: Store-level viability scoring, actions, ML layers, clustering |
| `profit_ml_layer.py` | Part 2b: Optional ML enhancement on ZCTA scores (XGBoost, SHAP) |

## Data Requirements

Put all data files in a `data/` directory alongside the scripts.

### Required
- **`data/pharmacy_data.csv`** — NPI pharmacy dataset with one-hot chain columns (Walgreens, CVS, etc.) and a `short_ZIP` column. This is buddy's export. ~156K rows.

### Auto-downloaded by Part 2 (internet required on first run)
- Census ACS (income, population, age, insurance) — pulled from Census API
- CDC PLACES (chronic disease prevalence) — pulled from CDC
- Zillow ZHVI (home prices) — `data/houseprice.csv` if pre-placed, otherwise imputed from income

### Optional
- **`data/houseprice.csv`** — Zillow Home Value Index by ZCTA. Only covers ~4,500 ZCTAs; the model imputes the remaining ~36,500 from state-level income ratios. If absent, all home prices are imputed.
- **`data/rent_data.csv`** — Median rent by ZCTA. Used in cost calculation if available.

## How to Run

### Full pipeline (recommended first run)
```bash
python run_complete_system.py --npi data/pharmacy_data.csv
```
Takes ~30 seconds. No GPU needed.

### With LLM-generated store briefs
```bash
python run_complete_system.py --npi data/pharmacy_data.csv --api-key sk-ant-xxx
```
Generates 50 natural-language store assessments via Claude API. Costs ~$0.50.

### Part 3 only (if Part 2 results already exist)
```bash
python run_complete_system.py --npi data/pharmacy_data.csv --walgreens-only
```
Skips ZCTA scoring, runs only Walgreens optimization. Requires `results_v2/profit_scores.csv` from a previous Part 2 run.

### Running scripts individually
```bash
# Part 2 alone
python profit_model_v2.py

# Part 3 alone (after Part 2)
python walgreens_optimizer_v2.py \
    --zcta-scores results_v2/profit_scores.csv \
    --npi data/pharmacy_data.csv
```

## Outputs

### `results_v2/` (Part 2)
- `profit_scores.csv` — 41,094 ZCTAs scored on revenue potential, cost, demand

### `results_walgreens/` (Part 3)
- `store_viability_scores.csv` — Main output. One row per Walgreens ZCTA with viability score, action, archetype
- `consolidation_pairs.csv` — ZCTAs with 2+ Walgreens where consolidation is recommended
- `archetype_centroids.csv` — Cluster centers for the 3 store archetypes
- `store_briefs.csv` — LLM-generated assessments (if API key provided)
- `portfolio_summary.json` — Aggregate stats

## Key Output Columns

In `store_viability_scores.csv`:

| Column | Description |
|--------|-------------|
| `store_viability` | 0–1 composite score. Higher = healthier store |
| `action` | Strategic recommendation: PROTECT & INVEST, EXPAND CLINICAL, MONITOR, REFORMAT, CONSOLIDATE, CLOSURE CANDIDATE |
| `archetype_name` | Cluster label: Fortress Store, Middle Market, or Cannibalization Zone |
| `store_revenue` | 0–1 revenue potential (population, income, health demand) |
| `store_cost` | 0–1 operating cost pressure (rent, home prices, wages) |
| `store_position` | 0–1 competitive position (market share, competition density) |
| `walgreens_count` | Number of Walgreens in this ZCTA |
| `ml_justification_gap` | How much the ML model disagrees with the rule-based score (high = potential hidden gem or hidden risk) |

## Expected Results (validated Feb 2026)

- ~6,664 stores scored (after filtering 7 zero-population ZCTAs)
- ~1,188 closure candidates (17.8%) — aligns with Walgreens' announced ~14% closure target
- ~1,520 protect & invest
- ~1,778 monitor
- ~1,489 consolidation candidates
- Top stores: Solo Walgreens in low-cost defensible markets (Appalachia, rural South)
- Bottom stores: Over-saturated ZCTAs with 3-10 Walgreens competing against themselves (Newark NJ, Lakeview Chicago, SoMa SF)

## Dependencies

```bash
pip install pandas numpy scikit-learn xgboost shap requests
```

Python 3.9+. No GPU required. Runs in ~6 seconds for Part 3, ~30 seconds for full pipeline.

## Known Limitations

1. **ZCTA-level, not store-level.** Without actual Walgreens store addresses, each ZCTA with N Walgreens is treated as a single decision unit. If you obtain a store-level CSV with `store_id` and `ZIP`, pass it via `--wag-stores` for true per-store scoring.

2. **Home price coverage.** Zillow data covers only ~11% of ZCTAs. The rest are imputed from state-level income-to-price ratios. Results in cost-heavy states (CA, NY, MA) are more reliable than rural areas.

3. **No actual store P&L.** Revenue and cost are estimated from public census/demographic data, not Walgreens financials. The model identifies *relative* viability, not absolute profitability.

4. **Cannibalization is ZCTA-based.** We measure how many Walgreens share the same ZCTA, not physical proximity. Two Walgreens 0.5 miles apart in different ZCTAs won't show cannibalization. This is conservative by design — false negatives are better than the false positives we had with centroid-based distance.
