# GLM+Expected-Access Model Integration Guide

## What Changed

The pharmacy desert prediction system now uses a research-grade **GLM (Poisson) + Expected-Access** approach with OOF cross-validation, per-state calibration, neighbor QA, and suspicious zero detection.

### Key Improvements

1. **OOF Poisson GLM**: Out-of-Fold cross-validation for unbiased expected pharmacy count predictions
2. **Per-State Calibration**: State-level adjustment factors (observed/expected) with national fallback
3. **Urban-Only Calibration**: Additional urban-specific calibration for QA purposes
4. **NB Variance**: Negative Binomial variance estimation (alpha) for realistic uncertainty
5. **Neighbor QA**: Geographic kNN (Haversine on lon/lat) or feature-space kNN for validation
6. **Suspicious Zero Detection**: Flags high-population ZIPs with zero pharmacies but high expected counts
7. **Deviance Residuals**: More robust than Pearson residuals, especially for zero counts
8. **Dual Scoring**: Main (residual-based) and alternative (deficit-based) IFAE scores

## How It Works

### Training Pipeline (`data/finalfinalfinal_training.py`)

The model trains in these steps:

1. **Load Data**: 
   - Pharmacy NPI bundle (with lon/lat for centroids)
   - Demographics (population, density)
   - Health (poor general health %)
   - Income (median HH income)
   - Optional: AQI (PM2.5), HHI (heat vulnerability)

2. **Feature Engineering**:
   - Convert to percentiles (0-1): income_pct_inv, health_pct, access_pct_inv, density_pct, aqi_pct, heat_pct
   - Center predictors: subtract 0.5 from percentiles
   - Create density spline (B-spline with df=3) if sufficient data
   - Compute composite score: weighted average of percentiles

3. **OOF GLM Training**:
   - Poisson GLM with exposure=population
   - GroupKFold by STATE (or KFold if <2 states)
   - Out-of-Fold predictions for unbiased μ estimates
   - Estimate NB alpha from OOF residuals

4. **Per-State Calibration**:
   - Compute calibration factor per state: `state_cal = Σ(observed) / Σ(μ_oof)` for each state
   - Fallback to national calibration if state has insufficient data
   - Apply: `μ_oof_cal = μ_oof * state_cal`
   - Urban-only calibration also computed for QA

5. **Neighbor QA**:
   - If ≥20% of ZIPs have lon/lat: use geographic kNN (Haversine distance)
   - Otherwise: use feature-space kNN (income, health, log population)
   - Compute median neighbor observed/expected rates per 10k
   - Suspicious zero flag: `pop ≥ 20k + pharmacies=0 + μ̂≥3 (or ≥2/10k) + neighbors≥1/10k`

6. **Final Refit (QA)**:
   - Refit GLM on all valid rows with STATE fixed effects (C(REGION))
   - Produces `glm_mu_expected_pharm_full` for planning/QA
   - Export coefficients to `glm_full_coefficients.csv`

7. **Scoring & Ranking**:
   - Deviance residuals: more robust for zeros than Pearson
   - Underserved score: `-deviance_resid` (higher = more underserved)
   - `glm_nb_score`: normalized underserved score
   - `IFAE_score_residual`: 50% composite + 50% glm_nb_score (main)
   - `IFAE_score_deficit`: 50% composite + 50% normalized deficit_rate10k (alt)
   - `IFAE_score = IFAE_score_residual` (main ranking)

### Files Output

- `results/national_ifae_rank.csv` - Main rankings (residual-based blend), sorted by IFAE_score desc
- `results/national_ifae_rank_alt_deficit.csv` - Alternative rankings (deficit-based blend)
- `results/topK_ifae_urban.csv` - Top 10 urban deserts (pop ≥ 1000, urban mask)
- `results/bottomK_ifae.csv` - Bottom 10 (pop > 0 only)
- `results/qa_expected_vs_observed.csv` - QA with neighbor metrics and suspicious zeros
- `results/glm_full_coefficients.csv` - Model coefficients with SE and z-scores

## Integration Points

### 1. Training Module (`data_processing/glm_training.py`)

Wrapper that executes the training script:

```python
from data_processing import run_glm_training

# Train the model (or skip if results exist)
success = run_glm_training(force_retrain=False)

# Force retrain even if results exist
success = run_glm_training(force_retrain=True)
```

### 2. Score Loading (`data_processing/ml_ifae.py`)

Loads GLM scores from `results/national_ifae_rank.csv`:

```python
from data_processing import load_ai_scores

df = load_ai_scores()  # Returns DataFrame with 'zip' and 'ai_score' columns
```

### 3. Streamlit App (`app/app.py`)

The main app integrates GLM in multiple ways:

- **Auto-load**: Loads GLM scores on startup (if available)
- **Retrain Button**: Triggers training via `run_glm_training(force_retrain=True)`
- **Scoring Modes**: Blended (Math + GLM), Math Only, GLM Only
- **Display**: Shows GLM score in the rankings table

## Training the Model

### Command Line

```bash
# Train the model directly
cd /Users/aryaanverma/pharmacy-deserts/pharmacy-deserts
python data/finalfinalfinal_training.py

# Or use the wrapper
python data_processing/glm_training.py --force
```

### From Streamlit App

1. Open sidebar
2. Find "🧠 ML Model (GLM+Expected-Access)" section
3. Click "🔄 Retrain Model"
4. Wait for training to complete (~2-5 minutes depending on data size)
5. App will auto-reload with new results

## Scoring Modes Explained

### Blended (Math + GLM) [Recommended]

Combines mathematical and GLM scores:
- Mathematical model uses hand-crafted features and weights
- GLM provides data-driven expected access predictions
- Final score: uses GLM's IFAE_score (which is 50% composite + 50% glm_nb_score)
- Best of both worlds: interpretability + statistical rigor

### Math Only

Uses only the traditional formula:
- Scarcity (pharmacies per 10k population)
- Health burden (poor health %)
- Income (median household income, inverted)
- Population density
- Drive time (GoodRx data)
- Education attainment (HS or lower %)
- Optional: Heat vulnerability
- User-adjustable weights via sliders

### GLM Only

Pure machine learning approach:
- Uses only the IFAE_score from GLM model
- Based on deviance residuals (underserved score)
- Accounts for expected pharmacy counts given demographics
- Per-state calibration ensures regional fairness

## Key Features

### 1. **Automatic Training**
- Model trains automatically on first run if results don't exist
- Caches results for fast subsequent loads
- Manual retrain available via UI button

### 2. **Flexible Scoring Modes**
- **Blended (Math + GLM)**: Combines mathematical model with ML predictions (recommended)
- **Math Only**: Traditional weighted formula with user-adjustable sliders
- **GLM Only**: Pure ML predictions (deviance residual-based)

### 3. **Per-State Calibration**
- OOF predictions prevent overfitting to training data
- State-level calibration factors: observed/expected per state
- National fallback for states with insufficient data
- Urban-only calibration available for QA purposes

### 4. **Neighbor QA & Suspicious Zeros**
- Geographic kNN (Haversine distance on lon/lat) when ≥20% coverage
- Feature-space kNN fallback (income, health, log population)
- Computes median neighbor observed/expected rates per 10k
- Flags suspicious zeros: high pop + 0 pharmacies + high expected + normal neighbors
- Helps identify data quality issues and true outliers

### 5. **Robust Error Handling**
- Graceful fallbacks if ML model unavailable
- Still provides mathematical scoring as backup
- Clear error messages in UI
- Progress tracking with tqdm (in terminal)

### 6. **Deviance Residuals**
- More robust than Pearson residuals for zero counts
- Properly handles log-likelihood in Poisson/NB models
- Signed: positive for underserved, negative for over-served

## Troubleshooting

### "No trained model found"

**Solution**: Click "🔄 Retrain Model" in the sidebar, or run:
```bash
python data/finalfinalfinal_training.py
```

### "Training failed"

Check these common issues:
1. Missing data files in `data/` directory
2. Incompatible pandas/numpy versions
3. Insufficient memory (GLM needs ~2-4GB RAM)
4. Check console output for specific error

### "Low coverage" warnings

The model will warn if:
- AQI coverage < 30%: AQI feature disabled
- Heat coverage < 30%: Heat feature disabled
- Density coverage < 20%: Density spline disabled
- Geo coverage < 20%: Falls back to feature-space kNN

This is normal and handled automatically.

### Scores seem off

Try these steps:
1. **Retrain**: Data may have changed since last training
2. **Check data quality**: Look at `qa_expected_vs_observed.csv`
3. **Review suspicious zeros**: Flag may indicate data issues
4. **Compare scores**: Check both residual and deficit-based rankings
5. **Neighbor QA**: Look at median neighbor rates for validation

## Technical Details

### Model Assumptions

1. **Poisson GLM**: Pharmacy counts follow Poisson distribution with mean proportional to population
2. **Log-Link**: `log(E[pharmacies|X]) = log(population) + β₀ + β₁·X₁ + ... + spline(density)`
3. **NB Variance**: `Var[Y] = μ + α·μ²` (overdispersion parameter α estimated from residuals)
4. **Calibration**: State-level multiplicative adjustment to correct for regional bias

### Feature Importance

Typical GLM coefficient magnitudes (from `glm_full_coefficients.csv`):
- `income_pct_inv_c`: -0.3 to -0.5 (lower income → fewer pharmacies expected)
- `health_pct_c`: 0.2 to 0.4 (higher health burden → more pharmacies expected)
- `density spline`: nonlinear effect (urban areas have different pharmacy density patterns)
- `aqi_pct_c`: small effect (~0.1) if included
- `heat_pct_c`: small effect (~0.1) if included
- `C(REGION)`: varies by state (some states have 2-3x higher baseline pharmacy density)

### Neighbor QA Interpretation

- **Median neighbor obs rate**: What nearby ZIPs actually have (per 10k)
- **Median neighbor exp rate**: What model expects for nearby ZIPs (per 10k)
- If `obs_rate << exp_rate` for neighbors: area is generally underserved
- If `obs_rate > exp_rate` for neighbors: area is generally well-served
- Suspicious zero: ZIP has 0 pharmacies but neighbors are normal → potential data error or opportunity

## Data Requirements

### Required Files
- `data/financial_data.csv`: ZCTA5, S1901_C01_012E (median HH income)
- `data/health_data.csv`: ZCTA5, GHLTH_CrudePrev (poor health %)
- `data/pharmacy_data.csv`: ZIP, NAME (pharmacy name); optional: STATE, lon, lat
- `data/population_data.csv`: ZCTA5, population, land_area (with skiprows=10)

### Optional Files
- `data/AQI_data.csv`: ZIP, Arithmetic Mean, Observation Count (PM2.5)
- `data/HHI_data.xlsx`: ZCTA, HHB_SCORE (heat vulnerability)

### Column Flexibility

The model auto-detects many column name variations:
- ZCTA: `ZCTA5`, `ZCTA`, `ZIP`, `Zip`, `NAME`, `name`, `GEOID`
- Region: `STATE`, `state`, `STATEFP`, `statefp`, `county`, `COUNTY`
- Population: `POP`, `Population`, `population`, `TOTAL_POP`, `DP05_0001E`
- Density: `land_area_km2`, `Land_Area_km2`, `ALAND_KM2`, `aland_km2`
- Lon/Lat: `lon`/`lat`, `longitude`/`latitude`, `LON`/`LAT`, `X`/`Y`

## Next Steps

1. **Run Training**: Click "🔄 Retrain Model" or `python new_training.py`
2. **Review QA**: Check `results/qa_expected_vs_observed.csv` for sanity
3. **Compare Modes**: Try all 3 scoring modes to see differences
4. **Adjust Weights**: If using "Math Only", tune sliders for your priorities
5. **Export Results**: Use the built-in CSV download from the app

## Support

For issues or questions:
1. Check console output for detailed error messages
2. Review `results/qa_expected_vs_observed.csv` for data quality
3. Inspect `results/glm_full_coefficients.csv` for model details
4. Compare `national_ifae_rank.csv` vs `national_ifae_rank_alt_deficit.csv` for sensitivity
