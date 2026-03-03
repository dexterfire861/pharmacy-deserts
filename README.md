# Pharmacy Desert Explorer

A research-grade application that identifies and ranks **pharmacy deserts** — areas with inadequate pharmacy access — across the United States. It combines a statistical GLM (Poisson) model with a traditional mathematical scoring approach to surface underserved ZIP codes for policy analysis and pharmacy expansion planning.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Data Sources & Features](#data-sources--features)
- [Model Architecture & Design Decisions](#model-architecture--design-decisions)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Deployment](#deployment)
- [Further Documentation](#further-documentation)

## Overview

Not all communities have equal access to pharmacies. This project quantifies pharmacy access gaps using demographic, health, and geographic data for every ZIP code in the US. It produces a ranked list of the most underserved areas through three complementary scoring approaches:

| Mode | Description | Best For |
|------|-------------|----------|
| **GLM Only** (recommended) | Pure statistical model — Poisson GLM with per-state calibration | Unbiased discovery of underserved areas |
| **Math Only** | Weighted formula with user-adjustable sliders | Stakeholder presentations, transparent prioritization |
| **Blended** | 50% Math + 50% GLM | Balancing interpretability with statistical rigor |

## Key Features

- **Hybrid AI + Mathematical scoring** with three selectable modes
- **Interactive Streamlit dashboard** with adjustable weight sliders
- **Folium map visualization** of top pharmacy deserts with pharmacy popups
- **Per-state calibration** to account for regional density differences
- **Neighbor QA** using geographic kNN (Haversine) to validate predictions
- **Suspicious zero detection** to flag potential data quality issues
- **CSV export** of full results or filtered top-K rankings
- **Docker deployment** with AWS S3 integration for production use
- **Password authentication** for access control

## Project Structure

```
pharmacy-deserts/
├── app/                        # Streamlit web application
│   ├── app.py                  #   Main UI (scoring modes, sliders, map)
│   ├── state.py                #   Cached data loaders (S3/local)
│   ├── config.py               #   Configuration management (env vars)
│   └── auth.py                 #   Password authentication
├── data/                       # Data files and training pipeline
│   ├── finalfinalfinal_training.py  #   GLM training script
│   ├── features.py             #   Preprocessing (normalization, percentiles)
│   ├── loaders.py              #   CSV/Excel loaders with auto-detection
│   ├── s3_loaders.py           #   AWS S3 data backend
│   ├── financial_data.csv      #   Median household income by ZCTA
│   ├── health_data.csv         #   Poor health prevalence by ZCTA
│   └── *.ipynb                 #   NPI data loading notebooks
├── models/                     # Scoring logic
│   ├── scoring.py              #   Mathematical scoring functions
│   └── ai_scores.py            #   IFAE score loading
├── viz/                        # Visualization
│   └── map_viz.py              #   Folium map rendering
├── utils/                      # Utilities
│   └── cache.py                #   Streamlit cache wrapper
├── results/                    # Model output files
│   ├── national_ifae_rank.csv  #   Main rankings (residual-based)
│   ├── national_ifae_rank_alt_deficit.csv  # Alt rankings (deficit-based)
│   ├── qa_expected_vs_observed.csv  #   QA with neighbor metrics
│   └── glm_full_coefficients.csv   #   Model coefficients with SE/z-scores
├── deploy/                     # Deployment configuration
│   ├── docker-compose.yml      #   Multi-service orchestration
│   ├── ec2-setup.sh            #   AWS EC2 setup automation
│   └── nginx.conf              #   Reverse proxy config (HTTPS)
├── Dockerfile                  # Container build definition
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration
├── QUICKSTART.md               # Quick start guide
├── INTEGRATION_GUIDE.md        # Detailed GLM model documentation
├── DEBUGGING_GEOGRAPHIC_BIAS.md  # Known bias issues and fixes
└── MODEL_INTEGRATION_SUCCESS.md  # Model verification results
```

## Data Sources & Features

The model ingests data from multiple public sources and merges them at the ZIP code (ZCTA) level:

| Feature | Source | Coverage | Role in Model |
|---------|--------|----------|---------------|
| **Pharmacy Counts** | NPI registry (61,970 pharmacies) | 14,940 ZIPs | Target variable (Poisson count) |
| **Population & Density** | ACS Census | ~41,000 ZIPs | Exposure term; B-spline predictor |
| **Median Household Income** | ACS Census API | ~33,800 ZIPs (97%) | Predictor (inverted — lower income → higher need) |
| **Health Burden** | CDC BRFSS (poor general health %) | ~32,500 ZIPs | Predictor |
| **Heat Vulnerability** | Heat Health Index (HHI) | ~32,100 ZIPs (78%) | Optional predictor |
| **Air Quality (PM2.5)** | EPA annual average | ~125 ZIPs (0.3%) | Disabled — insufficient coverage |
| **Drive Time** | GoodRx pharmacy access data | County-level | Optional filter (off by default to avoid geographic bias) |
| **Education** | ACS Census (% HS or lower) | Variable | Math model weight |

All continuous features are converted to **percentile ranks** (0–1) before modeling. This is more robust to outliers than raw values or z-scores.

## Model Architecture & Design Decisions

### Why a Hybrid Approach?

A single model cannot satisfy all stakeholders. Policy makers need **transparency** (adjustable weights, explainable scores), while researchers need **statistical validity** (unbiased predictions, proper uncertainty). The hybrid approach provides both:

- The **mathematical model** is fully transparent — stakeholders see exactly how weights combine.
- The **GLM model** is statistically rigorous — it finds underserved areas even where pharmacies exist but are insufficient for the population.

### GLM Pipeline (Statistical Model)

The core statistical model follows a research-grade pipeline:

1. **Poisson GLM with population exposure**
   - `log(E[pharmacies | X]) = log(population) + β₀ + β₁·income + β₂·health + spline(density)`
   - Chosen over linear regression because pharmacy counts are non-negative integers.

2. **Out-of-Fold (OOF) cross-validation** (GroupKFold by state)
   - Prevents training-set leakage and produces unbiased expected counts.
   - Groups by state to avoid data leakage from geographically adjacent ZIPs.

3. **Negative Binomial variance estimation**
   - Real pharmacy counts are overdispersed relative to Poisson. The NB alpha parameter (`Var = μ + α·μ²`) is estimated from OOF residuals for realistic uncertainty.

4. **Per-state calibration**
   - Computes `state_cal = Σ(observed) / Σ(expected)` for each state.
   - Corrects for regional differences in pharmacy density norms.
   - Falls back to national calibration for states with insufficient data.

5. **GBDT/XGBoost residual refinement**
   - A gradient-boosted model is trained on GLM residuals to capture non-linear patterns the GLM misses.
   - The hybrid score blends 50% residual-based + 50% composite demographic score.

6. **Deviance residuals** (instead of Pearson)
   - More robust for count models with many zeros.
   - Signed: positive = underserved, negative = over-served.

7. **Neighbor QA**
   - Geographic kNN (Haversine distance) when ≥20% of ZIPs have coordinates.
   - Falls back to feature-space kNN (income, health, log population).
   - Validates predictions against similar nearby areas.

8. **Suspicious zero detection**
   - Flags ZIPs with: population ≥ 20k, 0 pharmacies, expected count ≥ 3, and neighbors with ≥ 1 pharmacy per 10k.
   - Helps distinguish true deserts from data quality issues.

### Mathematical Model (Interpretable)

A weighted linear combination of seven feature percentiles:

- Pharmacy scarcity, health burden, income (inverted), population density, drive time, education, heat vulnerability
- Weights are user-adjustable via sliders and auto-normalize to 100%.

### Final Scoring

| Score | Formula | Use Case |
|-------|---------|----------|
| `IFAE_score_residual` | 50% composite + 50% GLM NB score | Main ranking (recommended) |
| `IFAE_score_deficit` | 50% composite + 50% deficit rate | Alternative ranking |
| `score_math` | Weighted sum of feature percentiles | Transparent, adjustable |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Percentile normalization over z-scores | Robust to skewed distributions and outliers |
| GroupKFold by state for CV | Prevents geographic leakage between train/test folds |
| Per-state calibration | States have structurally different pharmacy densities |
| GoodRx gate OFF by default | [GoodRx data introduced geographic bias](DEBUGGING_GEOGRAPHIC_BIAS.md) |
| GLM Only as default mode | Produces the most geographically diverse, unbiased results |
| Deviance over Pearson residuals | Better behaved for zero-inflated count data |
| Air quality feature disabled | Only 0.3% ZIP coverage — insufficient for reliable predictions |

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Local Installation

```bash
# Clone the repository
git clone https://github.com/dexterfire861/pharmacy-deserts.git
cd pharmacy-deserts

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

Generate the GLM scores before running the app:

```bash
python data/finalfinalfinal_training.py
```

This produces output files in `results/` (takes approximately 15–30 seconds).

### Launch the App

```bash
streamlit run app/app.py
```

The app opens in your browser at `http://localhost:8501`.

### Docker

```bash
# Build
docker build -t pharmacy-deserts .

# Run
docker run -p 8501:8501 --env-file .env pharmacy-deserts
```

## Usage

### Scoring Modes

Select a scoring mode from the sidebar:

- **GLM Only** — Uses the pre-computed IFAE score from the Poisson GLM. Best for unbiased national analysis.
- **Math Only** — Adjust feature weights with sliders to explore different prioritization strategies.
- **Blended** — Combines both models for a balanced ranking.

### Retraining

Click **🔄 Retrain Model** in the sidebar to retrain the GLM on current data, or run the training script directly:

```bash
python data/finalfinalfinal_training.py
```

### Exporting Results

The app provides CSV download buttons for the full results table and filtered top-K rankings.

### Model Outputs

After training, the `results/` directory contains:

| File | Contents |
|------|----------|
| `national_ifae_rank.csv` | Main rankings sorted by IFAE score (residual-based) |
| `national_ifae_rank_alt_deficit.csv` | Alternative rankings (deficit-based) |
| `qa_expected_vs_observed.csv` | QA metrics with neighbor validation and suspicious zero flags |
| `glm_full_coefficients.csv` | GLM coefficients with standard errors and z-scores |

## Deployment

### Docker Compose

```bash
cd deploy
docker-compose up -d
```

### AWS EC2

The `deploy/ec2-setup.sh` script automates EC2 instance setup. Configure your S3 bucket and credentials in `.env` before deploying.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | `production` or `development` | `production` |
| `AWS_S3_BUCKET` | S3 bucket for data files | `pharmacy-deserts-data` |
| `AWS_REGION` | AWS region | `us-east-1` |
| `REQUIRE_AUTH` | Enable password authentication | `true` |
| `APP_PASSWORD` | Password for app access | — |
| `DATA_DIR` | Data file directory | `raw_data` |
| `RESULTS_DIR` | Results output directory | `results` |

## Further Documentation

- **[QUICKSTART.md](QUICKSTART.md)** — Step-by-step setup and demo guide
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** — Detailed GLM model architecture and training pipeline
- **[MODEL_INTEGRATION_SUCCESS.md](MODEL_INTEGRATION_SUCCESS.md)** — Model verification results and top predictions
- **[DEBUGGING_GEOGRAPHIC_BIAS.md](DEBUGGING_GEOGRAPHIC_BIAS.md)** — Known geographic bias issues and how they were resolved
