"""
PHARMACY PROFIT MAXIMIZATION MODEL — v2
========================================
Economic-first approach: Profit = Revenue Potential × Capture Rate − Cost Pressure

Three pillars replace the old magic-number weighted percentiles:

  1. REVENUE POTENTIAL INDEX  — How much Rx + retail revenue can this ZCTA generate?
     Components: population-adjusted Rx demand, chronic disease Rx intensity,
     direct medication usage signals (BPMED), payer mix reimbursement quality

  2. COST PRESSURE INDEX — How expensive is it to operate a pharmacy here?
     Components: occupancy (rent/home price proxy), labor cost (state-level),
     urban premium factor

  3. COMPETITIVE CAPTURE RATE — What share of revenue can one pharmacy realistically get?
     Components: pharmacies per capita (saturation), chain vs independent mix,
     market concentration (HHI proxy)

Final:  profit_score = Revenue_Potential × Capture_Rate − Cost_Pressure
        (all normalized to [0, 1] before combination)

UI SLIDER MAPPING (for your dashboard):
  ┌─────────────────────────┬────────────────────────────────────┐
  │ Old Slider              │ New Slider (v2)                    │
  ├─────────────────────────┼────────────────────────────────────┤
  │ Health burden            │ Rx Demand Intensity               │
  │ Income                   │ Payer Quality / Reimbursement     │
  │ Scarcity                 │ Market Opportunity (1/saturation) │
  │ Drive Time               │ (removed — no drive-time data)    │
  │ Population Density       │ Market Size                       │
  │ Education                │ Cost Pressure (inverted)          │
  │ (new)                    │ Revenue vs Cost Balance (α)       │
  └─────────────────────────┴────────────────────────────────────┘

Usage:
  python profit_model_v2.py                          # run with defaults
  python profit_model_v2.py --out results_v2         # custom output dir
  python profit_model_v2.py --top-k 25               # top 25 locations

Inputs (same files as v1 — drop-in compatible):
  data/financial_data.csv   — ACS S1901 (income)
  data/health_data.csv      — CDC PLACES (ZCTA level)
  data/population_data.csv  — FOURFRONT or ACS population+density
  data/insurance.csv        — ACS S2703 (private coverage)
  data/houseprice.csv       — Zillow/other home price by ZIP
  data/pharmacy_data.csv    — NPI pharmacy locations (or Excel bundles)
  data/rent.csv             — (optional) median rent by ZCTA

Outputs:
  results_v2/profit_scores.csv              — full ZCTA-level scores
  results_v2/top_opportunities.csv          — top K by profit score
  results_v2/component_diagnostics.csv      — pillar-level breakdowns
  results_v2/slider_config.json             — default slider weights for UI
  results_v2/model_summary.json             — run metadata + statistics
"""

import argparse
import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Optional: statsmodels for GLM baseline (not required for v2 scoring)
try:
    import statsmodels.api as sm
    _SM_OK = True
except ImportError:
    _SM_OK = False

# ---------------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    financial_csv  = "data/financial_data.csv",
    health_csv     = "data/health_data.csv",
    population_csv = "data/population_data.csv",
    insurance_csv  = "data/insurance.csv",
    houseprice_csv = "data/houseprice.csv",
    pharmacy_csv_candidates = [
        "data/pharmacy_data.csv",
        "data/pharmacy.csv",
    ],
    rent_csv_candidates = [
        "data/rent.csv",
        "data/rent_data.csv",
        "data/zcta_rent.csv",
    ],
    excel_globs = [
        "data/**/subset*_Table*_filter.xlsx",
    ],
    out_dir  = "results_v2",
    top_k    = 10,
    min_pop  = 500,      # minimum population to score
)

# Default slider weights — these are what the UI sends / user adjusts
# Each weight scales [0.0, 2.0] with 1.0 = neutral
DEFAULT_SLIDER_WEIGHTS = {
    # Revenue sub-components
    "w_rx_demand":        1.0,   # chronic disease Rx intensity
    "w_payer_quality":    1.3,   # reimbursement quality of payer mix (boosted: Medicaid vs commercial matters)
    "w_market_size":      1.5,   # population / density (boosted: volume matters for profitability)
    "w_med_usage":        1.0,   # direct medication usage signals (BPMED etc)

    # Cost sub-components
    "w_occupancy_cost":   1.0,   # rent / home price
    "w_labor_cost":       1.0,   # state-level wage proxy

    # Capture rate sub-components
    "w_competition":      1.0,   # pharmacies per capita (higher = worse)
    "w_market_opportunity": 1.0, # inverse of saturation (scarcity bonus)

    # Master balance
    "alpha_revenue_vs_cost": 0.65,  # how much weight on revenue vs cost
}

# Chronic disease weights by prescription intensity
# (avg annual Rx per patient with condition — rough industry estimates)
RX_INTENSITY_WEIGHTS = {
    "DIABETES_CrudePrev":   12.0,   # ~12 Rx/yr (insulin, metformin, strips, etc.)
    "BPHIGH_CrudePrev":      8.0,   # ~8 Rx/yr (ACE inhibitors, ARBs, diuretics)
    "HIGHCHOL_CrudePrev":    4.0,   # ~4 Rx/yr (statins)
    "COPD_CrudePrev":       10.0,   # ~10 Rx/yr (inhalers, steroids, antibiotics)
    "ARTHRITIS_CrudePrev":   6.0,   # ~6 Rx/yr (NSAIDs, DMARDs)
    "CASTHMA_CrudePrev":     6.0,   # ~6 Rx/yr (inhalers, leukotriene modifiers)
    "DEPRESSION_CrudePrev":  4.0,   # ~4 Rx/yr (SSRIs, SNRIs)
    "OBESITY_CrudePrev":     2.0,   # ~2 Rx/yr (emerging GLP-1 class)
}

# Payer reimbursement quality multipliers (avg reimbursement per Rx, indexed)
# Source: industry benchmarks, CMS data
PAYER_REIMBURSEMENT = {
    "employer":      1.00,   # baseline — best reimbursement
    "direct":        0.85,   # individual market, slightly lower
    "tricare":       0.75,   # government rate, decent
    "medicare":      0.65,   # Part D, below commercial
    "medicaid":      0.45,   # lowest reimbursement
    "marketplace":   0.70,   # ACA exchange, moderate
    "uninsured":     0.30,   # cash pay, heavy discounts, bad debt risk
}

# State-level relative labor cost index (BLS Occupational Employment data)
# Indexed to national median = 1.00; pharmacist + tech wages
# This is a simplified version — full version would use BLS OEWS data
STATE_LABOR_INDEX = {
    "CA": 1.25, "NY": 1.22, "MA": 1.20, "CT": 1.18, "NJ": 1.17,
    "WA": 1.16, "DC": 1.25, "AK": 1.20, "HI": 1.15, "MD": 1.12,
    "OR": 1.10, "CO": 1.08, "MN": 1.06, "IL": 1.05, "VA": 1.04,
    "NH": 1.03, "RI": 1.03, "VT": 1.02, "DE": 1.02, "PA": 1.00,
    "WI": 0.98, "MI": 0.97, "OH": 0.96, "FL": 0.95, "TX": 0.95,
    "AZ": 0.95, "NV": 0.98, "GA": 0.93, "NC": 0.93, "UT": 0.95,
    "NE": 0.92, "IA": 0.91, "KS": 0.91, "MO": 0.90, "IN": 0.92,
    "TN": 0.90, "SC": 0.90, "KY": 0.89, "LA": 0.89, "OK": 0.88,
    "AL": 0.88, "AR": 0.87, "NM": 0.90, "ID": 0.90, "MT": 0.92,
    "ND": 0.93, "SD": 0.90, "WY": 0.93, "WV": 0.87, "MS": 0.85,
    "ME": 0.95, "PR": 0.70,
}

# ---------------------------------------------------------------------------
#  UTILITIES (carried forward from v1 with fixes)
# ---------------------------------------------------------------------------

_T0 = time.time()

def stamp(msg):
    now = datetime.now().strftime("%H:%M:%S")
    elapsed = time.time() - _T0
    print(f"[{now} +{elapsed:7.1f}s]  {msg}", flush=True)

def detect_sep(path):
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()
    return "\t" if head.count("\t") > head.count(",") else ","

def read_csv(path, **kw):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing: {path}")
    if "sep" not in kw:
        kw["sep"] = detect_sep(str(path))
    return pd.read_csv(str(path), low_memory=False, encoding="utf-8-sig", **kw)

def coerce_zcta(series):
    """Extract 3-5 digit ZIP and zero-pad to 5."""
    s = series.astype(str).str.extract(r"(\d{3,5})", expand=False)
    return s.fillna("").str.zfill(5)

def pick_col(df, *candidates):
    """Return first matching column name or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def to_num(s):
    """Coerce to numeric, treating (X) and blanks as NaN."""
    if isinstance(s, pd.Series):
        return pd.to_numeric(
            s.replace({"(X)": np.nan, "X": np.nan, "-": np.nan, "": np.nan}),
            errors="coerce"
        )
    return pd.to_numeric(s, errors="coerce")

def pct_rank(s):
    """Percentile rank in [0, 1]. NaN → 0.5."""
    return to_num(s).rank(pct=True, method="average").fillna(0.5)

def norm01(s):
    """Min-max normalize to [0, 1]."""
    s = pd.Series(s, dtype=float)
    lo = np.nanmin(s.values) if np.isfinite(s.values).any() else 0.0
    hi = np.nanmax(s.values) if np.isfinite(s.values).any() else 1.0
    return ((s - lo) / (hi - lo + 1e-12)).clip(0.0, 1.0)

def weighted_mean(values_and_weights):
    """Compute weighted mean from [(value_series, weight), ...]."""
    total_w = sum(w for _, w in values_and_weights if w > 0)
    if total_w == 0:
        return pd.Series(0.5, index=values_and_weights[0][0].index)
    result = sum(v.fillna(0.5) * w for v, w in values_and_weights if w > 0) / total_w
    return result.clip(0.0, 1.0)


# State inference from ZIP3 prefix
ZIP3_RANGES = [
    (5, 5, "NY"), (6, 9, "PR"),
    (10, 27, "MA"), (28, 29, "RI"), (30, 38, "NH"), (39, 49, "ME"),
    (50, 59, "VT"), (60, 69, "CT"), (70, 89, "NJ"),
    (100, 149, "NY"), (150, 196, "PA"), (197, 199, "DE"),
    (200, 205, "DC"), (206, 219, "MD"), (220, 246, "VA"), (247, 268, "WV"),
    (270, 289, "NC"), (290, 299, "SC"), (300, 319, "GA"), (320, 349, "FL"),
    (350, 369, "AL"), (370, 385, "TN"), (386, 397, "MS"), (398, 399, "GA"),
    (400, 427, "KY"), (430, 459, "OH"), (460, 479, "IN"), (480, 499, "MI"),
    (500, 528, "IA"), (530, 549, "WI"), (550, 567, "MN"), (570, 577, "SD"),
    (580, 588, "ND"), (590, 599, "MT"), (600, 629, "IL"), (630, 658, "MO"),
    (660, 679, "KS"), (680, 693, "NE"), (700, 715, "LA"), (716, 729, "AR"),
    (730, 749, "OK"), (750, 799, "TX"), (800, 816, "CO"), (820, 831, "WY"),
    (832, 838, "ID"), (840, 847, "UT"), (850, 865, "AZ"), (870, 884, "NM"),
    (885, 885, "TX"), (889, 898, "NV"), (900, 966, "CA"), (967, 968, "HI"),
    (970, 979, "OR"), (980, 994, "WA"), (995, 999, "AK"),
]

def infer_state(zcta5):
    s = str(zcta5).strip()
    if not re.fullmatch(r"\d{5}", s):
        return np.nan
    z3 = int(s[:3])
    for lo, hi, st in ZIP3_RANGES:
        if lo <= z3 <= hi:
            return st
    return np.nan


# ---------------------------------------------------------------------------
#  DATA LOADERS
# ---------------------------------------------------------------------------

def load_financial(path):
    """Load ACS S1901 income data."""
    df = read_csv(path)
    zc = pick_col(df, "ZCTA5", "zcta5", "ZCTA", "ZIP", "Zip", "NAME", "name")
    if zc is None:
        raise KeyError("financial_data.csv: no ZCTA column found")
    df["ZCTA5"] = coerce_zcta(df[zc])

    out = pd.DataFrame({"ZCTA5": df["ZCTA5"]})

    # Median household income
    if "S1901_C01_012E" in df.columns:
        out["median_income"] = to_num(df["S1901_C01_012E"])
    else:
        raise KeyError("financial_data.csv missing S1901_C01_012E (median income)")

    # Mean income (optional)
    if "S1901_C01_013E" in df.columns:
        out["mean_income"] = to_num(df["S1901_C01_013E"])

    # Households
    if "S1901_C01_001E" in df.columns:
        out["households"] = to_num(df["S1901_C01_001E"])

    # High-income brackets ($100K+, $150K+, $200K+)
    hi_cols = [c for c in ["S1901_C01_009E", "S1901_C01_010E", "S1901_C01_011E"]
               if c in df.columns]
    if hi_cols:
        hi_sum = sum(to_num(df[c]).fillna(0) for c in hi_cols)
        out["high_income_hh_count"] = hi_sum

    out = out.dropna(subset=["ZCTA5"]).drop_duplicates(subset=["ZCTA5"])
    stamp(f"FINANCIAL: {len(out):,} ZCTAs, median_income null={out['median_income'].isna().sum():,}")
    return out


def load_health(path):
    """Load CDC PLACES data — extract ALL useful measures, not just 6."""
    df = read_csv(path)
    zc = pick_col(df, "ZCTA5", "ZCTA", "ZIP", "Zip", "NAME", "name")
    if zc is None:
        raise KeyError("health_data.csv: no ZCTA column found")
    df["ZCTA5"] = coerce_zcta(df[zc])

    out = pd.DataFrame({"ZCTA5": df["ZCTA5"]})

    # General health
    for col in ["GHLTH_CrudePrev", "PHLTH_CrudePrev", "MHLTH_CrudePrev"]:
        if col in df.columns:
            out[col] = to_num(df[col])

    # Chronic disease prevalence (for Rx demand estimation)
    chronic_cols = []
    for col in RX_INTENSITY_WEIGHTS.keys():
        if col in df.columns:
            out[col] = to_num(df[col])
            chronic_cols.append(col)

    # Direct medication usage signals (gold for pharmacy demand)
    for col in ["BPMED_CrudePrev", "CHOLSCREEN_CrudePrev", "CHECKUP_CrudePrev"]:
        if col in df.columns:
            out[col] = to_num(df[col])

    # Disability measures (higher pharmacy utilization)
    for col in ["DISABILITY_CrudePrev", "MOBILITY_CrudePrev"]:
        if col in df.columns:
            out[col] = to_num(df[col])

    # Behavioral (moderate Rx relevance)
    for col in ["OBESITY_CrudePrev", "CSMOKING_CrudePrev", "SLEEP_CrudePrev"]:
        if col in df.columns:
            out[col] = to_num(df[col])

    # Population counts from PLACES
    for col in ["TotalPopulation", "TotalPop18plus"]:
        if col in df.columns:
            out[f"places_{col}"] = to_num(df[col])

    out = out.dropna(subset=["ZCTA5"]).drop_duplicates(subset=["ZCTA5"])
    stamp(f"HEALTH: {len(out):,} ZCTAs, {len(chronic_cols)} chronic disease columns found")
    return out, chronic_cols


def load_population(path):
    """Load population + density. Handles FOURFRONT format (skiprows)."""
    # Try FOURFRONT format (has header junk)
    try:
        df = read_csv(path, skiprows=10)
        if df.shape[1] < 3:
            raise ValueError("too few cols")
    except Exception:
        df = read_csv(path)

    zc = pick_col(df, "ZCTA5", "ZCTA", "ZIP", "Zip", "GEOID", "geoid", "NAME", "name")
    if zc is None:
        raise KeyError("population_data.csv: no ZCTA column found")
    df["ZCTA5"] = coerce_zcta(df[zc])

    out = pd.DataFrame({"ZCTA5": df["ZCTA5"]})

    # Population
    pop_col = pick_col(df, "population", "Population", "POP", "TOTAL_POP",
                       "TotPop", "DP05_0001E", "pop")
    if pop_col is None:
        # Guess: largest numeric column
        num_cols = [c for c in df.columns if c != "ZCTA5"]
        sums = {c: to_num(df[c]).sum() for c in num_cols}
        pop_col = max(sums, key=sums.get) if sums else None
    if pop_col:
        out["population"] = to_num(df[pop_col])

    # Density — FOURFRONT has it directly!
    dens_col = pick_col(df, "density", "Density", "pop_density", "DENSITY")
    if dens_col:
        out["pop_density"] = to_num(df[dens_col])
        stamp("  → Using direct density column from population data")

    # Land area fallback
    land_col = pick_col(df, "land_area_km2", "Land_Area_km2", "ALAND_KM2",
                        "aland_km2", "ALAND", "area", "AREA_KM2")
    if land_col and "pop_density" not in out.columns:
        area = to_num(df[land_col])
        # Check if in sq meters (ALAND from Census is in sq meters)
        med = area.median()
        if med > 1e5:
            area = area / 1e6  # convert sq m to sq km
        out["land_area_km2"] = area
        out["pop_density"] = np.where(area > 0, out["population"] / area, np.nan)

    # Lat/lon from FOURFRONT
    for coord in [("lat", "Lat", "LAT", "latitude"), ("long", "Lon", "LON", "longitude", "lng")]:
        cc = pick_col(df, *coord)
        if cc:
            out[coord[0] if coord[0] in ("lat",) else "lon"] = to_num(df[cc])

    # City/State from FOURFRONT
    st_col = pick_col(df, "St", "STATE", "State", "state")
    if st_col:
        out["state_from_pop"] = df[st_col].astype(str).str.strip().str.upper()

    out = out.dropna(subset=["ZCTA5"]).drop_duplicates(subset=["ZCTA5"])
    has_density = out["pop_density"].notna().sum() if "pop_density" in out.columns else 0
    stamp(f"POPULATION: {len(out):,} ZCTAs, density available for {has_density:,}")
    return out


def _nrm(s):
    return re.sub(r"\s+", " ", str(s).replace("\ufeff", "").strip().lower())

def _pick_acs_code(label_map, must=(), forbid=(), prefer=()):
    """Find ACS column code by searching label descriptions."""
    if label_map is None:
        return None
    must = [_nrm(x) for x in must]
    forbid = [_nrm(x) for x in forbid]
    prefer = [_nrm(x) for x in prefer]
    best, best_score = None, -1
    for code, lab in label_map.items():
        ln = _nrm(lab)
        if all(m in ln for m in must) and not any(f in ln for f in forbid):
            score = sum(1 for p in prefer if p in ln)
            if score > best_score:
                best_score, best = score, code
    return best


def load_insurance(path):
    """Parse ACS S2703 code-export for payer mix features."""
    df_raw = read_csv(path, dtype=str)
    df = df_raw.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    # Detect label row
    if df.shape[0] == 0:
        raise ValueError("insurance.csv is empty")

    first_row = [_nrm(str(df.iloc[0][c])) for c in df.columns[:4]]
    has_label_row = "geography" in first_row or "geographic area name" in " ".join(first_row)
    if not has_label_row:
        raise KeyError("insurance.csv doesn't look like ACS code export")

    label_map = {c: str(df.iloc[0][c]) for c in df.columns}
    df = df.iloc[1:].reset_index(drop=True)

    if "NAME" not in df.columns:
        raise KeyError("insurance.csv missing NAME column")

    df["ZCTA5"] = coerce_zcta(df["NAME"])
    df = df[df["ZCTA5"].str.match(r"^\d{5}$", na=False)].copy()

    def get_pct(must_kw, prefer_kw=()):
        code = _pick_acs_code(label_map, must=must_kw,
                              forbid=("margin of error",), prefer=prefer_kw)
        if code and code in df.columns:
            return (to_num(df[code]) / 100.0).clip(0, 1)
        return pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({"ZCTA5": df["ZCTA5"]})

    # Total population
    total_code = _pick_acs_code(
        label_map,
        must=("estimate", "total", "civilian noninstitutionalized population"),
        forbid=("margin of error", "coverage", "percent")
    )
    if total_code is None:
        total_code = "S2703_C01_001E" if "S2703_C01_001E" in df.columns else None
    if total_code:
        out["ins_total_pop"] = to_num(df[total_code]).clip(lower=1)

    # Payer shares (percent columns)
    out["ins_employer_pct"] = get_pct(
        ("percent", "employer-based health insurance alone or in combination"),
        ("percent private coverage",))

    out["ins_direct_pct"] = get_pct(
        ("percent", "direct-purchase health insurance alone or in combination"),
        ("percent private coverage",))

    out["ins_tricare_pct"] = get_pct(
        ("percent", "tricare"),
        ("percent private coverage",))

    out["ins_marketplace_pct"] = get_pct(
        ("percent", "subsidized", "market", "place", "coverage", "alone"),
        ("percent private coverage",))

    # Private coverage total
    out["ins_private_pct"] = get_pct(
        ("percent private coverage", "civilian noninstitutionalized population"),
        ("coverage alone or in combination",))

    # Try to get below-poverty private coverage rate
    out["ins_below_poverty_private_pct"] = get_pct(
        ("percent", "below 138 percent", "poverty"),
        ("percent private coverage",))

    out = out.dropna(subset=["ZCTA5"]).drop_duplicates(subset=["ZCTA5"])
    stamp(f"INSURANCE: {len(out):,} ZCTAs")
    return out


def load_houseprice(path):
    """Load home price data."""
    df = read_csv(path, dtype=str)
    zc = pick_col(df, "Zipcode", "ZIP", "zip", "ZCTA5", "zipcode")
    if zc is None:
        raise KeyError("houseprice.csv missing ZIP column")
    df["ZCTA5"] = coerce_zcta(df[zc])

    pc = pick_col(df, "Price", "price")
    if pc is None:
        raise KeyError("houseprice.csv missing Price column")

    df["home_price"] = to_num(df[pc].astype(str).str.replace(r"[^0-9.]", "", regex=True))
    out = df[["ZCTA5", "home_price"]].dropna(subset=["ZCTA5"]).drop_duplicates(subset=["ZCTA5"])
    stamp(f"HOUSEPRICE: {len(out):,} ZCTAs")
    return out


def load_rent(candidates):
    """Load optional rent data."""
    for path in candidates:
        if Path(path).exists():
            df = read_csv(path)
            zc = pick_col(df, "ZCTA5", "ZCTA", "ZIP", "zip", "Zip", "GEOID")
            if zc is None:
                continue
            df["ZCTA5"] = coerce_zcta(df[zc])
            # Find rent column
            rc = pick_col(df, "rent", "median_rent", "Rent", "MedianRent",
                         "DP04_0134E", "S2506_C01_001E")
            if rc is None:
                # guess most-populated numeric col
                for c in df.columns:
                    if c != "ZCTA5" and to_num(df[c]).notna().sum() > 100:
                        rc = c
                        break
            if rc is None:
                continue
            out = df[["ZCTA5"]].copy()
            out["rent"] = to_num(df[rc])
            out = out.dropna(subset=["ZCTA5"]).drop_duplicates(subset=["ZCTA5"])
            stamp(f"RENT: {len(out):,} ZCTAs from {path}")
            return out
    return None


def load_pharmacy(csv_candidates, excel_globs):
    """Load pharmacy counts per ZCTA. Try CSV first, then Excel bundles."""
    from glob import glob

    # Try CSV
    for path in csv_candidates:
        if Path(path).exists():
            df = read_csv(path)
            zc = pick_col(df, "ZCTA5", "ZCTA", "ZIP", "zip", "Zip", "GEOID", "short_ZIP", "Short_ZIP")
            if zc is None:
                continue
            df["ZCTA5"] = coerce_zcta(df[zc])

            # Count pharmacies per ZCTA
            if "location_id" in df.columns or "NPI" in df.columns:
                id_col = "location_id" if "location_id" in df.columns else "NPI"
                ph_cnt = (df.groupby("ZCTA5")[id_col].nunique()
                          .reset_index().rename(columns={id_col: "pharmacies_count"}))
            else:
                ph_cnt = (df.groupby("ZCTA5").size()
                          .reset_index(name="pharmacies_count"))

            # Try to get chain identity
            name_col = pick_col(df, "Provider Organization Name (Legal Business Name)",
                               "ORG_NAME", "org_name", "name", "Name")
            chain_info = None
            if name_col:
                chain_info = _tag_chains(df, name_col)

            stamp(f"PHARMACY (CSV): {ph_cnt['pharmacies_count'].sum():,} sites in {len(ph_cnt):,} ZCTAs")
            return ph_cnt, chain_info

    # Try Excel bundles
    all_files = []
    for g in excel_globs:
        all_files.extend(glob(g, recursive=True))

    if all_files:
        frames = []
        for fp in all_files:
            try:
                xl = pd.ExcelFile(fp)
                sheet = xl.sheet_names[0]
                x = xl.parse(sheet_name=sheet, dtype=str)
                frames.append(x)
            except Exception:
                continue

        if frames:
            df = pd.concat(frames, ignore_index=True)
            zc = pick_col(df, "Short_ZIP",
                         "Provider Business Practice Location Address Postal Code",
                         "ZIP", "Zip")
            if zc:
                df["ZCTA5"] = coerce_zcta(df[zc])
                # Filter to pharmacy taxonomy codes
                tax_cols = [c for c in df.columns if "taxonomy" in c.lower() and "code" in c.lower()]
                if tax_cols:
                    mask = df[tax_cols].apply(
                        lambda row: any(str(v).startswith("3336") for v in row if pd.notna(v)),
                        axis=1
                    )
                    df = df[mask]

                npi_col = pick_col(df, "NPI", "npi")
                if npi_col:
                    ph_cnt = (df.groupby("ZCTA5")[npi_col].nunique()
                              .reset_index().rename(columns={npi_col: "pharmacies_count"}))
                else:
                    ph_cnt = df.groupby("ZCTA5").size().reset_index(name="pharmacies_count")

                stamp(f"PHARMACY (Excel): {ph_cnt['pharmacies_count'].sum():,} sites in {len(ph_cnt):,} ZCTAs")
                return ph_cnt, None

    raise FileNotFoundError("No pharmacy data found")


def _tag_chains(df, name_col):
    """Tag pharmacy chains from organization name. Returns ZCTA-level chain counts."""
    CHAIN_PATTERNS = {
        "WALGREENS":  [r"WALGREEN", r"DUANE\s*READE"],
        "CVS":        [r"\bCVS\b", r"CAREMARK", r"TARGET\s+PHARMACY", r"OMNICARE"],
        "WALMART":    [r"WAL\s*MART", r"SAMS?\s*CLUB"],
        "RITE_AID":   [r"RITE\s*AID"],
        "KROGER":     [r"KROGER", r"HARRIS\s*TEETER", r"FRED\s*MEYER", r"RALPH"],
        "COSTCO":     [r"COSTCO"],
        "PUBLIX":     [r"PUBLIX"],
    }

    names = df[name_col].fillna("").str.upper()
    df_work = df[["ZCTA5"]].copy()

    for chain, patterns in CHAIN_PATTERNS.items():
        mask = pd.Series(False, index=df.index)
        for pat in patterns:
            mask = mask | names.str.contains(pat, regex=True, na=False)
        df_work[chain] = mask.astype(int)

    df_work["INDEPENDENT"] = (df_work[list(CHAIN_PATTERNS.keys())].sum(axis=1) == 0).astype(int)

    chain_cols = list(CHAIN_PATTERNS.keys()) + ["INDEPENDENT"]
    out = df_work.groupby("ZCTA5")[chain_cols].sum().reset_index()
    return out


# ---------------------------------------------------------------------------
#  PILLAR 1: REVENUE POTENTIAL INDEX
# ---------------------------------------------------------------------------

def compute_revenue_potential(df, chronic_cols, weights):
    """
    Revenue = f(Rx demand intensity, payer quality, market size, med usage signals)

    Each sub-component is a [0,1] percentile, then combined with slider weights.
    """
    n = len(df)
    stamp("  Computing Revenue Potential...")

    # --- A) Rx Demand Intensity ---
    # Weighted by prescription intensity per condition
    if chronic_cols:
        weighted_burden = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        for col in chronic_cols:
            if col in df.columns and col in RX_INTENSITY_WEIGHTS:
                w = RX_INTENSITY_WEIGHTS[col]
                weighted_burden += pct_rank(df[col]) * w
                total_weight += w
        if total_weight > 0:
            rx_demand = (weighted_burden / total_weight).clip(0, 1)
        else:
            rx_demand = pct_rank(df.get("GHLTH_CrudePrev", pd.Series(0.5, index=df.index)))
    else:
        rx_demand = pct_rank(df.get("GHLTH_CrudePrev", pd.Series(0.5, index=df.index)))

    # --- B) Direct Medication Usage ---
    # BPMED is the share of hypertensives actually taking meds — direct Rx demand
    med_signals = []
    if "BPMED_CrudePrev" in df.columns:
        med_signals.append(pct_rank(df["BPMED_CrudePrev"]))
    if "CHECKUP_CrudePrev" in df.columns:
        med_signals.append(pct_rank(df["CHECKUP_CrudePrev"]) * 0.5)  # lower weight
    if "DISABILITY_CrudePrev" in df.columns:
        med_signals.append(pct_rank(df["DISABILITY_CrudePrev"]) * 0.7)

    if med_signals:
        med_usage = pd.concat(med_signals, axis=1).mean(axis=1).clip(0, 1)
    else:
        med_usage = pd.Series(0.5, index=df.index)

    # --- C) Payer Quality (Reimbursement) ---
    payer_components = []
    if "ins_employer_pct" in df.columns:
        payer_components.append(
            (pct_rank(df["ins_employer_pct"]), PAYER_REIMBURSEMENT["employer"])
        )
    if "ins_direct_pct" in df.columns:
        payer_components.append(
            (pct_rank(df["ins_direct_pct"]), PAYER_REIMBURSEMENT["direct"])
        )
    if "ins_tricare_pct" in df.columns:
        payer_components.append(
            (pct_rank(df["ins_tricare_pct"]), PAYER_REIMBURSEMENT["tricare"])
        )
    if "ins_marketplace_pct" in df.columns:
        # Marketplace is moderate — not penalized like in v1
        payer_components.append(
            (pct_rank(df["ins_marketplace_pct"]), PAYER_REIMBURSEMENT["marketplace"])
        )

    if payer_components:
        payer_quality = weighted_mean(payer_components)
    else:
        payer_quality = pd.Series(0.5, index=df.index)

    # --- D) Market Size ---
    pop_pct = pct_rank(np.log1p(df["population"].fillna(0)))
    density_pct = pct_rank(df["pop_density"]) if "pop_density" in df.columns else pop_pct

    # Households is a better retail proxy than raw population
    if "households" in df.columns:
        hh_pct = pct_rank(np.log1p(df["households"].fillna(0)))
        market_size = (0.40 * pop_pct + 0.35 * density_pct + 0.25 * hh_pct).clip(0, 1)
    else:
        market_size = (0.55 * pop_pct + 0.45 * density_pct).clip(0, 1)

    # --- COMBINE with slider weights ---
    w = weights
    revenue = weighted_mean([
        (rx_demand,     w["w_rx_demand"]),
        (payer_quality, w["w_payer_quality"]),
        (market_size,   w["w_market_size"]),
        (med_usage,     w["w_med_usage"]),
    ])

    stamp(f"  Revenue: mean={revenue.mean():.3f} std={revenue.std():.3f}")

    return revenue, {
        "rx_demand": rx_demand,
        "payer_quality": payer_quality,
        "market_size": market_size,
        "med_usage": med_usage,
    }


# ---------------------------------------------------------------------------
#  PILLAR 2: COST PRESSURE INDEX
# ---------------------------------------------------------------------------

def compute_cost_pressure(df, weights):
    """
    Cost = f(occupancy cost, labor cost)

    Higher cost → lower profit. This index is [0,1] where 1 = most expensive.
    """
    stamp("  Computing Cost Pressure...")

    # --- A) Occupancy Cost ---
    # Use rent if available, home price as fallback (they're 0.85+ correlated)
    if "rent" in df.columns and df["rent"].notna().sum() > 100:
        occupancy = pct_rank(df["rent"])
        # Fill gaps with home_price rank
        if "home_price" in df.columns:
            hp_rank = pct_rank(df["home_price"])
            occupancy = occupancy.fillna(hp_rank)
    elif "home_price" in df.columns:
        occupancy = pct_rank(df["home_price"])
    else:
        occupancy = pd.Series(0.5, index=df.index)

    # --- B) Labor Cost ---
    # State-level index from BLS data
    if "state" in df.columns:
        labor = df["state"].map(STATE_LABOR_INDEX).fillna(1.0)
        labor = norm01(labor)
    else:
        labor = pd.Series(0.5, index=df.index)

    # --- COMBINE ---
    w = weights
    cost = weighted_mean([
        (occupancy, w["w_occupancy_cost"]),
        (labor,     w["w_labor_cost"]),
    ])

    stamp(f"  Cost: mean={cost.mean():.3f} std={cost.std():.3f}")

    return cost, {
        "occupancy_cost": occupancy,
        "labor_cost": labor,
    }


# ---------------------------------------------------------------------------
#  PILLAR 3: COMPETITIVE CAPTURE RATE
# ---------------------------------------------------------------------------

def compute_capture_rate(df, weights):
    """
    Capture = f(1/saturation, market_opportunity)

    Higher capture → better chance of grabbing available revenue.
    Where there are zero pharmacies, capture is maximum (greenfield).
    """
    stamp("  Computing Capture Rate...")

    pop = df["population"].clip(lower=1)
    pharm_count = df["pharmacies_count"].fillna(0)

    # --- A) Saturation (inverse = opportunity) ---
    # pharmacies per 10K people
    pharm_per_10k = (pharm_count * 10000 / pop).fillna(0)
    saturation = pct_rank(pharm_per_10k)

    # Opportunity = 1 - saturation (low pharmacy density = high opportunity)
    # Zero-pharmacy ZCTAs get a bonus, but GATED BY POPULATION:
    #   - Tiny towns (<1K) have no pharmacy because there's no market → modest opportunity
    #   - Mid-size (5K–20K) with no pharmacy → genuine desert, high opportunity
    #   - Large (20K+) with no pharmacy → clear gap, maximum opportunity
    market_opportunity = 1.0 - saturation

    # Population-scaled desert bonus (log scale, 0.15 floor, 1.0 ceiling)
    pop_log = np.log1p(pop)
    pop_floor, pop_ceil = np.log1p(2000), np.log1p(25000)
    desert_strength = np.clip(
        0.15 + 0.85 * (pop_log - pop_floor) / (pop_ceil - pop_floor),
        0.15, 1.0
    )
    market_opportunity = np.where(
        pharm_count == 0,
        pd.Series(desert_strength, index=df.index),
        market_opportunity
    )
    market_opportunity = pd.Series(market_opportunity, index=df.index).clip(0, 1)

    # --- B) Competition intensity ---
    # More pharmacies = revenue split more ways
    # Log transform because going from 0→1 competitors matters more than 10→11
    competition = pct_rank(np.log1p(pharm_count))

    # --- COMBINE ---
    # Capture rate: high when opportunity is high and competition is low
    w = weights
    capture = weighted_mean([
        (market_opportunity,  w["w_market_opportunity"]),
        (1.0 - competition,   w["w_competition"]),   # invert: less competition = better
    ])

    stamp(f"  Capture: mean={capture.mean():.3f} std={capture.std():.3f}")

    return capture, {
        "market_opportunity": market_opportunity,
        "competition_intensity": competition,
        "saturation": saturation,
        "pharm_per_10k": pharm_per_10k,
    }


# ---------------------------------------------------------------------------
#  FINAL PROFIT SCORE
# ---------------------------------------------------------------------------

def compute_profit_score(revenue, cost, capture, alpha, population):
    """
    profit = (alpha * revenue_adj + (1 - alpha) * (1 - cost)) * pop_adequacy

    Capture MODULATES revenue (±40%) rather than multiplying it from zero.
    Population adequacy ensures small markets can't dominate rankings over
    larger markets with real volume — a pharmacy in a 2K town can't outscore
    a pharmacy in a 30K suburb regardless of margin.

    alpha controls revenue-vs-cost emphasis (slider "Revenue vs Cost Balance").
    Default 0.65 means we weight revenue opportunity ~2x cost avoidance.
    """
    # Revenue-side: capture adjusts revenue by ±40%, not 0-100%
    # capture=1.0 → revenue × 1.0 (full), capture=0.0 → revenue × 0.6 (floor)
    revenue_adj = (revenue * (0.6 + 0.4 * capture)).clip(0, 1)

    # Cost-side: how cheap is it to operate? (invert: 1 = cheap)
    cost_advantage = (1.0 - cost).clip(0, 1)

    # Raw margin score
    margin = (alpha * revenue_adj + (1.0 - alpha) * cost_advantage).clip(0, 1)

    # Population adequacy factor: log-scaled, gentle curve
    #   <2K  → ~0.55 (can still score well, just can't dominate top)
    #   5K   → ~0.72
    #   10K  → ~0.82
    #   25K  → ~0.93
    #   50K+ → ~1.0 (no penalty)
    pop = population.clip(lower=500)
    pop_log = np.log1p(pop)
    pop_floor, pop_ceil = np.log1p(2000), np.log1p(50000)
    pop_adequacy = np.clip(
        0.55 + 0.45 * (pop_log - pop_floor) / (pop_ceil - pop_floor),
        0.55, 1.0
    )
    pop_adequacy = pd.Series(pop_adequacy, index=revenue.index)

    # Final: margin × population adequacy
    profit = (margin * pop_adequacy).clip(0, 1)

    # Normalize to full [0, 1] range for ranking
    profit = norm01(profit)

    return profit


# ---------------------------------------------------------------------------
#  MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(config=None, slider_weights=None):
    """Execute the full profit scoring pipeline."""

    cfg = {**DEFAULTS, **(config or {})}
    weights = {**DEFAULT_SLIDER_WEIGHTS, **(slider_weights or {})}
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp("=" * 60)
    stamp("PHARMACY PROFIT MODEL v2")
    stamp("=" * 60)

    # ── Load all data sources ──
    fin = load_financial(cfg["financial_csv"])
    hlth, chronic_cols = load_health(cfg["health_csv"])
    pop = load_population(cfg["population_csv"])
    ins = load_insurance(cfg["insurance_csv"])
    hp  = load_houseprice(cfg["houseprice_csv"])

    rent_df = load_rent(cfg["rent_csv_candidates"])

    try:
        ph_cnt, chain_info = load_pharmacy(
            cfg["pharmacy_csv_candidates"], cfg["excel_globs"])
    except FileNotFoundError as e:
        stamp(f"WARNING: {e} — proceeding with zero pharmacy counts")
        ph_cnt = pd.DataFrame(columns=["ZCTA5", "pharmacies_count"])
        chain_info = None

    # ── Merge ──
    stamp("Merging datasets...")
    df = (fin
          .merge(hlth, on="ZCTA5", how="outer")
          .merge(pop,  on="ZCTA5", how="outer")
          .merge(ins,  on="ZCTA5", how="left")
          .merge(hp,   on="ZCTA5", how="left")
          .merge(ph_cnt, on="ZCTA5", how="left"))

    if rent_df is not None:
        df = df.merge(rent_df, on="ZCTA5", how="left")

    if chain_info is not None:
        df = df.merge(chain_info, on="ZCTA5", how="left")

    # Fill pharmacy counts: missing = 0
    df["pharmacies_count"] = df["pharmacies_count"].fillna(0)
    df["has_pharmacy"] = df["pharmacies_count"] > 0

    # Assign state
    if "state_from_pop" in df.columns:
        df["state"] = df["state_from_pop"]
    else:
        df["state"] = df["ZCTA5"].apply(infer_state)

    # Fill critical nulls
    for col in ["population", "median_income"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ── Home Price Imputation & Outlier Capping ──
    # Zillow ZHVI only covers ~11% of ZCTAs. For the rest, impute from
    # median_income using state-level price-to-income ratios.
    if "home_price" in df.columns and "median_income" in df.columns:
        hp_before = df["home_price"].notna().sum()

        # Step 1: Cap outliers within each state using IQR
        if "state" in df.columns or "state_from_pop" in df.columns:
            state_col = "state" if "state" in df.columns else "state_from_pop"
        else:
            state_col = None

        if state_col and df[state_col].notna().any():
            for st, grp in df.groupby(state_col):
                hp_vals = grp["home_price"].dropna()
                if len(hp_vals) >= 5:
                    q25, q75 = hp_vals.quantile(0.25), hp_vals.quantile(0.75)
                    iqr = q75 - q25
                    cap = q75 + 2.0 * iqr  # generous 2× IQR cap
                    mask = (df[state_col] == st) & (df["home_price"] > cap)
                    if mask.any():
                        stamp(f"  Home price cap ({st}): {mask.sum()} ZCTAs capped at ${cap:,.0f}")
                        df.loc[mask, "home_price"] = cap

        # Step 2: Impute missing home prices from income
        # Compute state-level price-to-income ratio from ZCTAs that have both
        has_both = df["home_price"].notna() & df["median_income"].notna() & (df["median_income"] > 0)
        if has_both.sum() >= 50:
            if state_col and df[state_col].notna().any():
                # State-level ratios
                state_ratios = {}
                for st, grp in df[has_both].groupby(state_col):
                    if len(grp) >= 3:
                        ratio = (grp["home_price"] / grp["median_income"]).median()
                        state_ratios[st] = np.clip(ratio, 2.0, 15.0)  # sane range

                national_ratio = np.clip(
                    (df.loc[has_both, "home_price"] / df.loc[has_both, "median_income"]).median(),
                    2.0, 15.0
                )

                # Fill missing with state_ratio × income, fallback to national
                missing = df["home_price"].isna() & df["median_income"].notna() & (df["median_income"] > 0)
                if missing.any():
                    state_ratio_series = df.loc[missing, state_col].map(state_ratios).fillna(national_ratio)
                    df.loc[missing, "home_price"] = df.loc[missing, "median_income"] * state_ratio_series

            else:
                # No state info — use national ratio
                national_ratio = np.clip(
                    (df.loc[has_both, "home_price"] / df.loc[has_both, "median_income"]).median(),
                    2.0, 15.0
                )
                missing = df["home_price"].isna() & df["median_income"].notna() & (df["median_income"] > 0)
                if missing.any():
                    df.loc[missing, "home_price"] = df.loc[missing, "median_income"] * national_ratio

        hp_after = df["home_price"].notna().sum()
        stamp(f"  Home price: {hp_before:,} measured → {hp_after:,} after imputation (+{hp_after - hp_before:,})")


    # Density fallback
    if "pop_density" not in df.columns or df["pop_density"].notna().sum() < 100:
        df["pop_density"] = np.where(
            df.get("land_area_km2", pd.Series(np.nan, index=df.index)).gt(0),
            df["population"] / df["land_area_km2"],
            np.nan
        )

    stamp(f"Merged: {len(df):,} ZCTAs, {df.columns.size} columns")
    stamp(f"  has_pharmacy: {df['has_pharmacy'].sum():,} ({100*df['has_pharmacy'].mean():.1f}%)")
    stamp(f"  zero_pharmacy: {(~df['has_pharmacy']).sum():,} ({100*(~df['has_pharmacy']).mean():.1f}%)")

    # ── Deduplicate ZCTAs with identical stats (PO box / corporate mail ZCTAs) ──
    # These share a parent ZCTA's census data exactly. Use multiple columns to
    # avoid false positives — two real towns rarely match on all of these.
    dedup_cols = ["population", "median_income", "pharmacies_count", "home_price"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    if len(dedup_cols) >= 3:
        before = len(df)
        # Round floats to avoid floating-point near-misses
        dedup_key = df[dedup_cols].round(2)
        dupes = dedup_key.duplicated(keep="first")
        # Only drop if population > 0 (don't touch the zero-pop rows, they get filtered later)
        dupes = dupes & (df["population"].fillna(0) > 0)
        df = df[~dupes].copy()
        dropped = before - len(df)
        if dropped > 0:
            stamp(f"  Dedup: dropped {dropped:,} ZCTAs with identical census fingerprints")

    # ── Filter to scoreable ZCTAs ──
    scoreable = df["population"].fillna(0) >= cfg["min_pop"]
    stamp(f"Scoreable (pop >= {cfg['min_pop']}): {scoreable.sum():,}")

    # ── COMPUTE THREE PILLARS ──
    stamp("-" * 40)
    stamp("COMPUTING ECONOMIC PILLARS")
    stamp("-" * 40)

    revenue, rev_components = compute_revenue_potential(df, chronic_cols, weights)
    cost, cost_components   = compute_cost_pressure(df, weights)
    capture, cap_components = compute_capture_rate(df, weights)

    alpha = weights["alpha_revenue_vs_cost"]
    profit = compute_profit_score(revenue, cost, capture, alpha, df["population"].fillna(0))

    # ── Store results ──
    df["revenue_potential"] = revenue
    df["cost_pressure"]     = cost
    df["capture_rate"]      = capture
    df["profit_score"]      = profit

    # Sub-components for diagnostics + UI tooltips
    for name, val in rev_components.items():
        df[f"rev_{name}"] = val
    for name, val in cost_components.items():
        df[f"cost_{name}"] = val
    for name, val in cap_components.items():
        df[f"cap_{name}"] = val

    # ── Classify tiers ──
    df["tier"] = pd.cut(
        df["profit_score"],
        bins=[-0.01, 0.30, 0.60, 0.80, 1.01],
        labels=["Low", "Moderate", "Strong", "Premium"]
    )

    # ── Desert classification (keep for compatibility with your UI) ──
    df["is_pharmacy_desert"] = (df["pharmacies_count"] == 0) & (df["population"].fillna(0) >= 500)
    df["desert_opportunity"] = np.where(
        df["is_pharmacy_desert"],
        df["profit_score"],
        np.nan
    )

    # ── Rank ──
    df["profit_rank"] = df["profit_score"].rank(ascending=False, method="min").astype(int)

    # ── Outputs ──
    stamp("-" * 40)
    stamp("WRITING OUTPUTS")
    stamp("-" * 40)

    # 1. Full scores
    keep_cols = [
        "ZCTA5", "state", "population", "pop_density", "median_income",
        "home_price", "pharmacies_count", "has_pharmacy", "is_pharmacy_desert",
        # Three pillars
        "revenue_potential", "cost_pressure", "capture_rate", "profit_score",
        "profit_rank", "tier",
        # Revenue sub-components
        "rev_rx_demand", "rev_payer_quality", "rev_market_size", "rev_med_usage",
        # Cost sub-components
        "cost_occupancy_cost", "cost_labor_cost",
        # Capture sub-components
        "cap_market_opportunity", "cap_competition_intensity",
        "cap_saturation", "cap_pharm_per_10k",
    ]
    # Add lat/lon if available
    if "lat" in df.columns:
        keep_cols += ["lat", "lon"]
    # Add rent if available
    if "rent" in df.columns:
        keep_cols.append("rent")
    # Add chain columns if available
    chain_cols = [c for c in df.columns if c in
                  ["WALGREENS", "CVS", "WALMART", "RITE_AID", "KROGER",
                   "COSTCO", "PUBLIX", "INDEPENDENT"]]
    keep_cols += chain_cols

    keep_cols = [c for c in keep_cols if c in df.columns]
    result = df[keep_cols].sort_values("profit_score", ascending=False).reset_index(drop=True)

    result.to_csv(out_dir / "profit_scores.csv", index=False)
    stamp(f"  → {out_dir / 'profit_scores.csv'} ({len(result):,} rows)")

    # 2. Top opportunities
    top = result[result["population"].fillna(0) >= cfg["min_pop"]].head(cfg["top_k"])
    top.to_csv(out_dir / "top_opportunities.csv", index=False)
    stamp(f"  → {out_dir / 'top_opportunities.csv'} ({len(top)} rows)")

    # 3. Desert opportunities (for your existing UI)
    deserts = (result[result["is_pharmacy_desert"]]
               .sort_values("profit_score", ascending=False))
    deserts.to_csv(out_dir / "desert_opportunities.csv", index=False)
    stamp(f"  → {out_dir / 'desert_opportunities.csv'} ({len(deserts):,} deserts)")

    # 4. Component diagnostics
    diag_cols = [c for c in result.columns if c.startswith(("rev_", "cost_", "cap_"))]
    diag = result[["ZCTA5", "profit_score", "revenue_potential",
                    "cost_pressure", "capture_rate"] + diag_cols]
    diag.to_csv(out_dir / "component_diagnostics.csv", index=False)

    # 5. Slider config for UI
    slider_config = {
        "sliders": [
            {
                "id": "w_rx_demand",
                "label": "Rx Demand Intensity",
                "pillar": "revenue",
                "description": "Weight for chronic disease prescription burden",
                "min": 0.0, "max": 2.0, "step": 0.1,
                "default": weights["w_rx_demand"],
            },
            {
                "id": "w_payer_quality",
                "label": "Payer Quality",
                "pillar": "revenue",
                "description": "Weight for insurance reimbursement quality",
                "min": 0.0, "max": 2.0, "step": 0.1,
                "default": weights["w_payer_quality"],
            },
            {
                "id": "w_market_size",
                "label": "Market Size",
                "pillar": "revenue",
                "description": "Weight for population and density",
                "min": 0.0, "max": 2.0, "step": 0.1,
                "default": weights["w_market_size"],
            },
            {
                "id": "w_med_usage",
                "label": "Medication Usage",
                "pillar": "revenue",
                "description": "Weight for direct medication utilization signals",
                "min": 0.0, "max": 2.0, "step": 0.1,
                "default": weights["w_med_usage"],
            },
            {
                "id": "w_occupancy_cost",
                "label": "Occupancy Cost",
                "pillar": "cost",
                "description": "Weight for rent/real estate cost pressure",
                "min": 0.0, "max": 2.0, "step": 0.1,
                "default": weights["w_occupancy_cost"],
            },
            {
                "id": "w_labor_cost",
                "label": "Labor Cost",
                "pillar": "cost",
                "description": "Weight for state-level pharmacy labor costs",
                "min": 0.0, "max": 2.0, "step": 0.1,
                "default": weights["w_labor_cost"],
            },
            {
                "id": "w_competition",
                "label": "Competition Penalty",
                "pillar": "capture",
                "description": "Weight for competitive saturation effect",
                "min": 0.0, "max": 2.0, "step": 0.1,
                "default": weights["w_competition"],
            },
            {
                "id": "w_market_opportunity",
                "label": "Market Opportunity",
                "pillar": "capture",
                "description": "Weight for pharmacy scarcity bonus",
                "min": 0.0, "max": 2.0, "step": 0.1,
                "default": weights["w_market_opportunity"],
            },
            {
                "id": "alpha_revenue_vs_cost",
                "label": "Revenue vs Cost Balance",
                "pillar": "master",
                "description": "Higher = prioritize revenue potential over cost avoidance",
                "min": 0.0, "max": 1.0, "step": 0.05,
                "default": weights["alpha_revenue_vs_cost"],
            },
        ]
    }
    with open(out_dir / "slider_config.json", "w") as f:
        json.dump(slider_config, f, indent=2)

    # 6. Model summary
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "total_zctas": len(result),
        "scoreable_zctas": int(scoreable.sum()),
        "pharmacy_deserts": int(df["is_pharmacy_desert"].sum()),
        "avg_profit_score": float(result["profit_score"].mean()),
        "median_profit_score": float(result["profit_score"].median()),
        "tier_distribution": result["tier"].value_counts().to_dict(),
        "avg_revenue_potential": float(revenue.mean()),
        "avg_cost_pressure": float(cost.mean()),
        "avg_capture_rate": float(capture.mean()),
        "weights_used": weights,
        "data_coverage": {
            "financial": int(df["median_income"].notna().sum()),
            "health": int(df.get("GHLTH_CrudePrev", pd.Series()).notna().sum()),
            "insurance": int(df.get("ins_employer_pct", pd.Series()).notna().sum()),
            "home_price": int(df.get("home_price", pd.Series()).notna().sum()),
            "pharmacy": int((df["pharmacies_count"] > 0).sum()),
            "density": int(df.get("pop_density", pd.Series()).notna().sum()),
        },
        "chronic_disease_cols_used": chronic_cols,
    }
    with open(out_dir / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    stamp(f"  → {out_dir / 'model_summary.json'}")

    # ── Print top results ──
    stamp("=" * 60)
    stamp(f"TOP {cfg['top_k']} PROFIT OPPORTUNITIES")
    stamp("=" * 60)
    display_cols = ["ZCTA5", "state", "population", "profit_score",
                    "revenue_potential", "cost_pressure", "capture_rate",
                    "pharmacies_count", "tier"]
    display_cols = [c for c in display_cols if c in top.columns]
    print(top[display_cols].to_string(index=False))
    print()

    # Desert summary
    stamp(f"Pharmacy deserts: {df['is_pharmacy_desert'].sum():,}")
    if deserts.shape[0] > 0:
        stamp(f"Top desert opportunity: ZCTA {deserts.iloc[0]['ZCTA5']} "
              f"(profit={deserts.iloc[0]['profit_score']:.3f})")

    stamp("Pipeline complete.")
    return df, result, summary


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pharmacy Profit Model v2")
    parser.add_argument("--out", default=DEFAULTS["out_dir"], help="Output directory")
    parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    parser.add_argument("--min-pop", type=int, default=DEFAULTS["min_pop"])
    args = parser.parse_args()

    config = {
        "out_dir": args.out,
        "top_k": args.top_k,
        "min_pop": args.min_pop,
    }

    run_pipeline(config=config)
