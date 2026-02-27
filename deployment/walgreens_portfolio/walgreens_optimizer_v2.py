"""
WALGREENS DISTRIBUTION OPTIMIZER — v2
=======================================
Scores each Walgreens pharmacy on whether its market fundamentals justify
its existence, and what strategic action it suggests.

Same economic logic as profit_model_v2 but the question is INVERTED:
  Part 2:  "Where SHOULD a pharmacy exist?"  → score ZCTAs for new openings
  Part 3:  "Which existing stores to keep?"  → score each Walgreens on viability

Three Economic Pillars (adapted for existing-store context):

  1. STORE REVENUE POTENTIAL — How much can THIS Walgreens realistically earn?
     Same as Part 2 (Rx demand, payer quality, market size) BUT now we also
     factor in Walgreens-specific market share.

  2. STORE COST PRESSURE — How expensive is it to operate THIS location?
     Same as Part 2 (occupancy, labor).

  3. COMPETITIVE POSITION — Can this Walgreens defend its revenue?
     - CVS/Walmart presence in same ZCTA (direct threat)
     - Walgreens-on-Walgreens cannibalization (consolidation signal)
     - Independent pharmacy share (vulnerable to capture)
     - Chain diversity (fragmented vs. dominated market)

  Final:  store_viability = Revenue × Position − Cost

ML Layers (same architecture, ZCTA-level targets broadcast to stores):
  Layer 1: XGBoost — predict "expected Walgreens count" per ZCTA
  Layer 2: Isolation Forest — find anomalous ZCTA profiles
  Layer 3: SHAP — which features drive store viability?
  Layer 4: LLM briefs — per-store strategic recommendations

Input formats supported:
  A) Pre-tagged pharmacy DataFrame with 'chain' column → skips regex tagging
  B) Raw NPI pharmacy data → auto-tags chains via regex

Usage:
  python walgreens_optimizer_v2.py --npi data/pharm.csv
  python walgreens_optimizer_v2.py --api-key sk-ant-...
"""

import json
import math
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False

try:
    import anthropic
    _ANTHROPIC = True
except ImportError:
    _ANTHROPIC = False

try:
    from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    _SK = True
except ImportError:
    raise ImportError("scikit-learn required: pip install scikit-learn")

try:
    from sklearn.neighbors import BallTree
    _BALLTREE = True
except ImportError:
    _BALLTREE = False


# ---------------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------------

_T0 = time.time()

def stamp(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')} +{time.time()-_T0:7.1f}s]  {msg}", flush=True)

WALGREENS_PATTERNS = [
    r"WALGREEN\s*CO", r"WALGREENS", r"WALGREEN\s*BOOTS",
    r"WALGREENS\s*BOOTS\s*ALLIANCE", r"WALGREENS\s*SPECIALTY", r"DUANE\s*READE",
]

COMPETITOR_PATTERNS = {
    "CVS":        [r"\bCVS\b", r"CAREMARK", r"TARGET\s+PHARMACY", r"OMNICARE"],
    "Walmart":    [r"WAL\s*MART", r"SAMS?\s*CLUB\s*PHARMACY"],
    "Rite_Aid":   [r"RITE\s*AID"],
    "Kroger":     [r"KROGER", r"HARRIS\s*TEETER", r"FRED\s*MEYER", r"KING\s*SOOPERS",
                   r"RALPH", r"SMITH.?S\s*PHARMACY", r"FRY.?S\s*PHARMACY",
                   r"DILLONS", r"MARIANO"],
    "Albertsons": [r"ALBERTSONS?", r"SAFEWAY\s*PHARMACY", r"VONS\s*PHARMACY",
                   r"JEWEL.OSCO", r"ACME\s*PHARMACY", r"TOM\s*THUMB\s*PHARMACY"],
    "Costco":     [r"COSTCO"],
    "Publix":     [r"PUBLIX"],
    "HEB":        [r"H.?E.?B\s*PHARMACY"],
    "Amazon":     [r"AMAZON\s*PHARMACY", r"PILLPACK"],
}

PHARMACY_TAXONOMY_PREFIX = "3336"

DEFAULT_STORE_WEIGHTS = {
    "w_rx_demand": 1.0, "w_payer_quality": 1.0, "w_market_size": 1.0,
    "w_med_usage": 1.0, "w_front_end_revenue": 1.0,
    "w_occupancy_cost": 1.0, "w_labor_cost": 1.0,
    "w_walgreens_share": 1.0, "w_competitor_threat": 1.0,
    "w_cannibalization": 1.2, "w_independent_vuln": 0.8,
    "alpha_revenue_vs_cost": 0.60, "beta_position_weight": 0.25,
}

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
#  UTILITIES
# ---------------------------------------------------------------------------

def pct_rank(s):
    return pd.to_numeric(s, errors="coerce").rank(pct=True, method="average").fillna(0.5)

def norm01(s):
    s = pd.Series(s, dtype=float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    return ((s - lo) / (hi - lo + 1e-12)).clip(0, 1)

def weighted_mean(values_and_weights):
    total_w = sum(w for _, w in values_and_weights if w > 0)
    if total_w == 0:
        return pd.Series(0.5, index=values_and_weights[0][0].index)
    result = sum(v.fillna(0.5) * w for v, w in values_and_weights if w > 0) / total_w
    return result.clip(0, 1)

def coerce_zcta(series):
    s = series.astype(str).str.extract(r"(\d{3,5})", expand=False)
    return s.fillna("").str.zfill(5)

def _pick(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _to_float(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").astype(float).fillna(default)

def _haversine_miles(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return 3958.7613 * 2 * math.asin(min(1.0, math.sqrt(a)))


# ---------------------------------------------------------------------------
#  STEP 0: PHARMACY IDENTIFICATION (auto-detects pre-tagged vs raw NPI)
# ---------------------------------------------------------------------------

def tag_all_pharmacies(npi_df, org_col=None, dba_col=None, tax_col=None, zip_col=None):
    """
    Tag every pharmacy with chain affiliation.
    Auto-detects three input formats:
      A) Single 'chain' column already exists → use as-is
      B) One-hot binary columns (Walgreens=0/1, CVS=0/1, ...) → convert to single chain
      C) Raw NPI data → regex match against known chain name patterns
    """
    stamp("STEP 0: Tagging pharmacy chains")
    df = npi_df.copy()

    # ── FORMAT B: One-hot binary columns? ──
    # Detect buddy's format: columns named after chains with 0/1 values
    ONE_HOT_MAP = {
        # one-hot column name → our standard chain name
        "Walgreens":         "Walgreens",
        "CVS":               "CVS",
        "Walmart":           "Walmart",
        "Rite_Aid":          "Rite_Aid",
        "Kroger":            "Kroger",
        "Safeway":           "Albertsons",   # Safeway is Albertsons-owned
        "Costco":            "Costco",
        "ShopRite":          "Independent",  # small chain, treat as independent
        "Pathmark":          "Independent",  # defunct, treat as independent
        "Kaiser_Permanente": "Independent",  # closed pharmacies, treat as independent
        "Health_Mart":       "Independent",  # franchise network
        "Good_Neighbor":     "Independent",  # franchise network
        "Publix":            "Publix",
        "HEB":               "HEB",
        "Albertsons":        "Albertsons",
        "Amazon":            "Amazon",
    }
    onehot_cols = [c for c in ONE_HOT_MAP if c in df.columns]
    # Require at least 3 known chain columns + they look binary (0/1)
    if len(onehot_cols) >= 3:
        sample = df[onehot_cols].head(100)
        looks_binary = all(sample[c].dropna().isin([0, 1, "0", "1", 0.0, 1.0]).all() for c in onehot_cols)
        if looks_binary:
            stamp(f"  One-hot format detected ({len(onehot_cols)} chain columns)")
            # Convert one-hot → single chain column
            # Priority: first column with a 1 wins (order = onehot_cols)
            df["chain"] = "Independent"
            for oh_col in onehot_cols:
                mask = pd.to_numeric(df[oh_col], errors="coerce").fillna(0).astype(int) == 1
                df.loc[mask, "chain"] = ONE_HOT_MAP[oh_col]

            # Handle non_chain column if present
            if "non_chain" in df.columns:
                non_mask = pd.to_numeric(df["non_chain"], errors="coerce").fillna(0).astype(int) == 1
                # Only override if no chain was already assigned
                already_chain = df["chain"] != "Independent"
                df.loc[non_mask & ~already_chain, "chain"] = "Independent"

            # Ensure ZCTA5 — buddy uses "short_ZIP"
            z = _pick(df, "short_ZIP", "Short_ZIP", "ZCTA5", "zcta5", "zip", "ZIP",
                      "zipcode", "zip_code", "postal_code",
                      "Provider Business Practice Location Address Postal Code")
            df["ZCTA5"] = coerce_zcta(df[z]) if z else "00000"

            # State
            st = _pick(df, "Provider Business Practice Location Address State Name",
                       "state", "State", "STATE")
            if st: df["state"] = df[st].astype(str).str.strip()

            # NPI
            npi_col = _pick(df, "NPI", "npi")
            if npi_col: df["npi"] = df[npi_col].astype(str)

            chain_counts = df["chain"].value_counts()
            stamp(f"  Chain distribution:\n{chain_counts.to_string()}")
            stamp(f"  Total: {len(df):,} | Walgreens: {(df['chain'] == 'Walgreens').sum():,}")
            return df

    # ── FORMAT A: Single chain column already exists? ──
    chain_col = _pick(df, "chain", "Chain", "chain_label", "Chain_Label",
                      "pharmacy_chain", "label")
    if chain_col is not None:
        stamp(f"  Pre-tagged data detected (column: '{chain_col}')")
        df["chain"] = df[chain_col].astype(str).str.strip()
        # Normalize common label variations to our standard names
        chain_map = {}
        for raw in df["chain"].unique():
            u = raw.upper()
            if "WALGREEN" in u or "DUANE" in u:  chain_map[raw] = "Walgreens"
            elif "CVS" in u or "CAREMARK" in u:   chain_map[raw] = "CVS"
            elif "WAL" in u and "MART" in u:      chain_map[raw] = "Walmart"
            elif "RITE" in u and "AID" in u:      chain_map[raw] = "Rite_Aid"
            elif "KROGER" in u:                    chain_map[raw] = "Kroger"
            elif "ALBERTSON" in u or "SAFEWAY" in u: chain_map[raw] = "Albertsons"
            elif "COSTCO" in u:                    chain_map[raw] = "Costco"
            elif "PUBLIX" in u:                    chain_map[raw] = "Publix"
            elif "INDEPENDENT" in u or "OTHER" in u: chain_map[raw] = "Independent"
            else:                                  chain_map[raw] = raw
        df["chain"] = df["chain"].map(chain_map).fillna("Independent")

        # Ensure ZCTA5
        z = _pick(df, "short_ZIP", "Short_ZIP", "ZCTA5", "zcta5", "zip", "ZIP", "Zip", "zipcode",
                  "zip_code", "postal_code", "store_zip", "Short_ZIP",
                  "Provider Business Practice Location Address Postal Code")
        df["ZCTA5"] = coerce_zcta(df[z]) if z else "00000"

        # State
        st = _pick(df, "Provider Business Practice Location Address State Name",
                   "state", "State", "STATE")
        if st and "state" not in df.columns: df["state"] = df[st].astype(str).str.strip()

        npi_col = _pick(df, "NPI", "npi")
        if npi_col: df["npi"] = df[npi_col].astype(str)

        stamp(f"  Chain distribution:\n{df['chain'].value_counts().to_string()}")
        return df

    # ── RAW NPI: regex tagging ──
    stamp("  No 'chain' column — applying regex tagging")
    if org_col is None:
        org_col = _pick(df, "Provider Organization Name (Legal Business Name)",
                        "organization_name", "ORG_NAME", "org_name", "Name", "name")
    if dba_col is None:
        dba_col = _pick(df, "Provider Other Organization Name", "dba_name", "DBA_NAME")
    if tax_col is None:
        tax_col = _pick(df, "Healthcare Provider Taxonomy Code_1",
                        "taxonomy_code", "Taxonomy_Code", "taxonomy")
    if zip_col is None:
        zip_col = _pick(df, "Provider Business Practice Location Address Postal Code",
                        "Short_ZIP", "ZIP", "Zip", "zip", "postal_code")

    for col in [org_col, dba_col]:
        if col and col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()

    search = df[org_col].fillna("") if org_col and org_col in df.columns else pd.Series("", index=df.index)
    if dba_col and dba_col in df.columns:
        search = search + " | " + df[dba_col].fillna("")

    if tax_col and tax_col in df.columns:
        tax_cols = [c for c in df.columns if "taxonomy" in c.lower() and "code" in c.lower()]
        if tax_cols:
            tax_mask = df[tax_cols].apply(
                lambda row: any(str(v).startswith(PHARMACY_TAXONOMY_PREFIX) for v in row if pd.notna(v)), axis=1)
        else:
            tax_mask = df[tax_col].astype(str).str.startswith(PHARMACY_TAXONOMY_PREFIX)
        df = df[tax_mask].copy()
        search = search.loc[df.index]
        stamp(f"  Filtered to {len(df):,} pharmacy-taxonomy records")

    df["chain"] = "Independent"
    for chain_name, patterns in COMPETITOR_PATTERNS.items():
        if not patterns: continue
        mask = search.str.contains("|".join(patterns), case=False, na=False, regex=True)
        df.loc[mask, "chain"] = chain_name

    wag_mask = search.str.contains("|".join(WALGREENS_PATTERNS), case=False, na=False, regex=True)
    df.loc[wag_mask, "chain"] = "Walgreens"

    df["ZCTA5"] = coerce_zcta(df[zip_col]) if zip_col and zip_col in df.columns else "00000"
    npi_col = _pick(df, "NPI", "npi")
    if npi_col: df["npi"] = df[npi_col].astype(str)

    stamp(f"  Chain distribution:\n{df['chain'].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
#  STEP 1: COMPETITION METRICS PER ZCTA
# ---------------------------------------------------------------------------

def compute_competition_metrics(tagged_df):
    stamp("STEP 1: Computing competition metrics per ZCTA")
    df = tagged_df.copy()
    total = df.groupby("ZCTA5").size().rename("total_pharmacies")
    chain_pivot = df.groupby(["ZCTA5", "chain"]).size().unstack(fill_value=0)
    chain_pivot.columns = [f"{c.lower().replace(' ', '_')}_count" for c in chain_pivot.columns]

    result = pd.concat([total, chain_pivot], axis=1).fillna(0).reset_index()
    wag_col = "walgreens_count"
    if wag_col not in result.columns: result[wag_col] = 0

    result["non_walgreens_count"] = result["total_pharmacies"] - result[wag_col]
    result["walgreens_share"] = result[wag_col] / result["total_pharmacies"].clip(lower=1)
    result["walgreens_cannibalization"] = (result[wag_col] > 1).astype(int)
    result["chain_diversity"] = (chain_pivot > 0).sum(axis=1).values

    for comp in ["cvs", "walmart", "rite_aid", "kroger"]:
        col = f"{comp}_count"
        if col in result.columns:
            result[f"has_{comp}"] = (result[col] > 0).astype(int)

    ind_col = "independent_count"
    result["independent_share"] = result[ind_col] / result["total_pharmacies"].clip(lower=1) if ind_col in result.columns else 0.0

    stamp(f"  {len(result):,} ZCTAs | Cannibalization: {result['walgreens_cannibalization'].sum():,}")
    return result


# ---------------------------------------------------------------------------
#  STEP 2: MERGE WITH ZCTA SCORES + BUILD STORE ROWS
# ---------------------------------------------------------------------------

def merge_store_data(zcta_scores_path, competition_df, tagged_df, walgreens_stores=None):
    stamp("STEP 2: Merging store-level data")
    zcta = pd.read_csv(zcta_scores_path, dtype={"ZCTA5": str})
    zcta["ZCTA5"] = zcta["ZCTA5"].astype(str).str.zfill(5)
    stamp(f"  ZCTA scores: {len(zcta):,} rows")

    comp_cols = [c for c in competition_df.columns if c != "ZCTA5"]
    zcta_clean = zcta.drop(columns=[c for c in comp_cols if c in zcta.columns], errors="ignore")
    merged = zcta_clean.merge(competition_df, on="ZCTA5", how="left")
    for c in comp_cols:
        if c in merged.columns: merged[c] = merged[c].fillna(0)

    wag_col = "walgreens_count"
    if wag_col in merged.columns:
        wag_zctas = merged[merged[wag_col] > 0].copy()
    else:
        wag_zctas = merged.copy()
        wag_zctas[wag_col] = 1

    # ── Real store list → true store-level rows ──
    if walgreens_stores is not None:
        if isinstance(walgreens_stores, (str, Path)):
            walgreens_stores = pd.read_csv(walgreens_stores, dtype=str)
        st = walgreens_stores.copy()
        z_col = _pick(st, "ZCTA5", "zcta5", "zip", "zipcode", "ZIP", "Zip",
                      "postal_code", "zip_code", "store_zip")
        if z_col is None:
            raise ValueError("walgreens_stores must include a ZIP/ZCTA column.")
        st["ZCTA5"] = coerce_zcta(st[z_col])
        sid = _pick(st, "store_id", "store_number", "store_num", "location_id", "id", "npi")
        st["store_id"] = st[sid].astype(str).str.strip() if sid else [f"STORE_{i}" for i in range(len(st))]
        # Carry through coords if present
        for c in ["lat", "latitude", "store_lat"]:
            if c in st.columns: st["lat"] = _to_float(st[c]); break
        for c in ["lon", "lng", "longitude", "store_lon"]:
            if c in st.columns: st["lon"] = _to_float(st[c]); break

        stores = st.merge(merged, on="ZCTA5", how="left")
        n_per = stores.groupby("ZCTA5")["store_id"].transform("count").astype(int)
        stores["walgreens_count"] = n_per
        stores["store_index"] = stores.groupby("ZCTA5").cumcount().astype(int)
        stores["store_zcta_share"] = (1.0 / n_per.clip(lower=1)).astype(float)
        stores["decision_level"] = "store"
        stamp(f"  Loaded {len(stores):,} Walgreens stores (store-level)")
        return stores, merged

    # ── No store list → ZCTA-level decision units ──
    stores = wag_zctas.copy()
    stores["store_id"] = stores["ZCTA5"].astype(str) + "_WAG_ZCTA"
    stores["store_index"] = 0
    stores["store_zcta_share"] = 1.0
    stores["decision_level"] = "zcta"

    # Filter out zero-population ZCTAs (PO boxes, office parks, etc.)
    if "population" in stores.columns:
        zero_pop = stores["population"].fillna(0) <= 0
        if zero_pop.any():
            stamp(f"  Dropping {zero_pop.sum()} zero-population ZCTAs")
            stores = stores[~zero_pop].copy()

    stamp(f"  ZCTA-level decision units: {len(stores):,}")
    return stores, merged


# ---------------------------------------------------------------------------
#  STEP 3: THREE ECONOMIC PILLARS
# ---------------------------------------------------------------------------

def compute_store_revenue(stores, weights):
    stamp("  Computing Store Revenue Potential...")
    rx_demand    = _to_float(stores.get("rev_rx_demand",    pd.Series(dtype=float)), 0.5)
    payer_quality = _to_float(stores.get("rev_payer_quality", pd.Series(dtype=float)), 0.5)
    market_size  = _to_float(stores.get("rev_market_size",  pd.Series(dtype=float)), 0.5)
    med_usage    = _to_float(stores.get("rev_med_usage",    pd.Series(dtype=float)), 0.5)
    front_end    = pct_rank(stores["median_income"]) if "median_income" in stores.columns else pd.Series(0.5, index=stores.index)

    w = weights
    zcta_revenue = weighted_mean([
        (rx_demand,     w["w_rx_demand"]),     (payer_quality, w["w_payer_quality"]),
        (market_size,   w["w_market_size"]),    (med_usage,     w["w_med_usage"]),
        (front_end,     w["w_front_end_revenue"]),
    ])
    # Dampen by share: sole WAG=1.0×, 1-of-2=0.75×, 1-of-3=0.67×
    share = stores["store_zcta_share"].fillna(1.0).clip(0.01, 1.0)
    store_rev = (zcta_revenue * (0.5 + 0.5 * share)).clip(0, 1)
    stamp(f"    mean={store_rev.mean():.3f}")
    return store_rev, {"zcta_revenue": zcta_revenue, "front_end": front_end}


def compute_store_cost(stores, weights):
    stamp("  Computing Store Cost Pressure...")
    if "cost_occupancy_cost" in stores.columns: occ = _to_float(stores["cost_occupancy_cost"], 0.5)
    elif "home_price" in stores.columns:        occ = pct_rank(stores["home_price"])
    else:                                       occ = pd.Series(0.5, index=stores.index)

    if "cost_labor_cost" in stores.columns:     lab = _to_float(stores["cost_labor_cost"], 0.5)
    elif "state" in stores.columns:             lab = norm01(stores["state"].map(STATE_LABOR_INDEX).fillna(1.0))
    else:                                       lab = pd.Series(0.5, index=stores.index)

    # ── Sanity check: occupancy vs income ──
    # If a ZCTA has 95th percentile cost but only 50th percentile income,
    # the cost data is probably wrong (Zillow sparse coverage artifact).
    # Blend occupancy toward income-implied cost when they diverge badly.
    if "median_income" in stores.columns:
        income_pct = pct_rank(stores["median_income"]).fillna(0.5)
        # Income-implied cost: higher income areas → higher rent (correlation ~0.7)
        income_implied_cost = income_pct * 0.8 + 0.1  # scale to [0.1, 0.9]

        # Detect divergence: occupancy cost way above what income suggests
        divergence = (occ - income_implied_cost).clip(lower=0)
        # Where divergence > 0.3 (cost is 30+ percentile points above income),
        # pull cost back toward income-implied value
        correction_strength = (divergence - 0.20).clip(lower=0) / 0.30  # ramps 0→1
        correction_strength = correction_strength.clip(upper=0.7)  # max 70% correction
        occ = occ * (1 - correction_strength) + income_implied_cost * correction_strength

    cost = weighted_mean([(occ, weights["w_occupancy_cost"]), (lab, weights["w_labor_cost"])])

    # ── Affluence-adjusted cost cap ──
    # In high-income areas, high rent is "paid for" by high revenue.
    # Cap effective cost so it can never completely overwhelm revenue.
    if "median_income" in stores.columns:
        income_pct = pct_rank(stores["median_income"]).fillna(0.5)
        cost_ceiling = 0.4 + 0.6 * (1.0 - income_pct)  # affluent→cap 0.4, poor→cap 1.0
        cost = cost.clip(upper=cost_ceiling)

    stamp(f"    mean={cost.mean():.3f}")
    return cost, {"occupancy": occ, "labor": lab}


def compute_competitive_position(stores, weights, cannibal_radius_miles=3.0):
    stamp("  Computing Competitive Position...")
    share_score = _to_float(stores.get("walgreens_share", pd.Series(dtype=float)), 0.5).clip(0, 1)

    # Competitor threat — density-adjusted
    # Raw threat: which major chains are present?
    threat_parts = []
    for comp in ["cvs", "walmart", "kroger", "rite_aid"]:
        for col in [f"has_{comp}", f"{comp}_count"]:
            if col in stores.columns:
                threat_parts.append((_to_float(stores[col], 0) > 0).astype(float) * 0.15)
                break
    threat = sum(threat_parts).clip(0, 1) if threat_parts else pd.Series(0.0, index=stores.index)

    # ── Density adjustment ──
    # In dense/high-pop areas, competition is expected and doesn't hurt as much
    # because foot traffic volume compensates.
    # Scale threat down by up to 40% in the densest areas.
    if "population" in stores.columns and "total_pharmacies" in stores.columns:
        pop = _to_float(stores["population"], 5000).clip(lower=1)
        total_ph = _to_float(stores["total_pharmacies"], 1).clip(lower=0)
        # pharmacies per 10K people — the real competitive intensity signal
        ph_per_10k = (total_ph / pop) * 10000
        # National median is ~4-6 per 10K. Above 10 is genuinely saturated.
        density_factor = (ph_per_10k / 10.0).clip(0, 1)  # 0=no pressure, 1=saturated

        # Population scale factor: bigger population = more room for everyone
        pop_pct = pct_rank(pop)
        urban_relief = 0.4 * pop_pct  # up to 40% reduction in dense areas
        threat = (threat * (1.0 - urban_relief) + density_factor * urban_relief).clip(0, 1)

    competitor_safety = (1.0 - threat).clip(0, 1)

    # Cannibalization — ZCTA-level only (we don't have store-level coords)
    # True cannibalization = multiple Walgreens in the SAME ZCTA.
    # We cannot measure cross-ZCTA cannibalization without store addresses,
    # and ZCTA centroids are too imprecise to fake it.
    wag_in_zcta = _to_float(stores.get("walgreens_count",
                   pd.Series(1, index=stores.index)), 1).clip(lower=1)

    # Severity: 1 store = no cannibalization, 2 = moderate, 3+ = significant
    # But even 2 in a ZCTA isn't always bad — big ZCTAs (50K+ pop) can support it.
    excess = (wag_in_zcta - 1).clip(lower=0)

    # Population-adjusted: a ZCTA with 50K people can support 2 Walgreens
    if "population" in stores.columns:
        pop = _to_float(stores["population"], 10000).clip(lower=1000)
        supportable = (pop / 25000).clip(lower=1)  # 1 per 25K people
        excess_adj = (wag_in_zcta - supportable).clip(lower=0)
    else:
        excess_adj = excess

    # Gradual severity: 0 excess → 0.0, 1 excess → 0.35, 2 → 0.55, 3+ → 0.70
    cannibal_sev = np.where(excess_adj <= 0, 0.0,
                   np.clip(0.35 * excess_adj / (0.5 + excess_adj), 0.0, 0.70))

    stores["wag_nearby_count"] = wag_in_zcta.astype(int)
    stores["walgreens_cannibalization"] = (wag_in_zcta > 1).astype(int)
    cannibal_score = 1.0 - pd.Series(cannibal_sev, index=stores.index)

    ind_share = _to_float(stores.get("independent_share", pd.Series(dtype=float)), 0.5)
    ind_opp = pct_rank(ind_share)

    pos = weighted_mean([
        (share_score, weights["w_walgreens_share"]), (competitor_safety, weights["w_competitor_threat"]),
        (cannibal_score, weights["w_cannibalization"]), (ind_opp, weights["w_independent_vuln"]),
    ])
    stamp(f"    mean={pos.mean():.3f}")
    return pos, {"walgreens_share": share_score, "competitor_safety": competitor_safety,
                 "cannibal_score": cannibal_score, "independent_opportunity": ind_opp, "raw_threat": threat}


def compute_store_viability(revenue, cost, position, alpha, beta):
    alpha = min(alpha, 1.0); beta = min(beta, 1.0 - alpha); gamma = 1.0 - alpha - beta

    # ── Key fix: revenue contributes even when position is low ──
    # Old: revenue × position → if position=0.22, revenue gets crushed to 22%
    # New: revenue × (0.3 + 0.7 × position) → position=0.22 still keeps 45% of revenue
    # This reflects reality: a Georgetown Walgreens still has high foot traffic
    # even in a competitive market.
    rev_contribution = (revenue * (0.3 + 0.7 * position)).clip(0, 1)

    v = (alpha * rev_contribution + beta * position + gamma * (1.0 - cost).clip(0, 1)).clip(0, 1)

    # Normalize to 0-1
    v = norm01(v)

    # ── Revenue floor (AFTER norm01 so it actually sticks) ──
    # Stores with above-median revenue should never be the worst in the country.
    # This prevents the "Georgetown problem" where high cost + competition
    # crushes a fundamentally strong store to the bottom.
    rev_pct = revenue.rank(pct=True, na_option="keep").fillna(0.5)
    floor = (rev_pct - 0.30).clip(lower=0) * 0.35  # median rev → floor 0.07, top decile → floor 0.21
    v = v.clip(lower=floor)

    # Re-normalize gently — only if floor pushed values above 1.0 (shouldn't happen)
    v = v.clip(0, 1)

    return v


# ---------------------------------------------------------------------------
#  STEP 4: STRATEGIC ACTION FLAGS
# ---------------------------------------------------------------------------

def assign_strategic_actions(stores):
    stamp("STEP 4: Assigning strategic actions")
    v, rev, cost, pos = stores["store_viability"], stores["store_revenue"], stores["store_cost"], stores["store_position"]
    stores["action"] = "MONITOR"

    protect = (v >= v.quantile(0.70)) & (pos >= pos.quantile(0.60)) & (_to_float(stores.get("walgreens_cannibalization", pd.Series(0)), 0) == 0)
    stores.loc[protect, "action"] = "PROTECT & INVEST"

    if "rev_rx_demand" in stores.columns and "rev_payer_quality" in stores.columns:
        rx, pay = _to_float(stores["rev_rx_demand"], 0.5), _to_float(stores["rev_payer_quality"], 0.5)
        clinical = (rx >= rx.quantile(0.65)) & (pay >= pay.quantile(0.50)) & (v >= v.quantile(0.40)) & ~protect
        stores.loc[clinical, "action"] = "EXPAND CLINICAL"

    cannibal = (_to_float(stores.get("walgreens_cannibalization", pd.Series(0)), 0) > 0) & (v < v.quantile(0.80))
    stores.loc[cannibal, "action"] = "CONSOLIDATE"

    reformat = (rev >= rev.quantile(0.50)) & (v >= v.quantile(0.30)) & (v < v.quantile(0.60)) & ~cannibal & ~protect
    stores.loc[reformat, "action"] = "REFORMAT"

    closure = (v < v.quantile(0.20)) & ((cost >= cost.quantile(0.60)) | (pos < pos.quantile(0.25)))
    stores.loc[closure, "action"] = "CLOSURE CANDIDATE"

    for action, count in stores["action"].value_counts().items():
        stamp(f"    {action:25s} {count:6,}  ({100*count/len(stores):.1f}%)")
    return stores


# ---------------------------------------------------------------------------
#  STEP 5: CONSOLIDATION PAIRS
# ---------------------------------------------------------------------------

def find_consolidation_pairs(stores, tagged_df=None, cannibal_radius_miles=3.0):
    stamp("STEP 5: Finding consolidation pairs")
    dl = str(stores["decision_level"].iloc[0]) if "decision_level" in stores.columns else "zcta"

    if dl == "zcta":
        wc = _to_float(stores.get("walgreens_count", pd.Series(1, index=stores.index)), 1)
        multi = stores[wc > 1]
        if multi.empty:
            stamp("  No multi-Walgreens ZCTAs"); return pd.DataFrame()
        cols = [c for c in ["ZCTA5", "state", "population", "walgreens_count", "total_pharmacies", "store_viability"] if c in multi.columns]
        out = multi[cols].copy()
        out["note"] = "ZCTA has >1 Walgreens. Provide store list for true pairs."
        out["pair_type"] = "zcta_screen"
        stamp(f"  Consolidation screen: {len(out):,} ZCTAs")
        return out.sort_values("walgreens_count", ascending=False)

    # Store-level pairs
    pairs = []
    has_coords = {"lat", "lon"}.issubset(stores.columns) and stores["lat"].notna().any() and _BALLTREE

    if has_coords:
        stamp("  Distance-based consolidation")
        lat, lon = _to_float(stores["lat"]), _to_float(stores["lon"])
        ok = lat.notna() & lon.notna()
        idx_arr = stores.index[ok].to_numpy()
        if len(idx_arr) >= 2:
            coords = np.deg2rad(np.c_[lat.loc[idx_arr].values, lon.loc[idx_arr].values])
            tree = BallTree(coords, metric="haversine")
            neighbors = tree.query_radius(coords, r=cannibal_radius_miles / 3958.7613)
            visited = set()
            for i, nbrs in enumerate(neighbors):
                if i in visited: continue
                cluster = [int(j) for j in nbrs if int(j) not in visited]
                if len(cluster) < 2: continue
                for j in cluster: visited.add(j)
                cs = stores.loc[idx_arr[cluster]].sort_values("store_viability", ascending=False)
                keeper = cs.iloc[0]
                for _, cl in cs.iloc[1:].iterrows():
                    pairs.append({
                        "pair_type": "distance", "ZCTA5": keeper.get("ZCTA5", ""),
                        "keep_store_id": keeper["store_id"], "keep_viability": float(keeper["store_viability"]),
                        "close_store_id": cl["store_id"], "close_viability": float(cl["store_viability"]),
                        "viability_gap": float(keeper["store_viability"] - cl["store_viability"]),
                        "distance_miles": round(_haversine_miles(
                            float(keeper.get("lat", 0)), float(keeper.get("lon", 0)),
                            float(cl.get("lat", 0)), float(cl.get("lon", 0))), 2),
                    })

    if not pairs:
        for zcta in stores[_to_float(stores.get("walgreens_count", pd.Series(1)), 1).astype(int) > 1]["ZCTA5"].unique():
            zr = stores[stores["ZCTA5"] == zcta].sort_values("store_viability", ascending=False)
            if len(zr) < 2: continue
            keeper = zr.iloc[0]
            for _, cl in zr.iloc[1:].iterrows():
                pairs.append({
                    "pair_type": "within_zcta", "ZCTA5": zcta,
                    "keep_store_id": keeper["store_id"], "keep_viability": float(keeper["store_viability"]),
                    "close_store_id": cl["store_id"], "close_viability": float(cl["store_viability"]),
                    "viability_gap": float(keeper["store_viability"] - cl["store_viability"]),
                })

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty: pairs_df = pairs_df.sort_values("viability_gap", ascending=False)
    stamp(f"  {len(pairs_df):,} consolidation pairs")
    return pairs_df


# ---------------------------------------------------------------------------
#  STEP 6: ML LAYERS (ZCTA-level, broadcast to stores)
# ---------------------------------------------------------------------------

def build_store_features(stores):
    features, desc = {}, {}
    for col, nm, d in [("population", "log_pop", "Log population"), ("pop_density", "density_pct", "Density pct"),
                       ("median_income", "income_pct", "Income pct"), ("households", "log_hh", "Log households")]:
        if col in stores.columns:
            v = _to_float(stores[col], 0)
            features[nm] = np.log1p(v) if "log" in nm else v.rank(pct=True).fillna(0.5)
            desc[nm] = d
    for col in ["revenue_potential", "cost_pressure", "capture_rate", "rev_rx_demand",
                "rev_payer_quality", "rev_market_size", "rev_med_usage", "cost_occupancy_cost", "cost_labor_cost"]:
        if col in stores.columns: features[col] = _to_float(stores[col], 0.5); desc[col] = col
    for col in ["total_pharmacies", "non_walgreens_count", "walgreens_count", "walgreens_share",
                "chain_diversity", "independent_share", "walgreens_cannibalization"]:
        if col in stores.columns: features[col] = _to_float(stores[col], 0); desc[col] = col
    for comp in ["cvs", "walmart", "kroger", "rite_aid"]:
        f = f"has_{comp}"
        if f in stores.columns: features[f] = _to_float(stores[f], 0); desc[f] = f
    features["store_zcta_share"] = stores["store_zcta_share"].fillna(1.0)
    if "rev_rx_demand" in features and "walgreens_share" in features:
        features["demand_x_share"] = features["rev_rx_demand"] * features["walgreens_share"]
    if "income_pct" in features and "total_pharmacies" in features:
        features["affluence_x_competition"] = features.get("income_pct", 0.5) * (1.0 / (features["total_pharmacies"] + 1))

    X = pd.DataFrame(features, index=stores.index).replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.5)
    stamp(f"  Features: {X.shape[0]:,} × {X.shape[1]}")
    return X, list(X.columns), desc


def run_ml_layers(stores, X, feature_names, feature_desc, output_dir=None):
    stamp("─── ML Layer 1: XGBoost Justification (ZCTA-level) ───")
    zcta_key = stores["ZCTA5"].astype(str)
    Xz = X.copy(); Xz["_z"] = zcta_key.values; Xz = Xz.groupby("_z").median(numeric_only=True)

    pop_z = _to_float(stores.groupby("ZCTA5")["population"].first(), 1).clip(lower=1)
    wag_z = _to_float(stores.groupby("ZCTA5")["walgreens_count"].first(), 0)
    common = Xz.index.intersection(pop_z.index).intersection(wag_z.index)
    Xz, pop_z, wag_z = Xz.loc[common], pop_z.loc[common], wag_z.loc[common]
    y = np.log(wag_z / pop_z * 10000 + 0.1)

    leak = [c for c in Xz.columns if any(s in c.lower() for s in ["walgreens_count", "walgreens_cannibalization", "wag_nearby", "store_zcta_share"]) and c != "walgreens_share"]
    X_clean = Xz.drop(columns=leak, errors="ignore")
    mask = (pop_z >= 500) & np.isfinite(y) & np.isfinite(X_clean.sum(axis=1))
    X_tr, y_tr = X_clean.loc[mask], y.loc[mask]
    stamp(f"  Training on {len(X_tr):,} ZCTAs")

    cv = []
    for _, (ti, vi) in enumerate(KFold(5, shuffle=True, random_state=42).split(X_tr)):
        if _XGB:
            p = dict(objective="reg:squarederror", max_depth=5, eta=0.05, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1.5, min_child_weight=10, tree_method="hist", verbosity=0)
            bst = xgb.train(p, xgb.DMatrix(X_tr.iloc[ti], y_tr.iloc[ti]), 400,
                           evals=[(xgb.DMatrix(X_tr.iloc[vi], y_tr.iloc[vi]), "v")],
                           verbose_eval=False, early_stopping_rounds=40)
            pred = bst.predict(xgb.DMatrix(X_tr.iloc[vi]))
        else:
            m = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=400, random_state=42)
            m.fit(X_tr.iloc[ti], y_tr.iloc[ti]); pred = m.predict(X_tr.iloc[vi])
        cv.append(np.sqrt(mean_squared_error(y_tr.iloc[vi], pred)))
    stamp(f"  CV RMSE: {np.mean(cv):.4f} ± {np.std(cv):.4f}")

    if _XGB:
        model = xgb.train(p, xgb.DMatrix(X_tr, y_tr), 400)
        yp = pd.Series(model.predict(xgb.DMatrix(X_clean)), index=X_clean.index)
    else:
        model = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=400, random_state=42)
        model.fit(X_tr, y_tr); yp = pd.Series(model.predict(X_clean), index=X_clean.index)

    gap_z = (yp - y).rank(pct=True).fillna(0.5)
    stores["ml_justification_gap"] = zcta_key.map(gap_z.to_dict()).fillna(0.5).values

    stamp("─── ML Layer 2: Isolation Forest ───")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_clean.fillna(0))
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(Xs)
    a_score = pd.Series(-iso.decision_function(Xs), index=X_clean.index).rank(pct=True).fillna(0.5)
    a_flag = pd.Series(iso.predict(Xs), index=X_clean.index) == -1
    v_z = stores.groupby("ZCTA5")["store_viability"].max().reindex(X_clean.index).fillna(0.5)
    a_type = pd.Series("normal", index=X_clean.index)
    a_type[a_flag & (v_z > v_z.median())] = "hidden_gem"
    a_type[a_flag & (v_z <= v_z.median())] = "hidden_risk"
    stores["anomaly_score"] = zcta_key.map(a_score.to_dict()).fillna(0.5).values
    stores["anomaly_flag"] = zcta_key.map(a_flag.to_dict()).fillna(False).values
    stores["anomaly_type"] = zcta_key.map(a_type.to_dict()).fillna("normal").values
    stamp(f"  Anomalies: {int(a_flag.sum()):,}/{len(a_flag):,}")

    stamp("─── ML Layer 3: SHAP ───")
    if _SHAP and _XGB:
        try:
            sv = shap.TreeExplainer(model).shap_values(xgb.DMatrix(X_clean.sample(min(5000, len(X_clean)), random_state=42)))
            ma = pd.Series(np.abs(sv).mean(0), index=X_clean.columns).sort_values(ascending=False)
            for f, v in ma.head(6).items(): stamp(f"    {f:35s} {v:.4f}")
            if output_dir: ma.to_csv(Path(output_dir) / "shap_store_importance.csv")
        except Exception as e: stamp(f"  SHAP error: {e}")
    elif _XGB:
        try: stamp(f"  Top features: {dict(list(sorted(model.get_score(importance_type='gain').items(), key=lambda x:-x[1]))[:5])}")
        except: pass

    return stores, {"model": model, "cv_rmse": float(np.mean(cv))}


# ---------------------------------------------------------------------------
#  STEP 7–9: CLUSTERING, BRIEFS, SUMMARY (compact)
# ---------------------------------------------------------------------------

CLUSTER_FEATURES = ["store_revenue", "store_cost", "store_position", "store_viability",
                    "ml_justification_gap", "walgreens_count", "walgreens_share"]

def cluster_stores(stores, n_clusters=6):
    stamp("STEP 7: Archetype clustering")
    avail = [c for c in CLUSTER_FEATURES if c in stores.columns]
    if len(avail) < 3:
        stores["archetype_id"] = 0; stores["archetype_name"] = "Unknown"; return stores, pd.DataFrame()
    X = stores[avail].fillna(0.5).values; sc = StandardScaler(); Xs = sc.fit_transform(X)
    k = min(n_clusters, max(3, len(stores) // 50))
    km = KMeans(n_clusters=k, random_state=42, n_init=10); labels = km.fit_predict(Xs)
    stores["archetype_id"] = labels
    centroids = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=avail)
    centroids["cluster_id"] = range(k)
    centroids["count"] = pd.Series(labels).value_counts().sort_index().values

    # Compute distribution medians for relative thresholds
    med = {c: stores[c].median() for c in avail if c in stores.columns}

    def _name(r):
        v    = r.get("store_viability", .5)
        rev  = r.get("store_revenue", .5)
        cost = r.get("store_cost", .5)
        pos  = r.get("store_position", .5)
        gap  = r.get("ml_justification_gap", .5)
        wag  = r.get("walgreens_count", 1)
        share = r.get("walgreens_share", .5)

        v_med    = med.get("store_viability", .5)
        rev_med  = med.get("store_revenue", .5)
        cost_med = med.get("store_cost", .5)
        pos_med  = med.get("store_position", .5)

        # High viability + good position → Fortress
        if v > v_med * 1.3 and pos > pos_med * 1.1:
            return "Fortress Store"
        # High revenue + high cost → expensive but productive
        if rev > rev_med * 1.15 and cost > cost_med * 1.3:
            return "High-Revenue High-Cost"
        # Above-median viability, high ML gap → model underpredicts
        if v > v_med and gap > med.get("ml_justification_gap", .5) * 1.3:
            return "Hidden Potential"
        # Low viability + many Walgreens → over-saturated
        if v < v_med * 0.8 and wag > 2:
            return "Cannibalization Zone"
        # Low viability + low revenue → weak market
        if v < v_med * 0.7 and rev < rev_med * 0.85:
            return "Struggling Outpost"
        # Below-median viability, low position → competitive pressure
        if v < v_med and pos < pos_med * 0.9:
            return "Competitive Pressure"
        # High revenue but below-median viability → cost or position dragging
        if rev > rev_med * 1.1 and v < v_med:
            return "Operational Drag"
        return "Middle Market"

    centroids["archetype_name"] = [_name(r) for _, r in centroids.iterrows()]
    stores["archetype_name"] = stores["archetype_id"].map(dict(zip(centroids["cluster_id"], centroids["archetype_name"])))
    for _, r in centroids.iterrows(): stamp(f"    {r['archetype_name']:30s} n={int(r['count'])}")
    return stores, centroids


STORE_BRIEF_SYSTEM = "You are a Walgreens portfolio strategist. For each store: 3 sentences. 1) Market position. 2) Key factor. 3) Recommendation: PROTECT & INVEST / EXPAND CLINICAL / CONSOLIDATE / REFORMAT / MONITOR / CLOSE. Be blunt, use numbers."

def generate_store_briefs(stores, n=50, api_key=None):
    stamp("STEP 8: Store briefs")
    top = stores.nlargest(max(1, n // 2), "store_viability")
    bottom = stores.nsmallest(max(1, n // 2), "store_viability")
    subset = pd.concat([top, bottom]).drop_duplicates(subset=["store_id"])
    briefs = []
    for _, r in subset.iterrows():
        ctx = f"Store:{r['store_id']}|ZCTA:{r.get('ZCTA5','?')}({r.get('state','?')})|Pop:{r.get('population',0):,.0f}|Inc:${r.get('median_income',0):,.0f}|V:{r.get('store_viability',0):.3f}|Rev:{r.get('store_revenue',0):.3f}|Cost:{r.get('store_cost',0):.3f}|Pos:{r.get('store_position',0):.3f}|Pharm:{r.get('total_pharmacies',0):.0f}|WAG:{r.get('walgreens_count',1):.0f}|Share:{r.get('walgreens_share',0):.0%}|Action:{r.get('action','?')}|Arch:{r.get('archetype_name','?')}"
        briefs.append({"store_id": r["store_id"], "ZCTA5": r.get("ZCTA5", ""), "state": r.get("state", ""),
                       "store_viability": float(r.get("store_viability", 0)), "action": r.get("action", ""),
                       "archetype_name": r.get("archetype_name", ""), "prompt": ctx, "brief": None})

    if api_key and _ANTHROPIC:
        stamp("  LLM briefs via Claude API...")
        client = anthropic.Anthropic(api_key=api_key)
        for i in range(0, len(briefs), 8):
            batch = briefs[i:i+8]
            try:
                resp = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=1200, system=STORE_BRIEF_SYSTEM,
                    messages=[{"role": "user", "content": f"JSON array of {{store_id, brief, recommendation}} for:\n{json.dumps([{'store_id': b['store_id'], 'context': b['prompt']} for b in batch])}"}])
                data = json.loads(resp.content[0].text.strip())
                by_id = {d["store_id"]: d for d in data if isinstance(d, dict) and "store_id" in d}
                for b in batch:
                    d = by_id.get(b["store_id"])
                    if d and d.get("brief"): b["brief"] = str(d["brief"]).strip()
            except Exception as e: stamp(f"  LLM error: {str(e)[:100]}")

    for b in briefs:
        if not b["brief"]:
            m = stores[stores["store_id"] == b["store_id"]]
            if len(m): b["brief"] = _tpl_brief(m.iloc[0])
            else: b["brief"] = f"Store {b['store_id']}: viability {b['store_viability']:.3f}. {b['action']}."

    stamp(f"  {len(briefs)} briefs generated")
    return pd.DataFrame(briefs)

def _tpl_brief(r):
    v, rev, cost, pos, action = r.get("store_viability",.5), r.get("store_revenue",.5), r.get("store_cost",.5), r.get("store_position",.5), r.get("action","MONITOR")
    pop, share, st = r.get("population",0), r.get("walgreens_share",0), r.get("state","?")
    s1 = f"This {st} location serves {pop:,.0f} people with {'dominant' if share>.5 else 'moderate' if share>.25 else 'minor'} share ({share:.0%})."
    if r.get("walgreens_cannibalization",0)>0: s2=f"{r.get('walgreens_count',2):.0f} Walgreens in ZCTA create consolidation urgency."
    elif cost>.75 and rev<.5: s2=f"High costs ({cost:.2f}) with weak revenue ({rev:.2f}) = unsustainable."
    elif rev>.7 and pos>.6: s2=f"Strong revenue ({rev:.2f}) and position ({pos:.2f}) worth defending."
    else: s2=f"Revenue {rev:.2f}, cost {cost:.2f}, position {pos:.2f} → viability {v:.3f}."
    return f"{s1} {s2} Recommendation: {action}."


def portfolio_summary(stores, pairs_df, centroids):
    stamp("STEP 9: Portfolio summary")
    s = {"total_stores": len(stores), "mean_viability": float(stores["store_viability"].mean()),
         "action_distribution": stores["action"].value_counts().to_dict(),
         "archetype_distribution": stores["archetype_name"].value_counts().to_dict() if "archetype_name" in stores.columns else {},
         "consolidation_pairs": len(pairs_df), "closure_candidates": int((stores["action"]=="CLOSURE CANDIDATE").sum()),
         "protect_invest": int((stores["action"]=="PROTECT & INVEST").sum()),
         "anomaly_hidden_gems": int((stores.get("anomaly_type","")=="hidden_gem").sum()),
         "anomaly_hidden_risks": int((stores.get("anomaly_type","")=="hidden_risk").sum())}
    for k, v in s.items():
        if not isinstance(v, dict): stamp(f"  {k}: {v}")
    return s


# ---------------------------------------------------------------------------
#  MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_walgreens_pipeline(zcta_scores_path="results_v2/profit_scores.csv", npi_data=None,
    npi_excel_globs=None, output_dir="results_walgreens", n_clusters=6, n_briefs=50,
    api_key=None, slider_weights=None, walgreens_stores=None, cannibal_radius_miles=3.0):

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    weights = {**DEFAULT_STORE_WEIGHTS, **(slider_weights or {})}
    stamp("="*65); stamp("WALGREENS DISTRIBUTION OPTIMIZER v2"); stamp("="*65)

    if npi_data is None:
        from glob import glob
        for path in ["data/pharmacy_data.csv", "data/pharmacy.csv", "data/pharmacies.csv",
                     "data/npi_tagged.csv"] + glob("data/**/subset*_Table*_filter.xlsx", recursive=True):
            if Path(path).exists():
                stamp(f"Loading: {path}")
                npi_data = pd.read_excel(path, dtype=str) if path.endswith((".xlsx",".xls")) else pd.read_csv(path, dtype=str, low_memory=False)
                break
    if npi_data is None: raise FileNotFoundError("No pharmacy data found.")
    if isinstance(npi_data, (str, Path)):
        p = Path(npi_data)
        npi_data = pd.read_excel(p, dtype=str) if p.suffix in (".xlsx",".xls") else pd.read_csv(p, dtype=str, low_memory=False)

    tagged = tag_all_pharmacies(npi_data)
    competition = compute_competition_metrics(tagged)
    stores, merged = merge_store_data(zcta_scores_path, competition, tagged, walgreens_stores)

    stamp("="*65); stamp("ECONOMIC PILLARS"); stamp("="*65)
    stores["store_revenue"], rc = compute_store_revenue(stores, weights)
    stores["store_cost"], cc = compute_store_cost(stores, weights)
    stores["store_position"], pc = compute_competitive_position(stores, weights, cannibal_radius_miles)
    for n, v in {**{f"rev_{k}":v for k,v in rc.items()}, **{f"cost_{k}":v for k,v in cc.items()}, **{f"pos_{k}":v for k,v in pc.items()}}.items():
        stores[n] = v

    stores["store_viability"] = compute_store_viability(stores["store_revenue"], stores["store_cost"], stores["store_position"], weights["alpha_revenue_vs_cost"], weights["beta_position_weight"])
    stores["viability_rank"] = stores["store_viability"].rank(ascending=False, method="min").astype(int)
    stamp(f"Viability: mean={stores['store_viability'].mean():.3f} std={stores['store_viability'].std():.3f}")

    stores = assign_strategic_actions(stores)
    pairs_df = find_consolidation_pairs(stores, tagged, cannibal_radius_miles)

    stamp("="*65); stamp("ML LAYERS"); stamp("="*65)
    X, fn, fd = build_store_features(stores)
    stores, ml = run_ml_layers(stores, X, fn, fd, str(out))
    stores, centroids = cluster_stores(stores, n_clusters)
    briefs_df = generate_store_briefs(stores, n_briefs, api_key)
    stores["tier"] = pd.cut(stores["store_viability"], bins=[-0.01,.20,.40,.60,.80,1.01], labels=["Critical","At-Risk","Moderate","Strong","Fortress"])
    summary = portfolio_summary(stores, pairs_df, centroids)

    stamp("="*65); stamp("WRITING OUTPUTS"); stamp("="*65)
    stores.to_csv(out/"store_viability_scores.csv", index=False)
    pairs_df.to_csv(out/"consolidation_pairs.csv", index=False)
    centroids.to_csv(out/"archetype_centroids.csv", index=False)
    briefs_df.to_csv(out/"store_briefs.csv", index=False)
    with open(out/"slider_config.json","w") as f: json.dump({"sliders": [{"id":k,"default":v} for k,v in weights.items()]}, f, indent=2)
    with open(out/"portfolio_summary.json","w") as f: json.dump(summary, f, indent=2, default=lambda o: int(o) if isinstance(o,np.integer) else float(o) if isinstance(o,np.floating) else str(o))

    stamp("="*65); stamp("HIGHLIGHTS"); stamp("="*65)
    stamp(f"Stores: {len(stores):,} | Closures: {(stores['action']=='CLOSURE CANDIDATE').sum():,} | Pairs: {len(pairs_df):,} | Protect: {(stores['action']=='PROTECT & INVEST').sum():,}")
    cols = [c for c in ["store_id","state","store_viability","action","archetype_name"] if c in stores.columns]
    stamp("Top 5:"); print(stores.nlargest(5,"store_viability")[cols].to_string(index=False))
    stamp("Bottom 5:"); print(stores.nsmallest(5,"store_viability")[cols].to_string(index=False))
    stamp(f"All outputs in {out}/"); stamp("Done.")

    return {"stores": stores, "pairs": pairs_df, "centroids": centroids, "briefs": briefs_df, "summary": summary, "ml_artifacts": ml}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--zcta-scores", default="results_v2/profit_scores.csv")
    ap.add_argument("--npi", default=None)
    ap.add_argument("--wag-stores", default=None)
    ap.add_argument("--out", default="results_walgreens")
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--n-briefs", type=int, default=50)
    ap.add_argument("--cannibal-radius", type=float, default=3.0)
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()
    run_walgreens_pipeline(a.zcta_scores, a.npi, output_dir=a.out, n_clusters=a.n_clusters,
                          n_briefs=a.n_briefs, api_key=a.api_key, walgreens_stores=a.wag_stores,
                          cannibal_radius_miles=a.cannibal_radius)
