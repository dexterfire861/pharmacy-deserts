# =============================================================================
# IFAE ZIP/ZCTA Ranking — Composite + Expected-Access (Poisson mean + NB variance)
# Enhancements:
# - OOF μ calibration PER STATE with NATIONAL fallback (prevents zeroing 'ALL')
# - Also compute URBAN-ONLY per-state calibration (QA)
# - Deviance residuals for underserved score; Pearson also exported
# - Neighbor QA: geo kNN (lon/lat) -> fallback to feature kNN, median neighbor rates
# - Suspicious-zero flag: big pop, 0 stores, high μ̂, neighbors normal
# - Alternative blend: composite 50% + normalized positive deficit_rate10k 50%
# - tqdm progress; safe density spline; exposure=population
#
# Outputs:
#   results/national_ifae_rank.csv               (main: residual-based blend)
#   results/national_ifae_rank_alt_deficit.csv   (alt: deficit-rate blend)
#   results/topK_ifae_urban.csv
#   results/bottomK_ifae.csv                     (pop>0 only)
#   results/qa_expected_vs_observed.csv
#   results/glm_full_coefficients.csv
# =============================================================================

import time, warnings, math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
from patsy import dmatrices, bs  # expose 'bs' with eval_env=1

# tqdm
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kw): return x

# === CONFIG ===
FINANCIAL_CSV  = "data/financial_data.csv"      # needs: ZCTA/ZIP + S1901_C01_012E
HEALTH_CSV     = "data/health_data.csv"         # needs: ZCTA5, GHLTH_CrudePrev
PHARMACY_CSV   = "data/pharmacy_data.csv"       # needs: ZIP, NAME; optional STATE/COUNTY and lon/lat columns
POPULATION_CSV = "data/population_data.csv"     # teammate used skiprows=10; autodetect

# Optional (auto-disabled if thin/missing)
AQI_CSV        = "data/AQI_data.csv"            # needs: ZIP, Arithmetic Mean, Observation Count
HHI_XLSX       = "data/HHI_data.xlsx"           # needs: ZCTA, HHB_SCORE

# Prefer STATE over COUNTY for grouping
REGION_PRIORITY = ["STATE","state","STATEFP","statefp","county","COUNTY","county_fips"]

OUT_DIR        = "results"
TOP_K          = 10
MIN_POP_TOPK   = 1000

# Urban thresholds (km^2)
DENS_CORE_KM2   = 400.0
DENS_FRINGE_KM2 = 200.0
URBAN_USE_CORE  = False

CV_N_SPLITS_MAX = 3

# Neighbor config
NEIGHBOR_K                      = 10
GEO_COVERAGE_MIN                = 0.20   # need >=20% ZIPs with centroids to use geo kNN
SUSPICIOUS_ZERO_MIN_POP         = 20000
SUSPICIOUS_ZERO_MIN_EXPECTED    = 3.0    # expected count >= 3 OR expected rate >= 2 per 10k
SUSPICIOUS_ZERO_MIN_RATE10K     = 2.0
SUSPICIOUS_ZERO_MIN_NEIGH_RATE  = 1.0

# ---------- logging ----------
_T0 = time.time()
def stamp(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now} +{time.time()-_T0:6.2f}s] {msg}", flush=True)

class Step:
    def __init__(self, name): self.name=name; self.t0=None
    def __enter__(self): self.t0=time.time(); stamp(f"▶ {self.name} ..."); return self
    def __exit__(self, et, ev, tb):
        dt = time.time()-self.t0
        stamp(("✓ " if et is None else "✖ ") + f"{self.name} {'done' if et is None else 'failed'} in {dt:.2f}s")

# ---------- utils ----------
def read_csv_smart(path, **kw):
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"Missing file: {path}")
    try:
        return pd.read_csv(path, low_memory=False, **kw)
    except Exception as e:
        stamp(f"CSV read warning: {e}. Retrying vanilla.")
        return pd.read_csv(path)

def coerce_zcta(series):
    s = series.astype(str).str.extract(r"(\d{3,5})", expand=False)
    return s.fillna("").str.zfill(5)

def pct_rank(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    return s.rank(pct=True, method='average').fillna(0.5)

def norm01(x):
    x = pd.Series(x, dtype=float)
    return (x - x.min())/(x.max() - x.min() + 1e-12)

def safe_log1p(v):
    return np.log1p(np.maximum(v, 0.0))

def pick_first(*names, in_df=None):
    if in_df is None: return None
    for n in names:
        if n in in_df.columns: return n
    return None

# optional loaders
def read_aqi_data(file_path):
    df = read_csv_smart(file_path)
    zip_cols = [c for c in df.columns if str(c).upper().startswith('ZIP')]
    if not zip_cols: raise KeyError("AQI CSV needs a ZIP column.")
    zip_col = zip_cols[-1]
    df['zip'] = df[zip_col].astype(str).str.extract(r'(\d{5})')[0].fillna('').str.zfill(5)
    if 'Arithmetic Mean' not in df.columns or 'Observation Count' not in df.columns:
        raise KeyError("AQI CSV must include 'Arithmetic Mean' and 'Observation Count'.")
    val_col, w_col = 'Arithmetic Mean', 'Observation Count'
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df[w_col]   = pd.to_numeric(df[w_col], errors='coerce').fillna(0)
    if 'Parameter Name' in df.columns:
        df = df[df['Parameter Name'].astype(str).str.contains('PM2.5', na=False)]
    df = df.dropna(subset=['zip', val_col]); df = df[df[w_col] > 0]
    grp = df.groupby('zip', group_keys=False).apply(
        lambda g: pd.Series({'aqi': np.average(g[val_col], weights=g[w_col]),
                             'obs_total': g[w_col].sum()})
    ).reset_index()
    return grp[['zip','aqi','obs_total']]

def read_hhi_excel(file_path):
    df = pd.read_excel(file_path, dtype={'ZCTA': str})
    if 'ZCTA' not in df.columns: raise ValueError("HHI Excel must contain 'ZCTA'.")
    df['zip'] = df['ZCTA'].astype(str).str.extract(r'(\d{5})')[0].fillna('').str.zfill(5)
    out = pd.DataFrame({'zip': df['zip']})
    if 'HHB_SCORE' in df.columns: out['heat_hhb'] = pd.to_numeric(df['HHB_SCORE'], errors='coerce')
    if 'NBE_SCORE' in df.columns: out['nbe_score'] = pd.to_numeric(df['NBE_SCORE'], errors='coerce')
    if 'OVERALL_SCORE' in df.columns: out['hhi_overall'] = pd.to_numeric(df['OVERALL_SCORE'], errors='coerce')
    return out.dropna(subset=['zip']).drop_duplicates(subset=['zip'])

# --- geo helpers ---
def extract_lon_lat_cols(df):
    lon_candidates = ["lon","longitude","LON","LONGITUDE","X","x"]
    lat_candidates = ["lat","latitude","LAT","LATITUDE","Y","y"]
    lon_col = pick_first(*lon_candidates, in_df=df)
    lat_col = pick_first(*lat_candidates, in_df=df)
    if lon_col and lat_col:
        lon = pd.to_numeric(df[lon_col], errors='coerce')
        lat = pd.to_numeric(df[lat_col], errors='coerce')
        good = lat.between(-90,90) & lon.between(-180,180)
        return lon_col, lat_col, good
    return None, None, None

def build_zcta_centroids_from_pharm(ph):
    lon_col, lat_col, good = extract_lon_lat_cols(ph)
    if lon_col is None or lat_col is None:
        return pd.DataFrame(columns=["ZCTA5","centroid_lon","centroid_lat"])
    # average by ZCTA
    tmp = ph.loc[good].copy()
    tmp["ZCTA5"] = coerce_zcta(tmp["ZCTA5"])
    grp = tmp.groupby("ZCTA5", as_index=False).agg(
        centroid_lon=(lon_col, "mean"),
        centroid_lat=(lat_col, "mean")
    )
    # sanity ranges
    grp = grp[grp["centroid_lat"].between(-90,90) & grp["centroid_lon"].between(-180,180)]
    return grp

# ===========================
# LOAD + PREP
# ===========================
with Step("Load FINANCIAL (income)"):
    fin = read_csv_smart(FINANCIAL_CSV)
    zcta_col = next((c for c in ["ZCTA5","zcta5","ZCTA","zcta","ZIP","Zip","NAME","Name","name"] if c in fin.columns), None)
    if zcta_col is None: raise KeyError("No ZCTA/ZIP column in financial.")
    if "S1901_C01_012E" not in fin.columns: raise KeyError("Need S1901_C01_012E (median HH income).")
    fin = fin.rename(columns={zcta_col:"ZCTA5"}); fin["ZCTA5"] = coerce_zcta(fin["ZCTA5"])
    fin = fin[["ZCTA5","S1901_C01_012E"]].copy()
    fin["S1901_C01_012E"] = pd.to_numeric(fin["S1901_C01_012E"], errors="coerce")
    stamp(f"FIN rows={len(fin)}, null income={fin['S1901_C01_012E'].isna().sum()}")

with Step("Load HEALTH (poor general health %)"):
    hlth = read_csv_smart(HEALTH_CSV)
    if "ZCTA5" not in hlth.columns:
        for cand in ["ZCTA","zcta","ZIP","Zip","name","NAME"]:
            if cand in hlth.columns: hlth = hlth.rename(columns={cand:"ZCTA5"}); break
    if "ZCTA5" not in hlth.columns or "GHLTH_CrudePrev" not in hlth.columns:
        raise KeyError("HEALTH_CSV must contain ZCTA5 and GHLTH_CrudePrev.")
    hlth["ZCTA5"] = coerce_zcta(hlth["ZCTA5"])
    hlth["GHLTH_CrudePrev"] = pd.to_numeric(hlth["GHLTH_CrudePrev"], errors="coerce")
    hlth = hlth[["ZCTA5","GHLTH_CrudePrev"]]
    stamp(f"HEALTH rows={len(hlth)}, null health%={hlth['GHLTH_CrudePrev'].isna().sum()}")

with Step("Load PHARMACY (access)"):
    ph = read_csv_smart(PHARMACY_CSV)
    for needed in ["ZIP","NAME"]:
        if needed not in ph.columns: raise KeyError(f"PHARMACY_CSV must contain {needed}.")
    ph["ZCTA5"] = coerce_zcta(ph["ZIP"])
    ph_cnt = ph.groupby("ZCTA5", dropna=False)["NAME"].nunique(dropna=True).reset_index().rename(columns={"NAME":"pharmacies_count"})
    # region (STATE best)
    region_df = None
    for rcol in REGION_PRIORITY:
        if rcol in ph.columns:
            region_df = ph[["ZCTA5", rcol]].drop_duplicates("ZCTA5").rename(columns={rcol:"REGION"}); break
    if region_df is None:
        region_df = ph[["ZCTA5"]].drop_duplicates().assign(REGION="ALL")
    # attempt centroids
    centroids = build_zcta_centroids_from_pharm(ph).rename(columns={"centroid_lon":"LON","centroid_lat":"LAT"})
    stamp(f"PHARMACY unique ZCTA5={ph_cnt['ZCTA5'].nunique()}, pharmacies={ph_cnt['pharmacies_count'].sum()}")

with Step("Load POPULATION (skiprows=10 + autodetect)"):
    pop = read_csv_smart(POPULATION_CSV, skiprows=10)
    code_col = next((c for c in ["ZCTA5","ZCTA","ZIP","Zip","GEOID","geoid","NAME","name"] if c in pop.columns), None)
    if code_col is None:
        obj_cols = [c for c in pop.columns if pop[c].dtype == object]
        code_col = obj_cols[0] if obj_cols else pop.columns[0]
    pop = pop.rename(columns={code_col:"ZCTA5"}); pop["ZCTA5"] = coerce_zcta(pop["ZCTA5"])
    pop_col = next((c for c in ["POP","Population","population","TOTAL_POP","TotPop","DP05_0001E","pop"] if c in pop.columns), None)
    if pop_col is None:
        num_cols = [c for c in pop.columns if pd.api.types.is_numeric_dtype(pop[c])]
        sums = {c: pd.to_numeric(pop[c], errors="coerce").sum(skipna=True) for c in num_cols}
        pop_col = max(sums, key=sums.get) if sums else None
    if pop_col is None: raise KeyError("Could not infer population column in POPULATION_CSV.")
    land_col = next((c for c in ["land_area_km2","Land_Area_km2","ALAND_KM2","aland_km2","ALAND","area","AREA_KM2","ALAND_SQMI","AREALAND"] if c in pop.columns), None)
    keep_cols = ["ZCTA5", pop_col] + ([land_col] if land_col else [])
    pop = pop[keep_cols].copy().rename(columns={pop_col:"population"})
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    if land_col:
        pop = pop.rename(columns={land_col:"land_area_raw"})
        pop["land_area_raw"] = pd.to_numeric(pop["land_area_raw"], errors="coerce")
    stamp(f"POP rows={len(pop)}, null pop={pop['population'].isna().sum()}, land_col={land_col}")

# Optional: AQI + HHI
try:
    with Step("Load AQI (annual weighted PM2.5)"):
        aqi_annual = read_aqi_data(AQI_CSV); stamp(f"AQI rows={len(aqi_annual)} (coverage ~{100*len(aqi_annual)/max(1, pop['ZCTA5'].nunique()):.1f}%)")
except Exception as e:
    aqi_annual = None; warnings.warn(f"AQI not loaded: {e}")
try:
    with Step("Load HHI (heat vulnerability)"):
        hhi = read_hhi_excel(HHI_XLSX); stamp(f"HHI rows={len(hhi)}")
except Exception as e:
    hhi = None; warnings.warn(f"HHI not loaded: {e}")

# Merge
with Step("Merge all features by ZCTA/ZIP"):
    df = fin.merge(hlth, on="ZCTA5", how="outer") \
            .merge(pop, on="ZCTA5", how="outer") \
            .merge(ph_cnt, on="ZCTA5", how="left") \
            .merge(region_df, on="ZCTA5", how="left")
    if aqi_annual is not None and not aqi_annual.empty:
        df = df.merge(aqi_annual.rename(columns={'zip':'ZCTA5'}), on="ZCTA5", how="left")
    if hhi is not None and not hhi.empty:
        df = df.merge(hhi.rename(columns={'zip':'ZCTA5'}), on="ZCTA5", how="left")
    if not centroids.empty:
        df = df.merge(centroids, on="ZCTA5", how="left")
    df["REGION"] = df["REGION"].fillna("ALL").astype(str)
    stamp(f"Merged shape: (rows={df.shape[0]}, cols={df.shape[1]})")

# Feature engineering
with Step("Feature engineering"):
    df["median_income"]   = pd.to_numeric(df["S1901_C01_012E"], errors="coerce")
    df["poor_health_pct"] = pd.to_numeric(df["GHLTH_CrudePrev"], errors="coerce")
    df["pharmacies_count"]= pd.to_numeric(df["pharmacies_count"], errors="coerce").fillna(0)
    for col in ["median_income","poor_health_pct"]:
        med = df[col].median(skipna=True); df[col] = df[col].fillna(med)
    denom = df["pharmacies_count"].replace(0, 1)
    df["pop_per_pharmacy"] = np.where(df["population"].gt(0), df["population"]/denom, np.nan)
    # density
    if "land_area_raw" in df.columns:
        med_area = np.nanmedian(df["land_area_raw"])
        df["land_area_km2"] = np.where(np.isfinite(med_area) & (med_area > 1e5), df["land_area_raw"]/1e6, df["land_area_raw"])
        df["pop_density"] = np.where(df["land_area_km2"].gt(0), df["population"]/df["land_area_km2"], np.nan)
    else:
        df["pop_density"] = np.nan
    # optional extras
    if "aqi" in df.columns:       df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    if "heat_hhb" in df.columns:  df["heat_hhb"] = pd.to_numeric(df["heat_hhb"], errors="coerce")
    # percentiles
    df["income_pct_inv"]  = 1 - pct_rank(df["median_income"])
    df["health_pct"]      = pct_rank(df["poor_health_pct"])
    df["access_pct_inv"]  = pct_rank(df["pop_per_pharmacy"])
    df["density_pct"]     = pct_rank(df["pop_density"]) if df["pop_density"].notna().any() else 0.5
    df["aqi_pct"]         = pct_rank(df["aqi"])         if "aqi" in df.columns else 0.5
    df["heat_pct"]        = pct_rank(df["heat_hhb"])    if "heat_hhb" in df.columns else 0.5
    stamp("Feature percentiles constructed.")

# Composite (transparent)
with Step("Compute composite"):
    cov = {
        'aqi':    ("aqi" in df.columns) and df["aqi"].notna().mean(),
        'heat':   ("heat_hhb" in df.columns) and df["heat_hhb"].notna().mean(),
        'density': df["pop_density"].notna().mean()
    }
    use_aqi  = bool(cov["aqi"])  and (cov["aqi"]  >= 0.30)
    use_heat = bool(cov["heat"]) and (cov["heat"] >= 0.30)
    use_dens = bool(cov["density"]) and (cov["density"] >= 0.20)

    w_income, w_health, w_access = 0.30, 0.35, 0.30
    w_aqi   = 0.05 if use_aqi  else 0.00
    w_heat  = 0.05 if use_heat else 0.00
    w_dens  = 0.00 if not use_dens else 0.02
    Wsum = w_income + w_health + w_access + w_aqi + w_heat + w_dens
    w_income, w_health, w_access, w_aqi, w_heat, w_dens = [w/Wsum for w in [w_income, w_health, w_access, w_aqi, w_heat, w_dens]]

    df["composite"] = (
        w_income * df["income_pct_inv"] +
        w_health * df["health_pct"]     +
        w_access * df["access_pct_inv"] +
        w_aqi    * (df["aqi_pct"]     if use_aqi  else 0) +
        w_heat   * (df["heat_pct"]    if use_heat else 0) +
        w_dens   * (df["density_pct"] if use_dens else 0)
    )

# ==========================
# EXPECTED-ACCESS (Poisson mean + NB variance)
# ==========================
with Step("Expected-access model (OOF pooled CV by STATE; calibrated per-state with national fallback; final refit w/ STATE FE)"):
    region_col = "REGION"
    # Rows with pop > 0
    m = df["population"].fillna(0) > 0
    cols_base = ["ZCTA5","pharmacies_count","population",
                 "income_pct_inv","health_pct","density_pct","pop_density",
                 "aqi_pct","heat_pct", region_col, "LON","LAT"]
    work = df.loc[m, cols_base].copy()

    # Optionals → neutral 0.5
    for c in ["aqi_pct","heat_pct","density_pct"]:
        if c not in work.columns: work[c] = 0.5
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.5)

    # Center predictors
    for c in ["income_pct_inv","health_pct","aqi_pct","heat_pct","density_pct"]:
        if c in work.columns:
            work[c + "_c"] = (pd.to_numeric(work[c], errors="coerce").fillna(0.5) - 0.5)

    # Density spline guard
    use_raw_density = work["pop_density"].notna().any()
    include_density_spline = False
    density_spline_term = None
    if use_raw_density:
        work["pop_density_clipped"] = work["pop_density"].clip(
            lower=np.nanpercentile(work["pop_density"], 1),
            upper=np.nanpercentile(work["pop_density"], 99)
        )
        if work["pop_density_clipped"].notna().any():
            include_density_spline = True
            density_spline_term = "bs(pop_density_clipped, df=3, degree=3)"
    else:
        if work["density_pct"].nunique(dropna=True) > 1:
            include_density_spline = True
            density_spline_term = "bs(density_pct, df=3, degree=3)"

    # ---------- OOF CV pooled (NO C(REGION)) ----------
    terms_cv = ["1", "income_pct_inv_c", "health_pct_c", "aqi_pct_c", "heat_pct_c"]
    if include_density_spline: terms_cv.insert(1, density_spline_term)
    formula_cv = "pharmacies_count ~ " + " + ".join(terms_cv)

    y_cv, X_cv = dmatrices(formula_cv, work, return_type="dataframe", eval_env=1)
    valid_idx = X_cv.index
    y = y_cv.iloc[:, 0]
    exposure_all = work.loc[valid_idx, "population"].clip(lower=1.0).astype(float)
    groups = work.loc[valid_idx, region_col].astype(str)

    n_groups = groups.nunique(); n_params = X_cv.shape[1]; n_rows = len(valid_idx)
    n_splits = min(CV_N_SPLITS_MAX, n_groups) if n_groups >= 2 else min(3, max(2, n_rows // 2))
    print(f"Valid rows: {n_rows} | params(cv): {n_params} | groups: {n_groups} | folds: {n_splits}")

    fam = sm.families.Poisson()
    mu_oof = pd.Series(index=valid_idx, dtype=float); devs = []

    if n_groups >= 2:
        splitter = GroupKFold(n_splits=n_splits).split(X_cv, y, groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_cv, y)

    for tr, te in tqdm(splitter, total=n_splits, desc=f"GLM CV ({n_splits} folds)", unit="fold", leave=False):
        X_tr, X_te = X_cv.iloc[tr], X_cv.iloc[te]
        y_tr, y_te = y.iloc[tr],  y.iloc[te]
        exp_tr, exp_te = exposure_all.iloc[tr], exposure_all.iloc[te]

        model = sm.GLM(y_tr, X_tr, family=fam, exposure=exp_tr)
        res   = model.fit()
        mu_te = res.predict(X_te, exposure=exp_te)
        mu_oof.iloc[te] = mu_te

        devs.append(float(np.mean((y_te - mu_te)**2 / (mu_te + 1e-6))))

    # Estimate alpha from OOF (pre-calibration)
    def estimate_alpha(y_vec, mu_vec, p):
        yv = y_vec.values.astype(float); muv = mu_vec.values.astype(float)
        n  = np.isfinite(yv*muv).sum(); dof = max(n - p, 1)
        def f(alpha):
            denom = muv + alpha*(muv**2) + 1e-12
            return ( ((yv - muv)**2 / denom).sum() ) - dof
        lo, hi = 0.0, 1.0
        for _ in range(40):
            if f(hi) < 0: break
            hi *= 2.0
            if hi > 1e6: break
        for _ in range(60):
            mid = 0.5*(lo+hi); val = f(mid)
            lo, hi = (mid, hi) if val > 0 else (lo, mid)
        return max(0.0, 0.5*(lo+hi))

    alpha_hat = estimate_alpha(y, mu_oof, X_cv.shape[1])

    # ---------- Per-STATE calibration with NATIONAL fallback ----------
    obs = work.loc[valid_idx, "pharmacies_count"].astype(float)
    pop = exposure_all.astype(float)
    reg = work.loc[valid_idx, region_col].astype(str)

    by_state = pd.DataFrame({"y":obs, "mu":mu_oof, "REGION":reg})
    # national factor
    nat_y, nat_mu = by_state["y"].sum(), by_state["mu"].sum()
    nat_factor = float(nat_y / nat_mu) if nat_mu > 0 else 1.0

    grp_sum = by_state.groupby("REGION")[["y","mu"]].sum()
    grp_sum["cal"] = grp_sum.apply(lambda r: (r["y"]/r["mu"]) if r["mu"]>0 else np.nan, axis=1)
    # fallback: NA/0 -> national factor
    grp_sum["cal"] = grp_sum["cal"].where(np.isfinite(grp_sum["cal"]) & (grp_sum["cal"]>0), nat_factor)
    cal_map = grp_sum["cal"].to_dict()

    mu_oof_cal = mu_oof * reg.map(cal_map).astype(float).values

    # ---------- Urban-only per-STATE calibration (QA) ----------
    has_area   = df["pop_density"].notna()
    dens_rule  = DENS_CORE_KM2 if URBAN_USE_CORE else DENS_FRINGE_KM2
    urban_mask = ( (has_area & (df["pop_density"] >= dens_rule)) |
                   (~has_area & (df["population"].fillna(0) >= 5000)) )
    is_valid_urban = pd.Series(False, index=df.index); is_valid_urban.loc[valid_idx] = urban_mask.loc[valid_idx]
    urb_tab = pd.DataFrame({
        "y": obs,
        "mu": mu_oof,
        "REGION": reg,
        "is_urban": is_valid_urban.loc[valid_idx].values
    })
    urb_sum = urb_tab[urb_tab["is_urban"]].groupby("REGION")[["y","mu"]].sum()
    urb_sum["cal"] = urb_sum.apply(lambda r: (r["y"]/r["mu"]) if r["mu"]>0 else np.nan, axis=1)
    cal_map_urban = urb_sum["cal"].to_dict()
    # fallback to per-state cal, then to national factor
    cal_map_urban = {k:(float(v) if (v is not None and np.isfinite(v) and v>0) else cal_map.get(k, nat_factor)) for k,v in cal_map_urban.items()}
    mu_oof_cal_urban = mu_oof * reg.map(cal_map_urban).astype(float).values

    # Diagnostics
    stamp(f"CV mean scaled MSE (lower better): {np.mean(devs):.4f}")
    print(f"Alpha_hat ≈ {alpha_hat:.4f}")
    base_rate10k = 1e4 * float(obs.sum()) / max(float(pop.sum()), 1.0)
    def qd(series): return {k: float(series.quantile(k)) for k in [0.5, 0.95, 0.99]}
    rate_oof_raw  = (mu_oof       / pop).replace([np.inf, -np.inf], np.nan) * 1e4
    rate_oof_cal  = (mu_oof_cal   / pop).replace([np.inf, -np.inf], np.nan) * 1e4
    rate_oof_calU = (mu_oof_cal_urban / pop).replace([np.inf, -np.inf], np.nan) * 1e4
    print(f"Observed baseline ~ {base_rate10k:.2f} per 10k | OOF expected (raw) p50/95/99: {qd(rate_oof_raw)}")
    print(f"Per-state calibrated OOF expected p50/95/99: {qd(rate_oof_cal)}")
    print(f"Urban-only per-state calibrated OOF expected p50/95/99: {qd(rate_oof_calU)}")

    # NB variance + residuals (state-calibrated)
    var_mu_cal = mu_oof_cal + alpha_hat*(mu_oof_cal**2)
    pearson_cal = (obs - mu_oof_cal) / np.sqrt(np.maximum(var_mu_cal, 1e-12))

    # Deviance residuals (robust for zeros)
    yv = obs.values.astype(float); muv = mu_oof_cal.values
    eps = 1e-12
    term = np.where(yv > 0, yv * np.log(np.maximum(yv, eps) / np.maximum(muv, eps)) - (yv - muv), - (yv - muv))
    dev_resid = np.sign(yv - muv) * np.sqrt(2.0 * np.maximum(term, 0.0))

    # Underservice score (higher = more underserved)
    underserved = -pd.Series(dev_resid, index=valid_idx)
    glm_nb_score_local = norm01(underserved)

    # Deficits (OOF, per-state calibrated μ)
    deficit_count_oof = (mu_oof_cal - obs)
    deficit_rate10k_oof = 1e4 * deficit_count_oof / pop.clip(lower=1.0)

    # ---------- Final refit on all valid rows (OPTIONAL REGION FE for QA expectations) ----------
    terms_full = ["1", "income_pct_inv_c", "health_pct_c", "aqi_pct_c", "heat_pct_c"]
    if include_density_spline: terms_full.insert(1, density_spline_term)
    if region_col:
        terms_full.append(f"C({region_col})")
    formula_full = "pharmacies_count ~ " + " + ".join(terms_full)

    y_full, X_full = dmatrices(formula_full, work, return_type="dataframe", eval_env=1)
    idx_full = X_full.index
    exp_full = work.loc[idx_full, "population"].clip(lower=1.0).astype(float)

    fam = sm.families.Poisson()
    res_full = sm.GLM(y_full.iloc[:,0], X_full, family=fam, exposure=exp_full).fit()
    mu_full = pd.Series(res_full.predict(X_full, exposure=exp_full), index=idx_full)

    rate_full = (mu_full / exp_full).replace([np.inf, -np.inf], np.nan) * 1e4
    def qd3(series): return {k: float(series.quantile(k)) for k in [0.5, 0.95, 0.99]}
    print(f"FULL expected per 10k (p50/95/99): {qd3(rate_full)}")

    # Deficits with full refit (QA/planning)
    deficit_count_full = (mu_full - work.loc[idx_full, "pharmacies_count"].astype(float))
    deficit_rate10k_full = 1e4 * deficit_count_full / exp_full.clip(lower=1.0)

    # Export coefficients for QA
    coef = (pd.Series(res_full.params, name="coef")
            .to_frame()
            .join(res_full.bse.rename("se"))
            .assign(z=lambda d: d.coef/d.se))
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    coef_file = Path(OUT_DIR) / "glm_full_coefficients.csv"
    coef.to_csv(coef_file)
    stamp(f"Wrote {coef_file}")

    # --- write back to df ---
    df["glm_nb_score"] = np.nan; df.loc[valid_idx, "glm_nb_score"] = glm_nb_score_local.values
    df["glm_mu_expected_pharm_oof_cal_state"]  = np.nan; df.loc[valid_idx, "glm_mu_expected_pharm_oof_cal_state"]  = mu_oof_cal.values
    df["glm_mu_expected_pharm_oof_cal_state_urban"] = np.nan; df.loc[valid_idx, "glm_mu_expected_pharm_oof_cal_state_urban"] = mu_oof_cal_urban.values
    df["glm_mu_expected_pharm_full"] = np.nan; df.loc[idx_full,  "glm_mu_expected_pharm_full"] = mu_full.values
    df["glm_pearson_resid"] = np.nan; df.loc[valid_idx, "glm_pearson_resid"] = pearson_cal.values
    df["glm_deviance_resid"] = np.nan; df.loc[valid_idx, "glm_deviance_resid"] = dev_resid
    df["deficit_count_oof"] = np.nan;     df.loc[valid_idx, "deficit_count_oof"] = deficit_count_oof.values
    df["deficit_rate10k_oof"] = np.nan;   df.loc[valid_idx, "deficit_rate10k_oof"] = deficit_rate10k_oof.values
    df["deficit_count_full"] = np.nan;    df.loc[idx_full,  "deficit_count_full"] = deficit_count_full.values
    df["deficit_rate10k_full"] = np.nan;  df.loc[idx_full,  "deficit_rate10k_full"] = deficit_rate10k_full.values

# ==========================
# Neighbor QA (geo kNN -> feature kNN fallback)
# ==========================
with Step("Neighbor QA (geo kNN or feature-space fallback)"):
    # base (only valid rows)
    base = df.loc[df["glm_mu_expected_pharm_oof_cal_state"].notna(), ["ZCTA5","REGION","population","pharmacies_count",
                              "glm_mu_expected_pharm_oof_cal_state","LON","LAT",
                              "income_pct_inv","health_pct"]].copy()
    base["obs_rate10k"] = 1e4 * base["pharmacies_count"].astype(float) / base["population"].clip(lower=1.0)
    base["exp_rate10k"] = 1e4 * base["glm_mu_expected_pharm_oof_cal_state"].astype(float) / base["population"].clip(lower=1.0)

    # geo coverage
    geo_ok = base["LON"].notna() & base["LAT"].notna()
    geo_coverage = float(geo_ok.mean()) if len(base) else 0.0
    use_geo = geo_coverage >= GEO_COVERAGE_MIN
    stamp(f"Geo centroid coverage among valid rows: {geo_coverage:.1%} -> {'using GEO kNN' if use_geo else 'fall back to FEATURE kNN'}")

    # outputs
    nb_obs_median = pd.Series(np.nan, index=base.index)
    nb_exp_median = pd.Series(np.nan, index=base.index)

    for state, g in base.groupby("REGION"):
        if len(g) < 3:
            continue
        k = min(NEIGHBOR_K+1, len(g))  # +1 to include self
        if use_geo and (g["LON"].notna().any() and g["LAT"].notna().any()):
            gi = g[g["LON"].notna() & g["LAT"].notna()]
            if len(gi) < 3:  # fallback to feature if too small
                feat = np.c_[ (g["income_pct_inv"]-0.5), (g["health_pct"]-0.5), safe_log1p(g["population"]) ]
                nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(feat)
                dist, ind = nn.kneighbors(feat, return_distance=True)
                for idx_row, neighbors in zip(g.index, ind):
                    neighbors = [g.index[i] for i in neighbors if g.index[i] != idx_row][:NEIGHBOR_K]
                    nb_obs_median.loc[idx_row] = base.loc[neighbors, "obs_rate10k"].median()
                    nb_exp_median.loc[idx_row] = base.loc[neighbors, "exp_rate10k"].median()
            else:
                # haversine expects radians [lat, lon]
                coords = np.radians(gi[["LAT","LON"]].to_numpy())
                nn = NearestNeighbors(n_neighbors=k, metric="haversine").fit(coords)
                dist, ind = nn.kneighbors(coords, return_distance=True)
                gi_idx = gi.index.to_list()
                for row_idx, neighbors in zip(gi.index, ind):
                    nb_idx = [gi_idx[i] for i in neighbors if gi_idx[i] != row_idx][:NEIGHBOR_K]
                    nb_obs_median.loc[row_idx] = base.loc[nb_idx, "obs_rate10k"].median()
                    nb_exp_median.loc[row_idx] = base.loc[nb_idx, "exp_rate10k"].median()
                # fallback for rows without coords
                g_missing = g.index.difference(gi.index)
                if len(g_missing) > 0:
                    feat = np.c_[ (g["income_pct_inv"]-0.5), (g["health_pct"]-0.5), safe_log1p(g["population"]) ]
                    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(feat)
                    dist, ind = nn.kneighbors(feat, return_distance=True)
                    for idx_row, neighbors in zip(g.index, ind):
                        neighbors = [g.index[i] for i in neighbors if g.index[i] != idx_row][:NEIGHBOR_K]
                        nb_obs_median.loc[idx_row] = base.loc[neighbors, "obs_rate10k"].median()
                        nb_exp_median.loc[idx_row] = base.loc[neighbors, "exp_rate10k"].median()
        else:
            feat = np.c_[ (g["income_pct_inv"]-0.5), (g["health_pct"]-0.5), safe_log1p(g["population"]) ]
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(feat)
            dist, ind = nn.kneighbors(feat, return_distance=True)
            for idx_row, neighbors in zip(g.index, ind):
                neighbors = [g.index[i] for i in neighbors if g.index[i] != idx_row][:NEIGHBOR_K]
                nb_obs_median.loc[idx_row] = base.loc[neighbors, "obs_rate10k"].median()
                nb_exp_median.loc[idx_row] = base.loc[neighbors, "exp_rate10k"].median()

    # write back
    df["neighbor_obs_rate10k_median"] = np.nan; df.loc[nb_obs_median.index, "neighbor_obs_rate10k_median"] = nb_obs_median.values
    df["neighbor_exp_rate10k_median"] = np.nan; df.loc[nb_exp_median.index, "neighbor_exp_rate10k_median"] = nb_exp_median.values

    # Suspicious zero flag
    cond_big_pop   = df["population"].fillna(0) >= SUSPICIOUS_ZERO_MIN_POP
    cond_zero      = df["pharmacies_count"].fillna(0) == 0
    cond_high_mu   = (df["glm_mu_expected_pharm_oof_cal_state"].fillna(0) >= SUSPICIOUS_ZERO_MIN_EXPECTED) | \
                     ( (1e4 * df["glm_mu_expected_pharm_oof_cal_state"].fillna(0) / df["population"].clip(lower=1.0)) >= SUSPICIOUS_ZERO_MIN_RATE10K )
    cond_neighbors = df["neighbor_obs_rate10k_median"].fillna(0) >= SUSPICIOUS_ZERO_MIN_NEIGH_RATE
    df["suspicious_zero_flag"] = cond_big_pop & cond_zero & cond_high_mu & cond_neighbors

# ==========================
# Scores, blends, outputs
# ==========================
with Step("Blend composite with expected-access (residual) + alt deficit blend; finalize outputs"):
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)

    # Main blend: deviance residual score
    if "composite" not in df.columns or "glm_nb_score" not in df.columns:
        raise RuntimeError("Missing inputs for blend.")
    df["IFAE_score_residual"] = 0.5*df["composite"] + 0.5*norm01(df["glm_nb_score"])

    # Alt blend: deficit rate per 10k (positive part only)
    pos_def_rate = df["deficit_rate10k_oof"].clip(lower=0)
    df["glm_deficit_score"] = norm01(pos_def_rate)
    df["IFAE_score_deficit"] = 0.5*df["composite"] + 0.5*df["glm_deficit_score"]

    # Choose main score for ranking (residual-based)
    df["IFAE_score"] = df["IFAE_score_residual"]

    # Urban mask for presentation
    has_area   = df["pop_density"].notna()
    dens_rule  = DENS_CORE_KM2 if URBAN_USE_CORE else DENS_FRINGE_KM2
    urban_mask = ( (has_area & (df["pop_density"] >= dens_rule)) |
                   (~has_area & (df["population"].fillna(0) >= 5000)) )
    df["is_urbanish"] = urban_mask

    keep = [
        "ZCTA5","REGION","IFAE_score","IFAE_score_residual","IFAE_score_deficit","is_urbanish",
        "composite","glm_nb_score",
        "median_income","poor_health_pct","population","pharmacies_count",
        "pop_per_pharmacy","income_pct_inv","health_pct","access_pct_inv",
        "density_pct","pop_density",
        "glm_mu_expected_pharm_oof_cal_state","glm_mu_expected_pharm_oof_cal_state_urban","glm_mu_expected_pharm_full",
        "glm_pearson_resid","glm_deviance_resid",
        "deficit_count_oof","deficit_rate10k_oof",
        "deficit_count_full","deficit_rate10k_full",
        "neighbor_obs_rate10k_median","neighbor_exp_rate10k_median","suspicious_zero_flag",
    ]
    if "aqi" in df.columns:       keep += ["aqi","aqi_pct","obs_total"]
    if "heat_hhb" in df.columns:  keep += ["heat_hhb","heat_pct"]
    if "land_area_km2" in df.columns: keep += ["land_area_km2"]
    if "LON" in df.columns and "LAT" in df.columns: keep += ["LON","LAT"]

    ranked = df[keep].copy().sort_values("IFAE_score", ascending=False).reset_index(drop=True)

    # Files
    out_full_main   = out / "national_ifae_rank.csv"
    out_full_alt    = out / "national_ifae_rank_alt_deficit.csv"
    out_top_urban   = out / "topK_ifae_urban.csv"
    out_bottom      = out / "bottomK_ifae.csv"

    ranked.to_csv(out_full_main, index=False)
    ranked.sort_values("IFAE_score_deficit", ascending=False).to_csv(out_full_alt, index=False)

    eligible = ranked[(ranked["population"].fillna(0) >= MIN_POP_TOPK) & (ranked["is_urbanish"])]
    top_presentable = eligible.head(TOP_K)

    # Bottom list filtered to pop>0
    bottom_presentable = ranked[ ranked["population"].fillna(0) > 0 ].tail(TOP_K)

    qa_cols = [
        "ZCTA5","REGION","population","pharmacies_count",
        "glm_mu_expected_pharm_oof_cal_state","glm_mu_expected_pharm_oof_cal_state_urban","glm_mu_expected_pharm_full",
        "deficit_count_oof","deficit_rate10k_oof",
        "deficit_count_full","deficit_rate10k_full",
        "glm_pearson_resid","glm_deviance_resid",
        "neighbor_obs_rate10k_median","neighbor_exp_rate10k_median","suspicious_zero_flag",
        "IFAE_score","IFAE_score_deficit","composite","is_urbanish"
    ]
    qa = ranked[qa_cols].copy()
    qa_file = out / "qa_expected_vs_observed.csv"; qa.to_csv(qa_file, index=False)

    stamp(f"Wrote {out_full_main}")
    stamp(f"Wrote {out_full_alt}")
    stamp(f"Wrote {out_top_urban} (pop ≥ {MIN_POP_TOPK}, urban mask)")
    stamp(f"Wrote {out_bottom}")
    stamp(f"Wrote {qa_file}")

    print("\nTop (urban, pop ≥ {:,}) preview:\n".format(MIN_POP_TOPK), top_presentable.head(min(5, TOP_K)).to_string(index=False))
    print("\nBottom {} preview (pop>0 only):\n".format(TOP_K), bottom_presentable.tail(min(5, TOP_K)).to_string(index=False))

stamp("All done. 🚀")
