# =============================================================================
# IFAE ZIP/ZCTA Ranking — GLM (Poisson mean + NB variance proxy) + Hybrid Residual GBDT/XGB
#
# Reads PHARMACY from NPI Excel bundle split across 10 .xlsm/.xlsx files in:
#   data/Pharmacy_list_ZIP_fixed_final/subsetX_Table1_filter.xlsm (or .xlsx), X=1..10
# Each workbook contains a sheet named exactly: subsetX_Table1_filter
#
# Outputs into ./results:
#   - national_ifae_rank.csv
#   - national_ifae_rank_alt_deficit.csv
#   - topK_ifae_urban.csv
#   - bottomK_ifae.csv
#   - qa_expected_vs_observed.csv
#   - glm_full_coefficients.csv
# =============================================================================

import os, time, warnings, math, re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor

import statsmodels.api as sm
from patsy import dmatrices  # bs available if needed

# tqdm (optional)
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kw): return x

# Optional XGBoost
try:
    import xgboost as xgb
    _XGB_OK = True
except Exception:
    _XGB_OK = False

# =========================
# CONFIG
# =========================
FINANCIAL_CSV  = "data/financial_data.csv"      # Needs: ZCTA/ZIP/NAME + S1901_C01_012E (median HH income)
HEALTH_CSV     = "data/health_data.csv"         # Needs: ZCTA5/ZCTA/ZIP/NAME + GHLTH_CrudePrev
POPULATION_CSV = "data/population_data.csv"     # ACS ZCTA, will auto-infer population column; tries skiprows=10 then 0

# Optional extras (auto-disabled if missing)
AQI_CSV        = "data/AQI_data.csv"            # Needs: ZIP, Arithmetic Mean, Observation Count (pref PM2.5)
HHI_XLSX       = "data/HHI_data.xlsx"           # Needs: ZCTA + optional HHB_SCORE / NBE_SCORE / OVERALL_SCORE

# Pharmacy bundle (xlsm/xlsx; specific sheets like "subset3_Table1_filter")
NPI_GLOBS      = [
    "data/Pharmacy_list_ZIP_fixed_final/subset*_Table1_filter.xlsm",
    "data/Pharmacy_list_ZIP_fixed_final/subset*_Table1_filter.xlsx",
]

OUT_DIR        = "results"
TOP_K          = 10
MIN_POP_TOPK   = 1000

# Urban density thresholds (km^2)
DENS_CORE_KM2   = 400.0
DENS_FRINGE_KM2 = 200.0
URBAN_USE_CORE  = False

CV_N_SPLITS_MAX = 3

# Neighbor QA config
NEIGHBOR_K                      = 10
GEO_COVERAGE_MIN                = 0.20  # only used if LON/LAT exist
SUSPICIOUS_ZERO_MIN_POP         = 20000
SUSPICIOUS_ZERO_MIN_EXPECTED    = 3.0
SUSPICIOUS_ZERO_MIN_RATE10K     = 2.0
SUSPICIOUS_ZERO_MIN_NEIGH_RATE  = 1.0

# ---------- logging ----------
_T0 = time.time()
def stamp(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now} + {time.time()-_T0:6.2f}s] {msg}", flush=True)

class Step:
    def __init__(self, name): self.name=name; self.t0=None
    def __enter__(self): self.t0=time.time(); stamp(f"▶ {self.name} ..."); return self
    def __exit__(self, et, ev, tb):
        dt = time.time()-self.t0
        stamp(("✓ " if et is None else "✖ ") + f"{self.name} {'done' if et is None else 'failed'} in {dt:.2f}s")

# ---------- utils ----------
def read_csv_smart(path, **kw):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
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

    need = ['Arithmetic Mean', 'Observation Count']
    for n in need:
        if n not in df.columns:
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
    df = pd.read_excel(file_path, dtype={'ZCTA': str}, engine="openpyxl")
    if 'ZCTA' not in df.columns:
        raise ValueError("HHI Excel must contain 'ZCTA'.")
    df['zip'] = df['ZCTA'].astype(str).str.extract(r'(\d{5})')[0].fillna('').str.zfill(5)
    out = pd.DataFrame({'zip': df['zip']})
    if 'HHB_SCORE' in df.columns: out['heat_hhb'] = pd.to_numeric(df['HHB_SCORE'], errors='coerce')
    if 'NBE_SCORE' in df.columns: out['nbe_score'] = pd.to_numeric(df['NBE_SCORE'], errors='coerce')
    if 'OVERALL_SCORE' in df.columns: out['hhi_overall'] = pd.to_numeric(df['OVERALL_SCORE'], errors='coerce')
    return out.dropna(subset=['zip']).drop_duplicates(subset=['zip'])

# ------------------------------
# PHARMACY loader for xlsm/xlsx sheets named subset*_Table1_filter
# ------------------------------
from glob import glob

def _norm_upper(s):
    return s.astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)

def _normalize_colnames(cols):
    def norm_one(c):
        return re.sub(r"[^a-z0-9]+", "", str(c).lower())
    return {norm_one(c): c for c in cols}

def _collect_paths(globs):
    files = []
    for g in globs: files.extend(glob(g))
    return sorted(set(files))

def _candidate_sheet_name_for(path):
    base = Path(path).stem  # e.g., subset3_Table1_filter
    return base

def _match_subset_sheet(sheet_names):
    # Find any name matching pattern "subset{digits}_Table1_filter" (case-insensitive)
    for nm in sheet_names:
        if re.match(r"(?i)^subset\d+_table1_filter$", str(nm).strip()):
            return nm
    return None

def load_npi_pharmacy_bundle(glob_list):
    paths = _collect_paths(glob_list)
    if not paths:
        raise FileNotFoundError(f"No Excel files matched: {glob_list}")

    aliases = {
        "npi": ["npi"],
        "org_name": [
            "providerorganizationname(legalbusinessname)",
            "providerorganizationnamelegalbusinessname",
            "provider organization name (legal business name)",
            "providerorganizationname",
            "providername"
        ],
        "zip_full": ["providerbusinesspracticelocationaddresspostalcode",
                     "provider business practice location address postal code"],
        "state": ["providerbusinesspracticelocationaddressstate",
                  "providerbusinesspracticelocationaddressstatename",
                  "provider business practice location address state name",
                  "provider business practice location address state"],
        "city": ["providerbusinesspracticelocationaddresscityname",
                 "provider business practice location address city name",
                 "providerbusinesspracticelocationaddresscity"],
        "addr1": ["providerfirstlinebusinesspracticelocationaddress",
                  "provider first line business practice location address"],
        "addr2": ["providersecondlinebusinesspracticelocationaddress",
                  "provider second line business practice location address"],
        "phone": ["providerbusinesspracticelocationaddresstelephonenumber",
                  "provider business practice location address telephone number"],
    }

    bad = []
    frames = []
    chosen = []

    stamp("Sheet choices:")
    for p in paths:
        try:
            # First try exact sheet name = file stem
            sheet_try = _candidate_sheet_name_for(p)
            x = pd.read_excel(p, sheet_name=sheet_try, dtype=str, engine="openpyxl")
            chosen.append(f"  {Path(p).name} → {sheet_try} (exact via pandas)")
        except Exception:
            # Fall back: peek available names, pick regex match
            try:
                xl = pd.ExcelFile(p, engine="openpyxl")
                nm = _match_subset_sheet(xl.sheet_names)
                if nm is None:
                    raise RuntimeError(f"No sheet like subset*_Table1_filter in {p}")
                x = xl.parse(sheet_name=nm, dtype=str)
                chosen.append(f"  {Path(p).name} → {nm} (regex fallback)")
            except Exception as ee:
                bad.append(f"{Path(p).name}: {ee}")
                continue

        cmap = _normalize_colnames(x.columns)

        def get_by_alias(keys, default=""):
            for k in keys:
                nk = re.sub(r"[^a-z0-9]+", "", k.lower())
                if nk in cmap:
                    return x[cmap[nk]]
            return pd.Series(default, index=x.index, dtype=str)

        # taxonomy columns present as multiple Healthcare Provider Taxonomy Code columns
        tax_cols = []
        for c_norm, c_orig in cmap.items():
            if c_norm.startswith("healthcareprovidertaxonomycode"):
                tax_cols.append(c_orig)

        df = pd.DataFrame({
            "NPI": get_by_alias(aliases["npi"]),
            "ORG_NAME": _norm_upper(get_by_alias(aliases["org_name"])),
            "ZIP_FULL": get_by_alias(aliases["zip_full"]).astype(str),
            "STATE": _norm_upper(get_by_alias(aliases["state"])),
            "CITY": _norm_upper(get_by_alias(aliases["city"])),
            "ADDR1": _norm_upper(get_by_alias(aliases["addr1"])),
            "ADDR2": _norm_upper(get_by_alias(aliases["addr2"])),
            "PHONE": _norm_upper(get_by_alias(aliases["phone"])),
        })

        # 5-digit ZIP
        df["ZCTA5"] = df["ZIP_FULL"].str.extract(r"(\d{5})", expand=False).fillna("").str.zfill(5)

        # taxonomy list per row
        if tax_cols:
            tax = x[tax_cols].astype(str).apply(
                lambda s: [v.strip().upper() for v in s.values if isinstance(v, str) and v.strip()],
                axis=1
            )
        else:
            tax = pd.Series([[]]*len(x), index=x.index)
        df["__tax_list"] = tax

        frames.append(df)

    print("\n".join(chosen))
    if bad:
        for b in bad:
            print("  • WARNING", b)

    if not frames:
        raise RuntimeError("No usable NPI tables parsed. Verify the files aren't password-protected or empty.")

    ph = pd.concat(frames, ignore_index=True)

    # Keep pharmacies only: any 3336* (Pharmacy) or exact 333600000X
    # Exclude 3329* suppliers
    def is_pharmacy(tlist):
        if not isinstance(tlist, (list, tuple)): return False
        for code in tlist:
            c = str(code).upper()
            if c.startswith("3329"):  # DME suppliers etc
                continue
            if c.startswith("3336") or c == "333600000X":
                return True
        return False

    ph = ph[ph["__tax_list"].apply(is_pharmacy)].copy()

    # Drop rows without a valid ZCTA
    ph = ph[ph["ZCTA5"].str.match(r"^\d{5}$", na=False)].copy()

    # Build robust site id (NPI + address tokens; if NPI missing, address only)
    addr_key = (ph["ADDR1"] + "|" + ph["ADDR2"] + "|" + ph["CITY"] + "|" + ph["STATE"] + "|" + ph["ZCTA5"]).str.strip("|")
    ph["location_id"] = np.where(
        ph["NPI"].astype(str).str.len().gt(0),
        ph["NPI"].astype(str).str.strip() + "|" + addr_key,
        addr_key
    )

    # Deduplicate sites
    sites = ph.drop_duplicates(subset=["location_id"])

    # Per-ZCTA pharmacy counts
    ph_cnt = (
        sites.groupby("ZCTA5", dropna=False)["location_id"]
        .nunique(dropna=True).reset_index()
        .rename(columns={"location_id": "pharmacies_count"})
    )

    # REGION (State) — mode per ZCTA if present, else "ALL"
    if sites["STATE"].notna().any() and (sites["STATE"].astype(str).str.len()>0).any():
        region_df = (
            sites.groupby("ZCTA5")["STATE"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
            .reset_index().rename(columns={"STATE": "REGION"})
        )
    else:
        region_df = sites[["ZCTA5"]].drop_duplicates().assign(REGION="ALL")

    # No coordinates available from NPI here
    centroids = pd.DataFrame(columns=["ZCTA5","LON","LAT"])

    # Diagnostics
    per_sheet_counts = []
    try:
        for p in paths:
            try:
                nm = _candidate_sheet_name_for(p)
                x = pd.read_excel(p, sheet_name=nm, dtype=str, engine="openpyxl")
                cmap = _normalize_colnames(x.columns)
                tax_cols = [orig for norm, orig in cmap.items() if norm.startswith("healthcareprovidertaxonomycode")]
                if tax_cols:
                    tax = x[tax_cols].astype(str).apply(
                        lambda s: any(str(v).upper().startswith("3336") or str(v).upper()=="333600000X" for v in s.values),
                        axis=1
                    )
                    per_sheet_counts.append((Path(p).name, nm, int(tax.sum()), int(len(x))))
                else:
                    per_sheet_counts.append((Path(p).name, nm, 0, int(len(x))))
            except Exception:
                pass
        print("Per-sheet pharmacy rows (pharm/rows):")
        for fn, nm, a, b in per_sheet_counts:
            print(f"  {fn} [{nm}]: {a}/{b}")
    except Exception:
        pass

    stamp(f"NPI pharmacy bundle: rows={len(ph):,}, deduped sites={sites['location_id'].nunique():,}, ZCTAs={ph_cnt['ZCTA5'].nunique():,}")
    return ph_cnt, region_df, centroids

# ===========================
# LOAD + PREP
# ===========================
with Step("Load FINANCIAL (median income)"):
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
    hlth["GHLTH_CrudePrev"] = pd.to_numeric(hlth["GHLTH_CrudePrev"], errors='coerce')
    hlth = hlth[["ZCTA5","GHLTH_CrudePrev"]]
    stamp(f"HEALTH rows={len(hlth)}, null health%={hlth['GHLTH_CrudePrev'].isna().sum()}")

with Step("Load PHARMACY (CSV or NPI Excel bundle)"):
    # Try CSV first (simpler and more reliable)
    csv_path = Path("data/pharmacy_data.csv")
    if csv_path.exists():
        stamp(f"Using CSV pharmacy data: {csv_path}")
        ph = read_csv_smart(str(csv_path))
        if "ZIP" not in ph.columns or "NAME" not in ph.columns:
            raise KeyError("pharmacy_data.csv must contain ZIP and NAME columns")
        
        ph["ZCTA5"] = coerce_zcta(ph["ZIP"])
        ph_cnt = ph.groupby("ZCTA5", dropna=False)["NAME"].nunique(dropna=True).reset_index().rename(columns={"NAME":"pharmacies_count"})
        
        # Extract region (prefer STATE)
        region_df = None
        for rcol in ["STATE","state","STATEFP","statefp","county","COUNTY","county_fips"]:
            if rcol in ph.columns:
                region_df = ph[["ZCTA5", rcol]].drop_duplicates("ZCTA5").rename(columns={rcol:"REGION"})
                break
        if region_df is None:
            region_df = ph[["ZCTA5"]].drop_duplicates().assign(REGION="ALL")
        
        # Extract centroids from lon/lat if available
        centroids = pd.DataFrame(columns=["ZCTA5","LON","LAT"])
        lon_col = pick_first("lon","longitude","LON","LONGITUDE","X","x", in_df=ph)
        lat_col = pick_first("lat","latitude","LAT","LATITUDE","Y","y", in_df=ph)
        if lon_col and lat_col:
            tmp = ph[[lon_col, lat_col, "ZCTA5"]].copy()
            tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors='coerce')
            tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors='coerce')
            good = tmp[lon_col].between(-180,180) & tmp[lat_col].between(-90,90)
            if good.any():
                centroids = tmp[good].groupby("ZCTA5", as_index=False).agg(
                    LON=(lon_col, "mean"),
                    LAT=(lat_col, "mean")
                )
        
        TOTAL_PHARM = int(ph_cnt["pharmacies_count"].sum())
        stamp(f"PHARMACY unique ZCTA5={ph_cnt['ZCTA5'].nunique()}, pharmacies={TOTAL_PHARM:,}")
    else:
        # Fallback to Excel
        try:
            ph_cnt, region_df, centroids = load_npi_pharmacy_bundle(NPI_GLOBS)
            stamp(f"PHARMACY unique ZCTA5={ph_cnt['ZCTA5'].nunique()}, pharmacies={ph_cnt['pharmacies_count'].sum():,}")
            TOTAL_PHARM = int(ph_cnt["pharmacies_count"].sum())
        except Exception as e:
            stamp(f"WARNING: {e}")
            stamp("PHARMACY data missing. Will neutralize the access feature (set to 0.5).")
            ph_cnt = pd.DataFrame(columns=["ZCTA5","pharmacies_count"])
            region_df = pd.DataFrame(columns=["ZCTA5","REGION"])
            centroids = pd.DataFrame(columns=["ZCTA5","LON","LAT"])
            TOTAL_PHARM = 0

with Step("Load POPULATION (ACS ZCTA; try skiprows=10 then 0)"):
    try:
        pop = read_csv_smart(POPULATION_CSV, skiprows=10)
    except Exception:
        pop = read_csv_smart(POPULATION_CSV)
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

# Optional extras
try:
    with Step("Load AQI (annual weighted PM2.5)"):
        aqi_annual = read_aqi_data(AQI_CSV)
        cov = 100*len(aqi_annual)/max(1, pop['ZCTA5'].nunique())
        stamp(f"AQI rows={len(aqi_annual)} (coverage ~{cov:.1f}%)")
except Exception as e:
    aqi_annual = None
    warnings.warn(f"AQI not loaded: {e}")

try:
    with Step("Load HHI (heat vulnerability)"):
        hhi = read_hhi_excel(HHI_XLSX)
        stamp(f"HHI rows={len(hhi)}")
except Exception as e:
    hhi = None
    warnings.warn(f"HHI not loaded: {e}")

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
    df["median_income"]   = pd.to_numeric(df.get("S1901_C01_012E"), errors="coerce")
    df["poor_health_pct"] = pd.to_numeric(df.get("GHLTH_CrudePrev"), errors="coerce")
    df["pharmacies_count"]= pd.to_numeric(df.get("pharmacies_count"), errors="coerce").fillna(0)

    for col in ["median_income","poor_health_pct"]:
        med = df[col].median(skipna=True)
        df[col] = df[col].fillna(med)

    denom = df["pharmacies_count"].replace(0, 1)
    df["pop_per_pharmacy"] = np.where(df["population"].gt(0), df["population"]/denom, np.nan)

    # density (if land area present, allow unit-agnostic handling)
    if "land_area_raw" in df.columns:
        med_area = np.nanmedian(df["land_area_raw"])
        # If looks like square meters, convert to km^2
        df["land_area_km2"] = np.where(np.isfinite(med_area) and med_area>1e5, df["land_area_raw"]/1e6, df["land_area_raw"])
        df["pop_density"] = np.where(df["land_area_km2"].gt(0), df["population"]/df["land_area_km2"], np.nan)
    else:
        df["pop_density"] = np.nan

    if "aqi" in df.columns:       df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    if "heat_hhb" in df.columns:  df["heat_hhb"] = pd.to_numeric(df["heat_hhb"], errors="coerce")

    # percentiles
    df["income_pct_inv"]  = 1 - pct_rank(df["median_income"])
    df["health_pct"]      = pct_rank(df["poor_health_pct"])
    df["access_pct_inv"]  = pct_rank(df["pop_per_pharmacy"]) if df["pharmacies_count"].sum() > 0 else 0.5
    df["density_pct"]     = pct_rank(df["pop_density"]) if df["pop_density"].notna().any() else 0.5
    df["aqi_pct"]         = pct_rank(df["aqi"])         if "aqi" in df.columns else 0.5
    df["heat_pct"]        = pct_rank(df["heat_hhb"])    if "heat_hhb" in df.columns else 0.5
    stamp("Feature percentiles constructed.")

# Composite score
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
# EXPECTED ACCESS: GLM OOF + per-state cal → NB alpha → Hybrid residual model
# ==========================
glm_done = False
if df["pharmacies_count"].sum() > 0:
    with Step("GLM Poisson OOF (exposure=population) + per-state calibration"):
        region_col = "REGION"
        work = df.loc[df["population"].fillna(0) > 0, ["ZCTA5","pharmacies_count","population",
                                                       "income_pct_inv","health_pct","density_pct",
                                                       "aqi_pct","heat_pct", region_col]].copy()

        # Center predictors (neutral 0.5 baseline)
        for c in ["income_pct_inv","health_pct","aqi_pct","heat_pct","density_pct"]:
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.5)
            work[c + "_c"] = work[c] - 0.5

        # Simple, stable formula
        formula_cv = "pharmacies_count ~ 1 + income_pct_inv_c + health_pct_c + density_pct_c"

        y_cv, X_cv = dmatrices(formula_cv, work, return_type="dataframe", eval_env=0)
        valid_idx = X_cv.index
        y = y_cv.iloc[:, 0]
        exposure_all = work.loc[valid_idx, "population"].clip(lower=1.0).astype(float)
        groups = work.loc[valid_idx, region_col].astype(str)

        n_groups = groups.nunique()
        n_rows   = len(valid_idx)
        n_splits = min(CV_N_SPLITS_MAX, n_groups) if n_groups >= 2 else min(3, max(2, n_rows // 2))
        print(f"Valid rows: {n_rows} | params(cv): {X_cv.shape[1]} | groups: {n_groups} | folds: {n_splits}")

        fam = sm.families.Poisson()
        mu_oof = pd.Series(index=valid_idx, dtype=float)

        splitter = GroupKFold(n_splits=n_splits).split(X_cv, y, groups=groups) if n_groups >= 2 else KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_cv, y)

        for tr, te in tqdm(splitter, total=n_splits, desc=f"GLM CV ({n_splits} folds)", unit="fold", leave=False):
            X_tr, X_te = X_cv.iloc[tr], X_cv.iloc[te]
            y_tr, y_te = y.iloc[tr],  y.iloc[te]
            exp_tr, exp_te = exposure_all.iloc[tr], exposure_all.iloc[te]

            model = sm.GLM(y_tr, X_tr, family=fam, exposure=exp_tr)
            res   = model.fit()
            mu_te = res.predict(X_te, exposure=exp_te)
            mu_oof.iloc[te] = mu_te

        # Estimate NB alpha from OOF (Pearson chi-square)
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

        # Per-state calibration with national fallback
        obs = work.loc[valid_idx, "pharmacies_count"].astype(float)
        reg = work.loc[valid_idx, region_col].astype(str)
        by_state = pd.DataFrame({"y":obs, "mu":mu_oof, "REGION":reg})
        nat_y, nat_mu = by_state["y"].sum(), by_state["mu"].sum()
        nat_factor = float(nat_y / nat_mu) if nat_mu > 0 else 1.0

        grp_sum = by_state.groupby("REGION")[["y","mu"]].sum()
        grp_sum["cal"] = grp_sum.apply(lambda r: (r["y"]/r["mu"]) if r["mu"]>0 else np.nan, axis=1)
        grp_sum["cal"] = grp_sum["cal"].where(np.isfinite(grp_sum["cal"]) & (grp_sum["cal"]>0), nat_factor)
        cal_map = grp_sum["cal"].to_dict()

        mu_oof_cal = mu_oof * reg.map(cal_map).astype(float).values
        mu_oof_cal = pd.Series(mu_oof_cal, index=valid_idx)

        glm_done = True
else:
    stamp("Skipping GLM/Hybrid because pharmacy data is missing.")
    glm_done = False

# Hybrid residual model
if glm_done:
    with Step("Hybrid residual model (GBDT/XGB) + per-state calibration"):
        work = df.loc[df["population"].fillna(0) > 0, :]
        valid_idx = mu_oof_cal.index

        obs = work.loc[valid_idx, "pharmacies_count"].astype(float)

        X_tree = pd.DataFrame({
            "inc": work.loc[valid_idx, "income_pct_inv"].astype(float),
            "hlth": work.loc[valid_idx, "health_pct"].astype(float),
            "dens": work.loc[valid_idx, "density_pct"].astype(float),
            "aqi":  (work.loc[valid_idx, "aqi_pct"].astype(float)  if "aqi_pct"  in work else 0.5),
            "heat": (work.loc[valid_idx, "heat_pct"].astype(float) if "heat_pct" in work else 0.5),
            "log_pop": np.log1p(work.loc[valid_idx, "population"].astype(float)),
            "region_code": work.loc[valid_idx, "REGION"].astype("category").cat.codes.astype(int),
        }).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        eps = 0.5
        r_oof = np.log( (obs + eps) / (mu_oof_cal + eps) )

        r_pred = pd.Series(0.0, index=valid_idx)
        groups = work.loc[valid_idx, "REGION"].astype(str)
        n_splits = min(CV_N_SPLITS_MAX, max(2, groups.nunique()))
        gkf = GroupKFold(n_splits=n_splits)

        if _XGB_OK:
            # sanitize columns for xgboost
            X_tree.columns = [re.sub(r"[\[\]<>]", "_", c) for c in X_tree.columns]
            for tr, te in gkf.split(X_tree, r_oof, groups=groups):
                dtr = xgb.DMatrix(X_tree.iloc[tr], label=r_oof.iloc[tr])
                dte = xgb.DMatrix(X_tree.iloc[te], label=r_oof.iloc[te])
                params = dict(
                    objective="reg:squarederror",
                    max_depth=4, eta=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, reg_alpha=0.0,
                    nthread=0, tree_method="hist",
                )
                bst = xgb.train(params, dtr, num_boost_round=600)
                r_pred.iloc[te] = bst.predict(dte)
        else:
            for tr, te in gkf.split(X_tree, r_oof, groups=groups):
                mdl = HistGradientBoostingRegressor(
                    max_depth=4, learning_rate=0.05,
                    max_iter=600, min_samples_leaf=20
                )
                mdl.fit(X_tree.iloc[tr], r_oof.iloc[tr])
                r_pred.iloc[te] = mdl.predict(X_tree.iloc[te])

        mu_hybrid_oof = mu_oof_cal * np.exp(r_pred)

        # Per-state calibration on hybrid OOF
        by_state_h = pd.DataFrame({"y":obs, "mu":mu_hybrid_oof, "REGION":groups})
        nat_y_h, nat_mu_h = by_state_h["y"].sum(), by_state_h["mu"].sum()
        nat_factor_h = float(nat_y_h / max(nat_mu_h, 1e-9)) if nat_mu_h > 0 else 1.0

        grp_sum_h = by_state_h.groupby("REGION")[["y","mu"]].sum()
        grp_sum_h["cal"] = grp_sum_h.apply(lambda r: (r["y"]/r["mu"]) if r["mu"]>0 else np.nan, axis=1)
        grp_sum_h["cal"] = grp_sum_h["cal"].where(np.isfinite(grp_sum_h["cal"]) & (grp_sum_h["cal"]>0), nat_factor_h)
        cal_map_h = grp_sum_h["cal"].to_dict()

        mu_hybrid_oof_cal = mu_hybrid_oof * groups.map(cal_map_h).astype(float).values
        mu_hybrid_oof_cal = pd.Series(mu_hybrid_oof_cal, index=valid_idx)
else:
    mu_oof_cal = pd.Series(dtype=float)
    mu_hybrid_oof_cal = pd.Series(dtype=float)

# Final GLM refit + Hybrid full fit for QA
if glm_done:
    with Step("Final GLM refit and FULL hybrid prediction for QA"):
        region_col = "REGION"
        work = df.loc[df["population"].fillna(0) > 0, ["ZCTA5","pharmacies_count","population",
                                                       "income_pct_inv","health_pct","density_pct",
                                                       "aqi_pct","heat_pct", region_col]].copy()
        for c in ["income_pct_inv","health_pct","density_pct","aqi_pct","heat_pct"]:
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.5)
            work[c + "_c"] = work[c] - 0.5

        formula_full = "pharmacies_count ~ 1 + income_pct_inv_c + health_pct_c + density_pct_c + C(REGION)"
        y_full, X_full = dmatrices(formula_full, work, return_type="dataframe", eval_env=0)
        idx_full = X_full.index
        exp_full = work.loc[idx_full, "population"].clip(lower=1.0).astype(float)

        fam = sm.families.Poisson()
        res_full = sm.GLM(y_full.iloc[:,0], X_full, family=fam, exposure=exp_full).fit()
        mu_full = pd.Series(res_full.predict(X_full, exposure=exp_full), index=idx_full)

        # Export coefficients
        coef = (pd.Series(res_full.params, name="coef")
                .to_frame()
                .join(res_full.bse.rename("se"))
                .assign(z=lambda d: d.coef/d.se))
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        coef_file = Path(OUT_DIR) / "glm_full_coefficients.csv"
        coef.to_csv(coef_file)
        stamp(f"Wrote {coef_file}")

        # Per-state calibration baseline
        reg_full = work.loc[idx_full, region_col].astype(str)
        by_state_full = pd.DataFrame({"y": y_full.iloc[:,0].astype(float), "mu": mu_full, "REGION": reg_full})
        nat_y_f, nat_mu_f = by_state_full["y"].sum(), by_state_full["mu"].sum()
        nat_factor_f = float(nat_y_f / max(nat_mu_f, 1e-9)) if nat_mu_f > 0 else 1.0
        grp_sum_f = by_state_full.groupby("REGION")[["y","mu"]].sum()
        grp_sum_f["cal"] = grp_sum_f.apply(lambda r: (r["y"]/r["mu"]) if r["mu"]>0 else np.nan, axis=1)
        grp_sum_f["cal"] = grp_sum_f["cal"].where(np.isfinite(grp_sum_f["cal"]) & (grp_sum_f["cal"]>0), nat_factor_f)
        cal_map_f = grp_sum_f["cal"].to_dict()
        mu_full_cal = mu_full * reg_full.map(cal_map_f).astype(float).values
        mu_full_cal = pd.Series(mu_full_cal, index=idx_full)

        # FULL hybrid target + predict
        r_full = np.log( (y_full.iloc[:,0].astype(float) + 0.5) / (mu_full_cal + 0.5) )
        X_tree_full = pd.DataFrame({
            "inc": work.loc[idx_full, "income_pct_inv"].astype(float),
            "hlth": work.loc[idx_full, "health_pct"].astype(float),
            "dens": work.loc[idx_full, "density_pct"].astype(float),
            "aqi":  work.loc[idx_full, "aqi_pct"].astype(float) if "aqi_pct" in work else 0.5,
            "heat": work.loc[idx_full, "heat_pct"].astype(float) if "heat_pct" in work else 0.5,
            "log_pop": np.log1p(work.loc[idx_full, "population"].astype(float)),
            "region_code": work.loc[idx_full, "REGION"].astype("category").cat.codes.astype(int)
        }).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if _XGB_OK:
            X_tree_full.columns = [re.sub(r"[\[\]<>]", "_", c) for c in X_tree_full.columns]
            dfull = xgb.DMatrix(X_tree_full, label=r_full)
            params = dict(
                objective='reg:squarederror',
                max_depth=4, eta=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.0,
                nthread=0, tree_method="hist"
            )
            bst_full = xgb.train(params, dfull, num_boost_round=int(1.2 * 600))
            r_pred_full = pd.Series(bst_full.predict(xgb.DMatrix(X_tree_full)), index=idx_full)
        else:
            mdl_full = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                                     max_iter=int(1.2*600), min_samples_leaf=20)
            mdl_full.fit(X_tree_full, r_full)
            r_pred_full = pd.Series(mdl_full.predict(X_tree_full), index=idx_full)

        mu_hybrid_full = mu_full_cal * np.exp(r_pred_full)

        # Attach to df for QA/exports
        df["glm_mu_expected_pharm_oof_cal_state"] = np.nan; df.loc[mu_oof_cal.index, "glm_mu_expected_pharm_oof_cal_state"] = mu_oof_cal.values
        df["hybrid_mu_expected_oof_cal_state"]    = np.nan; df.loc[mu_hybrid_oof_cal.index, "hybrid_mu_expected_oof_cal_state"] = mu_hybrid_oof_cal.values
        df["glm_mu_expected_pharm_full"]          = np.nan; df.loc[idx_full,  "glm_mu_expected_pharm_full"] = mu_full.values
        df["hybrid_mu_expected_full"]             = np.nan; df.loc[idx_full,  "hybrid_mu_expected_full"]  = mu_hybrid_full.values
else:
    # no pharmacy → no expectations
    df["glm_mu_expected_pharm_oof_cal_state"] = np.nan
    df["hybrid_mu_expected_oof_cal_state"]    = np.nan
    df["glm_mu_expected_pharm_full"]          = np.nan
    df["hybrid_mu_expected_full"]             = np.nan

# ==========================
# Residuals, Neighbor QA, Scores, Outputs
# ==========================
with Step("Residuals + Neighbor QA + outputs"):
    # Deviance-residual score (use hybrid OOF if available)
    if glm_done:
        obs_all = df.loc[mu_hybrid_oof_cal.index, "pharmacies_count"].astype(float)
        mu_all  = df.loc[mu_hybrid_oof_cal.index, "hybrid_mu_expected_oof_cal_state"].astype(float)

        # Use alpha_hat from earlier OOF step if defined; else 0
        try:
            var_mu  = mu_all + alpha_hat*(mu_all**2)
        except Exception:
            var_mu  = mu_all

        yv = obs_all.values.astype(float); muv = mu_all.values
        eps = 1e-12
        term = np.where(yv > 0, yv * np.log(np.maximum(yv, eps) / np.maximum(muv, eps)) - (yv - muv), - (yv - muv))
        dev_resid = np.sign(yv - muv) * np.sqrt(2.0 * np.maximum(term, 0.0))
        underserved = -pd.Series(dev_resid, index=obs_all.index)
        glm_nb_score_hybrid = norm01(underserved)

        df["glm_nb_score_hybrid"] = np.nan
        df.loc[obs_all.index, "glm_nb_score_hybrid"] = glm_nb_score_hybrid.values
    else:
        df["glm_nb_score_hybrid"] = np.nan

    # Neighbor QA (feature space within state)
    base = df.loc[df["population"].fillna(0) > 0, ["ZCTA5","REGION","population","pharmacies_count",
                                                   "hybrid_mu_expected_oof_cal_state",
                                                   "income_pct_inv","health_pct"]].copy()
    base["obs_rate10k"] = 1e4 * base["pharmacies_count"].astype(float) / base["population"].clip(lower=1.0)
    base["exp_rate10k"] = 1e4 * base["hybrid_mu_expected_oof_cal_state"].astype(float) / base["population"].clip(lower=1.0)

    nb_obs_median = pd.Series(np.nan, index=base.index)
    nb_exp_median = pd.Series(np.nan, index=base.index)

    for state, g in base.groupby("REGION"):
        if len(g) < 3: continue
        k = min(NEIGHBOR_K+1, len(g))
        feat = np.c_[ (g["income_pct_inv"]-0.5), (g["health_pct"]-0.5), safe_log1p(g["population"]) ]
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(feat)
        dist, ind = nn.kneighbors(feat, return_distance=True)
        for idx_row, neighbors in zip(g.index, ind):
            neighbors = [g.index[i] for i in neighbors if g.index[i] != idx_row][:NEIGHBOR_K]
            nb_obs_median.loc[idx_row] = base.loc[neighbors, "obs_rate10k"].median()
            nb_exp_median.loc[idx_row] = base.loc[neighbors, "exp_rate10k"].median()

    df["neighbor_obs_rate10k_median"] = np.nan; df.loc[nb_obs_median.index, "neighbor_obs_rate10k_median"] = nb_obs_median.values
    df["neighbor_exp_rate10k_median"] = np.nan; df.loc[nb_exp_median.index, "neighbor_exp_rate10k_median"] = nb_exp_median.values

    # Suspicious zero flag
    cond_big_pop   = df["population"].fillna(0) >= SUSPICIOUS_ZERO_MIN_POP
    cond_zero      = df["pharmacies_count"].fillna(0) == 0
    cond_high_mu   = (df["hybrid_mu_expected_oof_cal_state"].fillna(0) >= SUSPICIOUS_ZERO_MIN_EXPECTED) | \
                     ( (1e4 * df["hybrid_mu_expected_oof_cal_state"].fillna(0) / df["population"].clip(lower=1.0)) >= SUSPICIOUS_ZERO_MIN_RATE10K )
    cond_neighbors = df["neighbor_obs_rate10k_median"].fillna(0) >= SUSPICIOUS_ZERO_MIN_NEIGH_RATE
    df["suspicious_zero_flag"] = cond_big_pop & cond_zero & cond_high_mu & cond_neighbors

    # Blends
    if "glm_nb_score_hybrid" in df.columns and df["glm_nb_score_hybrid"].notna().any():
        df["IFAE_score_residual"] = 0.5*df["composite"] + 0.5*norm01(df["glm_nb_score_hybrid"])
    else:
        df["IFAE_score_residual"] = df["composite"]

    pos_def_rate = (1e4 * (df["hybrid_mu_expected_oof_cal_state"].fillna(0) - df["pharmacies_count"].fillna(0)) /
                    df["population"].clip(lower=1.0)).clip(lower=0)
    df["glm_deficit_score"] = norm01(pos_def_rate)
    df["IFAE_score_deficit"] = 0.5*df["composite"] + 0.5*df["glm_deficit_score"]

    # Choose main
    df["IFAE_score"] = df["IFAE_score_residual"]

    # Urbanish mask
    has_area   = df["pop_density"].notna()
    dens_rule  = DENS_CORE_KM2 if URBAN_USE_CORE else DENS_FRINGE_KM2
    urban_mask = ( (has_area & (df["pop_density"] >= dens_rule)) |
                   (~has_area & (df["population"].fillna(0) >= 5000)) )
    df["is_urbanish"] = urban_mask

    keep = [
        "ZCTA5","REGION","IFAE_score","IFAE_score_residual","IFAE_score_deficit","is_urbanish",
        "composite","glm_nb_score_hybrid",
        "median_income","poor_health_pct","population","pharmacies_count",
        "pop_per_pharmacy","income_pct_inv","health_pct","access_pct_inv",
        "density_pct","pop_density",
        "glm_mu_expected_pharm_oof_cal_state","hybrid_mu_expected_oof_cal_state",
        "glm_mu_expected_pharm_full","hybrid_mu_expected_full",
        "neighbor_obs_rate10k_median","neighbor_exp_rate10k_median","suspicious_zero_flag"
    ]
    if "aqi" in df.columns:       keep += ["aqi","aqi_pct","obs_total"]
    if "heat_hhb" in df.columns:  keep += ["heat_hhb","heat_pct"]
    if "land_area_km2" in df.columns: keep += ["land_area_km2"]

    ranked = df[keep].copy().sort_values("IFAE_score", ascending=False).reset_index(drop=True)

    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    out_full_main   = out / "national_ifae_rank.csv"
    out_full_alt    = out / "national_ifae_rank_alt_deficit.csv"
    out_top_urban   = out / "topK_ifae_urban.csv"
    out_bottom      = out / "bottomK_ifae.csv"
    qa_file         = out / "qa_expected_vs_observed.csv"

    ranked.to_csv(out_full_main, index=False)
    ranked.sort_values("IFAE_score_deficit", ascending=False).to_csv(out_full_alt, index=False)

    eligible = ranked[(ranked["population"].fillna(0) >= MIN_POP_TOPK) & (ranked["is_urbanish"])]
    top_presentable = eligible.head(TOP_K)
    bottom_presentable = ranked[ ranked["population"].fillna(0) > 0 ].tail(TOP_K)

    qa_cols = [
        "ZCTA5","REGION","population","pharmacies_count",
        "glm_mu_expected_pharm_oof_cal_state","hybrid_mu_expected_oof_cal_state",
        "glm_mu_expected_pharm_full","hybrid_mu_expected_full",
        "neighbor_obs_rate10k_median","neighbor_exp_rate10k_median","suspicious_zero_flag",
        "IFAE_score","IFAE_score_deficit","composite","is_urbanish"
    ]
    ranked[qa_cols].to_csv(qa_file, index=False)

    stamp(f"Wrote {out_full_main}")
    stamp(f"Wrote {out_full_alt}")
    stamp(f"Wrote {out_top_urban} (pop ≥ {MIN_POP_TOPK}, urban mask)")
    stamp(f"Wrote {out_bottom}")
    stamp(f"Wrote {qa_file}")

    print(f"\nTop (urban, pop ≥ {MIN_POP_TOPK:,}) — Top {TOP_K}:")
    try:
        print(top_presentable.head(min(TOP_K, len(top_presentable))).to_string(index=False))
    except Exception:
        print(top_presentable.head(min(TOP_K, len(top_presentable))))

    print(f"\nBottom {TOP_K} (pop>0 only):")
    try:
        print(bottom_presentable.tail(min(TOP_K, len(bottom_presentable))).to_string(index=False))
    except Exception:
        print(bottom_presentable.tail(min(TOP_K, len(bottom_presentable))))

stamp("All done. 🚀")
