"""
Pharmacy Desert Analysis - Data Processing Module
==================================================
This module contains all data loading and processing functions for pharmacy desert analysis.
For the Streamlit UI, see app.py

Functions:
- read_*_data: Load data from various sources (financial, health, pharmacy, population, AQI, HHI)
- preprocess: Merge and clean all data sources
- score_candidates: Apply mathematical scoring model
- average_scores: Blend mathematical and AI scores
- render_top10_map: Display interactive map (requires Streamlit)
"""

import pandas as pd 
import numpy as np
import streamlit as st


@st.cache_data
def read_financial_data(file_path):
    """Read financial/income data from Census CSV"""
    df = pd.read_csv(file_path)
    df = df[['NAME', 'S1901_C01_012E']]
    df['zip'] = df['NAME'].str.extract(r'(\d{5})')
    df['zip'] = df['NAME'].str.extract(r'(\d{5})')
    return df


@st.cache_data
def read_hhi_excel(file_path):
    """
    Read Heat-Health Index (HHI) Excel data
    
    Returns: DataFrame with zip, heat_hhb (from HHB_SCORE), nbe_score, hhi_overall
    """
    df = pd.read_excel(file_path, dtype={'ZCTA': str})

    if 'ZCTA' not in df.columns:
        raise ValueError("HHI Excel must contain 'ZCTA' column.")
    
    df['zip'] = (
        df['ZCTA'].astype(str)
        .str.extract(r'(\d{5})')[0]
        .fillna('')
        .str.zfill(5)
    )

    out = pd.DataFrame({'zip': df['zip']})

    if 'HHB_SCORE' in df.columns:
        out['heat_hhb'] = pd.to_numeric(df['HHB_SCORE'], errors='coerce')
    if 'NBE_SCORE' in df.columns:
        out['nbe_score'] = pd.to_numeric(df['NBE_SCORE'], errors='coerce')
    if 'OVERALL_SCORE' in df.columns:
        out['hhi_overall'] = pd.to_numeric(df['OVERALL_SCORE'], errors='coerce')

    return out.dropna(subset=['zip']).drop_duplicates(subset=['zip'])


@st.cache_data
def read_aqi_data(file_path):
    """
    Read and aggregate Air Quality Index (PM2.5) data
    
    Returns: tuple of (aqi_monthly, aqi_annual) DataFrames
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Handle multiple ZIP columns
    zip_cols = [c for c in df.columns if c.upper().startswith('ZIP')]
    zip_col = zip_cols[-1]
    df['zip'] = df[zip_col].astype(str).str.extract(r'(\d{5})')[0].fillna('').str.zfill(5)

    val_col = 'Arithmetic Mean'
    w_col   = 'Observation Count'
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df[w_col]   = pd.to_numeric(df[w_col], errors='coerce').fillna(0)

    # Filter to PM2.5
    if 'Parameter Name' in df.columns:
        df = df[df['Parameter Name'].str.contains('PM2.5', na=False)]

    # Parse dates
    if 'Month' in df.columns:
        df['month'] = pd.to_datetime(df['Month'], format='%b-%y', errors='coerce')
    else:
        df['month'] = pd.to_datetime(df['Date Local'], errors='coerce').values.astype('datetime64[M]')

    df = df.dropna(subset=['zip', 'month', val_col])
    df = df[df[w_col] > 0]

    # Monthly weighted averages
    grp = df.groupby(['zip', 'month'], as_index=False).apply(
        lambda g: pd.Series({
            'aqi_monthly': np.average(g[val_col], weights=g[w_col]),
            'obs_month':   g[w_col].sum()
        })
    ).reset_index(drop=True)

    # Annual weighted averages
    grp_annual = df.groupby('zip', as_index=False).apply(
        lambda g: pd.Series({
            'aqi': np.average(g[val_col], weights=g[w_col]),
            'obs_total': g[w_col].sum()
        })
    ).reset_index(drop=True)

    grp['month_label'] = grp['month'].dt.strftime('%Y-%m')

    return grp[['zip', 'month', 'month_label', 'aqi_monthly', 'obs_month']], grp_annual[['zip', 'aqi', 'obs_total']]


@st.cache_data
def read_education_data_acs(year=2023, api_key=None):
    """
    Read ACS S1501 Educational Attainment by ZCTA.
    Returns % HS or lower (25+) and % less than HS (25+).
    """
    import requests, pandas as pd

    base = f"https://api.census.gov/data/{year}/acs/acs5/subject"
    vars_ = [
        "NAME",
        "S1501_C02_007E",  # < 9th grade (25+), percent
        "S1501_C02_008E",  # 9-12 no diploma (25+), percent
        "S1501_C02_009E",  # HS grad incl. GED (25+), percent
        "S1501_C02_014E",  # HS graduate or higher (25+), percent (optional check)
        "S1501_C02_015E"   # Bachelor's degree or higher (25+), percent (optional)
    ]

    params = {
        "get": ",".join(vars_),
        "for": "zip code tabulation area:*"
    }
    if api_key:
        params["key"] = api_key

    r = requests.get(base, params=params, timeout=120)
    #print(r.json())
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    # Clean names & types
    df = df.rename(columns={
        "zip code tabulation area": "zip",
        "S1501_C02_007E": "pct_less_9",
        "S1501_C02_008E": "pct_9to12_no_diploma",
        "S1501_C02_009E": "pct_hs_grad",
        "S1501_C02_014E": "pct_hs_or_higher",
        "S1501_C02_015E": "pct_ba_or_higher"
    })
    num_cols = ["pct_less_9","pct_9to12_no_diploma","pct_hs_grad","pct_hs_or_higher","pct_ba_or_higher"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["zip"] = df["zip"].astype(str).str.zfill(5)

    # Your targets
    df["edu_hs_or_lower_pct"]   = df[["pct_less_9","pct_9to12_no_diploma","pct_hs_grad"]].sum(axis=1)
    df["edu_less_than_hs_pct"]  = 100 - df["pct_hs_or_higher"]

    return df[["zip","edu_hs_or_lower_pct","edu_less_than_hs_pct","pct_ba_or_higher"]]

@st.cache_data
def read_population_data(file_path):
    """Read population density and location data"""
    df = pd.read_csv(file_path, skiprows=10)

    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    required = ["zip", "density", "lat", "long"]
    if any(k not in lower for k in required):
        raise ValueError(f"Expected columns Zip, density, lat, long; got {df.columns[:10].tolist()}")

    zip_col  = lower["zip"]
    dens_col = lower["density"]
    lat_col  = lower["lat"]
    lon_col  = lower["long"]

    out = pd.DataFrame({
        "zip": df[zip_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
        "pop_density": df[dens_col].astype(str).str.replace(",", "", regex=False),
        "lat": df[lat_col],
        "lon": df[lon_col],
    })
    out["pop_density"] = pd.to_numeric(out["pop_density"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    out = (out.dropna(subset=["zip"])
              .groupby("zip", as_index=False)
              .agg({"pop_density":"max",
                    "lat":"first",
                    "lon":"first"}))
    return out


@st.cache_data
def read_pharmacy_data(file_path):
    """Read pharmacy location data"""
    df = pd.read_csv(file_path)
    df = df[['ZIP', 'NAME', 'X', 'Y']]
    return df


@st.cache_data
def read_health_data(file_path):
    """Read health burden data from PLACES"""
    df = pd.read_csv(file_path)
    df = df[['ZCTA5', 'GHLTH_CrudePrev']]
    df['ZCTA5'] = df['ZCTA5'].astype(str).str.split('.').str[0].str.zfill(5)
    return df


@st.cache_data
def read_hud_zip_county_crosswalk(path):
    """
    Read HUD ZIP↔County crosswalk; return [zip, county, state, weight].
    Uses TOT_RATIO if present else RES_RATIO.
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, low_memory=False)

    cols = {c.lower(): c for c in df.columns}
    zip_col    = cols.get("zip") or cols.get("zipcode") or cols.get("zip_code")
    county_col = cols.get("county") or cols.get("county_fips") or cols.get("fips")
    state_col  = cols.get("state") or cols.get("stabbr") or cols.get("stusps")

    weight_col = None
    for cand in ["tot_ratio", "total_ratio", "res_ratio"]:
        if cand in cols:
            weight_col = cols[cand]
            break
    if not (zip_col and county_col and weight_col):
        raise ValueError("Crosswalk must have ZIP, COUNTY, and TOT_RATIO/RES_RATIO.")

    out = pd.DataFrame({
        "zip":    df[zip_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
        "county": df[county_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
        "state":  (df[state_col] if state_col else pd.Series(index=df.index, dtype="object")),
        "weight": pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    })
    out = out.dropna(subset=["zip","county"])
    out = out[out["weight"] > 0]
    return out



@st.cache_data
def read_county_desert_csv(path):
    """
    Read county-level desert dataset; return [county, county_desert, drive_time_min, desert_pop_pct].
    
    Extracts:
    - county_desert: Binary flag (0 or 1) indicating if county is a pharmacy desert
    - drive_time_min: Average drive time to nearest pharmacy in minutes
    - desert_pop_pct: Percentage of population living in pharmacy desert
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)
    fips_col = next((c for c in df.columns if "fips" in c.lower()), None)
    if not fips_col:
        raise ValueError("County dataset must include a county FIPS column.")
    df["county"] = df[fips_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)

    # Extract binary desert flag
    flag_col = next((c for c in df.columns if c.lower() in ["desert","is_desert","desert_flag","model1_pharm_desert","pharm_desert"]), None)
    if flag_col:
        val = pd.to_numeric(df[flag_col], errors="coerce").fillna(0.0).clip(0,1)
    else:
        score_col = next((c for c in df.columns if any(k in c.lower() for k in ["score","index","risk","prob"])), None)
        if not score_col:
            raise ValueError("No desert flag/score column found in county dataset.")
        raw = pd.to_numeric(df[score_col], errors="coerce")
        val = (raw - raw.min()) / (raw.max() - raw.min()) if raw.max() > raw.min() else 0.0

    # Extract drive time in minutes
    drive_time_col = next((c for c in df.columns if "drive_time" in c.lower() and "min" in c.lower()), None)
    drive_time = pd.to_numeric(df[drive_time_col], errors="coerce") if drive_time_col else pd.Series(index=df.index, dtype=float)
    
    # Extract desert population percentage
    desert_pop_col = next((c for c in df.columns if "desert_pop" in c.lower() and "pct" in c.lower()), None)
    desert_pop_pct = pd.to_numeric(df[desert_pop_col], errors="coerce") if desert_pop_col else pd.Series(index=df.index, dtype=float)

    out = pd.DataFrame({
        "county": df["county"], 
        "county_desert": val,
        "drive_time_min": drive_time,
        "desert_pop_pct": desert_pop_pct
    })
    return out.dropna(subset=["county"]).drop_duplicates(subset=["county"])


@st.cache_data
def downscale_county_to_zip(county_df, xwalk_df,
                            tiny_cutoff=0.01, min_coverage=0.60, threshold=0.50):
    xw = xwalk_df[xwalk_df["weight"] >= tiny_cutoff].copy()
    totals = xw.groupby("zip", as_index=False)["weight"].sum().rename(columns={"weight":"zip_total"})

    m = xw.merge(county_df, on="county", how="left")

    # availability flags per metric
    m["has_desert"] = m["county_desert"].notna()
    m["has_drive"]  = m["drive_time_min"].notna()
    m["has_pop"]    = m["desert_pop_pct"].notna()

    # weighted values
    m["w_desert"] = np.where(m["has_desert"], m["weight"] * m["county_desert"], 0.0)
    m["w_drive"]  = np.where(m["has_drive"],  m["weight"] * m["drive_time_min"], 0.0)
    m["w_pop"]    = np.where(m["has_pop"],    m["weight"] * m["desert_pop_pct"], 0.0)

    agg = (m.groupby("zip", as_index=False)
             .agg(zip_wsum=("weight","sum"),
                  wmatch_desert=("weight", lambda s: s[m.loc[s.index, "has_desert"]].sum()),
                  wmatch_drive =("weight", lambda s: s[m.loc[s.index, "has_drive"]].sum()),
                  wmatch_pop   =("weight", lambda s: s[m.loc[s.index, "has_pop"]].sum()),
                  wval_desert  =("w_desert","sum"),
                  wval_drive   =("w_drive","sum"),
                  wval_pop     =("w_pop","sum")))

    out = agg.merge(totals, on="zip", how="left")
    out["zip_total"] = out["zip_total"].replace(0, np.nan)

    # coverage per metric
    out["cov_desert"] = (out["wmatch_desert"] / out["zip_total"]).clip(0,1)
    out["cov_drive"]  = (out["wmatch_drive"]  / out["zip_total"]).clip(0,1)
    out["cov_pop"]    = (out["wmatch_pop"]    / out["zip_total"]).clip(0,1)

    # renormalize each metric by its own matched weight
    renorm_desert = out["wmatch_desert"] > 0
    renorm_drive  = out["wmatch_drive"]  > 0
    renorm_pop    = out["wmatch_pop"]    > 0

    out.loc[renorm_desert, "zip_desert_share"]   = out.loc[renorm_desert, "wval_desert"] / out.loc[renorm_desert, "wmatch_desert"]
    out.loc[renorm_drive,  "zip_drive_time"]     = out.loc[renorm_drive,  "wval_drive"]  / out.loc[renorm_drive,  "wmatch_drive"]
    out.loc[renorm_pop,    "zip_desert_pop_pct"] = out.loc[renorm_pop,    "wval_pop"]    / out.loc[renorm_pop,    "wmatch_pop"]

    # dominant-county fallback (per metric)
    dom = (m.sort_values(["zip","weight"], ascending=[True, False])
             .drop_duplicates("zip")[["zip","county_desert","drive_time_min","desert_pop_pct"]]
             .rename(columns={"county_desert":"dom_desert","drive_time_min":"dom_drive","desert_pop_pct":"dom_pop"}))
    out = out.merge(dom, on="zip", how="left")

    for col_out, col_dom, cond in [
        ("zip_desert_share", "dom_desert", out["zip_desert_share"].isna()),
        ("zip_drive_time", "dom_drive", out["zip_drive_time"].isna()),
        ("zip_desert_pop_pct", "dom_pop", out["zip_desert_pop_pct"].isna()),
    ]:
        out.loc[cond & out[col_dom].notna(), col_out] = out.loc[cond & out[col_dom].notna(), col_dom]

    # state median fallback (per metric)
    county_state = (xw.groupby(["county","state"], as_index=False)["weight"].sum()
                      .sort_values(["county","weight"], ascending=[True, False])
                      .drop_duplicates("county")[["county","state"]])
    cws = county_df.merge(county_state, on="county", how="left")
    state_med = (cws.dropna(subset=["state"])
                   .groupby("state")
                   .agg(state_desert=("county_desert","median"),
                        state_drive =("drive_time_min","median"),
                        state_pop   =("desert_pop_pct","median")))

    zip_state = (xw.groupby(["zip","state"], as_index=False)["weight"].sum()
                   .sort_values(["zip","weight"], ascending=[True, False])
                   .drop_duplicates("zip")[["zip","state"]])

    out = out.merge(zip_state, on="zip", how="left").merge(state_med, on="state", how="left")

    for col_out, col_state in [
        ("zip_desert_share", "state_desert"),
        ("zip_drive_time", "state_drive"),
        ("zip_desert_pop_pct", "state_pop"),
    ]:
        need = out[col_out].isna() & out[col_state].notna()
        out.loc[need, col_out] = out.loc[need, col_state]

    # coverage you use for gating = desert coverage (keep name stable)
    out["zip_alloc_coverage"] = out["cov_desert"].fillna(0.0)

    out["zip_desert_flag"] = (out["zip_desert_share"] >= threshold).astype("Int64")

    # return useful coverage columns for transparency
    return out[[
        "zip","zip_desert_share","zip_desert_flag",
        "zip_drive_time","zip_desert_pop_pct",
        "zip_alloc_coverage","cov_drive","cov_pop"
    ]].rename(columns={"cov_drive":"zip_alloc_cov_drive","cov_pop":"zip_alloc_cov_pop"})




def read_ai_scores_csv(path='data/ai_scores.csv'):
    """Read AI/ML scores from CSV (legacy function)"""
    try:
        df = pd.read_csv(path, dtype={'zip': str})
    except FileNotFoundError:
        return pd.DataFrame(columns=['zip','ai_score'])
    df['zip'] = df['zip'].astype(str).str.extract(r'(\d{5})')[0].str.zfill(5)
    df = df.dropna(subset=['zip']).drop_duplicates(subset=['zip'])
    return df.groupby('zip', as_index=False)['ai_score'].mean()


def export_math_scores_csv(ranked_df, path='data/math_scores.csv'):
    """Export mathematical scores to CSV"""
    out = ranked_df[['zip','score']].rename(columns={'score':'score_math'}).copy()
    out.to_csv(path, index=False)
    return out


def norm01(s):
    """Min-max normalize series to [0,1]"""
    s = pd.to_numeric(s, errors='coerce')
    if s.dropna().empty:
        return s.fillna(0)
    rng = s.max() - s.min()
    return (s - s.min())/rng if rng else s*0


@st.cache_data
def average_scores(math_df, ai_df, normalize=True):
    """
    Blend mathematical and AI scores
    
    Args:
        math_df: DataFrame with columns [zip, score_math]
        ai_df: DataFrame with columns [zip, ai_score]
        normalize: Whether to normalize before averaging
        
    Returns:
        DataFrame with columns [zip, final_score, score_math, ai_score]
    """
    merged = pd.merge(math_df, ai_df, on='zip', how='outer')

    if normalize:
        merged['math_n'] = norm01(merged['score_math'])
        merged['ai_n']   = norm01(merged['ai_score'])
        merged['final_score'] = merged[['math_n','ai_n']].mean(axis=1, skipna=True)
    else:
        merged['final_score'] = merged[['score_math','ai_score']].mean(axis=1, skipna=True)

    merged['final_score'] = merged['final_score'].fillna(0)

    return merged[['zip','final_score','score_math','ai_score']]


@st.cache_data
def preprocess(financial, health, pharmacy, population, aqi_annual=None, hhi=None):
    """
    Merge and preprocess all data sources
    
    Args:
        financial: Financial/income DataFrame
        health: Health burden DataFrame
        pharmacy: Pharmacy locations DataFrame
        population: Population density DataFrame
        aqi_annual: Optional air quality DataFrame
        hhi: Optional heat health index DataFrame
        
    Returns:
        Merged and cleaned DataFrame
    """
    # Income
    fin = financial.copy()
    fin['zip'] = fin['NAME'].str.extract(r'(\d{5})')
    fin = fin.rename(columns={'S1901_C01_012E':'median_income'})[['zip','median_income']]
    fin['median_income'] = pd.to_numeric(fin['median_income'], errors='coerce')
    fin = fin.dropna(subset=['zip']).drop_duplicates(subset=['zip'])

    # Health
    hlth = health.copy()
    hlth['zip'] = hlth['ZCTA5'].astype(str).str.split('.').str[0].str.zfill(5)
    hlth = hlth.rename(columns={'GHLTH_CrudePrev':'health_burden'})[['zip','health_burden']]
    hlth = hlth.dropna(subset=['zip']).drop_duplicates(subset=['zip'])

    # Pharmacies
    pharm = pharmacy.copy()
    pharm['zip'] = pharm['ZIP'].astype(str).str.zfill(5)
    pharm_counts = (
        pharm.dropna(subset=['zip'])
             .groupby('zip')
             .size()
             .reset_index(name='n_pharmacies')
    )

    # Population
    pop = population.copy()
    pop["pop_density"] = pd.to_numeric(pop["pop_density"], errors="coerce")
    pop["lat"] = pd.to_numeric(pop["lat"], errors="coerce")
    pop["lon"] = pd.to_numeric(pop["lon"], errors="coerce")
    pop = pop.dropna(subset=["zip"]).drop_duplicates(subset=["zip"])

    # Merge
    df = pharm_counts.merge(fin,  on='zip', how='outer') \
                     .merge(hlth, on='zip', how='outer') \
                     .merge(pop,  on='zip', how='outer')

    # Optional: AQI
    if aqi_annual is not None and not aqi_annual.empty:
        df = df.merge(aqi_annual[['zip','aqi']], on='zip', how='left')

    # Optional: HHI
    if hhi is not None and not hhi.empty:
        keep_cols = ['zip']
        if 'heat_hhb' in hhi.columns: keep_cols.append('heat_hhb')
        if 'nbe_score' in hhi.columns: keep_cols.append('nbe_score')
        if 'hhi_overall' in hhi.columns: keep_cols.append('hhi_overall')
        df = df.merge(hhi[keep_cols], on='zip', how='left')

    df['n_pharmacies'] = df['n_pharmacies'].fillna(0).astype(int)
    df['pop_density']  = df['pop_density'].fillna(0)
    
    return df



import streamlit as st
import pandas as pd
import numpy as np

# reuse your reader functions
# --- score: deserts first, then score ---
def score_candidates(df, w_scarcity, w_health, w_income, w_pop, w_aqi=0.0, w_heat=0.0):
    eps = 1e-6
    df['median_income'] = pd.to_numeric(df['median_income'], errors='coerce')
    df['health_burden'] = pd.to_numeric(df['health_burden'], errors='coerce')
    df['pop_density']   = pd.to_numeric(df['pop_density'], errors='coerce').fillna(0)

    per_density = df['n_pharmacies'] / (df['pop_density'] + eps)
    df['scarcity']   = 1 / (1 + per_density)

    df['scarcity_n'] = norm01(df['scarcity'])
    df['health_n']   = norm01(df['health_burden'])
    df['income_inv'] = 1 - norm01(df['median_income'])
    df['pop_norm']   = norm01(df['pop_density'])

    # AQI (higher is worse) — treat missing as neutral so ZIPs without AQI aren't penalized
    if 'aqi' in df.columns and df['aqi'].notna().any():
        df['aqi_norm'] = norm01(df['aqi'])
        neutral = df['aqi_norm'].mean(skipna=True)
        df['aqi_norm'] = df['aqi_norm'].fillna(neutral)
    else:
        df['aqi_norm'] = 0.0
        w_aqi = 0.0


    # HHI heat (higher is worse)
    if 'heat_hhb' in df.columns:
        df['heat_norm'] = norm01(df['heat_hhb'])
    else:
        df['heat_norm'] = 0.0
        w_heat = 0.0

    df['score'] = (w_scarcity*df['scarcity_n'].fillna(0) +
                   w_health  *df['health_n'].fillna(0)   +
                   w_income  *df['income_inv'].fillna(0) +
                   w_pop     *df['pop_norm'].fillna(0)   +
                   w_aqi     *df['aqi_norm']             +
                   w_heat    *df['heat_norm'])

    df['desert_flag'] = (df['n_pharmacies'] == 0)
    return df.sort_values(['desert_flag','score'], ascending=[False, False])

def read_population_labels(file_path):
    """Light read of population CSV just to attach City/State by ZIP."""
    df = pd.read_csv(file_path, skiprows=10)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    zip_col  = lower.get("zip")
    city_col = lower.get("city")
    st_col   = lower.get("st") or lower.get("state")

    if not zip_col:
        return pd.DataFrame(columns=["zip","city","state"])

    out = pd.DataFrame({"zip": df[zip_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)})
    if city_col: out["city"] = df[city_col].astype(str).str.strip()
    if st_col:   out["state"] = df[st_col].astype(str).str.strip()
    return out.dropna(subset=["zip"]).drop_duplicates(subset=["zip"])

def read_ifae_csv(path="results/national_ifae_rank.csv"):
    """
    Reads the AI CSV with columns:
    ZCTA5, IFAE_score, composite, iforest_anomaly, median_income, poor_health_pct,
    population, pharmacies_count, pop_per_pharmacy, income_pct_inv, health_pct,
    access_pct_inv, density_pct, pop_density, heat_hhb, heat_pct

    Returns: DataFrame with ['zip','ai_score'] (ai_score from IFAE_score).
    """
    try:
        df = pd.read_csv(path, low_memory=False, dtype={"ZCTA5": str})
    except FileNotFoundError:
        return pd.DataFrame(columns=["zip", "ai_score"])

    # 5-digit ZIP
    df["zip"] = (
        df["ZCTA5"].astype(str)
        .str.extract(r"(\d{5})")[0]
        .str.zfill(5)
    )

    # Use IFAE_score as the AI score (you can switch to 'composite' if you prefer)
    df["ai_score"] = pd.to_numeric(df["IFAE_score"], errors="coerce")

    # One row per zip
    out = df[["zip", "ai_score"]].dropna(subset=["zip"]).drop_duplicates(subset=["zip"])
    return out



def render_top10_map(top10: pd.DataFrame):
    """Show top 10 ZIPs with permanent labels; falls back if Folium isn't installed."""
    # attach City/State for labels
    labels = read_population_labels('data/population_data.csv')
    top10 = top10.merge(labels, on="zip", how="left")
    top10["place"] = (
        top10[["city", "state"]]
        .fillna("")
        .agg(lambda r: ", ".join([p for p in r if p]), axis=1)
        .replace("", "(unknown)")
    )

    # require only SOME lat/lon, not all
    has_latlon_cols = {"lat","lon"}.issubset(top10.columns)
    has_any_points = has_latlon_cols and top10[["lat","lon"]].notna().any().any()

    if not has_any_points:
        st.warning("No latitude/longitude data available in the population file.")
        return

    try:
        import folium
        from streamlit_folium import st_folium

        pts = top10.dropna(subset=["lat","lon"]).copy()

        # Fit map to the points
        fmap = folium.Map(location=[float(pts["lat"].mean()), float(pts["lon"].mean())],
                        zoom_start=4, control_scale=True)
        bounds = pts[["lat","lon"]].values.tolist()
        if bounds:
            fmap.fit_bounds(bounds, padding=(20, 20))

        for _, r in pts.iterrows():
            lat = float(r["lat"]); lon = float(r["lon"])
            place = (f'{r.get("city","")}, {r.get("state","")}'.strip(", ") or "(unknown)")

            # main dot sized by score
            popup = folium.Popup(
                        folium.IFrame(
                            html=f"""
                                <b>ZIP:</b> {r['zip']}<br>
                                <b>Place:</b> {place}<br>
                                <b>Final score:</b> {r.get('final_score', float('nan')):.3f}<br>
                                <b>Math score:</b> {r.get('score_math', float('nan')):.3f}<br>
                                <b>AI score:</b> {r.get('ai_score', float('nan')):.3f}<br>
                                <b>Pharmacies:</b> {int(r['n_pharmacies'])}<br>
                                <b>Pop density:</b> {r['pop_density']:.1f}
                            """,
                            width=240, height=170
                        ),
                        max_width=260
                    )
            folium.CircleMarker(
                        location=[lat, lon],
                        radius=max(5, min(20, 5 + 15*float(r.get("final_score", 0)))),
                        color=None, fill=True, fill_opacity=0.7,
                        popup=popup,
                    ).add_to(fmap)



        st_folium(fmap, width=None)
    except ModuleNotFoundError:
        st.info("For labeled markers, install: `pip install folium streamlit-folium`. Showing basic map instead.")
        st.map(top10.dropna(subset=["lat","lon"])[["lat","lon"]], zoom=4, use_container_width=True)
        keep = [c for c in ["zip","place","final_score","score_math","ai_score","n_pharmacies","pop_density"] if c in top10.columns]
        st.dataframe(top10[keep])


def score_candidates(df, w_scarcity, w_health, w_income, w_pop, w_aqi=0.0, w_heat=0.0, w_edu=0.0, w_drive_time=0.0):
    """
    Apply mathematical scoring model
    
    Args:
        df: Preprocessed DataFrame
        w_scarcity: Weight for pharmacy scarcity
        w_health: Weight for health burden
        w_income: Weight for income (inverted)
        w_pop: Weight for population density
        w_aqi: Weight for air quality
        w_heat: Weight for heat vulnerability
        w_edu: Weight for education (low attainment)
        w_drive_time: Weight for driving time to nearest pharmacy (PRIMARY METRIC)
        
    Returns:
        Scored and sorted DataFrame
    """
    eps = 1e-6
    df['median_income'] = pd.to_numeric(df['median_income'], errors='coerce')
    df['health_burden'] = pd.to_numeric(df['health_burden'], errors='coerce')
    df['pop_density']   = pd.to_numeric(df['pop_density'], errors='coerce').fillna(0)

    df['scarcity']   = 1 / (1 + df['n_pharmacies'])

    df['scarcity_n'] = norm01(df['scarcity'])
    df['health_n']   = norm01(df['health_burden'])
    df['income_inv'] = 1 - norm01(df['median_income'])
    df['pop_norm']   = norm01(df['pop_density'])
    df['edu_low_norm'] = norm01(df['edu_hs_or_lower_pct'])

    # Driving Time (PRIMARY METRIC) - higher minutes = worse access
    if 'zip_drive_time' in df.columns and df['zip_drive_time'].notna().any():
        df['drive_time_norm'] = norm01(df['zip_drive_time'])
        # Fill missing values with median (neutral)
        neutral = df['drive_time_norm'].median(skipna=True)
        df['drive_time_norm'] = df['drive_time_norm'].fillna(neutral)
    else:
        df['drive_time_norm'] = 0.0
        w_drive_time = 0.0

    # AQI
    if 'aqi' in df.columns and df['aqi'].notna().any():
        df['aqi_norm'] = norm01(df['aqi'])
        neutral = df['aqi_norm'].mean(skipna=True)
        df['aqi_norm'] = df['aqi_norm'].fillna(neutral)
    else:
        df['aqi_norm'] = 0.0
        w_aqi = 0.0

    # HHI heat
    if 'heat_hhb' in df.columns:
        df['heat_norm'] = norm01(df['heat_hhb'])
    else:
        df['heat_norm'] = 0.0
        w_heat = 0.0

    if 'edu_hs_or_lower_pct' in df.columns:
        df['edu_low_norm'] = norm01(df['edu_hs_or_lower_pct'])
    else:
        df['edu_low_norm'] = 0.0
        w_edu = 0.0


    
    #no more power scaling
    drive_time_score = df['drive_time_norm'] if w_drive_time > 0 else 0
    scarcity_score = df['scarcity_n'].fillna(0) if w_scarcity > 0 else 0
    health_score = df['health_n'].fillna(0) if w_health > 0 else 0
    income_score = df['income_inv'].fillna(0) if w_income > 0 else 0
    pop_score = df['pop_norm'].fillna(0) if w_pop > 0 else 0
    aqi_score = df['aqi_norm'] if w_aqi > 0 else 0
    heat_score = df['heat_norm'] if w_heat > 0 else 0
    edu_score = df['edu_low_norm'].fillna(0) if w_edu > 0 else 0
    
    df['score'] = (w_drive_time * drive_time_score +
                   w_scarcity  * scarcity_score    +
                   w_health    * health_score      +
                   w_income    * income_score      +
                   w_pop       * pop_score         +
                   w_aqi       * aqi_score         +
                   w_heat      * heat_score        +
                   w_edu       * edu_score)
    
    # Re-normalize final scores to [0, 1] for consistency
    score_min = df['score'].min()
    score_max = df['score'].max()
    if score_max > score_min:
        df['score'] = (df['score'] - score_min) / (score_max - score_min)

    df['desert_flag'] = (df['n_pharmacies'] == 0)
    
    return df.sort_values(['desert_flag','score'], ascending=[False, False])


@st.cache_data
def read_population_labels(file_path):
    """Read city/state labels for ZIPs"""
    df = pd.read_csv(file_path, skiprows=10)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    zip_col  = lower.get("zip")
    city_col = lower.get("city")
    st_col   = lower.get("st") or lower.get("state")

    if not zip_col:
        return pd.DataFrame(columns=["zip","city","state"])

    out = pd.DataFrame({"zip": df[zip_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)})
    if city_col: out["city"] = df[city_col].astype(str).str.strip()
    if st_col:   out["state"] = df[st_col].astype(str).str.strip()
    
    return out.dropna(subset=["zip"]).drop_duplicates(subset=["zip"])


@st.cache_data
def read_ifae_csv(path="results/national_ifae_rank.csv"):
    """
    Read Isolation Forest Autoencoder (IFAE) scores from notebook output
    
    Args:
        path: Path to national_ifae_rank.csv from IF_AE_training.ipynb
        
    Returns:
        DataFrame with columns [zip, ai_score]
    """
    try:
        df = pd.read_csv(path, low_memory=False, dtype={"ZCTA5": str})
    except FileNotFoundError:
        return pd.DataFrame(columns=["zip", "ai_score"])

    df["zip"] = (
        df["ZCTA5"].astype(str)
        .str.extract(r"(\d{5})")[0]
        .str.zfill(5)
    )

    df["ai_score"] = pd.to_numeric(df["IFAE_score"], errors="coerce")

    out = df[["zip", "ai_score"]].dropna(subset=["zip"]).drop_duplicates(subset=["zip"])
    return out


def render_top10_map(top10: pd.DataFrame):
    """
    Render interactive Folium map for top 10 ZIPs
    
    Args:
        top10: DataFrame with top 10 ranked ZIPs (must have lat/lon)
    """
    labels = read_population_labels('data/population_data.csv')
    top10 = top10.merge(labels, on="zip", how="left")
    top10["place"] = (
        top10[["city", "state"]]
        .fillna("")
        .agg(lambda r: ", ".join([p for p in r if p]), axis=1)
        .replace("", "(unknown)")
    )

    has_latlon_cols = {"lat","lon"}.issubset(top10.columns)
    has_any_points = has_latlon_cols and top10[["lat","lon"]].notna().any().any()

    if not has_any_points:
        st.warning("No latitude/longitude data available in the population file.")
        return

    try:
        import folium
        from streamlit_folium import st_folium

        pts = top10.dropna(subset=["lat","lon"]).copy()

        fmap = folium.Map(
            location=[float(pts["lat"].mean()), float(pts["lon"].mean())],
            zoom_start=4, 
            control_scale=True
        )
        
        bounds = pts[["lat","lon"]].values.tolist()
        if bounds:
            fmap.fit_bounds(bounds, padding=(20, 20))

        for _, r in pts.iterrows():
            lat = float(r["lat"])
            lon = float(r["lon"])
            place = (f'{r.get("city","")}, {r.get("state","")}'.strip(", ") or "(unknown)")

            # Build popup HTML with conditional drive time
            drive_time_html = ""
            if 'zip_drive_time' in r and pd.notna(r.get('zip_drive_time')):
                drive_time_html = f"<b>🚗 Drive Time:</b> {r['zip_drive_time']:.1f} min<br>"
            
            popup = folium.Popup(
                folium.IFrame(
                    html=f"""
                        <b>ZIP:</b> {r['zip']}<br>
                        <b>Place:</b> {place}<br>
                        {drive_time_html}
                        <b>Final score:</b> {r.get('final_score', float('nan')):.3f}<br>
                        <b>Math score:</b> {r.get('score_math', float('nan')):.3f}<br>
                        <b>AI score:</b> {r.get('ai_score', float('nan')):.3f}<br>
                        <b>Pharmacies:</b> {int(r['n_pharmacies'])}<br>
                        <b>Pop density:</b> {r['pop_density']:.1f}
                    """,
                    width=260, height=190
                ),
                max_width=280
            )
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(5, min(20, 5 + 15*float(r.get("final_score", 0)))),
                color=None, 
                fill=True, 
                fill_opacity=0.7,
                popup=popup,
            ).add_to(fmap)

        st_folium(fmap, width=None)
        
    except ModuleNotFoundError:
        st.info("For labeled markers, install: `pip install folium streamlit-folium`. Showing basic map instead.")
        st.map(top10.dropna(subset=["lat","lon"])[["lat","lon"]], zoom=4, use_container_width=True)
        keep = [c for c in ["zip","place","final_score","score_math","ai_score","n_pharmacies","pop_density"] if c in top10.columns]
        st.dataframe(top10[keep])


# ============================================================================
# NOTE: The Streamlit UI has been moved to app.py
# This file now contains only data processing functions
# Run the app with: streamlit run app.py
# ============================================================================
