import pandas as pd 


def read_financial_data(file_path):
    df = pd.read_csv(file_path)
    #print(df.columns)
    df = df[['NAME', 'S1901_C01_012E']]
    df['zip'] = df['NAME'].str.extract(r'(\d{5})')
    return df

import pandas as pd
import numpy as np

def read_hhi_excel(file_path):
    """
    Reads Heat-Health Index (HHI) Excel and returns:
      zip, heat_hhb (from HHB_SCORE), nbe_score (optional), hhi_overall (optional)
    """
    # Load – if there are multiple sheets, you can pass sheet_name=...
    df = pd.read_excel(file_path, dtype={'ZCTA': str})

    # Ensure 5-digit ZIP from ZCTA
    if 'ZCTA' not in df.columns:
        raise ValueError("HHI Excel must contain 'ZCTA' column.")
    df['zip'] = (
        df['ZCTA'].astype(str)
        .str.extract(r'(\d{5})')[0]     # <-- get the first group
        .fillna('')
        .str.zfill(5)
    )

    # Pick columns if present
    out = pd.DataFrame({'zip': df['zip']})

    if 'HHB_SCORE' in df.columns:
        out['heat_hhb'] = pd.to_numeric(df['HHB_SCORE'], errors='coerce')

    # Optional extras if you want them later
    if 'NBE_SCORE' in df.columns:
        out['nbe_score'] = pd.to_numeric(df['NBE_SCORE'], errors='coerce')
    if 'OVERALL_SCORE' in df.columns:
        out['hhi_overall'] = pd.to_numeric(df['OVERALL_SCORE'], errors='coerce')

    # Keep one row per ZIP
    out = out.dropna(subset=['zip']).drop_duplicates(subset=['zip'])

    return out


def read_aqi_data(file_path):
    """
    Reads hourly/daily PM2.5 rows and aggregates to:
      - monthly weighted averages per ZIP (by Observation Count)
      - annual weighted average per ZIP (same weight)
    Returns: aqi_monthly, aqi_annual
    """
    df = pd.read_csv(file_path, low_memory=False)

    # pick the right ZIP column (the file has ZIP twice; pandas may create 'ZIP' and 'ZIP.1')
    zip_cols = [c for c in df.columns if c.upper().startswith('ZIP')]
    zip_col = zip_cols[-1]  # use the last occurrence
    df['zip'] = df[zip_col].astype(str).str.extract(r'(\d{5})')[0].fillna('').str.zfill(5)

    # ensure numeric
    val_col = 'Arithmetic Mean'
    w_col   = 'Observation Count'
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df[w_col]   = pd.to_numeric(df[w_col], errors='coerce').fillna(0)

    # restrict to PM2.5 if multiple params exist
    if 'Parameter Name' in df.columns:
        df = df[df['Parameter Name'].str.contains('PM2.5', na=False)]

    # month parsing (your file has 'Month' like 'Jan-24'); fall back to Date Local if needed
    if 'Month' in df.columns:
        # robust parse for formats like 'Jan-24'
        df['month'] = pd.to_datetime(df['Month'], format='%b-%y', errors='coerce')
    else:
        df['month'] = pd.to_datetime(df['Date Local'], errors='coerce').values.astype('datetime64[M]')

    # drop junk rows
    df = df.dropna(subset=['zip', 'month', val_col])
    df = df[df[w_col] > 0]

    # ---- monthly weighted mean per ZIP ----
    # weighted mean = sum(val * weight) / sum(weight)
    grp = df.groupby(['zip', 'month'], as_index=False).apply(
        lambda g: pd.Series({
            'aqi_monthly': np.average(g[val_col], weights=g[w_col]),
            'obs_month':   g[w_col].sum()
        })
    ).reset_index(drop=True)

    # ---- annual weighted mean per ZIP (across all months) ----
    grp_annual = df.groupby('zip', as_index=False).apply(
        lambda g: pd.Series({
            'aqi': np.average(g[val_col], weights=g[w_col]),
            'obs_total': g[w_col].sum()
        })
    ).reset_index(drop=True)

    # nice month label for display (YYYY-MM)
    grp['month_label'] = grp['month'].dt.strftime('%Y-%m')

    return grp[['zip', 'month', 'month_label', 'aqi_monthly', 'obs_month']], grp_annual[['zip', 'aqi', 'obs_total']]



def read_population_data(file_path):
    # After 10 junk rows, the header starts with a leading comma
    df = pd.read_csv(file_path, skiprows=10)

    # Normalize column names (strip spaces)
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

    # If multiple rows per ZIP, keep the max density and the first valid lat/lon
    out = (out.dropna(subset=["zip"])
              .groupby("zip", as_index=False)
              .agg({"pop_density":"max",
                    "lat":"first",
                    "lon":"first"}))
    return out



def read_pharmacy_data(file_path):
    df = pd.read_csv(file_path)
    #print(df.columns)
    df = df[['ZIP', 'NAME', 'X', 'Y']]
    return df

def read_health_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['ZCTA5', 'GHLTH_CrudePrev']]
    df['ZCTA5'] = df['ZCTA5'].astype(str).str.split('.').str[0].str.zfill(5)
    return df

# --- read AI scores produced by your notebook ---
def read_ai_scores_csv(path='data/ai_scores.csv'):
    try:
        df = pd.read_csv(path, dtype={'zip': str})
    except FileNotFoundError:
        # no AI file yet; return empty frame so app still runs
        return pd.DataFrame(columns=['zip','ai_score'])
    df['zip'] = df['zip'].astype(str).str.extract(r'(\d{5})')[0].str.zfill(5)
    df = df.dropna(subset=['zip']).drop_duplicates(subset=['zip'])
    # If duplicates per ZIP exist, average them
    return df.groupby('zip', as_index=False)['ai_score'].mean()


# --- export the mathematical score in case you want a CSV of it too ---
def export_math_scores_csv(ranked_df, path='data/math_scores.csv'):
    out = ranked_df[['zip','score']].rename(columns={'score':'score_math'}).copy()
    out.to_csv(path, index=False)
    return out


# --- average AI + math into final_score (with optional normalization) ---
def average_scores(math_df, ai_df, normalize=True):
    # math_df: columns zip, score_math
    # ai_df:   columns zip, ai_score
    merged = pd.merge(math_df, ai_df, on='zip', how='outer')

    def _norm01(s):
        s = pd.to_numeric(s, errors='coerce')
        if s.dropna().empty:
            return s.fillna(0)
        rng = s.max() - s.min()
        return (s - s.min())/rng if rng else s*0

    if normalize:
        merged['math_n'] = _norm01(merged['score_math'])
        merged['ai_n']   = _norm01(merged['ai_score'])
        # row-wise mean ignoring NaNs; if one is missing, the other carries through
        merged['final_score'] = merged[['math_n','ai_n']].mean(axis=1, skipna=True)
    else:
        merged['final_score'] = merged[['score_math','ai_score']].mean(axis=1, skipna=True)

    # if BOTH missing, set to NaN → fill with 0 so sorting works
    merged['final_score'] = merged['final_score'].fillna(0)

    return merged[['zip','final_score','score_math','ai_score']]




import numpy as np

def norm01(s):
    """Min-max normalize to [0,1]."""
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return pd.Series(np.zeros(len(s)), index=s.index)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else pd.Series(np.zeros(len(s)), index=s.index)

def preprocess(financial, health, pharmacy, population, aqi_annual=None, hhi=None):
    # --- Income ---
    fin = financial.copy()
    fin['zip'] = fin['NAME'].str.extract(r'(\d{5})')
    fin = fin.rename(columns={'S1901_C01_012E':'median_income'})[['zip','median_income']]
    fin['median_income'] = pd.to_numeric(fin['median_income'], errors='coerce')
    fin = fin.dropna(subset=['zip']).drop_duplicates(subset=['zip'])

    # --- Health (PLACES) ---
    hlth = health.copy()
    hlth['zip'] = hlth['ZCTA5'].astype(str).str.split('.').str[0].str.zfill(5)
    hlth = hlth.rename(columns={'GHLTH_CrudePrev':'health_burden'})[['zip','health_burden']]
    hlth = hlth.dropna(subset=['zip']).drop_duplicates(subset=['zip'])

    # --- Pharmacies -> counts ---
    pharm = pharmacy.copy()
    pharm['zip'] = pharm['ZIP'].astype(str).str.zfill(5)
    pharm_counts = (
        pharm.dropna(subset=['zip'])
             .groupby('zip')
             .size()
             .reset_index(name='n_pharmacies')
    )

    # --- Population density ---
    pop = population.copy()  # has zip, pop_density, lat, lon
    pop["pop_density"] = pd.to_numeric(pop["pop_density"], errors="coerce")
    pop["lat"] = pd.to_numeric(pop["lat"], errors="coerce")
    pop["lon"] = pd.to_numeric(pop["lon"], errors="coerce")
    pop = pop.dropna(subset=["zip"]).drop_duplicates(subset=["zip"])
    # (optional) filter positives after you confirm values look right
    # pop = pop[pop["pop_density"] > 0]



    # --- Merge core ---
    df = pharm_counts.merge(fin,  on='zip', how='outer') \
                     .merge(hlth, on='zip', how='outer') \
                     .merge(pop,  on='zip', how='outer')

    # --- Optional AQI (annual) ---
    if aqi_annual is not None and not aqi_annual.empty:
        df = df.merge(aqi_annual[['zip','aqi']], on='zip', how='left')

    # --- Optional HHI (heat) ---
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
        st.dataframe(top10[["zip","place","score","n_pharmacies","pop_density"]])



def main():
    st.title("Pharmacy Desert Explorer")

    # --- load your cleaned dataframes ---
    financial_data = read_financial_data('data/financial_data.csv')
    health_data    = read_health_data('data/health_data.csv')
    pharmacy_data  = read_pharmacy_data('data/pharmacy_data.csv')
    population_data= read_population_data('data/population_data.csv')
    aqi_monthly, aqi_annual       = read_aqi_data('data/AQI_data.csv')
    hhi = read_hhi_excel('data/HHI_data.xlsx')

    df = preprocess(financial_data, health_data, pharmacy_data, population_data,
                aqi_annual=aqi_annual, hhi=hhi)




    # Sliders
    st.sidebar.header("Adjust Weights")
    w_scarcity = st.sidebar.slider("Scarcity (fewer pharmacies)", 0.0, 1.0, 0.30, 0.05)
    w_health   = st.sidebar.slider("Health burden",               0.0, 1.0, 0.20, 0.05)
    w_income   = st.sidebar.slider("Income (low → worse)",        0.0, 1.0, 0.10, 0.05)
    w_pop      = st.sidebar.slider("Population density",          0.0, 1.0, 0.25, 0.05)

    w_aqi  = st.sidebar.slider("Air quality (AQI)", 0.0, 1.0, 0.10, 0.05) if aqi_annual is not None else 0.0
    w_heat = st.sidebar.slider("Heat (HHI - HHB_SCORE)", 0.0, 1.0, 0.15, 0.05) if hhi is not None else 0.0

    total = w_scarcity + w_health + w_income + w_pop + w_aqi + w_heat
    w_scarcity, w_health, w_income, w_pop, w_aqi, w_heat = [w/total for w in [w_scarcity, w_health, w_income, w_pop, w_aqi, w_heat]]

    # --- your math ranking (already computed above) ---
    ranked = score_candidates(df, w_scarcity, w_health, w_income, w_pop, w_aqi=w_aqi, w_heat=w_heat)

    # Build a minimal math-scores df for merging
    math_df = ranked[['zip', 'score']].rename(columns={'score': 'score_math'}).copy()

    # Read AI CSV (IFAE)
    ai_df = read_ifae_csv("results/national_ifae_rank.csv")

    # Average (normalized) math + AI -> final_score
    combo = average_scores(math_df, ai_df, normalize=True)

    # Attach final_score, ai_score back to the full ranked frame
    ranked = ranked.merge(combo, on='zip', how='left')

    # Use final_score for ordering and for the map
    ranked = ranked.sort_values(['desert_flag','final_score'], ascending=[False, False])

    # Show a few extra HHI columns if present
    show_cols = ['zip','n_pharmacies','pop_density','median_income','health_burden']
    if 'aqi' in ranked.columns:       
        show_cols.append('aqi')
    if 'heat_hhb' in ranked.columns:  
        show_cols.append('heat_hhb')
    # add the math/AI/final columns
    show_cols += ['scarcity','pop_norm','income_inv','health_n','score_math','ai_score','final_score','desert_flag']
    st.dataframe(ranked[show_cols].head(50))


    st.write("### Top 10 ZIPs on the map")
    top10 = ranked.head(10).copy()
    render_top10_map(top10)


    
    st.download_button("Download full CSV", ranked.to_csv(index=False), "pharmacy_desert_candidates.csv", "text/csv")
if __name__ == "__main__":
    main()