import pandas as pd
import numpy as np
from utils.cache import cache_data

@cache_data
def read_hud_zip_county_crosswalk(path):
    import os
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_excel(path, dtype=str) if ext in (".xlsx",".xls") else pd.read_csv(path, dtype=str, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    zip_col    = cols.get("zip") or cols.get("zipcode") or cols.get("zip_code")
    county_col = cols.get("county") or cols.get("county_fips") or cols.get("fips")
    state_col  = cols.get("state") or cols.get("stabbr") or cols.get("stusps")
    weight_col = next((cols[c] for c in ["tot_ratio","total_ratio","res_ratio"] if c in cols), None)
    if not (zip_col and county_col and weight_col):
        raise ValueError("Crosswalk must have ZIP, COUNTY, and TOT_RATIO/RES_RATIO.")
    out = pd.DataFrame({
        "zip":    df[zip_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
        "county": df[county_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
        "state":  (df[state_col] if state_col else pd.Series(index=df.index, dtype="object")),
        "weight": pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    })
    out = out.dropna(subset=["zip","county"])
    return out[out["weight"] > 0]

@cache_data
def read_county_desert_csv(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    fips_col = next((c for c in df.columns if "fips" in c.lower()), None)
    if not fips_col:
        raise ValueError("County dataset must include a county FIPS column.")
    df["county"] = df[fips_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)

    flag_col = next((c for c in df.columns if c.lower() in ["desert","is_desert","desert_flag","model1_pharm_desert","pharm_desert"]), None)
    if flag_col:
        val = pd.to_numeric(df[flag_col], errors="coerce").fillna(0.0).clip(0,1)
    else:
        score_col = next((c for c in df.columns if any(k in c.lower() for k in ["score","index","risk","prob"])), None)
        if not score_col:
            raise ValueError("No desert flag/score column found in county dataset.")
        raw = pd.to_numeric(df[score_col], errors="coerce")
        val = (raw - raw.min())/(raw.max()-raw.min()) if raw.max()>raw.min() else 0.0

    drive_time_col  = next((c for c in df.columns if "drive_time" in c.lower() and "min" in c.lower()), None)
    pop_pct_col     = next((c for c in df.columns if "desert_pop" in c.lower() and "pct" in c.lower()), None)
    drive_time      = pd.to_numeric(df[drive_time_col], errors="coerce") if drive_time_col else pd.Series(index=df.index, dtype=float)
    desert_pop_pct  = pd.to_numeric(df[pop_pct_col], errors="coerce") if pop_pct_col else pd.Series(index=df.index, dtype=float)

    # Make pop % consistently 0–100
    if desert_pop_pct.notna().any() and desert_pop_pct.max(skipna=True) <= 1.0:
        desert_pop_pct = desert_pop_pct * 100.0

    out = pd.DataFrame({
        "county": df["county"],
        "county_desert": val,
        "drive_time_min": drive_time,
        "desert_pop_pct": desert_pop_pct
    })
    return out.dropna(subset=["county"]).drop_duplicates(subset=["county"])

@cache_data
def downscale_county_to_zip(county_df, xwalk_df,
                            tiny_cutoff=0.01, min_coverage=0.60, threshold=0.50):
    # filter tiny overlaps
    xw = xwalk_df[xwalk_df["weight"] >= tiny_cutoff].copy()
    totals = xw.groupby("zip", as_index=False)["weight"].sum().rename(columns={"weight":"zip_total"})

    m = xw.merge(county_df, on="county", how="left")
    m["has_desert"] = m["county_desert"].notna()
    m["has_drive"]  = m["drive_time_min"].notna()
    m["has_pop"]    = m["desert_pop_pct"].notna()

    m["w_desert"] = np.where(m["has_desert"], m["weight"] * m["county_desert"], 0.0)
    m["w_drive"]  = np.where(m["has_drive"],  m["weight"] * m["drive_time_min"], 0.0)
    m["w_pop"]    = np.where(m["has_pop"],    m["weight"] * m["desert_pop_pct"], 0.0)

    agg = (m.groupby("zip", as_index=False)
             .agg(zip_wsum=("weight","sum"),
                  wmatch_desert=("weight", lambda s: s[m.loc[s.index,"has_desert"]].sum()),
                  wmatch_drive =("weight", lambda s: s[m.loc[s.index,"has_drive"]].sum()),
                  wmatch_pop   =("weight", lambda s: s[m.loc[s.index,"has_pop"]].sum()),
                  wval_desert  =("w_desert","sum"),
                  wval_drive   =("w_drive","sum"),
                  wval_pop     =("w_pop","sum")))

    out = agg.merge(totals, on="zip", how="left")
    out["zip_total"] = out["zip_total"].replace(0, np.nan)

    out["cov_desert"] = (out["wmatch_desert"]/out["zip_total"]).clip(0,1)
    out["cov_drive"]  = (out["wmatch_drive"]/out["zip_total"]).clip(0,1)
    out["cov_pop"]    = (out["wmatch_pop"]/out["zip_total"]).clip(0,1)

    renorm_desert = out["wmatch_desert"] > 0
    renorm_drive  = out["wmatch_drive"]  > 0
    renorm_pop    = out["wmatch_pop"]    > 0

    out.loc[renorm_desert,"zip_desert_share"]   = out.loc[renorm_desert,"wval_desert"]/out.loc[renorm_desert,"wmatch_desert"]
    out.loc[renorm_drive, "zip_drive_time"]     = out.loc[renorm_drive, "wval_drive"]/out.loc[renorm_drive, "wmatch_drive"]
    out.loc[renorm_pop,   "zip_desert_pop_pct"] = out.loc[renorm_pop,   "wval_pop"]/out.loc[renorm_pop,   "wmatch_pop"]

    # dominant-county fallback
    dom = (m.sort_values(["zip","weight"], ascending=[True, False])
             .drop_duplicates("zip")[["zip","county_desert","drive_time_min","desert_pop_pct"]]
             .rename(columns={"county_desert":"dom_desert","drive_time_min":"dom_drive","desert_pop_pct":"dom_pop"}))
    out = out.merge(dom, on="zip", how="left")
    for col_out, col_dom in [("zip_desert_share","dom_desert"), ("zip_drive_time","dom_drive"), ("zip_desert_pop_pct","dom_pop")]:
        need = out[col_out].isna() & out[col_dom].notna()
        out.loc[need, col_out] = out.loc[need, col_dom]

    # state median fallback
    county_state = (xw.groupby(["county","state"], as_index=False)["weight"].sum()
                      .sort_values(["county","weight"], ascending=[True, False])
                      .drop_duplicates("county")[["county","state"]])
    cws = county_df.merge(county_state, on="county", how="left")
    state_med = (cws.dropna(subset=["state"]).groupby("state")
                   .agg(state_desert=("county_desert","median"),
                        state_drive =("drive_time_min","median"),
                        state_pop   =("desert_pop_pct","median")))

    zip_state = (xw.groupby(["zip","state"], as_index=False)["weight"].sum()
                   .sort_values(["zip","weight"], ascending=[True, False])
                   .drop_duplicates("zip")[["zip","state"]])

    out = out.merge(zip_state, on="zip", how="left").merge(state_med, on="state", how="left")
    for col_out, col_state in [("zip_desert_share","state_desert"), ("zip_drive_time","state_drive"), ("zip_desert_pop_pct","state_pop")]:
        need = out[col_out].isna() & out[col_state].notna()
        out.loc[need, col_out] = out.loc[need, col_state]

    out["zip_alloc_coverage"] = out["cov_desert"].fillna(0.0)
    out["zip_desert_flag"] = (out["zip_desert_share"] >= threshold).astype("Int64")

    return out[["zip","zip_desert_share","zip_desert_flag","zip_drive_time","zip_desert_pop_pct",
                "zip_alloc_coverage","cov_drive","cov_pop"]].rename(
        columns={"cov_drive":"zip_alloc_cov_drive","cov_pop":"zip_alloc_cov_pop"}
    )
