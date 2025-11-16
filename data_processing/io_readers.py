# pharmacy_deserts/data_processing/io_readers.py
import pandas as pd
import numpy as np
from utils.cache import cache_data

def norm01(s):
    """Percentile normalize series to [0,1]"""
    s = pd.to_numeric(s, errors='coerce')
    if s.dropna().empty:
        return s.fillna(0)
    return s.rank(pct=True, method = 'average')

@cache_data
def read_financial_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['NAME', 'S1901_C01_012E']]
    df['zip'] = df['NAME'].str.extract(r'(\d{5})')
    return df

@cache_data
def read_hhi_excel(file_path):
    df = pd.read_excel(file_path, dtype={'ZCTA': str})
    if 'ZCTA' not in df.columns:
        raise ValueError("HHI Excel must contain 'ZCTA' column.")
    df['zip'] = (
        df['ZCTA'].astype(str).str.extract(r'(\d{5})')[0].fillna('').str.zfill(5)
    )
    out = pd.DataFrame({'zip': df['zip']})
    if 'HHB_SCORE' in df.columns: out['heat_hhb'] = pd.to_numeric(df['HHB_SCORE'], errors='coerce')
    if 'NBE_SCORE' in df.columns: out['nbe_score'] = pd.to_numeric(df['NBE_SCORE'], errors='coerce')
    if 'OVERALL_SCORE' in df.columns: out['hhi_overall'] = pd.to_numeric(df['OVERALL_SCORE'], errors='coerce')
    return out.dropna(subset=['zip']).drop_duplicates(subset=['zip'])

@cache_data
def read_education_data_acs(year=2023, api_key=None):
    import requests
    base = f"https://api.census.gov/data/{year}/acs/acs5/subject"
    vars_ = ["NAME","S1501_C02_007E","S1501_C02_008E","S1501_C02_009E","S1501_C02_014E","S1501_C02_015E"]
    params = {"get": ",".join(vars_), "for": "zip code tabulation area:*"}
    if api_key: params["key"] = api_key
    r = requests.get(base, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0]).rename(columns={
        "zip code tabulation area":"zip",
        "S1501_C02_007E":"pct_less_9",
        "S1501_C02_008E":"pct_9to12_no_diploma",
        "S1501_C02_009E":"pct_hs_grad",
        "S1501_C02_014E":"pct_hs_or_higher",
        "S1501_C02_015E":"pct_ba_or_higher",
    })
    for c in ["pct_less_9","pct_9to12_no_diploma","pct_hs_grad","pct_hs_or_higher","pct_ba_or_higher"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df["edu_hs_or_lower_pct"]  = df[["pct_less_9","pct_9to12_no_diploma","pct_hs_grad"]].sum(axis=1)
    df["edu_less_than_hs_pct"] = 100 - df["pct_hs_or_higher"]
    return df[["zip","edu_hs_or_lower_pct","edu_less_than_hs_pct","pct_ba_or_higher"]]

@cache_data
def read_population_data(file_path):
    df = pd.read_csv(file_path, skiprows=10)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    required = ["zip", "population", "density", "lat", "long"]
    if any(k not in lower for k in required):
        raise ValueError(f"Expected columns Zip, population, density, lat, long; got {df.columns[:10].tolist()}")
    zip_col  = lower["zip"]; pop_col = lower["population"]; dens_col = lower["density"]; lat_col  = lower["lat"]; lon_col  = lower["long"]

    out = pd.DataFrame({
        "zip": df[zip_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
        "population": df[pop_col].astype(str).str.replace(",", "", regex=False),
        "pop_density": df[dens_col].astype(str).str.replace(",", "", regex=False),
        "lat": df[lat_col], "lon": df[lon_col],
    })
    out["population"] = pd.to_numeric(out["population"], errors="coerce")
    out["pop_density"] = pd.to_numeric(out["pop_density"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    return (out.dropna(subset=["zip"])
              .groupby("zip", as_index=False)
              .agg({"population":"sum","pop_density":"max","lat":"first","lon":"first"}))

@cache_data
def read_pharmacy_data(file_path):
    df = pd.read_csv(file_path)
    return df[['ZIP','NAME','X','Y']]

@cache_data
def read_health_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['ZCTA5','GHLTH_CrudePrev']]
    df['ZCTA5'] = df['ZCTA5'].astype(str).str.split('.').str[0].str.zfill(5)
    return df

@cache_data
def read_population_labels(file_path):
    df = pd.read_csv(file_path, skiprows=10)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    zip_col  = lower.get("zip"); city_col = lower.get("city"); st_col = lower.get("st") or lower.get("state")
    if not zip_col:
        return pd.DataFrame(columns=["zip","city","state"])
    out = pd.DataFrame({"zip": df[zip_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)})
    if city_col: out["city"] = df[city_col].astype(str).str.strip()
    if st_col:   out["state"] = df[st_col].astype(str).str.strip()
    return out.dropna(subset=["zip"]).drop_duplicates(subset=["zip"])
