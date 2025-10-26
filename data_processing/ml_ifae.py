# pharmacy_deserts/data_processing/ml_ifae.py
import pandas as pd
from utils.cache import cache_data

@cache_data
def read_ifae_csv(path="results/national_ifae_rank.csv"):
    try:
        df = pd.read_csv(path, low_memory=False, dtype={"ZCTA5": str})
    except FileNotFoundError:
        return pd.DataFrame(columns=["zip","ai_score"])
    df["zip"] = df["ZCTA5"].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)
    df["ai_score"] = pd.to_numeric(df["IFAE_score"], errors="coerce")
    return df[["zip","ai_score"]].dropna(subset=["zip"]).drop_duplicates(subset=["zip"])
