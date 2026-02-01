import pandas as pd

ZIP_ALIASES = {
    "zip", "zipcode", "zip_code", "zip code",
    "postal", "postalcode", "postal_code", "postal code",
    "zcta", "zcta5", "zcta_5",
}

def _norm(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def _find_zip_col(cols):
    norm_to_col = {_norm(c): c for c in cols}
    for a in ZIP_ALIASES:
        if _norm(a) in norm_to_col:
            return norm_to_col[_norm(a)]
    for c in cols:
        if any(k in _norm(c) for k in ["zip", "postal", "zcta"]):
            return c
    raise ValueError(f"No zip-like column found in: {list(cols)}")

def _extract_zip(series):
    # extract first 5-digit ZIP anywhere in the string
    z = series.astype(str).str.extract(r"(\d{5})", expand=False)
    return z

def join_on_zip(dfs, how="outer", zip_col="zip"):
    prepared = []

    for df in dfs:
        df = df.copy()
        zc = _find_zip_col(df.columns)
        df[zip_col] = _extract_zip(df[zc])
        if zc != zip_col:
            df = df.drop(columns=[zc])
        prepared.append(df)

    out = prepared[0]
    for i, df in enumerate(prepared[1:], start=2):
        out = out.merge(df, on=zip_col, how=how)

    return out

# --- Example with your actual files ---
financial_df = pd.read_csv("financial_data.csv")   # "ZCTA5 00601"
health_df = pd.read_csv("health_data.csv")         # 01001

merged_df = join_on_zip([financial_df, health_df])

print(merged_df.head())
