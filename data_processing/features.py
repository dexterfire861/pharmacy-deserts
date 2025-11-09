import pandas as pd
from utils.cache import cache_data

def norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    if s.dropna().empty: return s.fillna(0)
    rng = s.max() - s.min()
    return (s - s.min())/rng if rng else s*0

@cache_data
def preprocess(financial, health, pharmacy, population, hhi=None):
    fin = financial.copy()
    fin['zip'] = fin['NAME'].str.extract(r'(\d{5})')
    fin = fin.rename(columns={'S1901_C01_012E':'median_income'})[['zip','median_income']]
    fin['median_income'] = pd.to_numeric(fin['median_income'], errors='coerce')
    fin = fin.dropna(subset=['zip']).drop_duplicates(subset=['zip'])

    hlth = health.copy()
    hlth['zip'] = hlth['ZCTA5'].astype(str).str.split('.').str[0].str.zfill(5)
    hlth = hlth.rename(columns={'GHLTH_CrudePrev':'health_burden'})[['zip','health_burden']].dropna(subset=['zip']).drop_duplicates(subset=['zip'])

    pharm = pharmacy.copy()
    pharm['zip'] = pharm['ZIP'].astype(str).str.zfill(5)
    pharm_counts = (pharm.dropna(subset=['zip']).groupby('zip').size().reset_index(name='n_pharmacies'))

    pop = population.copy()
    for c in ("pop_density","lat","lon"):
        pop[c] = pd.to_numeric(pop[c], errors="coerce")
    pop = pop.dropna(subset=["zip"]).drop_duplicates(subset=["zip"])

    df = pharm_counts.merge(fin, on='zip', how='outer') \
                     .merge(hlth, on='zip', how='outer') \
                     .merge(pop,  on='zip', how='outer')

    if hhi is not None and not hhi.empty:
        keep_cols = ['zip'] + [c for c in ['heat_hhb','nbe_score','hhi_overall'] if c in hhi.columns]
        df = df.merge(hhi[keep_cols], on='zip', how='left')

    df['n_pharmacies'] = df['n_pharmacies'].fillna(0).astype(int)
    df['pop_density']  = df['pop_density'].fillna(0)
    return df
