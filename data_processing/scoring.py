# pharmacy_deserts/data_processing/scoring.py
import pandas as pd
from utils.cache import cache_data
from data_processing.features import norm01

def export_math_scores_csv(ranked_df, path='data/math_scores.csv'):
    out = ranked_df[['zip','score']].rename(columns={'score':'score_math'}).copy()
    out.to_csv(path, index=False); return out

@cache_data
def average_scores(math_df, ai_df, normalize=True):
    merged = pd.merge(math_df, ai_df, on='zip', how='outer')
    if normalize:
        merged['math_n'] = norm01(merged['score_math'])
        merged['ai_n']   = norm01(merged['ai_score'])
        merged['final_score'] = merged[['math_n','ai_n']].mean(axis=1, skipna=True)
    else:
        merged['final_score'] = merged[['score_math','ai_score']].mean(axis=1, skipna=True)
    merged['final_score'] = merged['final_score'].fillna(0)
    return merged[['zip','final_score','score_math','ai_score']]

def score_candidates(df, w_scarcity, w_health, w_income, w_pop, w_aqi=0.0, w_heat=0.0, w_edu=0.0, w_drive_time=0.0):
    df = df.copy()
    df['median_income'] = pd.to_numeric(df['median_income'], errors='coerce')
    df['health_burden'] = pd.to_numeric(df['health_burden'], errors='coerce')
    df['pop_density']   = pd.to_numeric(df['pop_density'],  errors='coerce').fillna(0)

    df['scarcity']    = 1 / (1 + df['n_pharmacies'])
    df['scarcity_n']  = norm01(df['scarcity'])
    df['health_n']    = norm01(df['health_burden'])
    df['income_inv']  = 1 - norm01(df['median_income'])
    df['pop_norm']    = norm01(df['pop_density'])

    if 'edu_hs_or_lower_pct' in df.columns:
        df['edu_low_norm'] = norm01(df['edu_hs_or_lower_pct'])
    else:
        df['edu_low_norm'] = 0.0; w_edu = 0.0

    if 'zip_drive_time' in df.columns and df['zip_drive_time'].notna().any():
        df['drive_time_norm'] = norm01(df['zip_drive_time'])
        df['drive_time_norm'] = df['drive_time_norm'].fillna(df['drive_time_norm'].median(skipna=True))
    else:
        df['drive_time_norm'] = 0.0; w_drive_time = 0.0

    if 'aqi' in df.columns and df['aqi'].notna().any():
        df['aqi_norm'] = norm01(df['aqi'])
        df['aqi_norm'] = df['aqi_norm'].fillna(df['aqi_norm'].mean(skipna=True))
    else:
        df['aqi_norm'] = 0.0; w_aqi = 0.0

    if 'heat_hhb' in df.columns:
        df['heat_norm'] = norm01(df['heat_hhb'])
    else:
        df['heat_norm'] = 0.0; w_heat = 0.0

    drive_time_score = df['drive_time_norm'] if w_drive_time > 0 else 0
    scarcity_score   = df['scarcity_n'].fillna(0) if w_scarcity > 0 else 0
    health_score     = df['health_n'].fillna(0)   if w_health   > 0 else 0
    income_score     = df['income_inv'].fillna(0) if w_income   > 0 else 0
    pop_score        = df['pop_norm'].fillna(0)   if w_pop      > 0 else 0
    aqi_score        = df['aqi_norm']             if w_aqi      > 0 else 0
    heat_score       = df['heat_norm']            if w_heat     > 0 else 0
    edu_score        = df['edu_low_norm'].fillna(0) if w_edu    > 0 else 0

    df['score'] = (w_drive_time*drive_time_score + w_scarcity*scarcity_score + w_health*health_score +
                   w_income*income_score + w_pop*pop_score + w_aqi*aqi_score + w_heat*heat_score + w_edu*edu_score)

    smin, smax = df['score'].min(), df['score'].max()
    if smax > smin:
        df['score'] = (df['score'] - smin) / (smax - smin)

    df['desert_flag'] = (df['n_pharmacies'] == 0)
    return df.sort_values(['desert_flag','score'], ascending=[False, False])
