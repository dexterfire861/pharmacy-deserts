import pandas as pd 



def read_financial_data(file_path):
    df = pd.read_csv(file_path)
    #print(df.columns)
    df = df[['NAME', 'S1901_C01_012E']]
    df['zip'] = df['NAME'].str.extract(r'(\d{5})')
    return df


def read_population_data(file_path):
    df = pd.read_csv(file_path, skiprows=10)

    #print(df.columns)
    df = df.iloc[:, [1, 2, 3]]
    return df

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



import numpy as np

def norm01(s):
    """Min-max normalize to [0,1]."""
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return pd.Series(np.zeros(len(s)), index=s.index)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else pd.Series(np.zeros(len(s)), index=s.index)

def preprocess(financial, health, pharmacy, population):
    # --- Income ---
    fin = financial.copy()
    fin['zip'] = fin['NAME'].str.extract(r'(\d{5})')
    fin = fin.rename(columns={'S1901_C01_012E':'median_income'})[['zip','median_income']]
    fin['median_income'] = pd.to_numeric(fin['median_income'], errors='coerce')

    # --- Health ---
    hlth = health.copy()
    hlth['zip'] = hlth['ZCTA5'].astype(str).str.split('.').str[0].str.zfill(5)
    hlth = hlth.rename(columns={'GHLTH_CrudePrev':'health_burden'})[['zip','health_burden']]

    # --- Pharmacies ---
    pharm = pharmacy.copy()
    pharm['zip'] = pharm['ZIP'].astype(str).str.zfill(5)
    pharm_counts = pharm.groupby('zip').size().reset_index(name='n_pharmacies')

    # --- Population density ---
    pop = population.copy()
    zip_col = pop.columns[1]
    dens_col = pop.columns[2]
    pop['zip'] = pop[zip_col].astype(str).str.extract(r'(\d{5})')
    pop = pop.rename(columns={dens_col:'pop_density'})[['zip','pop_density']]

    # --- Merge all ---
    df = pharm_counts.merge(fin, on='zip', how='outer') \
                     .merge(hlth, on='zip', how='outer') \
                     .merge(pop, on='zip', how='outer')

    df['n_pharmacies'] = df['n_pharmacies'].fillna(0).astype(int)
    return df

import streamlit as st
import pandas as pd
import numpy as np

# reuse your reader functions
def score_candidates(df, w_scarcity, w_health, w_income, w_pop):
    eps = 1e-6
    per_density = df['n_pharmacies'] / (df['pop_density'].fillna(0) + eps)
    df['scarcity'] = 1 / (1 + per_density)

    # normalize features
    df['scarcity_n'] = norm01(df['scarcity'])
    df['health_n']   = norm01(df['health_burden'])
    df['income_inv'] = 1 - norm01(df['median_income'])
    df['pop_norm']   = norm01(df['pop_density'])

    # weighted score
    df['score'] = (w_scarcity*df['scarcity_n'].fillna(0) +
                   w_health*df['health_n'].fillna(0) +
                   w_income*df['income_inv'].fillna(0) +
                   w_pop*df['pop_norm'].fillna(0))
    return df.sort_values('score', ascending=False)

def main():
    st.title("Pharmacy Desert Explorer")

    # --- load your cleaned dataframes ---
    financial_data = read_financial_data('data/financial_data.csv')
    health_data    = read_health_data('data/health_data.csv')
    pharmacy_data  = read_pharmacy_data('data/pharmacy_data.csv')
    population_data= read_population_data('data/population_data.csv')

    df = preprocess(financial_data, health_data, pharmacy_data, population_data)

    # --- sliders for weights ---
    st.sidebar.header("Adjust Weights")
    w_scarcity = st.sidebar.slider("Scarcity (fewer pharmacies)", 0.0, 1.0, 0.35, 0.05)
    w_health   = st.sidebar.slider("Health burden", 0.0, 1.0, 0.25, 0.05)
    w_income   = st.sidebar.slider("Income (low → worse)", 0.0, 1.0, 0.15, 0.05)
    w_pop      = st.sidebar.slider("Population density", 0.0, 1.0, 0.25, 0.05)

    # normalize so they sum to 1
    total = w_scarcity + w_health + w_income + w_pop
    w_scarcity, w_health, w_income, w_pop = [w/total for w in [w_scarcity, w_health, w_income, w_pop]]

    ranked = score_candidates(df, w_scarcity, w_health, w_income, w_pop)

    st.write("### Ranked ZIPs by Pharmacy Desert Score")
    st.dataframe(ranked.head(50))

    # export option
    st.download_button("Download full CSV", ranked.to_csv(index=False), "pharmacy_desert_candidates.csv", "text/csv")

if __name__ == "__main__":
    main()
