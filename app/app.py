# pharmacy_deserts/app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to sys.path so we can import our modules
# This is needed because streamlit runs app.py as a top-level script
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from data_processing import (
    read_financial_data, read_health_data, read_pharmacy_data, read_population_data,
    read_aqi_data, read_hhi_excel, read_ifae_csv, read_population_labels,
    read_education_data_acs, read_hud_zip_county_crosswalk, read_county_desert_csv,
    downscale_county_to_zip, preprocess, score_candidates, average_scores
)
from viz.map_viz import render_top10_map

def main():
    st.set_page_config(page_title="Pharmacy Desert Explorer", layout="wide", initial_sidebar_state="expanded")

    if 'prev_top10_zips' not in st.session_state: st.session_state.prev_top10_zips = []
    if 'prev_weights'   not in st.session_state: st.session_state.prev_weights = {}

    st.title("Pharmacy Desert Explorer")
    st.markdown("""
    ### Hybrid AI + Mathematical Approach
    1) **Math model** (weighted, adjustable)  •  2) **AI model** (Isolation Forest)  
    **Final Ranking** blends both for robust results.
    """)
    st.divider()

    with st.spinner("Loading data... (this only happens once per session)"):
        try:
            financial_data = read_financial_data('data/financial_data.csv')
            health_data    = read_health_data('data/health_data.csv')
            pharmacy_data  = read_pharmacy_data('data/pharmacy_data.csv')
            population_data= read_population_data('data/population_data.csv')
            aqi_monthly, aqi_annual = read_aqi_data('data/AQI_data.csv')
            hhi           = read_hhi_excel('data/HHI_data.xlsx')
            education     = read_education_data_acs(year=2023)
            hud_xwalk     = read_hud_zip_county_crosswalk("data/zip_county_cross.xlsx")
            county_df     = read_county_desert_csv("data/driving-time-desert.csv")
            zip_desert_df = downscale_county_to_zip(county_df, hud_xwalk, tiny_cutoff=0.01, min_coverage=0.60, threshold=0.50)

            df = preprocess(financial_data, health_data, pharmacy_data, population_data, aqi_annual=aqi_annual, hhi=hhi)
            df = df.merge(education[["zip","edu_hs_or_lower_pct"]], on='zip', how='left')
            df = df.merge(zip_desert_df, on='zip', how='left')
            st.success(f"Data loaded successfully! Analyzing {len(df):,} ZIP codes")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    # Sidebar: mode
    st.sidebar.header("Scoring Mode")
    scoring_mode = st.sidebar.radio("Choose ranking method:", ["Blended (Math + AI)", "Math Only", "AI Only"], index=0,
                                    help="Blended: 50% Math + 50% AI | Math Only: Show slider impact | AI Only: Isolation Forest anomalies")
    st.sidebar.divider()

    # Weights
    st.sidebar.header("Mathematical Model Weights")
    slider_disabled = (scoring_mode == "AI Only")
    if slider_disabled: st.sidebar.info("Sliders disabled in AI-only mode")
    else: st.sidebar.markdown("*Adjust priorities to see ranking changes:*")

    st.sidebar.markdown("**PRIMARY METRIC:**")
    w_drive_time = st.sidebar.slider("🚗 Drive Time (minutes)", 0.0, 1.0, 0.40, 0.05,
                                     help="Higher = prioritize longer drive times", disabled=slider_disabled) if 'zip_drive_time' in df.columns else 0.0
    st.sidebar.markdown("**SECONDARY METRICS:**")
    w_scarcity = st.sidebar.slider("Scarcity (fewer pharmacies)", 0.0, 1.0, 0.08, 0.05, disabled=slider_disabled)
    w_health   = st.sidebar.slider("Health burden", 0.0, 1.0, 0.18, 0.05, disabled=slider_disabled)
    w_income   = st.sidebar.slider("Income (low → worse)", 0.0, 1.0, 0.12, 0.05, disabled=slider_disabled)
    w_pop      = st.sidebar.slider("Population density", 0.0, 1.0, 0.05, 0.05, disabled=slider_disabled)

    st.sidebar.markdown("**TERTIARY METRICS (optional):**")
    w_aqi  = st.sidebar.slider("Air quality (AQI)", 0.0, 1.0, 0.03, 0.05, disabled=slider_disabled) if (('aqi' in df.columns) and df['aqi'].notna().any()) else 0.0
    w_heat = st.sidebar.slider("Heat vulnerability (HHI)", 0.0, 1.0, 0.05, 0.05, disabled=slider_disabled) if ('heat_hhb' in df.columns) else 0.0
    w_edu  = st.sidebar.slider("Education (low attainment)", 0.0, 1.0, 0.09, 0.05, disabled=slider_disabled) if ('edu_hs_or_lower_pct' in df.columns) else 0.0

    # GoodRx hard gate controls
    st.sidebar.header("GoodRx Desert Gate")
    gate_goodrx = st.sidebar.checkbox("Hard gate to GoodRx-defined deserts", value=True,
                                      help="When ON, only ZIPs that are GoodRx drive-time deserts (via county→ZIP downscale) are kept.")
    min_cov     = st.sidebar.slider("Minimum crosswalk coverage (HUD)", 0.0, 1.0, 0.60, 0.05)
    thr_goodrx  = st.sidebar.slider("Desert severity threshold (advanced)", 0.0, 1.0, 0.50, 0.05)

    # Normalize weights
    total = w_drive_time + w_scarcity + w_health + w_income + w_pop + w_aqi + w_heat + w_edu
    if total > 0:
        w_drive_time, w_scarcity, w_health, w_income, w_pop, w_aqi, w_heat, w_edu = [
            w/total for w in [w_drive_time, w_scarcity, w_health, w_income, w_pop, w_aqi, w_heat, w_edu]
        ]

    st.sidebar.markdown("---")
    st.sidebar.caption("**Normalized Weights:**")
    if w_drive_time > 0: st.sidebar.caption(f"🚗 Drive Time: {w_drive_time:.2%}")
    for name,val in [("Scarcity",w_scarcity),("Health",w_health),("Income",w_income),("Population",w_pop),
                     ("Air Quality",w_aqi),("Heat",w_heat),("Education",w_edu)]:
        if val > 0: st.sidebar.caption(f"{name}: {val:.2%}")

    # Hard gate (before scoring)
    if "zip_desert_share" in df.columns:
        df["zip_desert_flag_user"] = (df["zip_desert_share"] >= thr_goodrx).astype("Int64")
    else:
        df["zip_desert_flag_user"] = pd.NA

    if gate_goodrx:
        flag_col = "zip_desert_flag_user" if "zip_desert_flag_user" in df.columns else "zip_desert_flag"
        if flag_col not in df.columns:
            st.warning("GoodRx gate requested, but downscaled fields not found.")
        else:
            mask = (df[flag_col] == 1) & (df.get("zip_alloc_coverage", 0).fillna(0) >= min_cov)
            kept = df.loc[mask].copy(); dropped = len(df) - len(kept)
            st.info(f"GoodRx hard gate active → kept {len(kept):,} ZIPs, filtered out {dropped:,}.")
            if kept.empty:
                st.warning("No ZIPs pass the GoodRx gate. Lower coverage/threshold.")
                st.stop()
            df = kept
        st.caption(f"🔒 GoodRx hard gate ON · min HUD coverage ≥ {min_cov:.0%} · threshold ≥ {thr_goodrx:.0%}")

    # Score
    ranked = score_candidates(df, w_scarcity, w_health, w_income, w_pop, w_aqi=w_aqi, w_heat=w_heat, w_edu=w_edu, w_drive_time=w_drive_time)
    ranked['score'] = pd.to_numeric(ranked['score'], errors='coerce')
    ranked = ranked.dropna(subset=['score'])
    math_df = ranked[['zip','score']].rename(columns={'score':'score_math'}).copy()

    # AI merge
    st.sidebar.markdown("---")
    st.sidebar.header("AI Model (Isolation Forest)")
    ai_file_path = "results/national_ifae_rank.csv"
    try:
        ai_df = read_ifae_csv(ai_file_path)
        if os.path.exists(ai_file_path):
            mod_time = os.path.getmtime(ai_file_path)
            mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
            st.sidebar.success("AI scores loaded")
            st.sidebar.caption(f"**Last updated:** {mod_date}")
            st.sidebar.caption(f"**Coverage:** {len(ai_df):,} ZIP codes")
            if st.sidebar.button("Refresh AI Scores", help="Re-run IF_AE_training.ipynb to update"):
                st.sidebar.info("To update AI scores, run: `jupyter execute IF_AE_training.ipynb`")
        else:
            st.sidebar.warning("AI scores file not found")
            st.sidebar.info("Run `IF_AE_training.ipynb` to generate AI scores")
    except Exception as e:
        st.sidebar.error(f"Error loading AI scores: {e}")
        st.sidebar.info("Using mathematical scores only")
        ai_df = pd.DataFrame(columns=['zip','ai_score'])

    # Blend or pick
    if scoring_mode == "Math Only":
        ranked['final_score'] = ranked['score']; ranked['ai_score'] = np.nan; sort_col = 'final_score'; st.sidebar.success("Using Mathematical Model Only")
    elif scoring_mode == "AI Only":
        ranked['final_score'] = ranked[['zip']].merge(ai_df, on='zip', how='left')['ai_score']
        ranked['score_math'] = ranked['score']; sort_col = 'final_score'; st.sidebar.success("Using AI Model Only")
    else:
        combo = average_scores(math_df, ai_df, normalize=True)
        ranked = ranked.merge(combo, on='zip', how='left'); sort_col = 'final_score'; st.sidebar.success("Using Blended Approach")

    ranked[sort_col] = pd.to_numeric(ranked[sort_col], errors='coerce')
    ranked = ranked.dropna(subset=[sort_col])
    ranked['desert_flag'] = ranked['desert_flag'].astype(int)

    ranked = ranked.sort_values(['desert_flag', sort_col], ascending=[False, False], kind='mergesort', na_position='last').reset_index(drop=True)

    # ----- UI -----
    st.write("### Top Pharmacy Desert Candidates")
    st.caption(f"**Active Mode:** { {'Math Only':'🔢 Mathematical Model','AI Only':'🤖 AI Model (Isolation Forest)','Blended (Math + AI)':'⚖️ Hybrid: Math + AI'}[scoring_mode] }")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total ZIPs Analyzed", f"{len(ranked):,}")
    with col2: st.metric("Zero-Pharmacy Deserts", f"{(ranked['desert_flag']==1).sum():,}")
    with col3: st.metric("Avg Final Score", f"{ranked[sort_col].mean():.3f}")
    with col4:
        if 'ai_score' in ranked.columns and ranked['ai_score'].notna().any():
            ai_coverage = ranked['ai_score'].notna().sum()
            st.metric("AI Coverage", f"{100*ai_coverage/len(ranked):.1f}%")
        else:
            st.metric("AI Coverage", "0%")

    if scoring_mode != "AI Only":
        st.divider(); st.write("#### Weight Distribution")
        import plotly.graph_objects as go
        names, values = [], []
        if w_drive_time > 0: names.append('🚗 Drive Time'); values.append(w_drive_time)
        names += ['Scarcity','Health','Income','Population']; values += [w_scarcity,w_health,w_income,w_pop]
        if w_aqi>0: names.append('Air Quality'); values.append(w_aqi)
        if w_heat>0: names.append('Heat'); values.append(w_heat)
        if w_edu>0: names.append('Education'); values.append(w_edu)
        fig = go.Figure(data=[go.Bar(x=names, y=values, text=[f'{v:.1%}' for v in values], textposition='auto')])
        fig.update_layout(title="Current Weight Distribution (Normalized)", yaxis_title="Weight", height=300, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
        if scoring_mode == "Blended (Math + AI)": st.info("In Blended mode, Math weights affect 50% of the final score")

    # Optional Top-10 change tracker (Math Only) — unchanged from your UI, omitted here for brevity

    # Columns to show
    show_cols = ['zip','n_pharmacies','pop_density','median_income','health_burden']
    if 'zip_drive_time' in ranked.columns: show_cols.append('zip_drive_time')
    for c in ["zip_desert_share","zip_desert_flag","zip_desert_flag_user","zip_desert_pop_pct","zip_alloc_coverage","zip_alloc_method"]:
        if c in ranked.columns: show_cols.append(c)
    for c in ['aqi','heat_hhb','edu_hs_or_lower_pct']: 
        if c in ranked.columns: show_cols.append(c)
    for c in ['scarcity','pop_norm','income_inv','health_n','score_math','ai_score','final_score']:
        if c in ranked.columns: show_cols.append(c)
    show_cols += ['desert_flag']
    show_cols = [c for c in show_cols if c in ranked.columns]

    with st.expander("Understanding the Scores", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Scores**: `score_math` (weighted), `ai_score` (IFAE), `final_score` (blend), `desert_flag` (0/1).")
        with c2:
            st.markdown("**Features**: `scarcity`(inverse pharmacies), `health_n`, `income_inv`, `pop_norm`, optional: `drive_time_norm`, `aqi_norm`, `heat_norm`, `edu_low_norm`.")

    st.dataframe(ranked[show_cols].head(50), use_container_width=True, height=400)

    st.write("### Top 10 ZIPs on Interactive Map")
    render_top10_map(ranked.head(10).copy())

    st.write("### Export Results")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Full Results CSV", ranked.to_csv(index=False), "pharmacy_desert_candidates_full.csv", "text/csv")
    with col2:
        st.download_button("Download Top 100 CSV", ranked.head(100).to_csv(index=False), "pharmacy_desert_top100.csv", "text/csv")

    st.divider()
    st.caption("🏥 Pharmacy Desert Explorer | Hybrid AI + Mathematical Approach")
    st.caption("Built with Streamlit | Data refreshed on page load")

if __name__ == "__main__":
    main()
