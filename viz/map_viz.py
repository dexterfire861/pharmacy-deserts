# pharmacy_deserts/viz/map_viz.py
import pandas as pd
import streamlit as st
from data_processing.io_readers import read_population_labels

def render_top10_map(top10: pd.DataFrame):
    labels = read_population_labels('data/population_data.csv')
    top10 = top10.merge(labels, on="zip", how="left")
    top10["place"] = (
        top10[["city", "state"]].fillna("").agg(lambda r: ", ".join([p for p in r if p]), axis=1).replace("", "(unknown)")
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
        fmap = folium.Map(location=[float(pts["lat"].mean()), float(pts["lon"].mean())], zoom_start=4, control_scale=True)
        bounds = pts[["lat","lon"]].values.tolist()
        if bounds: fmap.fit_bounds(bounds, padding=(20, 20))

        for _, r in pts.iterrows():
            lat, lon = float(r["lat"]), float(r["lon"])
            place = (f'{r.get("city","")}, {r.get("state","")}'.strip(", ") or "(unknown)")
            drive_time_html = ""
            if ('zip_drive_time' in r.index) and pd.notna(r.get('zip_drive_time')):
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
                    """, width=260, height=190
                ),
                max_width=280
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(5, min(20, 5 + 15*float(r.get("final_score", 0)))),
                color=None, fill=True, fill_opacity=0.7, popup=popup
            ).add_to(fmap)

        st_folium(fmap, width=None)
    except ModuleNotFoundError:
        st.info("For labeled markers, install: `pip install folium streamlit-folium`. Showing basic map instead.")
        st.map(top10.dropna(subset=["lat","lon"])[["lat","lon"]], zoom=4, use_container_width=True)
        keep = [c for c in ["zip","place","final_score","score_math","ai_score","n_pharmacies","pop_density"] if c in top10.columns]
        st.dataframe(top10[keep])
