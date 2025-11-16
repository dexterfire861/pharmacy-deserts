# pharmacy_deserts/viz/map_viz.py
import pandas as pd
import streamlit as st
from data_processing.io_readers import read_population_labels
from data_processing.pharmacist_loader import load_all_pharmacist_data, get_pharmacists_for_zip

# Import health data parser
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
    from health_data_parser import format_health_stats_html
except ImportError:
    def format_health_stats_html(zip_code):
        return ""  # Fallback if health data not available

def render_top10_map(top10: pd.DataFrame, pharmacist_df=None):
    labels = read_population_labels('data/population_data.csv')
    top10 = top10.merge(labels, on="zip", how="left")
    
    # Create place string from city and state
    def make_place(row):
        city = str(row.get('city', '')).strip() if pd.notna(row.get('city')) else ''
        state = str(row.get('state', '')).strip() if pd.notna(row.get('state')) else ''
        if city and state:
            return f"{city}, {state}"
        elif city:
            return city
        elif state:
            return state
        else:
            return "(unknown)"
    
    top10["place"] = top10.apply(make_place, axis=1)

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
            
            # Get pharmacists for this ZIP
            pharmacist_html = ""
            if pharmacist_df is not None and not pharmacist_df.empty:
                pharmacists = get_pharmacists_for_zip(r['zip'], pharmacist_df)
                if pharmacists:
                    pharmacist_count = len(pharmacists)
                    
                    # Count awarded pharmacists
                    awarded_count = sum(1 for _, has_award, _, _ in pharmacists if has_award)
                    award_note = f" ({awarded_count} award winner{'s' if awarded_count != 1 else ''} ⭐)" if awarded_count > 0 else ""
                    
                    # Build scrollable HTML table with all pharmacists
                    table_rows = []
                    for idx, (name, has_award, phone, address) in enumerate(pharmacists, 1):
                        award_badge = '<span style="color: #FFD700; font-size: 12px;">⭐</span>' if has_award else ''
                        # Truncate address if too long for display
                        display_address = address[:35] + '...' if len(address) > 35 else address if address else '<span style="color: #999;">—</span>'
                        display_phone = phone if phone else '<span style="color: #999;">—</span>'
                        
                        table_rows.append(
                            f'<tr style="border-bottom: 1px solid #e8e8e8;">'
                            f'<td style="padding: 4px 4px; text-align: center; color: #666; font-size: 10px;">{idx}</td>'
                            f'<td style="padding: 4px 6px; font-size: 11px; font-weight: {("bold" if has_award else "normal")};">{name}</td>'
                            f'<td style="padding: 4px 4px; font-size: 9px; color: #555;">{display_phone}</td>'
                            f'<td style="padding: 4px 4px; font-size: 9px; color: #555;">{display_address}</td>'
                            f'<td style="padding: 4px 3px; text-align: center;">{award_badge}</td>'
                            f'</tr>'
                        )
                    
                    # Scrollable table with compact design
                    table_html = f'''
                        <div style="max-height: 200px; overflow-y: auto; overflow-x: hidden; margin-top: 6px; border: 1px solid #ddd; border-radius: 4px;">
                            <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                                <thead style="position: sticky; top: 0; background-color: #f8f8f8; z-index: 1;">
                                    <tr style="border-bottom: 2px solid #ccc;">
                                        <th style="padding: 5px 4px; text-align: center; width: 22px; font-size: 9px;">#</th>
                                        <th style="padding: 5px 6px; text-align: left; font-size: 9px;">Name</th>
                                        <th style="padding: 5px 4px; text-align: left; width: 85px; font-size: 9px;">Phone</th>
                                        <th style="padding: 5px 4px; text-align: left; width: 110px; font-size: 9px;">Location</th>
                                        <th style="padding: 5px 3px; text-align: center; width: 25px; font-size: 9px;">⭐</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {''.join(table_rows)}
                                </tbody>
                            </table>
                        </div>
                    '''
                    
                    pharmacist_html = f"<br><b style='font-size: 12px;'>👨‍⚕️ Pharmacists ({pharmacist_count} total{award_note}):</b>{table_html}"
                else:
                    pharmacist_html = f"<br><b style='font-size: 12px;'>👨‍⚕️ Pharmacists:</b> <i>None found</i>"
            else:
                pharmacist_html = "<br><b style='font-size: 12px;'>👨‍⚕️ Pharmacists:</b> <i>Data not loaded</i>"
            
            # Compact popup with enough width for location column
            popup_height = 400
            popup_width = 480
            
            # Get health statistics for this ZIP
            health_html = format_health_stats_html(r['zip'])
            
            popup = folium.Popup(
                folium.IFrame(
                    html=f"""
                        <b>ZIP:</b> {r['zip']}<br>
                        <b>Place:</b> {place}<br>
                        {drive_time_html}
                        <b>Final score:</b> {r.get('final_score', float('nan')):.3f}<br>
                        <b>Math score:</b> {r.get('score_math', float('nan')):.3f}<br>
                        <b>AI score:</b> {r.get('ai_score', float('nan')):.3f}<br>
                        <b>Pharmacies:</b> {int(r.get('n_pharmacies', r.get('pharmacies_count', 0)))}<br>
                        <b>Pop density:</b> {r['pop_density']:.1f}
                        {health_html}
                        {pharmacist_html}
                    """, width=popup_width, height=popup_height + 80
                ),
                max_width=popup_width + 20
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
