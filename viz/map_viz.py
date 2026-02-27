# pharmacy_deserts/viz/map_viz.py
import pandas as pd
import streamlit as st
from data.loaders import read_population_labels, get_pharmacists_for_zip, get_pharmacies_for_zip
from app.config import get_config
import re
import json
from pathlib import Path

# Import health data parser
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'raw_data'))
    from health_data_parser import format_health_stats_html
except ImportError:
    def format_health_stats_html(zip_code):
        return ""  # Fallback if health data not available


def _parse_city_state_from_csv(path, skiprows=10):
    """Extract zip/city/state from a population CSV file."""
    df = pd.read_csv(path, skiprows=skiprows)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    if "zip" not in lower:
        return pd.DataFrame(columns=["zip", "city", "state"])

    city_col = lower.get("city") or lower.get("place")
    state_col = lower.get("state") or lower.get("st")

    result = pd.DataFrame({
        "zip": df[lower["zip"]].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
        "city": df[city_col].astype(str) if city_col else "",
        "state": df[state_col].astype(str) if state_col else "",
    })
    return result.dropna(subset=["zip"]).drop_duplicates(subset=["zip"])


def _get_population_labels():
    """
    Get population labels (city/state) for ZIP codes.
    Tries legacy path first, then falls back to uploaded dataset files.
    """
    config = get_config()
    empty = pd.DataFrame(columns=["zip", "city", "state"])

    # --- Attempt 1: S3 or legacy local file ---
    try:
        if config.is_production and config.aws_s3_bucket:
            from data.s3_loaders import parse_s3_path, download_s3_file_to_memory
            import io
            s3_path = config.get_population_data_path()
            bucket, key = parse_s3_path(s3_path)
            data = download_s3_file_to_memory(bucket, key)
            result = _parse_city_state_from_csv(io.BytesIO(data))
            if not result.empty:
                return result
        else:
            local_path = Path('raw_data/population_data.csv')
            if local_path.exists():
                result = read_population_labels(str(local_path))
                if not result.empty:
                    return result
    except Exception as e:
        print(f"Warning: Legacy population labels failed: {e}")

    # --- Attempt 2: uploaded population file inside dataset version ---
    try:
        import json
        latest_path = Path('raw_data/datasets/pharmacy_data/LATEST.json')
        if latest_path.exists():
            ver = json.loads(latest_path.read_text()).get('latest_version')
            if ver:
                uploaded = Path(f'raw_data/datasets/pharmacy_data/versions/{ver}/files/population_data.csv')
                if uploaded.exists():
                    result = _parse_city_state_from_csv(uploaded)
                    if not result.empty:
                        return result
    except Exception as e:
        print(f"Warning: Uploaded population labels failed: {e}")

    return empty


def _normalize_col_key(col_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col_name or "").lower()).strip("_")


def _classify_revenue_column(col_name: str) -> str | None:
    """
    Classify a column as a revenue metric.

    Returns one of:
    - "with_insurance"
    - "without_insurance"
    - "generic"
    """
    normalized = _normalize_col_key(col_name)
    if not normalized:
        return None

    has_insurance_token = ("insurance" in normalized) or ("insurace" in normalized)

    if "revenue_with_insurance" in normalized:
        return "with_insurance"
    if "revenue_without_insurance" in normalized:
        return "without_insurance"
    if "revenue_potential" in normalized:
        # Backward-compatible alias for "with insurance" value.
        return "with_insurance"

    if "grand_total" in normalized and "without_cancer" in normalized:
        if has_insurance_token:
            return "with_insurance"
        return "without_insurance"

    if "revenue" in normalized and has_insurance_token:
        return "with_insurance"
    if "revenue" in normalized and "without" in normalized:
        return "without_insurance"
    if "revenue" in normalized:
        return "generic"
    return None


def _extract_numeric_from_row(row: pd.Series, column_name: str):
    raw = row.get(column_name)
    if isinstance(raw, pd.Series):
        raw = raw.iloc[0] if len(raw) else None
    value = pd.to_numeric(raw, errors="coerce")
    if pd.notna(value):
        return float(value)
    return None


def _pick_revenue_value(row: pd.Series, candidate_columns: list[str]):
    if not candidate_columns:
        return None

    ordered = sorted(
        candidate_columns,
        key=lambda c: (
            "revenue_potential" not in _normalize_col_key(c),
            "revenue_with_insurance" not in _normalize_col_key(c),
            len(str(c)),
        ),
    )
    for column_name in ordered:
        value = _extract_numeric_from_row(row, column_name)
        if value is not None:
            return value
    return None


def _extract_revenue_values(row: pd.Series) -> dict[str, float | None]:
    buckets: dict[str, list[str]] = {
        "with_insurance": [],
        "without_insurance": [],
        "generic": [],
    }
    for col in row.index:
        klass = _classify_revenue_column(str(col))
        if klass and klass in buckets:
            buckets[klass].append(str(col))

    with_insurance = _pick_revenue_value(row, buckets["with_insurance"])
    without_insurance = _pick_revenue_value(row, buckets["without_insurance"])
    generic = _pick_revenue_value(row, buckets["generic"])

    # Fallback when only one generic revenue value exists.
    if with_insurance is None and without_insurance is None and generic is not None:
        with_insurance = generic

    return {
        "with_insurance": with_insurance,
        "without_insurance": without_insurance,
    }


def _normalize_zip_value(value) -> str | None:
    """Normalize any ZIP-like value to a 5-digit ZIP string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    # Numeric-like ZIPs can arrive as 4-digit strings/ints (e.g., 1001 for 01001).
    numeric_like = re.fullmatch(r"\d+(?:\.0+)?", text)
    if numeric_like:
        base = text.split(".", 1)[0]
        if 1 <= len(base) <= 5:
            return base.zfill(5)

    match5 = re.search(r"(\d{5})", text)
    if match5:
        return match5.group(1).zfill(5)

    match_short = re.search(r"\b(\d{1,4})\b", text)
    if match_short:
        return match_short.group(1).zfill(5)
    return None


def _pick_zip_column(columns: list[str]) -> str | None:
    preferred = ["Short_ZIP", "ZIP", "zip", "Zip", "Zipcode", "zipcode", "ZCTA5", "zcta5"]
    for col in preferred:
        if col in columns:
            return col
    for col in columns:
        lowered = str(col).lower()
        if "zip" in lowered or "zcta" in lowered:
            return col
    return None


def _is_with_insurance_column(col_name: str) -> bool:
    normalized = _normalize_col_key(col_name)
    return (
        ("grand_total" in normalized and "without_cancer" in normalized and ("insurance" in normalized or "insurace" in normalized))
        or "revenue_with_insurance" in normalized
        or "revenue_potential" in normalized
    )


def _is_without_insurance_column(col_name: str) -> bool:
    normalized = _normalize_col_key(col_name)
    return (
        ("grand_total" in normalized and "without_cancer" in normalized and ("insurance" not in normalized and "insurace" not in normalized))
        or "revenue_without_insurance" in normalized
    )


def _preferred_revenue_column(columns: list[str], kind: str) -> str | None:
    if not columns:
        return None
    if kind == "with_insurance":
        ordered = sorted(
            columns,
            key=lambda c: (
                not _is_with_insurance_column(c),
                "revenue_with_insurance" not in _normalize_col_key(c),
                "revenue_potential" not in _normalize_col_key(c),
                len(str(c)),
            ),
        )
        return ordered[0]
    if kind == "without_insurance":
        ordered = sorted(
            columns,
            key=lambda c: (
                not _is_without_insurance_column(c),
                "revenue_without_insurance" not in _normalize_col_key(c),
                len(str(c)),
            ),
        )
        return ordered[0]
    return None


def _revenue_preset_candidate_paths() -> list[Path]:
    """
    Build candidate paths for hard-coded ZIP revenue presets.
    Priority is local repo file, then most recent uploaded custom health file.
    """
    candidates = [Path("raw_data/health_data_updated.xlsm")]
    latest_json = Path("raw_data/datasets/pharmacy_data/LATEST.json")
    try:
        if latest_json.exists():
            latest_version = json.loads(latest_json.read_text()).get("latest_version")
            if latest_version:
                files_dir = Path(f"raw_data/datasets/pharmacy_data/versions/{latest_version}/files")
                if files_dir.exists():
                    for pattern in ("custom_*health*.xlsm", "custom_*health*.xlsx", "custom_*health*.xls"):
                        for p in sorted(files_dir.glob(pattern)):
                            if p not in candidates:
                                candidates.append(p)
    except Exception:
        pass
    return candidates


@st.cache_data(show_spinner=False)
def _load_revenue_presets() -> pd.DataFrame:
    """
    Load hard-coded revenue values by ZIP (with/without insurance).
    """
    empty = pd.DataFrame(columns=["zip", "with_insurance", "without_insurance"])
    for path in _revenue_preset_candidate_paths():
        if not path.exists():
            continue
        try:
            if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
                excel_file = pd.ExcelFile(path)
                sheet_name = next(
                    (s for s in excel_file.sheet_names if "health" in str(s).lower()),
                    excel_file.sheet_names[0] if excel_file.sheet_names else None,
                )
                if sheet_name is None:
                    continue
                raw = pd.read_excel(excel_file, sheet_name=sheet_name)
            else:
                raw = pd.read_csv(path)
        except Exception:
            continue

        raw.columns = [str(c).strip() for c in raw.columns]
        if raw.empty:
            continue

        zip_col = _pick_zip_column(list(raw.columns))
        if not zip_col:
            continue

        with_ins_cols = [c for c in raw.columns if _classify_revenue_column(c) == "with_insurance"]
        without_ins_cols = [c for c in raw.columns if _classify_revenue_column(c) == "without_insurance"]

        with_col = _preferred_revenue_column(with_ins_cols, "with_insurance")
        without_col = _preferred_revenue_column(without_ins_cols, "without_insurance")

        if not with_col and not without_col:
            continue

        out = pd.DataFrame({"zip": raw[zip_col].map(_normalize_zip_value)})
        if with_col:
            out["with_insurance"] = pd.to_numeric(raw[with_col], errors="coerce")
        else:
            out["with_insurance"] = pd.NA
        if without_col:
            out["without_insurance"] = pd.to_numeric(raw[without_col], errors="coerce")
        else:
            out["without_insurance"] = pd.NA

        out = out.dropna(subset=["zip"])
        out = out.groupby("zip", as_index=False).agg(
            with_insurance=("with_insurance", "mean"),
            without_insurance=("without_insurance", "mean"),
        )
        out = out.dropna(subset=["with_insurance", "without_insurance"], how="all")
        if not out.empty:
            return out
    return empty


def render_top10_map(top10: pd.DataFrame, pharmacist_df=None, pharmacy_df=None):
    """
    Render an interactive map of top pharmacy desert ZIPs.
    Note: Returns HTML, so parent should handle display to avoid reruns on interaction.
    """
    # Check if city/state columns already exist in the dataframe
    has_city = 'city' in top10.columns and top10['city'].notna().any()
    has_state = 'state' in top10.columns and top10['state'].notna().any()
    
    # Only merge with labels file if city/state not already present
    if not (has_city and has_state):
        labels = _get_population_labels()
        if not labels.empty:
            # Merge but don't overwrite existing columns
            merge_cols = ['zip']
            if not has_city and 'city' in labels.columns:
                merge_cols.append('city')
            if not has_state and 'state' in labels.columns:
                merge_cols.append('state')
            if len(merge_cols) > 1:
                top10 = top10.merge(labels[merge_cols], on="zip", how="left")
    
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

    # Check for lat/lon columns (might be named differently)
    lat_col = None
    lon_col = None
    
    # Check all possible column name variations
    lat_names = ['lat', 'latitude', 'Lat', 'Latitude', 'LAT', 'LATITUDE']
    lon_names = ['lon', 'lng', 'long', 'longitude', 'Lon', 'Lng', 'Long', 'Longitude', 'LON', 'LNG', 'LONG', 'LONGITUDE']
    
    for col in top10.columns:
        if col in lat_names or col.lower() in ['lat', 'latitude']:
            lat_col = col
        elif col in lon_names or col.lower() in ['lon', 'lng', 'long', 'longitude']:
            lon_col = col
    
    has_latlon_cols = lat_col is not None and lon_col is not None
    
    # Standardize column names if found
    if has_latlon_cols:
        # Handle potential duplicate column names from merging
        if lat_col != 'lat':
            # Ensure we get a Series, not a DataFrame (handle duplicate column names)
            lat_series = top10[lat_col]
            if isinstance(lat_series, pd.DataFrame):
                # If multiple columns match, take the first one
                lat_series = lat_series.iloc[:, 0]
            # Remove any existing 'lat' column to avoid duplicates
            if 'lat' in top10.columns:
                top10 = top10.drop(columns=['lat'])
            top10['lat'] = lat_series
        if lon_col != 'lon':
            # Ensure we get a Series, not a DataFrame (handle duplicate column names)
            lon_series = top10[lon_col]
            if isinstance(lon_series, pd.DataFrame):
                # If multiple columns match, take the first one
                lon_series = lon_series.iloc[:, 0]
            # Remove any existing 'lon' column to avoid duplicates
            if 'lon' in top10.columns:
                top10 = top10.drop(columns=['lon'])
            top10['lon'] = lon_series
        
        # Ensure lat and lon are proper Series (not DataFrames)
        if 'lat' in top10.columns and isinstance(top10['lat'], pd.DataFrame):
            top10['lat'] = top10['lat'].iloc[:, 0]
        if 'lon' in top10.columns and isinstance(top10['lon'], pd.DataFrame):
            top10['lon'] = top10['lon'].iloc[:, 0]
    
    has_any_points = has_latlon_cols and top10[["lat","lon"]].notna().any().any()
    
    if not has_any_points:
        # Show what columns ARE available for debugging
        available_cols = list(top10.columns)
        st.info(f"📍 Map unavailable - no lat/lon data found. Available columns: {available_cols[:15]}{'...' if len(available_cols) > 15 else ''}")
        
        # Show a simple table of the top ZIPs instead
        display_cols = ['zip', 'place']
        for col in ['score', 'final_score', 'population', 'n_pharmacies', 'pharm_per_10k']:
            if col in top10.columns:
                display_cols.append(col)
        display_cols = [c for c in display_cols if c in top10.columns]
        
        if display_cols:
            st.dataframe(top10[display_cols], use_container_width=True)
        return

    try:
        import folium
        from streamlit_folium import st_folium
        pts = top10.dropna(subset=["lat","lon"]).copy()
        
        # Clean up duplicate columns before processing
        # When merging datasets, duplicate column names can cause issues
        # Remove duplicates, keeping only the first occurrence of each column name
        if pts.columns.duplicated().any():
            # Get unique column names, keeping first occurrence
            unique_cols = []
            seen = set()
            for col in pts.columns:
                if col not in seen:
                    unique_cols.append(col)
                    seen.add(col)
            pts = pts[unique_cols]
        
        # Ensure 'lat' and 'lon' exist and are Series (not DataFrame)
        if 'lat' in pts.columns:
            if isinstance(pts['lat'], pd.DataFrame):
                # If DataFrame, take first column
                pts['lat'] = pts['lat'].iloc[:, 0]
        
        if 'lon' in pts.columns:
            if isinstance(pts['lon'], pd.DataFrame):
                # If DataFrame, take first column
                pts['lon'] = pts['lon'].iloc[:, 0]
        
        # Ensure lat and lon are proper Series (not DataFrame) - handle duplicate columns from merging
        lat_series = pts["lat"]
        lon_series = pts["lon"]
        
        if isinstance(lat_series, pd.DataFrame):
            lat_series = lat_series.iloc[:, 0]
        if isinstance(lon_series, pd.DataFrame):
            lon_series = lon_series.iloc[:, 0]
        
        # Convert to numeric and get mean
        lat_mean = float(pd.to_numeric(lat_series, errors='coerce').mean())
        lon_mean = float(pd.to_numeric(lon_series, errors='coerce').mean())
        
        fmap = folium.Map(location=[lat_mean, lon_mean], zoom_start=4, control_scale=True)
        
        # Get bounds safely (ensure we have Series, not DataFrame)
        bounds_df = pts[["lat","lon"]].copy()
        # If lat or lon are DataFrames, extract first column
        if isinstance(bounds_df["lat"], pd.DataFrame):
            bounds_df["lat"] = bounds_df["lat"].iloc[:, 0]
        if isinstance(bounds_df["lon"], pd.DataFrame):
            bounds_df["lon"] = bounds_df["lon"].iloc[:, 0]
        
        bounds = bounds_df[["lat","lon"]].values.tolist()
        if bounds: fmap.fit_bounds(bounds, padding=(20, 20))

        revenue_presets = _load_revenue_presets()
        revenue_lookup = (
            revenue_presets.set_index("zip")[["with_insurance", "without_insurance"]].to_dict("index")
            if not revenue_presets.empty
            else {}
        )

        for _, r in pts.iterrows():
            # Safely extract lat/lon values (handle duplicate columns from merging)
            # Use .get() with default, then check if it's a Series
            lat_val = r.get("lat")
            lon_val = r.get("lon")
            
            # Handle case where duplicate columns cause Series return
            if isinstance(lat_val, pd.Series):
                lat_val = lat_val.iloc[0] if len(lat_val) > 0 else None
            elif lat_val is None:
                continue
            
            if isinstance(lon_val, pd.Series):
                lon_val = lon_val.iloc[0] if len(lon_val) > 0 else None
            elif lon_val is None:
                continue
            
            # Convert to float safely
            try:
                lat = float(pd.to_numeric(lat_val, errors='coerce'))
                lon = float(pd.to_numeric(lon_val, errors='coerce'))
                if pd.isna(lat) or pd.isna(lon):
                    continue  # Skip if conversion resulted in NaN
            except (ValueError, TypeError):
                continue  # Skip this row if lat/lon can't be converted
            place = (f'{r.get("city","")}, {r.get("state","")}'.strip(", ") or "(unknown)")
            drive_time_html = ""
            if ('zip_drive_time' in r.index) and pd.notna(r.get('zip_drive_time')):
                drive_time_html = f"<b>🚗 Drive Time:</b> {r['zip_drive_time']:.1f} min<br>"
            revenue_values = _extract_revenue_values(r)
            # Hard-coded ZIP preset values take precedence when available.
            zip_for_lookup = _normalize_zip_value(r.get("zip"))
            if zip_for_lookup and zip_for_lookup in revenue_lookup:
                preset_values = revenue_lookup.get(zip_for_lookup, {})
                preset_with = pd.to_numeric(preset_values.get("with_insurance"), errors="coerce")
                preset_without = pd.to_numeric(preset_values.get("without_insurance"), errors="coerce")
                if pd.notna(preset_with):
                    revenue_values["with_insurance"] = float(preset_with)
                if pd.notna(preset_without):
                    revenue_values["without_insurance"] = float(preset_without)
            revenue_html_lines = []
            if revenue_values["with_insurance"] is not None:
                revenue_html_lines.append(
                    f"<b>💰 Revenue (With Insurance):</b> ${revenue_values['with_insurance']:,.0f}<br>"
                )
            if revenue_values["without_insurance"] is not None:
                revenue_html_lines.append(
                    f"<b>💵 Revenue (Without Insurance):</b> ${revenue_values['without_insurance']:,.0f}<br>"
                )
            revenue_html = "".join(revenue_html_lines)
            
            # Get pharmacists for this ZIP
            pharmacist_html = ""
            if pharmacist_df is not None and not pharmacist_df.empty:
                pharmacists = get_pharmacists_for_zip(r['zip'], pharmacist_df)
                if pharmacists:
                    pharmacist_count = len(pharmacists)

                    awarded_count = sum(1 for _, has_award, _, _ in pharmacists if has_award)
                    award_note = (
                        f" ({awarded_count} award winner{'s' if awarded_count != 1 else ''} ⭐)"
                        if awarded_count > 0
                        else ""
                    )

                    table_rows = []
                    for idx, (name, has_award, phone, address) in enumerate(pharmacists, 1):
                        award_badge = '<span style="color: #FFD700; font-size: 12px;">⭐</span>' if has_award else ''
                        display_address = (
                            address[:35] + '...' if len(address) > 35
                            else address if address
                            else '<span style="color: #999;">—</span>'
                        )
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

                    table_html = f'''
                        <div style="max-height: 180px; overflow-y: auto; overflow-x: hidden; margin-top: 6px; border: 1px solid #ddd; border-radius: 4px;">
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

                    pharmacist_html = (
                        f"<br><b style='font-size: 12px;'>👨‍⚕️ Pharmacists "
                        f"({pharmacist_count} total{award_note}):</b>{table_html}"
                    )
                else:
                    pharmacist_html = "<br><b style='font-size: 12px;'>👨‍⚕️ Pharmacists:</b> <i>None found</i>"
            else:
                pharmacist_html = "<br><b style='font-size: 12px;'>👨‍⚕️ Pharmacists:</b> <i>Data not loaded</i>"

            # Get pharmacies for this ZIP
            pharmacy_html = ""
            if pharmacy_df is not None and not pharmacy_df.empty:
                pharmacies = get_pharmacies_for_zip(r["zip"], pharmacy_df)
                if pharmacies:
                    pharmacy_rows = []
                    for idx, (name, chain, phone, address) in enumerate(pharmacies, 1):
                        display_phone = phone if phone else '<span style="color: #999;">—</span>'
                        display_address = (
                            address[:35] + "..." if len(address) > 35
                            else address if address
                            else '<span style="color: #999;">—</span>'
                        )
                        chain_badge = chain if chain and chain != "Independent" else "Independent"
                        pharmacy_rows.append(
                            f'<tr style="border-bottom: 1px solid #e8e8e8;">'
                            f'<td style="padding: 4px 4px; text-align: center; color: #666; font-size: 10px;">{idx}</td>'
                            f'<td style="padding: 4px 6px; font-size: 11px;">{name}</td>'
                            f'<td style="padding: 4px 4px; font-size: 9px; color: #555;">{chain_badge}</td>'
                            f'<td style="padding: 4px 4px; font-size: 9px; color: #555;">{display_phone}</td>'
                            f'<td style="padding: 4px 4px; font-size: 9px; color: #555;">{display_address}</td>'
                            f"</tr>"
                        )

                    pharmacy_table_html = f'''
                        <div style="max-height: 180px; overflow-y: auto; overflow-x: hidden; margin-top: 6px; border: 1px solid #ddd; border-radius: 4px;">
                            <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                                <thead style="position: sticky; top: 0; background-color: #f8f8f8; z-index: 1;">
                                    <tr style="border-bottom: 2px solid #ccc;">
                                        <th style="padding: 5px 4px; text-align: center; width: 22px; font-size: 9px;">#</th>
                                        <th style="padding: 5px 6px; text-align: left; font-size: 9px;">Pharmacy</th>
                                        <th style="padding: 5px 4px; text-align: left; width: 70px; font-size: 9px;">Chain</th>
                                        <th style="padding: 5px 4px; text-align: left; width: 85px; font-size: 9px;">Phone</th>
                                        <th style="padding: 5px 4px; text-align: left; width: 110px; font-size: 9px;">Location</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {''.join(pharmacy_rows)}
                                </tbody>
                            </table>
                        </div>
                    '''
                    pharmacy_html = (
                        f"<br><b style='font-size: 12px;'>🏥 Pharmacy Locations "
                        f"({len(pharmacies)} total):</b>{pharmacy_table_html}"
                    )
                else:
                    pharmacy_html = "<br><b style='font-size: 12px;'>🏥 Pharmacy Locations:</b> <i>None found</i>"
            else:
                pharmacy_html = "<br><b style='font-size: 12px;'>🏥 Pharmacy Locations:</b> <i>Data not loaded</i>"

            # Compact popup with enough width for location column
            popup_height = 620
            popup_width = 520
            
            # Get health statistics for this ZIP
            health_html = format_health_stats_html(r['zip'])
            
            popup = folium.Popup(
                folium.IFrame(
                    html=f"""
                        <b>ZIP:</b> {r['zip']}<br>
                        <b>Place:</b> {place}<br>
                        {drive_time_html}
                        {revenue_html}
                        <b>Final score:</b> {r.get('final_score', float('nan')):.3f}<br>
                        <b>Math score:</b> {r.get('score_math', float('nan')):.3f}<br>
                        <b>AI score:</b> {r.get('ai_score', float('nan')):.3f}<br>
                        <b>Pharmacies:</b> {int(r.get('n_pharmacies', r.get('pharmacies_count', 0)))}<br>
                        <b>Pop density:</b> {r.get('pop_density', r.get('density', 0)):.1f}
                        {health_html}
                        {pharmacist_html}
                        {pharmacy_html}
                    """, width=popup_width, height=popup_height
                ),
                max_width=popup_width + 20
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(5, min(20, 5 + 15*float(r.get("final_score", 0)))),
                color=None, fill=True, fill_opacity=0.7, popup=popup
            ).add_to(fmap)

        # Use key to prevent reruns on map interaction
        st_folium(fmap, width=None, key="pharmacy_map", returned_objects=[])
    except ModuleNotFoundError:
        st.info("For labeled markers, install: `pip install folium streamlit-folium`. Showing basic map instead.")
        # Create a clean DataFrame with just lat/lon for st.map()
        map_data = top10.dropna(subset=["lat","lon"]).copy()
        # Ensure we have clean lat/lon columns (handle duplicate column names)
        if 'lat' in map_data.columns and 'lon' in map_data.columns:
            # Get lat/lon as Series, handling potential duplicates
            lat_series = map_data['lat']
            lon_series = map_data['lon']
            if isinstance(lat_series, pd.DataFrame):
                lat_series = lat_series.iloc[:, 0]
            if isinstance(lon_series, pd.DataFrame):
                lon_series = lon_series.iloc[:, 0]
            # Create clean DataFrame with unique column names
            clean_map_data = pd.DataFrame({
                'lat': lat_series,
                'lon': lon_series
            })
            st.map(clean_map_data, zoom=4, use_container_width=True)
        keep = [c for c in ["zip","place","final_score","score_math","ai_score","n_pharmacies","pop_density"] if c in top10.columns]
        st.dataframe(top10[keep])
