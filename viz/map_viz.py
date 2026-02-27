# pharmacy_deserts/viz/map_viz.py
import pandas as pd
import streamlit as st
from data.loaders import read_population_labels
from app.config import get_config
import re
import json
import html
from pathlib import Path


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


def _get_row_value(row: pd.Series, column_name: str):
    raw = row.get(column_name)
    if isinstance(raw, pd.Series):
        return raw.iloc[0] if len(raw) else None
    return raw


def _pick_text_value(row: pd.Series, candidate_columns: list[str]) -> str | None:
    for col in candidate_columns:
        if col not in row.index:
            continue
        raw = _get_row_value(row, col)
        if raw is None or pd.isna(raw):
            continue
        text = str(raw).strip()
        if text and text.lower() != "nan":
            return text
    return None


def _pick_numeric_value(row: pd.Series, candidate_columns: list[str]) -> float | None:
    for col in candidate_columns:
        if col in row.index:
            value = _extract_numeric_from_row(row, col)
            if value is not None:
                return value
    return None


def _format_decimal(value: float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _format_int(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,}"


def _format_income(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.0f}"


def _as_yes_no(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return "Yes"
        if lowered in {"0", "false", "no", "n"}:
            return "No"
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric):
        return "Yes" if float(numeric) >= 0.5 else "No"
    return "N/A"


def _detect_popup_mode(row: pd.Series) -> str:
    if (
        "store_viability" in row.index
        or "action" in row.index
        or "archetype_name" in row.index
    ):
        return "optimizer"
    if (
        "profit_score" in row.index
        or "revenue_potential" in row.index
        or "capture_rate" in row.index
    ):
        return "profit"
    return "default"


def _build_compact_popup_html(row: pd.Series, place: str) -> str:
    mode = _detect_popup_mode(row)
    zip_code = _normalize_zip_value(_get_row_value(row, "zip")) or str(_get_row_value(row, "zip") or "N/A")
    state = _pick_text_value(row, ["state", "st"])
    if not state and "," in place:
        state = place.split(",")[-1].strip()
    state = state or "N/A"

    rows: list[tuple[str, str]] = [("ZCTA / State", f"{zip_code} / {state}")]

    if mode == "profit":
        profit_score = _pick_numeric_value(row, ["profit_score", "final_score", "score"])
        tier = _pick_text_value(row, ["tier"])
        score_text = _format_decimal(profit_score)
        if tier:
            score_text = f"{score_text} ({tier})"
        rows.extend(
            [
                ("Profit Score / Tier", score_text),
                (
                    "Population / Pop density",
                    f"{_format_int(_pick_numeric_value(row, ['population']))} / {_format_decimal(_pick_numeric_value(row, ['pop_density', 'density']), 1)}",
                ),
                ("Median income", _format_income(_pick_numeric_value(row, ["median_income", "income"]))),
                ("Pharmacies in ZCTA", _format_int(_pick_numeric_value(row, ["n_pharmacies", "pharmacies_count", "total_pharmacies"]))),
                ("Revenue potential", _format_decimal(_pick_numeric_value(row, ["revenue_potential"]))),
                ("Cost pressure", _format_decimal(_pick_numeric_value(row, ["cost_pressure"]))),
                ("Capture rate", _format_decimal(_pick_numeric_value(row, ["capture_rate"]))),
                ("Pharmacy desert", _as_yes_no(_get_row_value(row, "is_pharmacy_desert") if "is_pharmacy_desert" in row.index else _get_row_value(row, "desert_flag"))),
            ]
        )
        title = "Profit Model Summary"
    elif mode == "optimizer":
        walgreens_count = _pick_numeric_value(row, ["walgreens_count"])
        total_pharmacies = _pick_numeric_value(row, ["total_pharmacies", "n_pharmacies", "pharmacies_count"])
        non_walgreens = _pick_numeric_value(row, ["non_walgreens_count"])
        if non_walgreens is None and walgreens_count is not None and total_pharmacies is not None:
            non_walgreens = max(0.0, float(total_pharmacies) - float(walgreens_count))

        rows.extend(
            [
                ("Viability Score", _format_decimal(_pick_numeric_value(row, ["store_viability", "final_score", "score"]))),
                ("Action", _pick_text_value(row, ["action"]) or "N/A"),
                ("Archetype", _pick_text_value(row, ["archetype_name", "archetype"]) or "N/A"),
                ("Population", _format_int(_pick_numeric_value(row, ["population"]))),
                ("Median income", _format_income(_pick_numeric_value(row, ["median_income", "income"]))),
                ("Walgreens count", _format_int(walgreens_count)),
                ("Non-Walgreens count", _format_int(non_walgreens)),
                ("Revenue", _format_decimal(_pick_numeric_value(row, ["store_revenue", "revenue"]))),
                ("Cost", _format_decimal(_pick_numeric_value(row, ["store_cost", "cost"]))),
                ("Position", _format_decimal(_pick_numeric_value(row, ["store_position", "position"]))),
            ]
        )
        title = "Optimization Model Summary"
    else:
        rows.extend(
            [
                ("Final Score", _format_decimal(_pick_numeric_value(row, ["final_score", "score"]))),
                ("Population", _format_int(_pick_numeric_value(row, ["population"]))),
                ("Median income", _format_income(_pick_numeric_value(row, ["median_income", "income"]))),
                ("Pharmacies in ZCTA", _format_int(_pick_numeric_value(row, ["n_pharmacies", "pharmacies_count", "total_pharmacies"]))),
            ]
        )
        title = "ZIP Summary"

    rows_html = "".join(
        [
            (
                "<div style='margin: 0 0 4px 0; white-space: nowrap;'>"
                f"<span style='font-weight: 600;'>{html.escape(label)}:</span> "
                f"{html.escape(value)}"
                "</div>"
            )
            for label, value in rows
        ]
    )

    return (
        "<div style='"
        "font-family: Helvetica, Arial, sans-serif;"
        "font-size: 13px;"
        "line-height: 1.35;"
        "color: #111111;"
        "white-space: nowrap;"
        "min-width: 360px;"
        "max-width: 520px;"
        "'>"
        f"<div style='font-size: 14px; font-weight: 600; margin: 0 0 7px 0; white-space: nowrap;'>{html.escape(title)}</div>"
        f"{rows_html}"
        "</div>"
    )


def render_top10_map(top10: pd.DataFrame, pharmacist_df=None, pharmacy_df=None, map_key: str = "pharmacy_map"):
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

        fmap.get_root().html.add_child(
            folium.Element(
                """
                <style>
                .leaflet-popup-content-wrapper {
                    border-radius: 10px !important;
                }
                .leaflet-popup-content {
                    margin: 10px 12px !important;
                    width: auto !important;
                }
                </style>
                """
            )
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
            popup = folium.Popup(
                _build_compact_popup_html(r, place),
                min_width=360,
                max_width=560,
            )
            score_val = float(r.get("final_score", 0) or 0)
            # Keep markers comfortably clickable even in low-score/worst views.
            marker_radius = max(11, min(20, 11 + 9 * score_val))
            folium.CircleMarker(
                location=[lat, lon],
                radius=marker_radius,
                color=None, fill=True, fill_opacity=0.7, popup=popup
            ).add_to(fmap)

        # Use key to prevent reruns on map interaction
        st_folium(fmap, width=None, key=map_key, returned_objects=[])
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
