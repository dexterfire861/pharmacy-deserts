#!/usr/bin/env python3
"""
Pharmacy Desert Finder (ZIP-level)
---------------------------------
Quick, hackathon-ready pipeline to score U.S. ZIP areas (ZCTA) for unmet pharmacy access
using extreme heat, air quality, sickness levels, population density, and income — plus
proximity to existing pharmacies.

Inputs (flexible):
- A GeoJSON of ZCTA geometries (recommended): properties must include a ZCTA/ZIP key like `ZCTA5CE10`.
- A CSV of pharmacy point locations with `lat,lon` columns.
- One or more indicator CSVs (any schema) that each contain a ZIP column and a numeric value to merge.

Output:
- `desert_candidates.csv` — ranked table with composite score.
- `desert_candidates.geojson` — same rows with geometry.
- `desert_map.html` — quick interactive Folium map.

Usage examples
--------------
# 1) Minimal (use defaults and a few indicators)
python pharmacy_desert_finder.py \
  --zcta-geo data/zcta.geojson --zip-prop ZCTA5CE10 \
  --pharmacies data/pharmacies.csv \
  --join data/heat.csv,zip,heat_index,heat \
  --join data/aqi.csv,zip,aqi,aqi \
  --join data/sickness.csv,zip,rate_per_1k,sickness \
  --join data/pop_density.csv,zip,pop_per_km2,pop_density \
  --join data/income.csv,zip,median_income,median_income

# 2) Customize weights
python pharmacy_desert_finder.py ... \
  --weight heat=0.25 --weight aqi=0.2 --weight sickness=0.25 \
  --weight distance_km=0.2 --weight income_inv=0.1

# 3) Filter low-viability areas (e.g., require pop density ≥ 200/km2)
python pharmacy_desert_finder.py ... --min-pop-density 200

Notes on weights
----------------
- Higher score = worse conditions / higher need. We invert income to reflect that lower income → higher need.
- If some indicators are missing, weights are re-normalized over the provided ones.

"""
from __future__ import annotations
import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception as e:  # pragma: no cover (geopandas may not be installed yet)
    gpd = None

from sklearn.neighbors import KDTree
import folium

# -----------------------------
# Utilities
# -----------------------------

def read_pharmacies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to find latitude/longitude column names robustly
    colmap = {c.lower(): c for c in df.columns}
    lat_col = colmap.get('lat') or colmap.get('latitude')
    lon_col = colmap.get('lon') or colmap.get('lng') or colmap.get('longitude')
    if not lat_col or not lon_col:
        raise ValueError("Pharmacy CSV must include 'lat' and 'lon' (or 'latitude'/'longitude').")
    df = df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
    df = df.dropna(subset=['lat', 'lon'])
    return df


def parse_join_arg(arg: str) -> Tuple[str, str, str, str]:
    """Parse a --join value formatted as "path,zip_col,val_col,new_name"""
    parts = [p.strip() for p in arg.split(',')]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--join must be 'path,zip_col,val_col,new_name'")
    return parts[0], parts[1], parts[2], parts[3]


def safe_zip_str(x) -> str:
    """Normalize ZIP/ZCTA to 5-digit string (keeps leading zeros)."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    # Remove any extended ZIP+4
    if '-' in s:
        s = s.split('-')[0]
    # Remove decimals like 12345.0
    if s.endswith('.0'):
        s = s[:-2]
    # Keep only digits, pad to length 5 if plausible
    s = ''.join(ch for ch in s if ch.isdigit())
    if len(s) == 0:
        return None
    if len(s) < 5:
        s = s.zfill(5)
    if len(s) > 5:
        s = s[:5]
    return s


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))


def nearest_dist_km(query_pts: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
    """Fast nearest-neighbor distances in kilometers using KDTree (haversine)."""
    if len(ref_pts) == 0:
        return np.full(len(query_pts), np.nan)
    # KDTree expects radians when using haversine metric
    tree = KDTree(np.radians(ref_pts), metric='haversine')
    dist_rad, _ = tree.query(np.radians(query_pts), k=1)
    return dist_rad.flatten() * 6371.0088


def robust_minmax(series: pd.Series, clip_pct=(5, 95)) -> pd.Series:
    """Min-max scale after clipping to percentiles to reduce outlier effects."""
    s = series.copy()
    low, high = np.nanpercentile(s.dropna(), clip_pct)
    s = s.clip(lower=low, upper=high)
    denom = (s.max() - s.min())
    if denom == 0 or np.isnan(denom):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / denom


def reweight(weights: Dict[str, float], available_cols: List[str]) -> Dict[str, float]:
    usable = {k: v for k, v in weights.items() if k in available_cols}
    total = sum(usable.values())
    if total <= 0:
        # fallback: equal weights
        eq = 1.0 / max(1, len(available_cols))
        return {k: eq for k in available_cols}
    return {k: v / total for k, v in usable.items()}

# -----------------------------
# Main pipeline
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Score ZIPs (ZCTAs) for potential pharmacy deserts.")
    p.add_argument('--zcta-geo', required=True, help='Path to ZCTA GeoJSON (or any ZIP polygon file).')
    p.add_argument('--zip-prop', default='ZCTA5CE10', help='Property/column in the GeoJSON with 5-digit ZIP/ZCTA (default: ZCTA5CE10).')
    p.add_argument('--pharmacies', required=True, help='CSV with pharmacy points (lat,lon required).')
    p.add_argument('--join', action='append', default=[], type=parse_join_arg,
                   help="Repeatable. Format: path,zip_col,val_col,new_name")
    p.add_argument('--min-pop-density', type=float, default=None,
                   help='Optional: filter out rows below this population density (per km^2).')
    p.add_argument('--top-n', type=int, default=200, help='How many top ZIPs to keep in outputs.')
    p.add_argument('--out-prefix', default='desert_candidates', help='Output filename prefix.')
    p.add_argument('--weight', action='append', default=[],
                   help='Repeatable. e.g., heat=0.25, aqi=0.2, sickness=0.25, distance_km=0.2, income_inv=0.1')
    return p


def parse_weights(weight_args: List[str]) -> Dict[str, float]:
    if not weight_args:
        return {}
    w = {}
    for arg in weight_args:
        if '=' not in arg:
            raise argparse.ArgumentTypeError("--weight must be key=value, e.g. heat=0.25")
        k, v = arg.split('=', 1)
        w[k.strip()] = float(v)
    return w


def main():
    args = build_parser().parse_args()

    if gpd is None:
        raise RuntimeError("geopandas is required. Install with: pip install geopandas shapely fiona rtree scikit-learn folium")

    # 1) Load ZCTA polygons and derive centroids
    gdf = gpd.read_file(args.zcta_geo)
    if args.zip_prop not in gdf.columns:
        raise ValueError(f"ZIP property '{args.zip_prop}' not in ZCTA file columns: {list(gdf.columns)[:12]}...")
    gdf['zip'] = gdf[args.zip_prop].apply(safe_zip_str)
    gdf = gdf[~gdf['zip'].isna()].copy()
    # Ensure projected CRS for area/density (use EPSG:5070 Albers US) then back to EPSG:4326
    gdf = gdf.set_crs(gdf.crs or 'EPSG:4326')
    try:
        gdf_area = gdf.to_crs('EPSG:5070')
        gdf['area_km2'] = gdf_area.geometry.area / 1e6
    except Exception:
        gdf['area_km2'] = np.nan
    # Centroids in WGS84 for distance calc
    gdf_cent = gdf.to_crs('EPSG:4326').copy()
    gdf_cent['centroid'] = gdf_cent.geometry.centroid
    gdf_cent['cent_lat'] = gdf_cent['centroid'].y
    gdf_cent['cent_lon'] = gdf_cent['centroid'].x

    base = gdf_cent[['zip', 'cent_lat', 'cent_lon', 'area_km2']].copy()

    # 2) Merge indicator CSVs
    for path, zip_col, val_col, new_name in args.join:
        df = pd.read_csv(path)
        if zip_col not in df.columns or val_col not in df.columns:
            raise ValueError(f"Join file {path} must have columns '{zip_col}' and '{val_col}'.")
        df = df[[zip_col, val_col]].copy()
        df['zip'] = df[zip_col].apply(safe_zip_str)
        df = df.dropna(subset=['zip'])
        df = df.rename(columns={val_col: new_name})[['zip', new_name]]
        base = base.merge(df, on='zip', how='left')

    # 3) Pharmacies distance (nearest)
    pharm = read_pharmacies(args.pharmacies)
    ref_pts = pharm[['lat', 'lon']].to_numpy()
    query = base[['cent_lat', 'cent_lon']].to_numpy()
    base['distance_km'] = nearest_dist_km(query, ref_pts)

    # 4) Derived/pop density if not supplied
    if 'pop_density' not in base.columns and 'population' in base.columns and base['area_km2'].notna().any():
        base['pop_density'] = base['population'] / base['area_km2']

    if args.min_pop_density is not None and 'pop_density' in base.columns:
        base = base[(base['pop_density'] >= args.min_pop_density) | (base['pop_density'].isna())].copy()

    # 5) Build normalized features
    feature_specs = []
    # convention: higher = worse need
    if 'heat' in base.columns:
        base['heat_norm'] = robust_minmax(base['heat'])
        feature_specs.append('heat_norm')
    if 'aqi' in base.columns:
        base['aqi_norm'] = robust_minmax(base['aqi'])
        feature_specs.append('aqi_norm')
    if 'sickness' in base.columns:
        base['sickness_norm'] = robust_minmax(base['sickness'])
        feature_specs.append('sickness_norm')
    if 'distance_km' in base.columns:
        base['distance_norm'] = robust_minmax(base['distance_km'])
        feature_specs.append('distance_norm')
    # income: lower income ⇒ higher need ⇒ invert after normalize
    if 'median_income' in base.columns:
        inc_norm = robust_minmax(base['median_income'])
        base['income_inv'] = 1.0 - inc_norm
        feature_specs.append('income_inv')

    if not feature_specs:
        raise RuntimeError("No features available to score. Ensure your --join files provided at least one indicator.")

    # 6) Weights
    default_w = {
        'heat_norm': 0.25,
        'aqi_norm': 0.20,
        'sickness_norm': 0.25,
        'distance_norm': 0.20,
        'income_inv': 0.10,
    }
    # Allow CLI override using raw keys; map common aliases
    alias_map = {
        'heat': 'heat_norm',
        'aqi': 'aqi_norm',
        'sickness': 'sickness_norm',
        'distance_km': 'distance_norm',
        'income': 'income_inv',
        'income_inv': 'income_inv',
    }
    user_w_raw = parse_weights(args.weight)
    user_w = {}
    for k, v in user_w_raw.items():
        user_w[alias_map.get(k, k)] = v

    all_cols = feature_specs
    weights = reweight({**default_w, **user_w}, all_cols)

    # 7) Composite score
    score = np.zeros(len(base))
    for k, w in weights.items():
        if k not in base.columns:
            continue
        score += w * base[k].fillna(base[k].median())
    base['score'] = score

    # 8) Attach geometry for export
    out_gdf = gdf.merge(base, on='zip', how='right')

    # 9) Rank + export
    out_gdf = out_gdf.sort_values('score', ascending=False)
    top_n = max(1, args.top_n)
    out_top = out_gdf.head(top_n).copy()

    csv_path = f"{args.out_prefix}.csv"
    geojson_path = f"{args.out_prefix}.geojson"
    map_path = f"{args.out_prefix}.html"

    cols_export = ['zip', 'score', 'cent_lat', 'cent_lon', 'distance_km', 'area_km2'] + \
        [c for c in ['heat','aqi','sickness','pop_density','median_income','heat_norm','aqi_norm','sickness_norm','distance_norm','income_inv'] if c in out_top.columns]

    out_top[cols_export].to_csv(csv_path, index=False)
    out_top.to_file(geojson_path, driver='GeoJSON')

    # 10) Quick Folium map
    try:
        # Center map roughly at continental US
        m = folium.Map(location=[39.5, -98.35], zoom_start=4, control_scale=True)
        folium.Choropleth(
            geo_data=geojson_path,
            name='Pharmacy Desert Score',
            data=out_top[['zip', 'score']],
            columns=['zip', 'score'],
            key_on='feature.properties.zip',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Composite Score (higher = higher need)'
        ).add_to(m)

        # Pharmacy markers (sample up to 2000 to keep map light)
        sample = len(pharm) > 2000
        ph_vis = pharm.sample(2000, random_state=42) if sample else pharm
        for _, r in ph_vis.iterrows():
            folium.CircleMarker([r['lat'], r['lon']], radius=2, opacity=0.6).add_to(m)

        folium.LayerControl().add_to(m)
        m.save(map_path)
    except Exception as e:
        print(f"[WARN] Could not build Folium map: {e}")

    # 11) Console summary
    print("Saved:")
    print(f"- {csv_path}")
    print(f"- {geojson_path}")
    if os.path.exists(map_path):
        print(f"- {map_path}")


if __name__ == '__main__':
    main()
