"""
Upload Data — Streamlit Page

Simple upload page for known data file types.
Runs the existing data pipeline (readers → preprocess → merge)
and saves a unified dataset CSV.
"""
import streamlit as st
import pandas as pd
import io
import sys
import logging
import tempfile
import traceback
import re
from pathlib import Path

logger = logging.getLogger(__name__)

parent_dir = Path(__file__).parent.parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from app.config import get_config
from app.auth import is_authenticated, login_form
from data.loaders import (
    read_financial_data, read_health_data,
    read_population_data, read_hhi_excel, read_education_data_acs,
    read_hud_zip_county_crosswalk, read_county_desert_csv,
    downscale_county_to_zip,
)
from data.features import preprocess
from storage.datasets import (
    get_storage,
    generate_version_id,
    generate_source_id,
    build_dataset_config,
)

st.set_page_config(page_title="Upload Data", page_icon="📤", layout="wide")

config = get_config()
if config.require_auth and not is_authenticated():
    login_form()
    st.stop()

DATASET_ID = "pharmacy_data"

# ── Known data-file slots ────────────────────────────────────────────────────
# (key, label, accepted_extensions, required, help_text)
SLOTS = [
    ("financial",     "Financial / Income Data *",   ["csv"],                 True,
     "Census CSV — must have NAME and S1901_C01_012E columns"),
    ("health",        "Health Burden Data *",         ["csv"],                 True,
     "PLACES CSV — must have ZCTA5 and GHLTH_CrudePrev columns"),
    ("pharmacy",      "Pharmacy Locations *",         ["csv", "xlsx", "xlsm"], True,
     "File with a ZIP column — one row per pharmacy location"),
    ("population",    "Population / Density Data *",  ["csv"],                 True,
     "CSV with Zip, Population, Density, Lat, Long (10-row header skip)"),
    ("hhi",           "Heat-Health Index (HHI)",      ["xlsx", "xls"],         False,
     "Excel with ZCTA, HHB_SCORE columns"),
    ("hud_crosswalk", "HUD ZIP↔County Crosswalk",    ["xlsx", "xls", "csv"],  False,
     "HUD file with ZIP, COUNTY, TOT_RATIO columns"),
    ("county_desert", "County Desert Data",           ["csv"],                 False,
     "CSV with county FIPS and desert flag/score column"),
]

SCORING_MAPPING_TARGETS = [
    {
        "component": "pharmacy_count",
        "label": "Pharmacy count input",
        "default_column": "n_pharmacies",
        "help": "Used by the Pharmacy Scarcity slider.",
    },
    {
        "component": "population",
        "label": "Population input",
        "default_column": "population",
        "help": "Used for population context and area filters.",
    },
    {
        "component": "income",
        "label": "Income input",
        "default_column": "median_income",
        "help": "Used by the Income (inverted) slider.",
    },
    {
        "component": "health_burden",
        "label": "Health burden input",
        "default_column": "health_burden",
        "help": "Used by the Health Burden slider.",
    },
    {
        "component": "pop_density",
        "label": "Population density input",
        "default_column": "pop_density",
        "help": "Used by the Population Density slider.",
    },
    {
        "component": "education_low",
        "label": "Low education input",
        "default_column": "edu_hs_or_lower_pct",
        "help": "Optional slider input for educational vulnerability.",
    },
    {
        "component": "drive_time",
        "label": "Drive-time input",
        "default_column": "zip_drive_time",
        "help": "Optional slider input for pharmacy drive-time burden.",
    },
    {
        "component": "heat_vulnerability",
        "label": "Heat vulnerability input",
        "default_column": "heat_hhb",
        "help": "Optional slider input for heat-health burden.",
    },
    {
        "component": "latitude",
        "label": "Latitude input",
        "default_column": "lat",
        "help": "Used for map display.",
    },
    {
        "component": "longitude",
        "label": "Longitude input",
        "default_column": "lon",
        "help": "Used for map display.",
    },
]

CORE_DEFAULT_COLUMNS = [
    "zip",
    "n_pharmacies",
    "median_income",
    "health_burden",
    "population",
    "pop_density",
    "lat",
    "lon",
]

DEFAULT_CORE_SKIP_ROWS = {
    "financial": 0,
    "health": 0,
    "pharmacy": 0,
    "population": 10,
    "hhi": 0,
    "hud_crosswalk": 0,
    "county_desert": 0,
}

ZIP_NORMALIZATION_LABELS = {
    "extract_5_digit_regex": "Extract first 5-digit sequence (default)",
    "already_5_digit": "Treat values as ZIPs and zero-pad to 5 digits (supports 4-digit inputs)",
    "zip_plus_4": "Prefer ZIP+4 format, fallback to first 5 digits",
}

FILL_UNCOVERED_LABELS = {
    "none": "Leave uncovered ZIPs as missing values",
    "mean": "Fill uncovered ZIPs with mean of numeric columns from this file",
    "median": "Fill uncovered ZIPs with median of numeric columns from this file",
}

DEFAULT_CUSTOM_FEATURE_WEIGHT = 0.05


def _normalize_skip_rows(value: int, default: int = 0) -> int:
    """Normalize skip_rows input to a safe non-negative int."""
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(0, out)


def _zip_mode_label(mode: str) -> str:
    return ZIP_NORMALIZATION_LABELS.get(mode, mode)


def _is_numeric_scoring_candidate(values: pd.Series, min_valid_ratio: float = 0.50) -> bool:
    """Check whether a column is suitable as a weighted numeric scoring feature."""
    if pd.api.types.is_numeric_dtype(values):
        return values.notna().any()

    numeric = pd.to_numeric(values, errors="coerce")
    observed = values.notna()
    observed_count = int(observed.sum())
    if observed_count == 0:
        return False
    valid_ratio = float(numeric[observed].notna().mean())
    return valid_ratio >= min_valid_ratio


def _coerce_numeric_like_series(
    values: pd.Series,
    min_valid_ratio: float = 0.80,
) -> tuple[pd.Series, bool]:
    """
    Convert numeric-like string columns (currency, percent, comma-separated)
    to numeric when conversion is reliable.
    """
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce"), True

    cleaned = values.astype(str).str.strip()
    cleaned = cleaned.replace(
        {
            "": None,
            "nan": None,
            "NaN": None,
            "None": None,
            "none": None,
            "N/A": None,
            "n/a": None,
            "(X)": None,
        }
    )
    cleaned = (
        cleaned.str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    observed = cleaned.notna()
    observed_count = int(observed.sum())
    if observed_count == 0:
        return values, False
    valid_ratio = float(numeric[observed].notna().mean())
    if valid_ratio >= min_valid_ratio:
        return numeric, True
    return values, False


def _read_custom_bytes(content: bytes, filename: str, skip_rows: int = 0) -> pd.DataFrame:
    """Read a custom file from raw bytes (CSV or Excel)."""
    ext = Path(filename).suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(io.BytesIO(content), skiprows=skip_rows)
    else:
        df = pd.read_csv(io.BytesIO(content), skiprows=skip_rows)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalize_zip_series(values: pd.Series, mode: str = "extract_5_digit_regex") -> pd.Series:
    """Normalize ZIP values using a selected strategy."""
    s = values.astype(str).str.strip()

    if mode == "already_5_digit":
        out = s.str.split(".", n=1).str[0]
        out = out.where(out.str.fullmatch(r"\d{1,5}", na=False))
        out = out.str.zfill(5)
    elif mode == "zip_plus_4":
        zip5 = s.str.extract(r"^(\d{5})(?:[-\s]?\d{4})?$")[0]
        out = zip5.where(zip5.notna(), s.str.extract(r"(\d{5})")[0])
    else:
        out = s.str.extract(r"(\d{5})")[0]

    return out.where(out.notna(), None)


def _sanitize_column_prefix(value: str) -> str:
    """Sanitize a custom prefix so merged columns are predictable."""
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _suggest_short_feature_name(value: str, max_len: int = 40) -> str:
    """Generate a compact, readable output name for long source columns."""
    text = str(value or "").strip()
    text = text.replace("!!", " ")
    text = re.sub(
        r"\b(estimate|margin of error|civilian noninstitutionalized population|"
        r"coverage alone or in combination|coverage alone|private health insurance alone or in combination|"
        r"private coverage|percent private coverage|total)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[^0-9A-Za-z]+", "_", text.lower())
    text = re.sub(r"_+", "_", text).strip("_")

    stop_words = {
        "or", "in", "and", "of", "the", "with", "without",
        "civilian", "noninstitutionalized", "population",
        "estimate", "margin", "error", "total", "coverage",
        "private", "percent",
    }
    parts = [p for p in text.split("_") if p and p not in stop_words]
    compact = "_".join(parts) if parts else text

    compact = _sanitize_column_prefix(compact) or "feature"
    if len(compact) > max_len:
        compact = compact[:max_len].rstrip("_")
    return compact or "feature"


def _final_output_name_for_selected_column(source_col: str, rename_map: dict | None) -> str:
    """Resolve the output feature name for a selected source column."""
    if isinstance(rename_map, dict):
        candidate = rename_map.get(source_col)
        if isinstance(candidate, str) and candidate.strip():
            return _sanitize_column_prefix(candidate) or source_col
    return source_col


def _resolve_column_rename_map(selected_columns: list[str], rename_map: dict | None) -> dict[str, str]:
    """Resolve selected column output names with sanitized, unique targets."""
    if not selected_columns:
        return {}
    if not isinstance(rename_map, dict):
        return {c: c for c in selected_columns}

    resolved: dict[str, str] = {}
    used_names: set[str] = set()
    for src_col in selected_columns:
        base_name = _final_output_name_for_selected_column(src_col, rename_map)
        base_name = _sanitize_column_prefix(base_name) or _sanitize_column_prefix(src_col) or "feature"
        target_name = base_name
        idx = 2
        while target_name in used_names:
            target_name = f"{base_name}_{idx}"
            idx += 1
        used_names.add(target_name)
        resolved[src_col] = target_name
    return resolved


def _prepare_custom_dataframe(raw_df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, dict]:
    """
    Normalize and prepare a custom upload for merging on ZIP.

    Supports feature selection, optional prefixing, and duplicate ZIP aggregation.
    """
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    zip_col = meta.get("zip_col")
    if not zip_col or zip_col not in df.columns:
        raise ValueError(
            f"ZIP column '{zip_col}' not found in {meta.get('name', 'custom file')}. "
            f"Available columns: {list(df.columns)}"
        )

    selected_columns = meta.get("selected_columns")
    if selected_columns:
        selected_columns = [c for c in selected_columns if c in df.columns and c != zip_col]
    else:
        selected_columns = [c for c in df.columns if c != zip_col]

    zip_mode = meta.get("zip_normalization_mode", "extract_5_digit_regex")
    df["zip"] = _normalize_zip_series(df[zip_col], mode=zip_mode)
    df = df.dropna(subset=["zip"])

    keep_cols = ["zip"] + selected_columns
    df = df[[c for c in keep_cols if c in df.columns]]

    # Convert ACS-style "(X)" markers to missing values so they do not
    # appear as literal strings in the final dataset or scoring inputs.
    x_marker_cells_cleaned = 0
    for col in [c for c in selected_columns if c in df.columns]:
        mask = df[col].astype(str).str.fullmatch(r"\s*\(X\)\s*", na=False)
        x_marker_cells_cleaned += int(mask.sum())
        if mask.any():
            df.loc[mask, col] = pd.NA

    column_renames_input = meta.get("column_renames", {})
    resolved_column_renames: dict[str, str] = {}
    if isinstance(column_renames_input, dict) and selected_columns:
        resolved_column_renames = _resolve_column_rename_map(selected_columns, column_renames_input)
        rename_map = {
            src_col: dst_col
            for src_col, dst_col in resolved_column_renames.items()
            if src_col != dst_col and src_col in df.columns
        }
        resolved_selected_columns = [resolved_column_renames.get(c, c) for c in selected_columns if c in df.columns]

        if rename_map:
            df = df.rename(columns=rename_map)
        selected_columns = resolved_selected_columns

    coerced_numeric_columns: list[str] = []
    for col in [c for c in df.columns if c != "zip"]:
        coerced, did_coerce = _coerce_numeric_like_series(df[col])
        if did_coerce:
            df[col] = coerced
            coerced_numeric_columns.append(col)

    column_prefix = _sanitize_column_prefix(meta.get("column_prefix", ""))
    rename_map = {}
    if column_prefix:
        rename_map = {col: f"{column_prefix}__{col}" for col in selected_columns if col in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

    duplicate_policy = meta.get("duplicate_policy", "first")
    duplicate_policy = duplicate_policy if duplicate_policy in {"first", "mean", "sum", "max", "min"} else "first"

    if duplicate_policy == "first":
        df = df.drop_duplicates(subset=["zip"], keep="first")
    else:
        feature_cols = [c for c in df.columns if c != "zip"]
        agg_map = {}
        for col in feature_cols:
            agg_map[col] = duplicate_policy if pd.api.types.is_numeric_dtype(df[col]) else "first"
        if agg_map:
            df = df.groupby("zip", as_index=False).agg(agg_map)
        else:
            df = df.drop_duplicates(subset=["zip"], keep="first")

    info = {
        "rows": len(df),
        "columns": [c for c in df.columns if c != "zip"],
        "column_prefix": column_prefix,
        "column_renames": resolved_column_renames,
        "duplicate_policy": duplicate_policy,
        "zip_normalization_mode": zip_mode,
        "x_marker_cells_cleaned": x_marker_cells_cleaned,
        "coerced_numeric_columns": coerced_numeric_columns,
    }
    return df, info


def _merge_custom_into_dataset(
    base_df: pd.DataFrame, custom_df: pd.DataFrame, meta: dict
) -> tuple[pd.DataFrame, int, int]:
    """Merge a prepared custom dataframe into the base dataset with collision handling."""
    df = base_df
    custom = custom_df.copy()
    join_mode = meta.get("join_mode", "outer")
    join_mode = join_mode if join_mode in {"left", "outer", "inner"} else "outer"
    fill_uncovered_strategy = meta.get("fill_uncovered_strategy", "none")
    fill_uncovered_strategy = (
        fill_uncovered_strategy
        if fill_uncovered_strategy in {"none", "mean", "median"}
        else "none"
    )

    source_suffix = _sanitize_column_prefix(Path(meta.get("name", "custom")).stem) or "custom"
    custom_feature_cols = [c for c in custom.columns if c != "zip"]
    conflicts = [c for c in custom_feature_cols if c in df.columns]

    if conflicts:
        collision_renames = {}
        used_names = set(df.columns) | {c for c in custom.columns if c not in conflicts}
        for original in conflicts:
            renamed = f"{original}_{source_suffix}"
            while renamed in used_names:
                renamed = f"{renamed}_x"
            collision_renames[original] = renamed
            used_names.add(renamed)
        custom = custom.rename(columns=collision_renames)

    fill_values: dict[str, float] = {}
    if fill_uncovered_strategy in {"mean", "median"}:
        for col in [c for c in custom.columns if c != "zip"]:
            series = pd.to_numeric(custom[col], errors="coerce")
            if series.notna().any():
                if fill_uncovered_strategy == "mean":
                    fill_values[col] = float(series.mean())
                else:
                    fill_values[col] = float(series.median())

    before_cols = len(df.columns)
    use_indicator = bool(fill_values)
    merged = df.merge(custom, on="zip", how=join_mode, indicator=use_indicator)

    filled_cells = 0
    if use_indicator and "_merge" in merged.columns:
        uncovered_mask = merged["_merge"] == "left_only"
        if uncovered_mask.any():
            for col, value in fill_values.items():
                if col not in merged.columns:
                    continue
                needs_fill = uncovered_mask & merged[col].isna()
                filled_cells += int(needs_fill.sum())
                if needs_fill.any():
                    merged.loc[needs_fill, col] = value
        merged = merged.drop(columns=["_merge"])

    added_cols = len(merged.columns) - before_cols
    return merged, added_cols, filled_cells


def _predict_custom_output_columns(custom_uploads: list[dict]) -> list[str]:
    """Predict merged custom column names from current upload selections."""
    predicted: list[str] = []
    for upload in custom_uploads or []:
        selected_columns = upload.get("selected_columns") or []
        column_renames = upload.get("column_renames", {})
        resolved_column_names = _resolve_column_rename_map(selected_columns, column_renames)
        prefix = _sanitize_column_prefix(upload.get("column_prefix", ""))
        for col in selected_columns:
            out_col = resolved_column_names.get(col, col)
            predicted.append(f"{prefix}__{out_col}" if prefix else out_col)
    return predicted


def _build_mapping_column_options(
    prev_config: dict,
    custom_uploads: list[dict],
    prev_custom: list[dict],
    replace_prev_custom: bool,
) -> list[str]:
    """Build candidate source columns for constrained scoring mappings."""
    candidates: set[str] = set(CORE_DEFAULT_COLUMNS)

    if prev_config:
        for col in prev_config.get("unified_columns", []):
            if isinstance(col, str):
                candidates.add(col)

    for col in _predict_custom_output_columns(custom_uploads):
        candidates.add(col)

    if prev_custom and not replace_prev_custom:
        for meta in prev_custom:
            if not isinstance(meta, dict):
                continue
            for col in meta.get("merged_columns", []):
                if isinstance(col, str):
                    candidates.add(col)

    return sorted(candidates)


def _read_pharmacy_upload(
    path: str,
    skip_rows: int = 0,
    zip_normalization_mode: str = "extract_5_digit_regex",
) -> pd.DataFrame:
    """Read an uploaded pharmacy file (CSV or Excel) and ensure a ZIP column exists."""
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, skiprows=skip_rows)
    else:
        df = pd.read_csv(path, skiprows=skip_rows)

    df.columns = [str(c).strip() for c in df.columns]

    for candidate in ["ZIP", "zip", "Zip", "ZCTA5", "zcta5", "Short_ZIP", "Short_ ZIP"]:
        if candidate in df.columns:
            if candidate != "ZIP":
                df = df.rename(columns={candidate: "ZIP"})
            break
    else:
        for c in df.columns:
            if "zip" in c.lower() or "zcta" in c.lower():
                df = df.rename(columns={c: "ZIP"})
                break
        else:
            raise ValueError(f"No ZIP column found. Available: {df.columns.tolist()[:10]}")

    df["ZIP"] = _normalize_zip_series(df["ZIP"], mode=zip_normalization_mode)
    valid_zip_count = int(df["ZIP"].notna().sum())
    if valid_zip_count == 0:
        raise ValueError(
            "No valid ZIP values found in pharmacy file after normalization. "
            f"Mode used: {zip_normalization_mode}"
        )

    return df.dropna(subset=["ZIP"])


# ── Main page ────────────────────────────────────────────────────────────────

def main():
    st.title("Upload Data")
    st.markdown(
        "Upload your data files to build the analysis dataset. "
        "Required files are marked with **\\***. "
        "Files from the previous version are reused automatically if you don't upload a replacement."
    )

    storage = get_storage()
    latest_version = storage.get_latest_version(DATASET_ID)
    prev_config = None
    prev_files: dict = {}

    # ── Current dataset status ───────────────────────────────────────────
    if latest_version:
        prev_config = storage.get_config(DATASET_ID, latest_version)
        if prev_config:
            prev_files = prev_config.get("uploaded_files", {})
            rows = prev_config.get("unified_rows")
            ncols = len(prev_config.get("unified_columns", []))
            desc = prev_config.get("description", "")
            rows_str = f"{rows:,}" if isinstance(rows, int) else str(rows or "?")
            st.success(
                f"Current dataset: v`{latest_version}` — "
                f"{rows_str} rows × {ncols} columns"
                + (f" — _{desc}_" if desc else "")
            )
            with st.expander("View current columns"):
                st.write(prev_config.get("unified_columns", []))
    else:
        st.info("No dataset yet. Upload the required files below to create one.")

    st.divider()

    # ── File uploaders ───────────────────────────────────────────────────
    st.header("Data Files")

    uploads: dict = {}  # key → UploadedFile
    for key, label, exts, _required, help_text in SLOTS:
        existing_tag = ""
        if key in prev_files:
            existing_tag = f"  ✓ _({prev_files[key]} from previous version)_"

        uploaded = st.file_uploader(
            f"{label}{existing_tag}", type=exts, key=f"up_{key}", help=help_text
        )
        if uploaded:
            uploads[key] = uploaded

    replace_prev_core = st.checkbox(
        "Replace previously uploaded core files (do not carry forward old core files)",
        value=False,
        help=(
            "When enabled, only files uploaded in this run are used for core slots. "
            "Required core files must be uploaded again."
        ),
    )

    if replace_prev_core:
        st.info("Core replace mode is ON. Required core files must be uploaded in this run.")

    prev_core_parse_options = prev_config.get("core_parse_options", {}) if prev_config else {}
    core_skip_rows: dict[str, int] = {}
    with st.expander("Core parsing options (row skipping and ZIP handling)", expanded=False):
        st.caption(
            "Use these controls when files include extra metadata/header rows "
            "or non-standard ZIP formats."
        )
        for key, label, *_ in SLOTS:
            default_skip = _normalize_skip_rows(
                prev_core_parse_options.get(key, DEFAULT_CORE_SKIP_ROWS.get(key, 0)),
                default=DEFAULT_CORE_SKIP_ROWS.get(key, 0),
            )
            core_skip_rows[key] = int(
                st.number_input(
                    f"{label.rstrip(' *')} rows to skip before header",
                    min_value=0,
                    max_value=500,
                    value=default_skip,
                    step=1,
                    key=f"core_skip_{key}",
                )
            )

        zip_mode_options = list(ZIP_NORMALIZATION_LABELS.keys())
        prev_zip_mode = prev_core_parse_options.get(
            "pharmacy_zip_normalization_mode", "extract_5_digit_regex"
        )
        if prev_zip_mode not in zip_mode_options:
            prev_zip_mode = "extract_5_digit_regex"
        pharmacy_zip_normalization_mode = st.selectbox(
            "Pharmacy ZIP normalization",
            options=zip_mode_options,
            index=zip_mode_options.index(prev_zip_mode),
            key="core_zip_mode_pharmacy",
            format_func=_zip_mode_label,
        )

    # Check which required slots are satisfied (new upload OR previous version)
    required_keys = {key for key, _, _, req, _ in SLOTS if req}
    available_keys = set(uploads.keys()) if replace_prev_core else set(uploads.keys()) | set(prev_files.keys())
    missing = required_keys - available_keys

    if missing:
        pretty = ", ".join(
            next(lbl.rstrip(" *") for k, lbl, *_ in SLOTS if k == m)
            for m in sorted(missing)
        )
        st.warning(f"Missing required: **{pretty}**")

    st.divider()

    # ── Custom / Proprietary Data ────────────────────────────────────────
    st.header("Custom / Proprietary Data")
    st.markdown(
        "Upload any additional CSV or Excel files your organization has. "
        "Each file must contain a **ZIP / ZCTA column** so it can be joined "
        "to the core dataset. You can choose which columns to include per file."
    )
    st.caption(
        "Numeric columns from uploaded custom files are automatically added as "
        "optional weighted features in Math/Blended scoring."
    )

    prev_custom = prev_config.get("custom_files", []) if prev_config else []
    prev_custom_by_name = {
        pc.get("filename"): pc for pc in prev_custom if isinstance(pc, dict) and pc.get("filename")
    }

    custom_files = st.file_uploader(
        "Upload custom data files",
        type=["csv", "xlsx", "xlsm", "xls"],
        accept_multiple_files=True,
        key="custom_uploads",
        help="Each file needs a column with 5-digit ZIP or ZCTA codes.",
    )

    custom_uploads_ready: list[dict] = []

    if custom_files:
        for ix, cf in enumerate(custom_files):
            file_key = f"{ix}_{cf.name}"
            prev_custom_meta = prev_custom_by_name.get(cf.name, {})
            with st.expander(f"📄 {cf.name}", expanded=True):
                try:
                    ext = Path(cf.name).suffix.lower()
                    skip_rows = int(
                        st.number_input(
                            "Rows to skip before header",
                            min_value=0,
                            max_value=500,
                            value=_normalize_skip_rows(prev_custom_meta.get("skip_rows", 0)),
                            step=1,
                            key=f"custom_skip_{file_key}",
                        )
                    )

                    cf.seek(0)
                    if ext in (".xlsx", ".xlsm", ".xls"):
                        preview_df = pd.read_excel(cf, skiprows=skip_rows)
                    else:
                        preview_df = pd.read_csv(cf, skiprows=skip_rows)
                    preview_df.columns = [str(c).strip() for c in preview_df.columns]

                    if preview_df.empty:
                        st.warning("No rows found after applying skip rows.")
                        continue

                    st.dataframe(preview_df.head(10), use_container_width=True, height=150)

                    # Auto-detect ZIP column, preferring prior selection if present
                    zip_candidates = [""] + list(preview_df.columns)
                    auto_zip = prev_custom_meta.get("zip_col", "")
                    if auto_zip not in preview_df.columns:
                        auto_zip = ""
                    if not auto_zip:
                        for candidate in ["zip", "ZIP", "Zip", "ZCTA5", "zcta5", "ZCTA", "Zipcode", "zipcode"]:
                            if candidate in preview_df.columns:
                                auto_zip = candidate
                                break
                    if not auto_zip:
                        for c in preview_df.columns:
                            if "zip" in c.lower() or "zcta" in c.lower():
                                auto_zip = c
                                break

                    default_idx = zip_candidates.index(auto_zip) if auto_zip in zip_candidates else 0
                    zip_col = st.selectbox(
                        "ZIP / ZCTA column",
                        options=zip_candidates,
                        index=default_idx,
                        key=f"custom_zip_{file_key}",
                    )

                    zip_mode_options = list(ZIP_NORMALIZATION_LABELS.keys())
                    default_zip_mode = prev_custom_meta.get(
                        "zip_normalization_mode", "extract_5_digit_regex"
                    )
                    if default_zip_mode not in zip_mode_options:
                        default_zip_mode = "extract_5_digit_regex"
                    zip_normalization_mode = st.selectbox(
                        "ZIP normalization strategy",
                        options=zip_mode_options,
                        index=zip_mode_options.index(default_zip_mode),
                        key=f"custom_zip_mode_{file_key}",
                        format_func=_zip_mode_label,
                    )

                    if zip_col:
                        normalized_zips = _normalize_zip_series(
                            preview_df[zip_col],
                            mode=zip_normalization_mode,
                        )
                        valid_count = int(normalized_zips.notna().sum())
                        total_count = len(preview_df)
                        source_sample = preview_df[zip_col].dropna().head(5).tolist()
                        normalized_sample = normalized_zips.dropna().head(5).tolist()
                        st.caption(f"Sample source ZIP values: {source_sample}")
                        st.caption(f"Sample normalized ZIP values: {normalized_sample}")

                        zip_preview = pd.DataFrame(
                            {
                                "source_zip": preview_df[zip_col].head(8).astype(str),
                                "normalized_zip": normalized_zips.head(8),
                            }
                        )
                        st.dataframe(zip_preview, use_container_width=True, height=180)

                        if valid_count == 0:
                            st.error(
                                f"No valid ZIPs found in '{zip_col}' with mode "
                                f"'{_zip_mode_label(zip_normalization_mode)}'."
                            )
                            continue
                        elif valid_count < total_count * 0.5:
                            st.warning(
                                f"Only {valid_count:,}/{total_count:,} rows have a valid ZIP "
                                "with current normalization."
                            )
                        else:
                            st.caption(f"ZIP coverage: {valid_count:,}/{total_count:,} rows")

                        feature_cols = [c for c in preview_df.columns if c != zip_col]
                        numeric_feature_cols = [
                            c
                            for c in feature_cols
                            if _coerce_numeric_like_series(preview_df[c].head(5000))[1]
                        ]
                        st.caption(
                            f"Available columns ({len(feature_cols)}): {', '.join(feature_cols[:10])}"
                            + ("…" if len(feature_cols) > 10 else "")
                        )
                        st.caption(
                            f"Numeric columns detected: {len(numeric_feature_cols)} "
                            f"(useful for scoring/weighting if you include them)"
                        )

                        selected_default = prev_custom_meta.get("selected_columns") or feature_cols
                        selected_default = [c for c in selected_default if c in feature_cols]
                        if not selected_default:
                            selected_default = feature_cols

                        selected_columns = st.multiselect(
                            "Columns to include in merged dataset",
                            options=feature_cols,
                            default=selected_default,
                            key=f"custom_cols_{file_key}",
                            help="Only selected columns will be merged into the unified dataset.",
                        )

                        naming_options = ["auto_shorten", "keep_original", "custom_edit"]
                        default_naming_strategy = prev_custom_meta.get("naming_strategy", "auto_shorten")
                        if default_naming_strategy not in naming_options:
                            default_naming_strategy = "auto_shorten"
                        naming_strategy = st.selectbox(
                            "Output feature naming",
                            options=naming_options,
                            index=naming_options.index(default_naming_strategy),
                            key=f"custom_naming_{file_key}",
                            format_func=lambda x: {
                                "auto_shorten": "Auto-shorten long names (recommended)",
                                "keep_original": "Keep original source names",
                                "custom_edit": "Manually edit output names",
                            }.get(x, x),
                            help=(
                                "Controls how selected columns are named before prefixing. "
                                "Useful for very long ACS-style column names."
                            ),
                        )

                        column_renames: dict[str, str] = {}
                        if naming_strategy == "auto_shorten":
                            column_renames = {
                                col: _suggest_short_feature_name(col) for col in selected_columns
                            }
                            if selected_columns:
                                preview_names = pd.DataFrame(
                                    {
                                        "source_column": selected_columns,
                                        "output_name": [column_renames[c] for c in selected_columns],
                                    }
                                )
                                st.dataframe(
                                    preview_names.head(12),
                                    use_container_width=True,
                                    height=180,
                                )
                        elif naming_strategy == "custom_edit" and selected_columns:
                            prev_rename_map = prev_custom_meta.get("column_renames", {})
                            if not isinstance(prev_rename_map, dict):
                                prev_rename_map = {}
                            rename_editor_df = pd.DataFrame(
                                {
                                    "source_column": selected_columns,
                                    "output_name": [
                                        prev_rename_map.get(c, _suggest_short_feature_name(c))
                                        for c in selected_columns
                                    ],
                                }
                            )
                            edited = st.data_editor(
                                rename_editor_df,
                                key=f"custom_rename_editor_{file_key}",
                                use_container_width=True,
                                hide_index=True,
                                disabled=["source_column"],
                            )
                            for _, row in edited.iterrows():
                                src = str(row.get("source_column", "")).strip()
                                out = str(row.get("output_name", "")).strip()
                                if src in selected_columns and out:
                                    column_renames[src] = out

                        x_marker_count = 0
                        for col in selected_columns:
                            if col in preview_df.columns:
                                x_marker_count += int(
                                    preview_df[col].astype(str).str.fullmatch(r"\s*\(X\)\s*", na=False).sum()
                                )
                        if x_marker_count > 0:
                            st.info(
                                f"Detected {x_marker_count:,} '(X)' values in selected columns. "
                                "These mean 'not available / not applicable' in ACS-style files and "
                                "will be converted to missing values during processing."
                            )

                        default_prefix = prev_custom_meta.get("column_prefix") or _sanitize_column_prefix(
                            Path(cf.name).stem
                        )
                        column_prefix = st.text_input(
                            "Column prefix (recommended to avoid name collisions)",
                            value=default_prefix,
                            key=f"custom_prefix_{file_key}",
                            help="Columns will be renamed like prefix__column. Leave blank to keep original names.",
                        )

                        join_options = ["left", "outer"]
                        default_join = prev_custom_meta.get("join_mode", "left")
                        if default_join not in join_options:
                            default_join = "left"
                        join_mode = st.selectbox(
                            "How to merge this file",
                            options=join_options,
                            index=join_options.index(default_join),
                            key=f"custom_join_{file_key}",
                            format_func=lambda x: (
                                "left (supplement existing ZIPs only)"
                                if x == "left"
                                else "outer (also add ZIPs not already in core data)"
                            ),
                        )

                        duplicate_options = ["first", "mean", "sum", "max", "min"]
                        default_duplicate = prev_custom_meta.get("duplicate_policy", "first")
                        if default_duplicate not in duplicate_options:
                            default_duplicate = "first"
                        duplicate_policy = st.selectbox(
                            "If multiple rows share a ZIP",
                            options=duplicate_options,
                            index=duplicate_options.index(default_duplicate),
                            key=f"custom_dupes_{file_key}",
                            help=(
                                "For mean/sum/max/min, numeric columns are aggregated. "
                                "Non-numeric columns keep the first value."
                            ),
                        )

                        fill_options = list(FILL_UNCOVERED_LABELS.keys())
                        default_fill = prev_custom_meta.get("fill_uncovered_strategy", "none")
                        if default_fill not in fill_options:
                            default_fill = "none"
                        fill_uncovered_strategy = st.selectbox(
                            "If this file does not cover all base ZIPs",
                            options=fill_options,
                            index=fill_options.index(default_fill),
                            key=f"custom_fill_{file_key}",
                            format_func=lambda x: FILL_UNCOVERED_LABELS.get(x, x),
                            help=(
                                "Applies to numeric columns from this file for ZIPs present in "
                                "the base dataset but missing from this custom file."
                            ),
                        )

                        cf.seek(0)
                        if not selected_columns:
                            st.warning("Select at least one column to include this file.")
                        else:
                            custom_uploads_ready.append(
                                {
                                    "name": cf.name,
                                    "content": cf.getvalue(),
                                    "skip_rows": skip_rows,
                                    "zip_col": zip_col,
                                    "zip_normalization_mode": zip_normalization_mode,
                                    "selected_columns": selected_columns,
                                    "naming_strategy": naming_strategy,
                                    "column_renames": column_renames,
                                    "column_prefix": column_prefix,
                                    "join_mode": join_mode,
                                    "duplicate_policy": duplicate_policy,
                                    "fill_uncovered_strategy": fill_uncovered_strategy,
                                }
                            )
                    else:
                        st.warning("Select a ZIP column to include this file.")
                except Exception as e:
                    st.error(f"Could not read {cf.name}: {e}")

    # Show count of previous custom files
    replace_prev_custom = st.checkbox(
        "Replace previously uploaded custom files (instead of supplementing them)",
        value=False,
        help="By default, previous custom files are carried forward and combined with new uploads.",
    )

    if prev_custom and not custom_files:
        st.info(
            f"{len(prev_custom)} custom file(s) from previous version will be carried forward. "
            "Upload new files to supplement them, or enable replace mode to start fresh."
        )

    st.divider()

    # ── Constrained scoring mappings (fixed main-app inputs) ────────────
    st.header("Scoring Input Mapping (Optional)")
    st.caption(
        "Map dataset columns to the fixed scoring inputs used by the main app sliders. "
        "This keeps scoring behavior consistent while allowing client-specific schemas."
    )

    prev_scoring = prev_config.get("scoring_config", {}) if prev_config else {}
    prev_map_by_component = {}
    for mapping in prev_scoring.get("column_mappings", []):
        if isinstance(mapping, dict):
            component = mapping.get("target_component")
            source_col = mapping.get("source_column")
            if component and source_col:
                prev_map_by_component[component] = source_col

    mapping_options = _build_mapping_column_options(
        prev_config=prev_config,
        custom_uploads=custom_uploads_ready,
        prev_custom=prev_custom,
        replace_prev_custom=replace_prev_custom,
    )

    if prev_map_by_component:
        st.caption("Previous mapping detected and pre-filled below.")

    scoring_component_map: dict[str, str] = {}
    for target in SCORING_MAPPING_TARGETS:
        component = target["component"]
        default_col = target["default_column"]
        options = [""] + mapping_options

        suggested = prev_map_by_component.get(component, "")
        if not suggested and default_col in mapping_options:
            suggested = default_col

        default_index = options.index(suggested) if suggested in options else 0
        selected_source = st.selectbox(
            target["label"],
            options=options,
            index=default_index,
            key=f"map_{component}",
            help=target["help"],
            format_func=lambda value: "(not mapped)" if value == "" else value,
        )
        if selected_source:
            scoring_component_map[component] = selected_source

    st.divider()

    # ── Options ──────────────────────────────────────────────────────────
    description = st.text_input(
        "Version description (optional)",
        placeholder="e.g., Updated financial data for 2024",
    )
    fetch_education = st.checkbox(
        "Fetch education data from Census API",
        value=True,
        help="Calls the ACS API for education attainment by ZIP. Requires internet.",
    )

    # ── Process button ───────────────────────────────────────────────────
    if st.button("Process & Save", type="primary", disabled=bool(missing)):
        try:
            _process_and_save(
                uploads, custom_uploads_ready, prev_config, latest_version,
                description, fetch_education, replace_prev_custom,
                replace_prev_core, scoring_component_map,
                core_skip_rows, pharmacy_zip_normalization_mode,
            )
        except Exception as e:
            st.error(f"Processing failed: {e}")
            logger.error("Upload processing failed", exc_info=True)
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def _process_and_save(
    uploads,
    custom_uploads,
    prev_config,
    prev_version,
    description,
    fetch_education,
    replace_prev_custom,
    replace_prev_core,
    scoring_component_map,
    core_skip_rows,
    pharmacy_zip_normalization_mode,
):
    """Run the existing data pipeline on uploaded files and save the unified result."""
    storage = get_storage()
    version_id = generate_version_id()
    prev_files = prev_config.get("uploaded_files", {}) if prev_config else {}
    prev_custom = prev_config.get("custom_files", []) if prev_config else []
    core_skip_rows = core_skip_rows or {}
    resolved_core_skip_rows = {
        key: _normalize_skip_rows(
            core_skip_rows.get(key, DEFAULT_CORE_SKIP_ROWS.get(key, 0)),
            default=DEFAULT_CORE_SKIP_ROWS.get(key, 0),
        )
        for key, *_ in SLOTS
    }
    if pharmacy_zip_normalization_mode not in ZIP_NORMALIZATION_LABELS:
        pharmacy_zip_normalization_mode = "extract_5_digit_regex"

    progress = st.progress(0, text="Starting…")

    with tempfile.TemporaryDirectory() as work_dir:
        work = Path(work_dir)
        file_paths: dict = {}   # slot key → local path
        file_map: dict = {}     # slot key → filename (saved in config for next time)

        # ── Resolve each slot: new upload wins, else previous version ────
        progress.progress(0.05, text="Preparing files…")
        all_keys = [key for key, *_ in SLOTS]

        for key in all_keys:
            if key in uploads:
                uf = uploads[key]
                fname = uf.name
                dest = work / fname
                content = uf.getvalue()
                dest.write_bytes(content)
                file_paths[key] = str(dest)
                file_map[key] = fname
                storage.upload_file(DATASET_ID, version_id, fname, content)

            elif (not replace_prev_core) and key in prev_files and prev_version:
                fname = prev_files[key]
                try:
                    content = storage.download_file(DATASET_ID, prev_version, fname)
                    dest = work / fname
                    dest.write_bytes(content)
                    file_paths[key] = str(dest)
                    file_map[key] = fname
                    storage.upload_file(DATASET_ID, version_id, fname, content)
                except FileNotFoundError:
                    required = {k for k, _, _, r, _ in SLOTS if r}
                    if key in required:
                        raise ValueError(
                            f"Required file '{key}' ({fname}) missing from previous version"
                        )

        required_keys = {key for key, _, _, req, _ in SLOTS if req}
        missing_required = sorted(k for k in required_keys if k not in file_paths)
        if missing_required:
            raise ValueError(f"Missing required core files: {', '.join(missing_required)}")

        # ── Read sources using existing readers ──────────────────────────
        progress.progress(0.10, text="Reading financial data…")
        financial = read_financial_data(
            file_paths["financial"],
            skip_rows=resolved_core_skip_rows.get("financial", 0),
        )

        progress.progress(0.20, text="Reading health data…")
        health = read_health_data(
            file_paths["health"],
            skip_rows=resolved_core_skip_rows.get("health", 0),
        )

        progress.progress(0.30, text="Reading pharmacy data…")
        pharmacy = _read_pharmacy_upload(
            file_paths["pharmacy"],
            skip_rows=resolved_core_skip_rows.get("pharmacy", 0),
            zip_normalization_mode=pharmacy_zip_normalization_mode,
        )

        progress.progress(0.40, text="Reading population data…")
        population = read_population_data(
            file_paths["population"],
            skip_rows=resolved_core_skip_rows.get("population", DEFAULT_CORE_SKIP_ROWS["population"]),
        )

        hhi = None
        if "hhi" in file_paths:
            progress.progress(0.45, text="Reading HHI data…")
            hhi = read_hhi_excel(
                file_paths["hhi"],
                skip_rows=resolved_core_skip_rows.get("hhi", 0),
            )

        # ── Core merge (preprocess) ──────────────────────────────────────
        progress.progress(0.50, text="Merging core datasets…")
        df = preprocess(financial, health, pharmacy, population, hhi=hhi)

        # ── Education data from Census API ───────────────────────────────
        if fetch_education:
            progress.progress(0.55, text="Fetching education data (Census API)…")
            try:
                edu = read_education_data_acs(year=2023)
                df = df.merge(edu[["zip", "edu_hs_or_lower_pct"]], on="zip", how="left")
            except Exception as e:
                st.warning(f"Education data fetch failed (non-fatal): {e}")

        # ── County desert downscaling ────────────────────────────────────
        if "hud_crosswalk" in file_paths and "county_desert" in file_paths:
            progress.progress(0.60, text="Processing county desert data…")
            hud = read_hud_zip_county_crosswalk(
                file_paths["hud_crosswalk"],
                skip_rows=resolved_core_skip_rows.get("hud_crosswalk", 0),
            )
            county = read_county_desert_csv(
                file_paths["county_desert"],
                skip_rows=resolved_core_skip_rows.get("county_desert", 0),
            )
            zip_desert = downscale_county_to_zip(county, hud)
            df = df.merge(zip_desert, on="zip", how="left")

        # ── Merge custom / proprietary datasets ──────────────────────────
        custom_file_meta: list[dict] = []

        prev_custom_by_name = {
            pc.get("filename"): pc for pc in prev_custom if isinstance(pc, dict) and pc.get("filename")
        }
        new_custom_by_name = {cu["name"]: cu for cu in custom_uploads} if custom_uploads else {}
        carried_prev_names = set()

        custom_sources_to_merge: list[tuple[str, dict, bytes]] = []

        if prev_custom and prev_version and not replace_prev_custom:
            for fname, pc in prev_custom_by_name.items():
                if fname in new_custom_by_name:
                    continue  # replaced by new upload with same filename
                try:
                    content = storage.download_file(DATASET_ID, prev_version, f"custom_{fname}")
                    custom_sources_to_merge.append(("previous", pc, content))
                    carried_prev_names.add(fname)
                except Exception as e:
                    st.warning(f"Could not carry forward custom file '{fname}': {e}")

        for cu in custom_uploads or []:
            custom_sources_to_merge.append(("new", cu, cu["content"]))

        if custom_sources_to_merge:
            progress.progress(0.68, text="Merging custom datasets…")

        for origin, custom_meta_input, content in custom_sources_to_merge:
            fname = custom_meta_input["filename"] if origin == "previous" else custom_meta_input["name"]
            try:
                raw_custom_df = _read_custom_bytes(
                    content,
                    fname,
                    skip_rows=_normalize_skip_rows(custom_meta_input.get("skip_rows", 0)),
                )

                merge_meta = {
                    "name": fname,
                    "zip_col": custom_meta_input.get("zip_col"),
                    "zip_normalization_mode": custom_meta_input.get(
                        "zip_normalization_mode", "extract_5_digit_regex"
                    ),
                    "selected_columns": custom_meta_input.get("selected_columns"),
                    "column_renames": custom_meta_input.get("column_renames"),
                    "column_prefix": custom_meta_input.get("column_prefix", Path(fname).stem),
                    "join_mode": custom_meta_input.get("join_mode", "outer"),
                    "duplicate_policy": custom_meta_input.get("duplicate_policy", "first"),
                    "fill_uncovered_strategy": custom_meta_input.get("fill_uncovered_strategy", "none"),
                }
                prepared_custom_df, prep_info = _prepare_custom_dataframe(raw_custom_df, merge_meta)

                if prepared_custom_df.empty:
                    st.warning(f"Skipping custom file '{fname}' because no rows contained valid ZIP values.")
                    continue

                if not prep_info["columns"]:
                    st.warning(f"Skipping custom file '{fname}' because no columns were selected.")
                    continue

                df, new_cols, filled_cells = _merge_custom_into_dataset(df, prepared_custom_df, merge_meta)
                logger.info(
                    "Merged custom file '%s' (%s): +%s columns, %s rows, join=%s, dupes=%s, fill=%s, cells=%s",
                    fname,
                    origin,
                    new_cols,
                    len(prepared_custom_df),
                    merge_meta["join_mode"],
                    prep_info["duplicate_policy"],
                    merge_meta["fill_uncovered_strategy"],
                    filled_cells,
                )

                storage.upload_file(DATASET_ID, version_id, f"custom_{fname}", content)
                custom_file_meta.append({
                    "filename": fname,
                    "skip_rows": _normalize_skip_rows(custom_meta_input.get("skip_rows", 0)),
                    "zip_col": merge_meta["zip_col"],
                    "zip_normalization_mode": prep_info["zip_normalization_mode"],
                    "selected_columns": custom_meta_input.get("selected_columns"),
                    "naming_strategy": custom_meta_input.get("naming_strategy", "keep_original"),
                    "column_renames": prep_info.get("column_renames", {}),
                    "column_prefix": prep_info["column_prefix"],
                    "join_mode": merge_meta["join_mode"],
                    "duplicate_policy": prep_info["duplicate_policy"],
                    "fill_uncovered_strategy": merge_meta["fill_uncovered_strategy"],
                    "filled_uncovered_cells": filled_cells,
                    "x_marker_cells_cleaned": prep_info.get("x_marker_cells_cleaned", 0),
                    "coerced_numeric_columns": prep_info.get("coerced_numeric_columns", []),
                    "merged_columns": prep_info["columns"],
                    "source_origin": origin,
                })
            except Exception as e:
                st.warning(f"Could not process custom file '{fname}': {e}")

        if carried_prev_names and custom_uploads and not replace_prev_custom:
            st.info(
                f"Supplemented with {len(custom_uploads)} new custom file(s) and "
                f"carried forward {len(carried_prev_names)} previous custom file(s)."
            )

        # ── Fill NaN for core columns introduced by outer joins ─────────
        for col, default in [
            ("n_pharmacies", 0), ("population", 0), ("pop_density", 0),
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(default)
        if "n_pharmacies" in df.columns:
            df["n_pharmacies"] = df["n_pharmacies"].astype(int)

        # ── Save unified CSV ─────────────────────────────────────────────
        progress.progress(0.80, text="Saving unified dataset…")
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        csv_bytes = buf.getvalue()
        storage.upload_unified_dataset(DATASET_ID, version_id, csv_bytes)

        # ── Save config ──────────────────────────────────────────────────
        progress.progress(0.90, text="Saving configuration…")
        sources = []
        for slot_key, fname in file_map.items():
            sources.append({
                "source_id": generate_source_id(fname),
                "filename": fname,
                "source_kind": "core",
                "slot": slot_key,
                "skip_rows": resolved_core_skip_rows.get(slot_key, 0),
                "zip_normalization_mode": (
                    pharmacy_zip_normalization_mode if slot_key == "pharmacy" else None
                ),
            })

        for custom_meta in custom_file_meta:
            original_name = custom_meta.get("filename")
            stored_name = f"custom_{original_name}"
            sources.append({
                "source_id": generate_source_id(stored_name),
                "filename": stored_name,
                "source_kind": "custom",
                "original_filename": original_name,
                "zip_column": custom_meta.get("zip_col"),
                "skip_rows": custom_meta.get("skip_rows", 0),
                "zip_normalization_mode": custom_meta.get("zip_normalization_mode"),
                "naming_strategy": custom_meta.get("naming_strategy"),
                "column_renames": custom_meta.get("column_renames", {}),
                "join_mode": custom_meta.get("join_mode"),
                "duplicate_policy": custom_meta.get("duplicate_policy"),
                "fill_uncovered_strategy": custom_meta.get("fill_uncovered_strategy"),
                "coerced_numeric_columns": custom_meta.get("coerced_numeric_columns", []),
                "x_marker_cells_cleaned": custom_meta.get("x_marker_cells_cleaned", 0),
                "merged_columns": custom_meta.get("merged_columns", []),
            })

        cfg = build_dataset_config(DATASET_ID, version_id, sources=sources)
        cfg["unified"] = True
        cfg["unified_rows"] = len(df)
        cfg["unified_columns"] = list(df.columns)
        cfg["uploaded_files"] = file_map
        cfg["custom_files"] = custom_file_meta
        cfg["core_parse_options"] = {
            **resolved_core_skip_rows,
            "pharmacy_zip_normalization_mode": pharmacy_zip_normalization_mode,
        }
        prev_scoring = prev_config.get("scoring_config", {}) if prev_config else {}
        mapped_columns = []
        dropped_mappings = []
        if scoring_component_map:
            for component, source_column in scoring_component_map.items():
                if source_column in df.columns:
                    mapped_columns.append({
                        "source_column": source_column,
                        "target_component": component,
                        "transform": None,
                    })
                else:
                    dropped_mappings.append(f"{component}->{source_column}")
        else:
            for mapping in prev_scoring.get("column_mappings", []) if isinstance(prev_scoring, dict) else []:
                if not isinstance(mapping, dict):
                    continue
                source_column = mapping.get("source_column")
                target_component = mapping.get("target_component")
                if not source_column or not target_component:
                    continue
                if source_column in df.columns:
                    mapped_columns.append({
                        "source_column": source_column,
                        "target_component": target_component,
                        "transform": mapping.get("transform"),
                    })
                else:
                    dropped_mappings.append(f"{target_component}->{source_column}")

        if dropped_mappings:
            st.warning(
                "Some mappings were not saved because columns were not in the final dataset: "
                + ", ".join(dropped_mappings)
            )

        mapped_source_columns = {m["source_column"] for m in mapped_columns if "source_column" in m}
        prev_custom_feature_weights = (
            prev_scoring.get("custom_feature_weights", {}) if isinstance(prev_scoring, dict) else {}
        )
        custom_feature_weights: dict[str, float] = {}

        # Auto-register numeric custom-uploaded columns as additional scorable features.
        for custom_meta in custom_file_meta:
            for col in custom_meta.get("merged_columns", []):
                if (
                    not isinstance(col, str)
                    or col not in df.columns
                    or col in mapped_source_columns
                    or not _is_numeric_scoring_candidate(df[col])
                ):
                    continue
                try:
                    prior_weight = float(
                        prev_custom_feature_weights.get(col, DEFAULT_CUSTOM_FEATURE_WEIGHT)
                    )
                except Exception:
                    prior_weight = DEFAULT_CUSTOM_FEATURE_WEIGHT
                custom_feature_weights[col] = min(1.0, max(0.0, prior_weight))

        # Preserve previous custom-feature weights for columns that still exist.
        for col, weight in prev_custom_feature_weights.items():
            if (
                isinstance(col, str)
                and col in df.columns
                and col not in mapped_source_columns
                and _is_numeric_scoring_candidate(df[col])
                and col not in custom_feature_weights
            ):
                try:
                    weight_value = float(weight)
                except Exception:
                    continue
                custom_feature_weights[col] = min(1.0, max(0.0, weight_value))

        cfg["scoring_config"] = {
            "column_mappings": mapped_columns,
            "weight_overrides": prev_scoring.get("weight_overrides", {})
            if isinstance(prev_scoring, dict)
            else {},
            "custom_feature_weights": custom_feature_weights,
            "desert_threshold": prev_scoring.get("desert_threshold", 2.0)
            if isinstance(prev_scoring, dict)
            else 2.0,
            "normalize_scores": prev_scoring.get("normalize_scores", True)
            if isinstance(prev_scoring, dict)
            else True,
        }
        if description:
            cfg["description"] = description
        elif prev_config and prev_config.get("description"):
            cfg["description"] = prev_config["description"]
        storage.upload_config(DATASET_ID, version_id, cfg)

        # ── Update LATEST ────────────────────────────────────────────────
        storage.update_latest(DATASET_ID, version_id)
        st.cache_data.clear()
        progress.progress(1.0, text="Done!")

    # ── Results ──────────────────────────────────────────────────────────
    st.success(
        f"Dataset saved! Version `{version_id}` — "
        f"**{len(df):,} rows × {len(df.columns)} columns**"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", len(df.columns))
    with c3:
        st.metric("Core Files", len(file_map))
    with c4:
        st.metric("Custom Files", len(custom_file_meta))

    st.markdown("### Preview")
    st.dataframe(df.head(50), use_container_width=True, height=400)
    st.markdown(f"**Columns ({len(df.columns)}):** `{', '.join(df.columns)}`")

    st.download_button(
        "Download unified dataset",
        csv_bytes,
        f"pharmacy_desert_data_{version_id}.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
