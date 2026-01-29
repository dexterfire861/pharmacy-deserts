"""
Dataset Onboarding Wizard - Streamlit Page

Allows users to:
- Upload multiple files (CSV, Excel, JSON, Parquet, ZIP)
- Configure column mappings and ZIP normalization
- Select and rename feature columns
- Apply cleaning rules
- Map columns to scoring components
- Upload to S3 with versioning
"""
import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from app.config import get_config
from app.auth import is_authenticated, login_form
from ingestion.parsers import (
    get_file_type, get_excel_sheet_names, parse_file, 
    extract_zip_contents, get_column_types, get_column_stats
)
from ingestion.normalize import (
    NORMALIZATION_MODES, normalize_zip_column, 
    detect_zip_column, suggest_normalization_mode
)
from storage.datasets import (
    get_storage, generate_version_id, build_dataset_config, build_file_mapping
)
from models.schema import (
    SCORING_COMPONENTS, COMPONENT_CATEGORIES, REQUIRED_COMPONENTS,
    SCORED_COMPONENTS, ScoreDirection, ScoringConfig, ColumnMapping
)

# Page configuration
st.set_page_config(
    page_title="Upload Data - Pharmacy Desert Explorer",
    page_icon="📤",
    layout="wide"
)

# Authentication check
config = get_config()
if config.require_auth and not is_authenticated():
    login_form()
    st.stop()


def init_session_state():
    """Initialize session state variables."""
    if 'upload_files' not in st.session_state:
        st.session_state.upload_files = {}  # filename -> {content, df, config}
    if 'dataset_id' not in st.session_state:
        st.session_state.dataset_id = "pharmacy_data"  # Fixed combined dataset
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'file_configs' not in st.session_state:
        st.session_state.file_configs = {}  # filename -> config dict
    if 'scoring_mappings' not in st.session_state:
        st.session_state.scoring_mappings = {}  # component_name -> source_column
    if 'weight_overrides' not in st.session_state:
        st.session_state.weight_overrides = {}
    if 'version_description' not in st.session_state:
        st.session_state.version_description = ""  # Description for this version


def render_step_indicator(current_step: int, total_steps: int = 5):
    """Render a visual step indicator."""
    steps = [
        "📁 Upload Files", 
        "⚙️ Configure Columns", 
        "🧹 Cleaning Rules", 
        "🎯 Scoring Config",
        "✅ Review & Submit"
    ]
    cols = st.columns(total_steps)
    for i, (col, step_name) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current_step:
                st.success(f"✓ {step_name}")
            elif i == current_step:
                st.info(f"→ {step_name}")
            else:
                st.markdown(f"○ {step_name}")


def render_file_upload_section():
    """Render the file upload section (Step 1)."""
    st.header("📁 Step 1: Upload Data Files")
    
    # Fixed dataset ID - all files go to one combined dataset
    DATASET_ID = "pharmacy_data"
    st.session_state.dataset_id = DATASET_ID
    
    # Explanation
    st.info("""
    **How it works:** Upload your data files here. Each file should contain ZIP/ZCTA codes.
    All files will be automatically merged on ZCTA5 (outer join) to create a combined dataset.
    
    **Common data types:**
    - 📊 Financial data (income, poverty rates)
    - 🏥 Health data (health outcomes, insurance coverage)  
    - 💊 Pharmacy locations
    - 👥 Population/demographics
    - 🗺️ Geographic data (lat/lon coordinates)
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Show existing sources
        storage = get_storage()
        try:
            existing_config = storage.get_config(DATASET_ID)
            if existing_config:
                existing_sources = existing_config.get('sources', [])
                if existing_sources:
                    st.success(f"✓ {len(existing_sources)} existing data source(s)")
                    with st.expander("View existing sources"):
                        for src in existing_sources:
                            st.write(f"• {src.get('filename', 'unknown')}")
        except:
            pass
    
    with col2:
        st.markdown("### Environment")
        env_badge = "🟢 Production (S3)" if config.is_production else "🔵 Development (Local)"
        st.info(env_badge)
    
    st.markdown("---")
    
    # Version description
    st.markdown("### Version Description")
    version_desc = st.text_input(
        "What's in this version?",
        value=st.session_state.version_description,
        placeholder="e.g., Added health data, fixed income column mapping",
        help="A short description to help identify this version later"
    )
    st.session_state.version_description = version_desc
    
    st.markdown("---")
    
    # Data Source Tabs: File Upload vs API
    source_tab = st.radio(
        "Add data from:",
        ["📁 Upload Files", "🌐 Connect API"],
        horizontal=True,
        key="source_type_tab"
    )
    
    st.markdown("---")
    
    if source_tab == "📁 Upload Files":
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload data files",
            type=['csv', 'xlsx', 'xlsm', 'json', 'parquet', 'zip'],
            accept_multiple_files=True,
            help="Supported formats: CSV, Excel (xlsx/xlsm), JSON, Parquet, ZIP archives"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                file_content = uploaded_file.read()
                file_type = get_file_type(filename)
                
                # Handle ZIP files
                if file_type == 'zip':
                    st.info(f"📦 Extracting ZIP archive: {filename}")
                    try:
                        extracted = extract_zip_contents(file_content)
                        for inner_name, inner_content in extracted.items():
                            inner_type = get_file_type(inner_name)
                            if inner_type in ['csv', 'xlsx', 'xlsm', 'json', 'parquet']:
                                st.session_state.upload_files[inner_name] = {
                                    'content': inner_content,
                                    'type': inner_type,
                                    'df': None,
                                    'from_zip': filename
                                }
                                st.success(f"  ↳ Extracted: {inner_name}")
                    except Exception as e:
                        st.error(f"Failed to extract {filename}: {e}")
                else:
                    st.session_state.upload_files[filename] = {
                        'content': file_content,
                        'type': file_type,
                        'df': None,
                        'from_zip': None
                    }
    
    # Show uploaded files (shown for both tabs)
    if st.session_state.upload_files:
        st.markdown("### Uploaded Files")
        
        for filename, file_info in st.session_state.upload_files.items():
            with st.expander(f"📄 {filename} ({file_info['type'].upper()})", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Excel sheet selection
                    sheet_name = None
                    if file_info['type'] in ['xlsx', 'xlsm']:
                        try:
                            sheets = get_excel_sheet_names(file_info['content'])
                            sheet_name = st.selectbox(
                                f"Select sheet for {filename}",
                                options=sheets,
                                key=f"sheet_{filename}"
                            )
                        except Exception as e:
                            st.error(f"Error reading sheets: {e}")
                    
                    # Skip rows for CSV/Excel
                    skip_rows = 0
                    if file_info['type'] in ['csv', 'xlsx', 'xlsm']:
                        skip_rows = st.number_input(
                            f"Skip rows (header offset)",
                            min_value=0,
                            max_value=100,
                            value=0,
                            key=f"skip_{filename}",
                            help="Number of rows to skip before the header row"
                        )
                    
                    # Parse button
                    if st.button(f"Load Preview", key=f"parse_{filename}"):
                        try:
                            with st.spinner("Parsing file..."):
                                df = parse_file(
                                    filename,
                                    file_info['content'],
                                    file_type=file_info['type'],
                                    sheet_name=sheet_name,
                                    skip_rows=skip_rows
                                )
                                st.session_state.upload_files[filename]['df'] = df
                                st.session_state.upload_files[filename]['sheet_name'] = sheet_name
                                st.session_state.upload_files[filename]['skip_rows'] = skip_rows
                                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
                        except Exception as e:
                            st.error(f"Error parsing file: {e}")
                
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{filename}"):
                        del st.session_state.upload_files[filename]
                        st.rerun()
                
                # Show preview if parsed
                if file_info.get('df') is not None:
                    df = file_info['df']
                    st.markdown(f"**Preview** (first 50 rows of {len(df):,} total)")
                    st.dataframe(df.head(50), use_container_width=True, height=300)
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            # Can proceed if we have files and a dataset ID (preview is optional here)
            has_files = len(st.session_state.upload_files) > 0
            can_proceed = has_files and st.session_state.dataset_id
            
            def go_to_step_2():
                st.session_state.current_step = 2
            
            st.button(
                "Next: Configure Columns →", 
                type="primary", 
                use_container_width=True,
                disabled=not can_proceed,
                on_click=go_to_step_2 if can_proceed else None
            )
            
            if not st.session_state.dataset_id:
                st.caption("⚠️ Enter a Dataset ID")
            if not has_files:
                st.caption("⚠️ Upload at least one file or connect an API")
    
    # API Source Section (when API tab is selected)
    if source_tab == "🌐 Connect API":
        render_api_source_section()


def render_api_source_section():
    """Render the API source connection section."""
    from ingestion.api_connectors import (
        CensusACSConnector, HUDConnector, CENSUS_PRESETS, US_STATES,
        list_census_presets, get_connector
    )
    
    st.markdown("### Connect to API Data Source")
    
    # API Type Selection
    api_type = st.selectbox(
        "Select API type",
        options=["census_acs", "hud", "custom"],
        format_func=lambda x: {
            "census_acs": "🇺🇸 US Census ACS (American Community Survey)",
            "hud": "🏠 HUD (Fair Market Rents, Income Limits)",
            "custom": "🔧 Custom API (Advanced)"
        }.get(x, x),
        key="api_type_select"
    )
    
    if api_type == "census_acs":
        st.info("""
        **Census ACS** provides demographic, social, economic, and housing data 
        for ZIP Code Tabulation Areas (ZCTAs) across the United States.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Year selection
            year = st.selectbox(
                "Data Year",
                options=list(range(2022, 2014, -1)),
                index=0,
                help="Select the ACS 5-year data release year"
            )
        
        with col2:
            # API Key (optional)
            api_key = st.text_input(
                "Census API Key (optional)",
                type="password",
                help="Get a free key at api.census.gov. Optional but recommended for heavy usage."
            )
        
        # Preset selection
        st.markdown("#### Select Data Category")
        presets = list_census_presets()
        
        preset_cols = st.columns(3)
        selected_presets = []
        
        for i, preset in enumerate(presets):
            col_idx = i % 3
            with preset_cols[col_idx]:
                if st.checkbox(
                    f"**{preset['name']}**",
                    key=f"preset_{preset['id']}",
                    help=preset['description']
                ):
                    selected_presets.append(preset['id'])
        
        # Fetch button
        st.markdown("---")
        
        if selected_presets:
            st.success(f"Selected: {', '.join(selected_presets)}")
            
            if st.button("🔄 Fetch Data from Census", type="primary"):
                connector = CensusACSConnector(api_key=api_key if api_key else None)
                
                for preset_id in selected_presets:
                    preset_info = CENSUS_PRESETS.get(preset_id, {})
                    with st.spinner(f"Fetching {preset_info.get('name', preset_id)}..."):
                        try:
                            df, metadata = connector.fetch(
                                year=year,
                                preset=preset_id,
                                api_key=api_key if api_key else None
                            )
                            
                            # Store as if it were an uploaded file
                            source_name = f"census_{preset_id}_{year}.csv"
                            st.session_state.upload_files[source_name] = {
                                'content': df.to_csv(index=False).encode('utf-8'),
                                'type': 'csv',
                                'df': df,
                                'from_api': True,
                                'api_metadata': metadata
                            }
                            
                            st.success(f"✓ {preset_info.get('name', preset_id)}: {len(df):,} rows")
                            
                            # Show preview
                            with st.expander(f"Preview: {source_name}"):
                                st.dataframe(df.head(10), use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Failed to fetch {preset_id}: {e}")
        else:
            st.warning("Select at least one data category to fetch")
    
    elif api_type == "hud":
        st.info("""
        **HUD API** provides housing data including Fair Market Rents and Income Limits.
        
        ⚠️ **Requires free API token:** [Register here](https://www.huduser.gov/hudapi/public/register)
        """)
        
        # API Token (required)
        hud_token = st.text_input(
            "HUD API Token (required)",
            type="password",
            help="Get a free token at huduser.gov/hudapi/public/register"
        )
        
        if not hud_token:
            st.warning("⚠️ HUD API token is required. Register for free at the link above.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                hud_dataset = st.selectbox(
                    "Dataset",
                    options=["fmr", "il"],
                    format_func=lambda x: {
                        "fmr": "Fair Market Rents",
                        "il": "Income Limits"
                    }.get(x, x),
                    help="FMR: Rental housing costs | IL: Income thresholds for housing programs"
                )
                
                hud_year = st.selectbox(
                    "Fiscal Year",
                    options=list(range(2024, 2018, -1)),
                    index=0
                )
            
            with col2:
                # State selection
                state_options = list(US_STATES.keys())
                selected_states = st.multiselect(
                    "Select States",
                    options=state_options,
                    default=["CA"],
                    format_func=lambda x: f"{x} - {US_STATES.get(x, x)}",
                    help="Select one or more states to fetch data for"
                )
            
            if selected_states:
                st.info(f"Will fetch {hud_dataset.upper()} data for: {', '.join(selected_states)}")
                
                if st.button("🔄 Fetch HUD Data", type="primary"):
                    connector = HUDConnector(api_token=hud_token)
                    
                    with st.spinner(f"Fetching HUD {hud_dataset.upper()} data for {len(selected_states)} state(s)..."):
                        try:
                            df, metadata = connector.fetch(
                                api_token=hud_token,
                                dataset=hud_dataset,
                                year=hud_year,
                                states=selected_states
                            )
                            
                            # Store as data source
                            source_name = f"hud_{hud_dataset}_{hud_year}_{'_'.join(selected_states)}.csv"
                            st.session_state.upload_files[source_name] = {
                                'content': df.to_csv(index=False).encode('utf-8'),
                                'type': 'csv',
                                'df': df,
                                'from_api': True,
                                'api_metadata': metadata
                            }
                            
                            st.success(f"✓ Fetched {len(df):,} records")
                            
                            # Show preview
                            with st.expander(f"Preview: {source_name}", expanded=True):
                                st.write(f"**Columns:** {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
                                st.dataframe(df.head(20), use_container_width=True)
                                
                            if metadata.get("errors"):
                                st.warning(f"Some states had errors: {metadata['errors']}")
                                
                        except Exception as e:
                            st.error(f"Failed to fetch HUD data: {e}")
            else:
                st.warning("Select at least one state")
    
    elif api_type == "custom":
        st.warning("""
        **Custom API** allows connecting to any REST API that returns JSON data.
        This is an advanced feature for integrations like McKesson or other data providers.
        """)
        
        url = st.text_input(
            "API URL",
            placeholder="https://api.example.com/data",
            help="The endpoint URL to fetch data from"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("HTTP Method", ["GET", "POST"])
            auth_type = st.selectbox(
                "Authentication",
                ["none", "api_key_header", "api_key_param", "bearer"]
            )
        
        with col2:
            if auth_type != "none":
                auth_key = st.text_input("API Key/Token", type="password")
            else:
                auth_key = None
            
            zip_column = st.text_input(
                "ZIP Code Column",
                value="zip",
                help="Name of the column containing ZIP codes in the API response"
            )
        
        data_path = st.text_input(
            "JSON Data Path (optional)",
            placeholder="results.data",
            help="Dot-notation path to the data array in the JSON response"
        )
        
        if url:
            if st.button("🔄 Fetch Data", type="primary"):
                try:
                    connector = get_connector("custom")
                    with st.spinner("Fetching data..."):
                        df, metadata = connector.fetch(
                            url=url,
                            method=method,
                            auth_type=auth_type,
                            auth_key=auth_key,
                            data_path=data_path,
                            zip_column=zip_column
                        )
                        
                        # Store as if it were an uploaded file
                        source_name = f"api_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        st.session_state.upload_files[source_name] = {
                            'content': df.to_csv(index=False).encode('utf-8'),
                            'type': 'csv',
                            'df': df,
                            'from_api': True,
                            'api_metadata': metadata
                        }
                        
                        st.success(f"✓ Fetched {len(df):,} rows")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Failed to fetch: {e}")
    
    # Show all sources (files + API)
    if st.session_state.upload_files:
        st.markdown("---")
        st.markdown("### All Data Sources")
        for name, info in st.session_state.upload_files.items():
            source_type = "🌐 API" if info.get('from_api') else "📄 File"
            row_count = len(info['df']) if info.get('df') is not None else "?"
            st.write(f"- {source_type} **{name}** ({row_count} rows)")


def render_column_config_section():
    """Render the column configuration section (Step 2)."""
    st.header("⚙️ Step 2: Configure Columns")
    
    # Initialize file_configs for any files that don't have it
    for filename in st.session_state.upload_files:
        if filename not in st.session_state.file_configs:
            st.session_state.file_configs[filename] = {
                'zip_column': '',
                'norm_mode': 'already_5_digit',
                'features': [],
                'renames': {}
            }
    
    # Auto-load previews for files that haven't been parsed yet
    files_to_parse = []
    for filename, file_info in st.session_state.upload_files.items():
        if file_info.get('df') is None:
            files_to_parse.append(filename)
    
    if files_to_parse:
        with st.spinner(f"Loading {len(files_to_parse)} file(s)..."):
            for filename in files_to_parse:
                file_info = st.session_state.upload_files[filename]
                try:
                    df = parse_file(
                        filename,
                        file_info['content'],
                        file_type=file_info['type'],
                        sheet_name=file_info.get('sheet_name'),
                        skip_rows=file_info.get('skip_rows', 0)
                    )
                    st.session_state.upload_files[filename]['df'] = df
                except Exception as e:
                    st.error(f"Failed to parse {filename}: {e}")
    
    for filename, file_info in st.session_state.upload_files.items():
        if file_info.get('df') is None:
            st.warning(f"Could not load {filename}. Please go back and configure parsing options.")
            continue
        
        df = file_info['df']
        current_config = st.session_state.file_configs.get(filename, {})
        
        with st.expander(f"📄 {filename}", expanded=True):
            # Data Preview at the top
            st.markdown(f"**Data Preview** ({len(df):,} rows × {len(df.columns)} columns)")
            st.dataframe(df.head(30), use_container_width=True, height=200)
            
            st.markdown("---")
            st.markdown(f"**Available Columns ({len(df.columns)}):** `{', '.join(df.columns)}`")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # ZIP column selection
                detected_zip = detect_zip_column(df)
                default_zip = current_config.get('zip_column') or (detected_zip if detected_zip in df.columns else '')
                
                zip_options = [''] + list(df.columns)
                zip_index = zip_options.index(default_zip) if default_zip in zip_options else 0
                
                zip_column = st.selectbox(
                    "ZIP/ZCTA Column *",
                    options=zip_options,
                    index=zip_index,
                    key=f"zip_col_{filename}",
                    help="Select the column containing ZIP codes"
                )
                
                # Show sample values
                if zip_column:
                    sample_values = df[zip_column].dropna().head(5).tolist()
                    st.caption(f"Sample values: {sample_values}")
            
            with col2:
                # Normalization mode
                suggested_mode = suggest_normalization_mode(df, zip_column) if zip_column else "already_5_digit"
                default_mode = current_config.get('norm_mode', suggested_mode)
                mode_options = list(NORMALIZATION_MODES.keys())
                mode_index = mode_options.index(default_mode) if default_mode in mode_options else 0
                
                norm_mode = st.selectbox(
                    "ZIP Normalization Mode",
                    options=mode_options,
                    index=mode_index,
                    key=f"norm_mode_{filename}",
                    help="""
                    - **already_5_digit**: Values are already 5-digit ZIPs (handles leading zeros)
                    - **extract_5_digit_regex**: Extract ZIP from longer strings
                    - **zip_plus_4**: Handle ZIP+4 format (12345-6789)
                    """
                )
                
                # Preview normalization
                if zip_column and st.button("Preview Normalization", key=f"preview_norm_{filename}"):
                    normalized = normalize_zip_column(df, zip_column, norm_mode)
                    preview_df = pd.DataFrame({
                        'Original': df[zip_column].head(10),
                        'Normalized': normalized.head(10)
                    })
                    st.dataframe(preview_df)
            
            st.markdown("---")
            
            # Feature column selection
            st.markdown("**Select Feature Columns** *")
            
            # Multi-select for features (exclude ZIP column)
            non_zip_cols = [c for c in df.columns if c != zip_column]
            default_features = [f for f in current_config.get('features', []) if f in non_zip_cols]
            
            # Select All / Clear All buttons
            feat_col1, feat_col2, feat_col3 = st.columns([1, 1, 2])
            with feat_col1:
                if st.button("Select All", key=f"select_all_{filename}", use_container_width=True):
                    st.session_state[f"features_{filename}"] = non_zip_cols
                    st.rerun()
            with feat_col2:
                if st.button("Clear All", key=f"clear_all_{filename}", use_container_width=True):
                    st.session_state[f"features_{filename}"] = []
                    st.rerun()
            
            selected_features = st.multiselect(
                "Features to include",
                options=non_zip_cols,
                default=default_features,
                key=f"features_{filename}",
                help="Select columns you want to use as features in the dataset"
            )
            
            # Column renaming
            renames = {}
            if selected_features:
                st.markdown("**Rename Columns (optional)**")
                
                num_cols = min(3, len(selected_features))
                rename_cols = st.columns(num_cols)
                
                for i, col in enumerate(selected_features):
                    with rename_cols[i % num_cols]:
                        # Get previous rename if exists
                        prev_rename = current_config.get('renames', {}).get(col, col)
                        new_name = st.text_input(
                            f"'{col}' →",
                            value=prev_rename,
                            key=f"rename_{filename}_{col}"
                        )
                        if new_name and new_name != col:
                            renames[col] = new_name
                
                if renames:
                    st.info(f"Column renames: {renames}")
            
            # Store config immediately on any change
            st.session_state.file_configs[filename] = {
                'zip_column': zip_column,
                'norm_mode': norm_mode,
                'features': selected_features,
                'renames': renames
            }
            
            # Show current config status
            if zip_column and selected_features:
                st.success(f"✓ Configured: {len(selected_features)} features selected")
            else:
                missing = []
                if not zip_column:
                    missing.append("ZIP column")
                if not selected_features:
                    missing.append("feature columns")
                st.warning(f"⚠️ Please select: {', '.join(missing)}")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    def go_to_step_1():
        st.session_state.current_step = 1
    
    def go_to_step_3():
        st.session_state.current_step = 3
    
    with col1:
        st.button("← Back to Upload", use_container_width=True, on_click=go_to_step_1)
    
    with col3:
        # Validate configs
        all_valid = all(
            cfg.get('zip_column') and cfg.get('features')
            for cfg in st.session_state.file_configs.values()
            if cfg  # Skip empty configs
        )
        can_proceed = all_valid and st.session_state.file_configs
        
        st.button(
            "Next: Cleaning Rules →", 
            type="primary", 
            use_container_width=True,
            disabled=not can_proceed,
            on_click=go_to_step_3 if can_proceed else None
        )
        
        if not can_proceed:
            st.caption("⚠️ Select ZIP column and features for all files")


def render_cleaning_rules_section():
    """Render the cleaning rules section (Step 3)."""
    st.header("🧹 Step 3: Cleaning Rules")
    
    for filename, file_info in st.session_state.upload_files.items():
        if file_info.get('df') is None:
            continue
        
        df = file_info['df']
        file_config = st.session_state.file_configs.get(filename, {})
        features = file_config.get('features', [])
        
        with st.expander(f"📄 {filename}", expanded=True):
            st.markdown("**Drop rows where selected columns are NULL:**")
            
            all_cols = [file_config.get('zip_column')] + features
            all_cols = [c for c in all_cols if c]
            
            drop_null_cols = st.multiselect(
                "Columns for NULL check",
                options=all_cols,
                default=[file_config.get('zip_column')] if file_config.get('zip_column') else [],
                key=f"drop_null_{filename}",
                help="Rows with NULL values in these columns will be dropped"
            )
            
            # Show impact preview
            if drop_null_cols:
                null_counts = df[drop_null_cols].isnull().any(axis=1).sum()
                st.caption(f"This will drop {null_counts:,} rows with NULL values")
            
            # Store cleaning rules
            if 'cleaning_rules' not in st.session_state.file_configs[filename]:
                st.session_state.file_configs[filename]['cleaning_rules'] = {}
            st.session_state.file_configs[filename]['cleaning_rules']['drop_null_columns'] = drop_null_cols
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    def go_to_step_2():
        st.session_state.current_step = 2
    
    def go_to_step_4():
        st.session_state.current_step = 4
    
    with col1:
        st.button("← Back to Columns", use_container_width=True, on_click=go_to_step_2)
    with col3:
        st.button("Next: Scoring Config →", type="primary", use_container_width=True, on_click=go_to_step_4)


def render_scoring_config_section():
    """Render the scoring configuration section (Step 4)."""
    st.header("🎯 Step 4: Scoring Component Mapping")
    
    st.markdown("""
    Map your data columns to **scoring components**. The scoring model uses these 
    mappings to calculate the pharmacy desert score for each ZIP code.
    
    **All components are optional** - map whatever data you have available.
    The model will use whatever components are mapped and ignore the rest.
    """)
    
    # Collect all available columns across all files (with renames applied)
    available_columns = []
    for filename, file_config in st.session_state.file_configs.items():
        features = file_config.get('features', [])
        renames = file_config.get('renames', {})
        
        for col in features:
            # Use renamed name if exists
            display_name = renames.get(col, col)
            available_columns.append({
                'column': display_name,
                'original': col,
                'file': filename
            })
    
    # Also add 'zcta5' as the normalized ZIP column (always available)
    column_options = ['(not mapped)'] + [c['column'] for c in available_columns]
    
    if not available_columns:
        st.warning("No feature columns available. Go back and select features in Step 2.")
        
        def go_to_step_2():
            st.session_state.current_step = 2
        st.button("← Back to Columns", on_click=go_to_step_2)
        return
    
    # Show available columns
    with st.expander("📋 Available Columns from Your Data", expanded=False):
        cols_df = pd.DataFrame(available_columns)
        st.dataframe(cols_df, use_container_width=True)
    
    st.markdown("---")
    
    # All components by category (nothing is required)
    st.markdown("### Available Scoring Components")
    st.markdown("*Map your columns to include them in the scoring model.*")
    
    # Organize all components by category
    components_by_category = {}
    for comp_name in SCORED_COMPONENTS:
        comp = SCORING_COMPONENTS[comp_name]
        cat = comp.category
        if cat not in components_by_category:
            components_by_category[cat] = []
        components_by_category[cat].append((comp_name, comp))
    
    for category, components in components_by_category.items():
        category_label = COMPONENT_CATEGORIES.get(category, category.title())
        
        with st.expander(f"**{category_label}** ({len(components)} components)", expanded=False):
            opt_cols = st.columns(2)
            for i, (comp_name, comp) in enumerate(components):
                with opt_cols[i % 2]:
                    current = st.session_state.scoring_mappings.get(comp_name, '(not mapped)')
                    default_idx = column_options.index(current) if current in column_options else 0
                    
                    direction_icon = "📈" if comp.direction == ScoreDirection.HIGHER_IS_WORSE else "📉"
                    
                    selected = st.selectbox(
                        f"{direction_icon} {comp.display_name}",
                        options=column_options,
                        index=default_idx,
                        key=f"score_map_{comp_name}",
                        help=f"{comp.description}\n\nDefault weight: {comp.default_weight:.0%}"
                    )
                    
                    if selected != '(not mapped)':
                        st.session_state.scoring_mappings[comp_name] = selected
                    elif comp_name in st.session_state.scoring_mappings:
                        del st.session_state.scoring_mappings[comp_name]
    
    st.markdown("---")
    
    # Weight customization (optional)
    with st.expander("⚖️ Customize Default Weights (Advanced)", expanded=False):
        st.markdown("""
        Adjust the default weights for scoring components. These can also be 
        adjusted in the main app using the sidebar sliders.
        """)
        
        for comp_name in st.session_state.scoring_mappings:
            if comp_name in SCORING_COMPONENTS:
                comp = SCORING_COMPONENTS[comp_name]
                if comp.default_weight > 0:
                    current_weight = st.session_state.weight_overrides.get(
                        comp_name, comp.default_weight
                    )
                    new_weight = st.slider(
                        f"{comp.display_name} weight",
                        0.0, 1.0, current_weight, 0.05,
                        key=f"weight_{comp_name}"
                    )
                    if new_weight != comp.default_weight:
                        st.session_state.weight_overrides[comp_name] = new_weight
    
    # Summary of mappings
    st.markdown("---")
    st.markdown("### Mapping Summary")
    
    if st.session_state.scoring_mappings:
        mapping_data = []
        for comp_name, col_name in st.session_state.scoring_mappings.items():
            comp = SCORING_COMPONENTS.get(comp_name)
            if comp:
                mapping_data.append({
                    'Component': comp.display_name,
                    'Your Column': col_name,
                    'Direction': '↑ = worse' if comp.direction == ScoreDirection.HIGHER_IS_WORSE else '↑ = better',
                    'Weight': f"{st.session_state.weight_overrides.get(comp_name, comp.default_weight):.0%}"
                })
        
        st.dataframe(pd.DataFrame(mapping_data), use_container_width=True, hide_index=True)
    else:
        st.info("No mappings configured yet.")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    def go_to_step_3():
        st.session_state.current_step = 3
    
    def go_to_step_5():
        st.session_state.current_step = 5
    
    with col1:
        st.button("← Back to Cleaning", use_container_width=True, on_click=go_to_step_3)
    
    with col3:
        # No validation required - all components are optional
        mapped_count = len(st.session_state.scoring_mappings)
        st.button(
            "Next: Review & Submit →", 
            type="primary", 
            use_container_width=True,
            on_click=go_to_step_5
        )
        
        if mapped_count == 0:
            st.caption("💡 No mappings yet - you can still proceed")
        else:
            st.caption(f"✓ {mapped_count} component(s) mapped")


def render_review_submit_section():
    """Render the review and submit section (Step 5)."""
    st.header("✅ Step 5: Review & Submit")
    
    # Summary
    st.markdown("### Configuration Summary")
    
    st.markdown(f"**Dataset ID:** `{st.session_state.dataset_id}`")
    version_id = generate_version_id()
    st.markdown(f"**Version ID:** `{version_id}`")
    if st.session_state.version_description:
        st.markdown(f"**Description:** {st.session_state.version_description}")
    
    # File summaries
    sources = []
    for filename, file_info in st.session_state.upload_files.items():
        if file_info.get('df') is None:
            continue
        
        df = file_info['df']
        file_config = st.session_state.file_configs.get(filename, {})
        
        with st.expander(f"📄 {filename}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Rows:** {len(df):,}")
                st.markdown(f"**ZIP Column:** `{file_config.get('zip_column')}`")
                st.markdown(f"**Normalization:** `{file_config.get('norm_mode')}`")
            with col2:
                st.markdown(f"**Features:** {len(file_config.get('features', []))}")
                st.markdown(f"**Renames:** {len(file_config.get('renames', {}))}")
                drop_cols = file_config.get('cleaning_rules', {}).get('drop_null_columns', [])
                st.markdown(f"**NULL check columns:** {len(drop_cols)}")
        
        # Build mapping for this file
        mapping = build_file_mapping(
            filename=filename,
            file_type=file_info['type'],
            zip_column=file_config.get('zip_column', ''),
            normalization_mode=file_config.get('norm_mode', 'already_5_digit'),
            feature_columns=file_config.get('features', []),
            column_renames=file_config.get('renames', {}),
            cleaning_rules=file_config.get('cleaning_rules', {}),
            sheet_name=file_info.get('sheet_name'),
            skip_rows=file_info.get('skip_rows', 0)
        )
        sources.append(mapping)
    
    # Scoring config summary
    st.markdown("### Scoring Configuration")
    
    if st.session_state.scoring_mappings:
        mapping_count = len(st.session_state.scoring_mappings)
        required_count = len([c for c in st.session_state.scoring_mappings if c in REQUIRED_COMPONENTS])
        optional_count = mapping_count - required_count
        
        st.markdown(f"**Mapped components:** {mapping_count} ({required_count} required, {optional_count} optional)")
        
        # Build scoring config for storage
        scoring_config = ScoringConfig(
            column_mappings=[
                ColumnMapping(source_column=col, target_component=comp)
                for comp, col in st.session_state.scoring_mappings.items()
            ],
            weight_overrides=st.session_state.weight_overrides
        )
        
        with st.expander("View Scoring Configuration JSON"):
            st.json(scoring_config.to_dict())
    else:
        st.warning("No scoring configuration! The dataset won't have model mappings.")
        scoring_config = None
    
    # Submit button
    st.markdown("---")
    
    def go_to_step_4():
        st.session_state.current_step = 4
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("← Back to Scoring", use_container_width=True, on_click=go_to_step_4)
    
    with col3:
        if st.button("🚀 Upload Dataset", type="primary", use_container_width=True):
            try:
                with st.spinner("Uploading to storage..."):
                    storage = get_storage()
                    dataset_id = st.session_state.dataset_id
                    
                    progress = st.progress(0)
                    status = st.empty()
                    
                    total_files = len(st.session_state.upload_files)
                    uploaded_paths = []
                    
                    # Upload files and mappings
                    for i, (filename, file_info) in enumerate(st.session_state.upload_files.items()):
                        if file_info.get('df') is None:
                            continue
                        
                        status.text(f"Uploading {filename}...")
                        
                        # Upload original file
                        file_path = storage.upload_file(
                            dataset_id, version_id, filename, file_info['content']
                        )
                        uploaded_paths.append(file_path)
                        
                        # Upload mapping
                        file_config = st.session_state.file_configs.get(filename, {})
                        mapping = build_file_mapping(
                            filename=filename,
                            file_type=file_info['type'],
                            zip_column=file_config.get('zip_column', ''),
                            normalization_mode=file_config.get('norm_mode', 'already_5_digit'),
                            feature_columns=file_config.get('features', []),
                            column_renames=file_config.get('renames', {}),
                            cleaning_rules=file_config.get('cleaning_rules', {}),
                            sheet_name=file_info.get('sheet_name'),
                            skip_rows=file_info.get('skip_rows', 0)
                        )
                        storage.upload_mapping(dataset_id, version_id, filename, mapping)
                        
                        progress.progress((i + 1) / (total_files + 2))
                    
                    # Upload dataset config (includes scoring config)
                    status.text("Uploading dataset configuration...")
                    dataset_config = build_dataset_config(
                        dataset_id=dataset_id,
                        version_id=version_id,
                        sources=sources
                    )
                    
                    # Add version description
                    if st.session_state.version_description:
                        dataset_config['description'] = st.session_state.version_description
                    
                    # Add scoring config to dataset config
                    if scoring_config:
                        dataset_config['scoring_config'] = scoring_config.to_dict()
                    
                    storage.upload_config(dataset_id, version_id, dataset_config)
                    progress.progress((total_files + 1) / (total_files + 2))
                    
                    # Update LATEST.json
                    status.text("Updating LATEST pointer...")
                    storage.update_latest(dataset_id, version_id)
                    
                    progress.progress(1.0)
                    status.empty()
                    
                st.success(f"""
                ✅ **Dataset uploaded successfully!**
                
                - **Dataset ID:** `{dataset_id}`
                - **Version ID:** `{version_id}`
                - **Files uploaded:** {len(uploaded_paths)}
                - **Scoring components:** {len(st.session_state.scoring_mappings)}
                """)
                
                # Auto-trigger model training
                st.markdown("---")
                st.markdown("### 🤖 Model Training")
                
                train_automatically = st.checkbox(
                    "Automatically train model with new data",
                    value=True,
                    help="Triggers the ML training pipeline with the uploaded data"
                )
                
                if train_automatically:
                    try:
                        with st.spinner("Training model with new data..."):
                            from training.orchestrator import trigger_training
                            model_version = trigger_training(
                                dataset_id=dataset_id,
                                dataset_version=version_id
                            )
                            
                        if model_version:
                            st.success(f"""
                            🎉 **Model trained successfully!**
                            
                            - **Model Version:** `{model_version}`
                            - **Based on:** Dataset `{dataset_id}` v`{version_id}`
                            """)
                        else:
                            st.warning("Training completed but no model version returned. Check logs.")
                            
                    except Exception as train_error:
                        st.warning(f"""
                        ⚠️ **Training not available yet**
                        
                        The training script needs to be adapted for automated training.
                        Your ML partner can:
                        1. Check `training/orchestrator.py` for the integration points
                        2. Adapt `new_training.py` to read from TRAINING_CONFIG env var
                        
                        Error: {train_error}
                        """)
                else:
                    st.info("""
                    💡 **Manual training:** Your ML partner can train the model using:
                    ```bash
                    python new_training.py
                    ```
                    Or trigger training programmatically:
                    ```python
                    from training.orchestrator import trigger_training
                    trigger_training(dataset_id="{dataset_id}", dataset_version="{version_id}")
                    ```
                    """)
                
                # Show paths
                with st.expander("📂 Uploaded paths"):
                    for path in uploaded_paths:
                        st.code(path)
                
                # Reset button
                def reset_wizard():
                    st.session_state.upload_files = {}
                    st.session_state.file_configs = {}
                    st.session_state.dataset_id = ""
                    st.session_state.current_step = 1
                    st.session_state.scoring_mappings = {}
                    st.session_state.weight_overrides = {}
                    st.session_state.version_description = ""
                
                st.button("📤 Upload Another Dataset", on_click=reset_wizard)
                    
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_load_existing_config():
    """Render section to load existing configuration."""
    st.markdown("---")
    st.header("📂 Load Existing Configuration")
    
    storage = get_storage()
    
    # List datasets
    datasets = storage.list_datasets()
    
    if not datasets:
        st.info("No existing datasets found.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_dataset = st.selectbox(
            "Select Dataset",
            options=[''] + datasets,
            key="load_dataset_id"
        )
    
    with col2:
        if selected_dataset:
            versions = storage.list_versions(selected_dataset)
            latest = storage.get_latest_version(selected_dataset)
            
            # Mark latest version and include descriptions
            version_options = []
            for v in versions:
                config = storage.get_config(selected_dataset, v)
                desc = config.get('description', '') if config else ''
                if v == latest:
                    label = f"{v} (LATEST)" + (f" - {desc}" if desc else "")
                else:
                    label = v + (f" - {desc}" if desc else "")
                version_options.append((v, label))
            
            selected_version = st.selectbox(
                "Select Version",
                options=[v[0] for v in version_options],
                format_func=lambda x: next((v[1] for v in version_options if v[0] == x), x),
                key="load_version_id"
            )
    
    if selected_dataset and selected_version:
        if st.button("Load Configuration"):
            config_data = storage.get_config(selected_dataset, selected_version)
            
            if config_data:
                desc = config_data.get('description', '')
                st.success(f"Loaded configuration for {selected_dataset} v{selected_version}" + (f": {desc}" if desc else ""))
                
                # Show scoring config if present
                if 'scoring_config' in config_data:
                    st.markdown("### Scoring Configuration")
                    scoring = config_data['scoring_config']
                    
                    if 'column_mappings' in scoring:
                        st.markdown("**Column Mappings:**")
                        for m in scoring['column_mappings']:
                            comp = SCORING_COMPONENTS.get(m['target_component'], {})
                            comp_name = comp.display_name if hasattr(comp, 'display_name') else m['target_component']
                            st.write(f"- `{m['source_column']}` → **{comp_name}**")
                
                with st.expander("Full Configuration JSON"):
                    st.json(config_data)
            else:
                st.error("Configuration not found")


def main():
    """Main function to render the upload wizard."""
    init_session_state()
    
    st.title("📤 Dataset Upload Wizard")
    st.markdown("Upload and configure new datasets for the Pharmacy Desert analysis.")
    
    # Ensure current_step is an integer
    current_step = int(st.session_state.current_step)
    
    # Step indicator (now 5 steps)
    render_step_indicator(current_step, total_steps=5)
    st.markdown("---")
    
    # Render current step
    if current_step == 1:
        render_file_upload_section()
    elif current_step == 2:
        render_column_config_section()
    elif current_step == 3:
        render_cleaning_rules_section()
    elif current_step == 4:
        render_scoring_config_section()
    elif current_step == 5:
        render_review_submit_section()
    
    # Load existing config section (only on step 1 or 5 to avoid interference)
    if current_step in [1, 5]:
        try:
            render_load_existing_config()
        except Exception as e:
            st.warning(f"Could not load existing configs: {e}")


if __name__ == "__main__":
    main()
