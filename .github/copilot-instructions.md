# Copilot Instructions for Pharmacy Desert Explorer

## Architecture Overview
This is a **Streamlit-based geospatial analysis platform** for identifying "pharmacy deserts" - areas with limited pharmacy access. The system combines flexible data ingestion, mathematical scoring, and machine learning models.

### Core Data Flow
1. **Data Ingestion** (`ingestion/`) → Parse/normalize user uploads via config-driven mappings
2. **Storage** (`storage/`) → Version datasets in filesystem/S3 with JSON metadata
3. **Scoring** (`models/scoring.py`) → Flexible mathematical scoring using `ScoringConfig` schema
4. **Training** (`training/`) → GLM + ensemble models for pharmacy access predictions
5. **Visualization** (`app/`) → Streamlit interface with cached state management

## Key Patterns & Conventions

### Configuration-Driven Data Mapping
- **Schema**: `models/schema.py` defines `ScoringComponent` objects that map arbitrary column names to standardized scoring features
- **Example**: User CSV column "median_income" → maps to standard component "income" via `ColumnMapping`
- **Flexibility**: All components are optional; system works with partial data

### Dual-Mode Data Loading
- **GLM Mode**: Fast startup with pre-computed results (`load_glm_results()`)
- **Math/Blended Mode**: Full dataset loading with ACS API calls (`load_math_dataset_bundle()`)
- **State Management**: `app/state.py` uses `@st.cache_data` for expensive operations

### Dataset Versioning Pattern
```python
# Structure: raw_data/datasets/{dataset_id}/versions/{timestamp}/
{
  "dataset_id": "pharmacy_data",
  "version_id": "20260129_172450", 
  "sources": [{"filename": "...", "column_renames": {...}}]
}
```

### Environment-Aware Configuration
- **Local Dev**: Filesystem storage, mock data
- **Production**: S3 storage, AWS integration via `app/config.py`
- **Authentication**: Optional password protection via `REQUIRE_AUTH`

## Critical Commands & Workflows

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (loads from filesystem)
streamlit run app/app.py

# Train new model with dataset
python new_training.py
```

### Data Upload Workflow
1. Navigate to "Upload_Data" page in Streamlit
2. Upload CSV/Excel files → parsed by `ingestion/parsers.py`
3. Configure ZIP normalization modes (`ingestion/normalize.py`)
4. Map columns to scoring components via UI
5. System generates versioned dataset config + uploads to storage

### Model Training Pipeline
- **Trigger**: `training/orchestrator.py` → exports data → calls `new_training.py`
- **Algorithm**: Negative Binomial GLM with spatial features, cross-validated
- **Output**: GLM coefficients + IFAE scores saved to `results/`

## Integration Points

### External APIs
- **Census ACS**: `data/loaders.py:read_education_data_acs()` - educational attainment by ZIP
- **S3 Storage**: `storage/` handles versioned dataset uploads when `AWS_S3_BUCKET` set

### Scoring Components
- **Required Understanding**: `models/schema.py:SCORING_COMPONENTS` defines the "vocabulary" of features
- **Flexible Weighting**: Users can adjust component weights via Streamlit sliders
- **Direction Handling**: `ScoreDirection.HIGHER_IS_WORSE` vs `HIGHER_IS_BETTER`

### File Processing Chain
1. **Parse** (`ingestion/parsers.py`) → Detect file type, extract data
2. **Normalize** (`ingestion/normalize.py`) → Standardize ZIP codes to ZCTA5 format  
3. **Map** (`ingestion/dataset_loader.py`) → Apply column mappings per saved config
4. **Score** (`models/scoring.py:score_with_config()`) → Generate desert scores

## Common Debugging Patterns
- **Data Issues**: Check `raw_data/datasets/{id}/versions/{timestamp}/dataset_config.json` for mapping errors
- **Scoring Problems**: Verify component mappings in `models/schema.py` match uploaded column names
- **Performance**: Large datasets trigger full recomputation - use GLM mode for faster iteration
- **Authentication**: Set `REQUIRE_AUTH=false` in development to bypass login

## File Patterns to Follow
- **Module Imports**: Always add parent directory to `sys.path` for relative imports
- **Caching**: Use `@st.cache_data` decorator for expensive data operations  
- **Error Handling**: Log via `logging.getLogger(__name__)` for consistent debugging
- **Config Access**: Use `get_config()` instead of direct environment variable access