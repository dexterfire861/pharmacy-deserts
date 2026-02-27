# Platform Evaluation - Current State Assessment

## ✅ What's Working

### 1. Core Data Flow
- ✅ **Default Dataset Loading**: Always loads default dataset (financial, health, population, etc.)
- ✅ **Data Merging**: Uploaded data merges with default (outer join on zip/zcta5)
- ✅ **Dataset Versioning**: Timestamp-based versions with LATEST.json pointer
- ✅ **Source Accumulation**: New uploads add to existing dataset (doesn't replace)
- ✅ **Storage Backends**: Both local (development) and S3 (production) supported

### 2. Upload System
- ✅ **5-Step Upload Wizard**: File upload → Column config → Scoring config → Review & Submit
- ✅ **File Parsing**: Supports CSV, Excel, JSON, Parquet, ZIP
- ✅ **ZIP Normalization**: Multiple normalization modes
- ✅ **Metadata Column Handling**: Can mark columns as metadata (excluded from training)
- ✅ **Source ID Generation**: Stable IDs for tracking sources across versions

### 3. Scoring System
- ✅ **Feature-Based Scoring**: Weights apply directly to any numeric feature column
- ✅ **GLM Mode**: Loads pre-computed GLM results (fast startup)
- ✅ **Math Mode**: Full dataset with adjustable feature weights
- ✅ **Blended Mode**: Combines Math + GLM scores
- ✅ **Weight Sliders**: Dynamic UI showing all available features

### 4. Visualization
- ✅ **Interactive Map**: Folium-based map with markers
- ✅ **Top 10 Rankings**: Display top pharmacy deserts
- ✅ **Feature Display**: Shows all features in map popups

### 5. ML Training Pipeline
- ✅ **Training Orchestrator**: Exports data and triggers training
- ✅ **Model Registry**: Versioned model storage (local/S3)
- ✅ **Auto-Training**: Can trigger after data upload

---

## ⚠️ Potential Issues / Missing Pieces

### 1. Default Dataset Requirements
**Status**: ⚠️ **NEEDS VERIFICATION**

**Issue**: Default dataset requires specific files in `raw_data/`:
- `financial_data.csv`
- `health_data.csv`
- `population_data.csv`
- `HHI_data.xlsx`
- `zip_county_cross.xlsx`
- `driving-time-desert.csv`
- `Pharmacy_list_ZIP_fixed_final/` (directory)

**Question**: Do these files exist? If not, the app will show empty dataset.

**Location**: `app/state.py` line 547 checks for `raw_data/financial_data.csv`

---

### 2. GLM Results File
**Status**: ⚠️ **REQUIRED FOR GLM MODE**

**Issue**: GLM mode requires `results/national_ifae_rank.csv` (or S3 equivalent)

**Current Behavior**:
- If file missing → Shows error and redirects to Upload Data page
- App won't crash, but GLM mode won't work

**Location**: `app/state.py` line 307-310, `app/app.py` line 328-344

---

### 3. S3 Configuration
**Status**: ✅ **IMPLEMENTED** (needs environment setup)

**Requirements for Production**:
- `ENVIRONMENT=production` in `.env`
- `AWS_S3_BUCKET=<your-bucket>` in `.env`
- `AWS_REGION=<region>` in `.env` (defaults to us-east-1)
- AWS credentials configured (via IAM role, credentials file, or env vars)

**Location**: `app/config.py` lines 102-112

---

### 4. Map Rendering
**Status**: ✅ **FIXED** (duplicate column handling added)

**Recent Fix**: Added duplicate column cleanup to handle merged datasets
- Removes duplicate column names before processing
- Handles Series vs scalar values safely

**Location**: `viz/map_viz.py` lines 163-189

---

### 5. Feature Weight Sliders
**Status**: ✅ **IMPLEMENTED**

**Current Behavior**:
- Shows sliders for ALL numeric features in dataset
- Excludes: zip, zcta5, city, state, lat, lon, score, desert_flag
- Weights apply directly to features (no component mapping needed)

**Location**: `app/app.py` lines 49-115, `models/scoring.py` lines 55-88

---

## 🔍 What to Verify

### Critical Checks

1. **Default Data Files Exist?**
   ```bash
   ls raw_data/financial_data.csv
   ls raw_data/health_data.csv
   ls raw_data/population_data.csv
   # etc.
   ```
   - If missing → App will work but show empty dataset
   - Solution: Upload data via Upload Data page

2. **GLM Results Exist?**
   ```bash
   ls results/national_ifae_rank.csv
   ```
   - If missing → GLM mode won't work
   - Solution: Train model first (or upload data and trigger training)

3. **S3 Configuration (for production)**
   ```bash
   # Check .env file
   cat .env | grep -E "ENVIRONMENT|AWS_S3_BUCKET|AWS_REGION"
   ```
   - Should have: `ENVIRONMENT=production`, `AWS_S3_BUCKET=<bucket>`

4. **Data Loading Works?**
   - Open app → Select "Math Only" or "Blended" mode
   - Should load default dataset + any uploaded data
   - Check sidebar: Should show "X data source(s) loaded"

---

## 📋 Functional Checklist

### Data Upload Flow
- [ ] Can upload CSV/Excel files
- [ ] Can configure ZIP column and normalization
- [ ] Can select feature columns
- [ ] Can mark metadata columns
- [ ] Can map to scoring components (optional, legacy)
- [ ] Upload creates new version
- [ ] New version accumulates with previous data
- [ ] Files are stored correctly (local or S3)

### Data Loading Flow
- [ ] Default dataset loads automatically
- [ ] Uploaded data merges with default
- [ ] All features available in weight sliders
- [ ] Map displays with lat/lon data
- [ ] No duplicate column errors

### Scoring Flow
- [ ] Feature weight sliders appear for all numeric features
- [ ] Adjusting weights updates scores
- [ ] Math mode calculates scores correctly
- [ ] GLM mode loads pre-computed results
- [ ] Blended mode combines both

### Visualization Flow
- [ ] Map renders without errors
- [ ] Top 10 rankings display
- [ ] Map markers show correct locations
- [ ] Popups show feature data
- [ ] No TypeError on lat/lon access

### Production Readiness
- [ ] S3 storage works when `ENVIRONMENT=production`
- [ ] Default data loads from S3
- [ ] Uploaded data saves to S3
- [ ] GLM results load from S3
- [ ] Model training saves to S3

---

## 🚨 Known Issues / Edge Cases

### 0. Column Name Mismatch (FIXED)
**Issue**: `map_viz.py` was using `r['density']` but dataset uses `pop_density`
**Status**: ✅ **FIXED** - Now uses `r.get('pop_density', r.get('density', 0))`
**Location**: `viz/map_viz.py` line 317

### 1. Empty Default Dataset
**Scenario**: Default data files don't exist
**Impact**: App loads but shows empty dataset
**Workaround**: Upload data via Upload Data page
**Fix Needed**: Better error message / guidance

### 2. Missing GLM Results
**Scenario**: GLM results file doesn't exist
**Impact**: GLM mode shows error, redirects to Upload
**Workaround**: Use Math/Blended mode, or train model first
**Fix Needed**: Clearer messaging about training requirement

### 3. Duplicate Column Names
**Status**: ✅ **FIXED**
**Fix**: Added duplicate column cleanup in map_viz.py

### 4. Column Name Conflicts (default vs uploaded)
**Status**: ✅ **HANDLED**
**Behavior**: Uploaded data takes precedence (default columns dropped)
**Location**: `app/state.py` lines 607-617

---

## 🎯 What You Need to Test

### Test 1: Fresh Start (No Data)
1. Clear all uploaded data
2. Open app
3. **Expected**: Should load default dataset (if files exist) OR show empty state
4. **Check**: Can you see the map? Do feature sliders appear?

### Test 2: Upload First Dataset
1. Go to Upload Data page
2. Upload a CSV with ZIP codes and some features
3. Configure and submit
4. **Expected**: Creates version 1, merges with default
5. **Check**: Return to main app - do you see default + uploaded features?

### Test 3: Upload Additional Data
1. Upload another CSV (different features)
2. **Expected**: Creates version 2, accumulates with version 1 + default
3. **Check**: All features (default + both uploads) available?

### Test 4: Feature Weighting
1. Go to main app, Math/Blended mode
2. **Expected**: See weight sliders for all numeric features
3. Adjust weights
4. **Expected**: Scores update, rankings change
5. **Check**: Map updates with new rankings?

### Test 5: GLM Mode
1. Ensure `results/national_ifae_rank.csv` exists (or train model)
2. Select "GLM Only" mode
3. **Expected**: Fast load, shows GLM rankings
4. **Check**: Map displays correctly?

### Test 6: Production Mode (S3)
1. Set `ENVIRONMENT=production` and `AWS_S3_BUCKET=<bucket>` in `.env`
2. Restart app
3. **Expected**: Loads from S3, saves to S3
4. **Check**: Data persists correctly?

---

## 📊 Current Architecture Summary

```
User Opens App
  ↓
Authentication Check
  ↓
Mode Selection (GLM / Math / Blended)
  ↓
┌─────────────────┬──────────────────┐
│ GLM Mode        │ Math/Blended     │
│                 │                  │
│ Load GLM        │ Load Default     │
│ Results         │ Dataset          │
│ (fast)          │                  │
│                 │ + Merge Uploaded │
│                 │                  │
│ Show Rankings   │ Feature Weights  │
│                 │ → Score          │
│                 │                  │
│                 │ + GLM (if        │
│                 │   Blended)       │
└─────────────────┴──────────────────┘
  ↓
Display Map + Rankings
```

---

## 🔧 Quick Fixes Needed (If Issues Found)

### If Default Dataset Doesn't Load
**Check**: `raw_data/` directory has required files
**Fix**: Either add default files OR update code to handle missing files gracefully

### If GLM Mode Fails
**Check**: `results/national_ifae_rank.csv` exists
**Fix**: Train model OR provide better error message

### If S3 Doesn't Work
**Check**: 
- `.env` file has correct settings
- AWS credentials configured
- S3 bucket exists and is accessible
**Fix**: Verify S3 setup, check IAM permissions

### If Map Doesn't Render
**Check**: DataFrame has 'lat' and 'lon' columns
**Fix**: Ensure default dataset includes lat/lon OR uploaded data has them

---

## ✅ Summary: What's Functional

**Core Functionality**: ✅ **WORKING**
- Data upload and accumulation
- Default + uploaded data merging
- Feature-based scoring
- Map visualization
- GLM and Math modes
- S3 storage support

**What You Can Do Now**:
1. Upload data files → Creates versions, accumulates
2. View map with default + uploaded features
3. Adjust feature weights → See score changes
4. Use GLM mode (if results exist) OR Math/Blended mode
5. Deploy to production with S3

**What Might Need Attention**:
1. Verify default data files exist (or handle gracefully)
2. Verify GLM results exist (or train model)
3. Test S3 configuration for production
4. Test full upload → merge → scoring → visualization flow

---

## 🎯 Next Steps to Verify Functionality

1. **Test the happy path**: Upload data → See map → Adjust weights → View results
2. **Test edge cases**: Missing default data, missing GLM results, duplicate columns
3. **Test production**: S3 configuration, data persistence
4. **Identify blockers**: What prevents you from using it as intended?
