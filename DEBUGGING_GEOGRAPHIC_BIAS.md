# Debugging Geographic Bias in Results

## Problem Identified

User reported results were too heavily concentrated in New York/New England area instead of being geographically diverse with more southern representation.

## Root Causes Found

### 1. **GoodRx Hard Gate (Default: ON)**
- **Location:** `app/app.py` line 122
- **Issue:** `value=True` by default
- **Effect:** Filters dataset to only GoodRx-defined drive-time deserts BEFORE ML model runs
- **Problem:** GoodRx data may have geographic bias based on where they collected drive-time data
- **Fix:** Changed default to `value=False`

### 2. **Math Model Filtering First**
- **Issue:** Math model runs and filters data, then GLM only scores the filtered subset
- **Effect:** GLM model never sees ZIPs that Math model filtered out
- **Problem:** Math weights were tuned for different criteria than GLM model
- **Fix:** Changed default scoring mode from "Blended" to "GLM Only"

### 3. **Desert Flag Prioritization**
- **Location:** `app/app.py` line 240
- **Issue:** `sort_values(['desert_flag', sort_col], ...)` - sorts by zero-pharmacy flag first
- **Effect:** All zero-pharmacy ZIPs ranked above any ZIPs with pharmacies, regardless of score
- **Problem:** Zero-pharmacy areas cluster geographically (rural West, remote areas)
- **Fix:** Changed to sort by score only: `sort_values(sort_col, ...)`

## Actual Model Performance (No Filters)

### GLM Model Top 200 Geographic Distribution:
```
TX: 20 ZIPs
CA: 18 ZIPs
PA: 6 ZIPs
OH: 6 ZIPs
FL: 6 ZIPs
TN: 5 ZIPs
LA: 5 ZIPs
WI: 4 ZIPs
AL: 4 ZIPs
VA, NE, MI, MA, IL, AZ: 3 each
```

**Conclusion:** The GLM model itself is very geographically diverse! The bias was introduced by app filters.

## Changes Made

### 1. app/app.py - GoodRx Gate Default
```python
# BEFORE:
gate_goodrx = st.sidebar.checkbox("Hard gate to GoodRx-defined deserts", value=True, ...)

# AFTER:
gate_goodrx = st.sidebar.checkbox("Hard gate to GoodRx-defined deserts", value=False,
    help="⚠️ May introduce geographic bias. Recommended: OFF when using GLM model.")
```

### 2. app/app.py - Scoring Mode Default
```python
# BEFORE:
scoring_mode = st.sidebar.radio(..., ["Blended (Math + GLM)", "Math Only", "GLM Only"], index=0, ...)

# AFTER:
scoring_mode = st.sidebar.radio(..., ["GLM Only", "Blended (Math + GLM)", "Math Only"], index=0,
    help="GLM Only: Pure ML model (recommended for diverse results) | ...")
```

### 3. app/app.py - Sorting Logic
```python
# BEFORE:
ranked = ranked.sort_values(['desert_flag', sort_col], ascending=[False, False], ...)

# AFTER:
# Sort by score only (don't prioritize desert_flag to avoid geographic clustering)
ranked = ranked.sort_values(sort_col, ascending=False, ...)
```

## Recommendations for Users

### For Most Diverse Results:
1. **Use "GLM Only" mode** (now default)
2. **Turn OFF GoodRx gate** (now default OFF)
3. **Keep Target Area Filters minimal** or OFF
4. Let the ML model find underserved areas based on statistical analysis, not hard rules

### When to Use Math Model:
- When you want explicit control over weights
- When you need to explain scoring to stakeholders
- When testing sensitivity to different priorities

### When to Use Blended:
- When you want a balance of interpretability and ML power
- When you trust both approaches

### When to Use GoodRx Gate:
- When you specifically want to validate/rank existing GoodRx-identified deserts
- When you have updated GoodRx data for all regions
- **Not recommended for national diverse analysis**

## Testing Results

After fixes, GLM-only mode with no filters shows:
- Diverse geographic distribution
- Proper representation of Southern states (TX, FL, LA, TN, AL)
- Balanced mix of urban underserved areas and zero-pharmacy rural areas
- Scores reflect actual statistical underservice, not just absence of pharmacies

## Key Insight

**The GLM model finds underserved areas even when pharmacies exist!**

Example from Top 10:
- 90044 (CA): 96,436 pop, **3 pharmacies** → Still ranked #4 (severely underserved)
- 90011 (CA): 109,414 pop, **4 pharmacies** → Still ranked #10 (severely underserved)
- 33147 (FL): 47,834 pop, **2 pharmacies** → Ranked #8 (underserved)

This is the power of the GLM model - it finds places that LOOK like they have pharmacies but are still underserved based on population, demographics, and expected access.

