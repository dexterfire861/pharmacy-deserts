# pharmacy_deserts/models/scoring.py
"""
Flexible mathematical scoring functions for pharmacy desert analysis.

This module provides both:
1. The new flexible scoring system (score_with_config) that works with any column names
2. The legacy scoring function (score_candidates) for backwards compatibility
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from utils.cache import cache_data
from data.features import norm01
from models.schema import (
    ScoringConfig, ScoringComponent, ScoreDirection,
    SCORING_COMPONENTS, get_default_scoring_config
)


def export_math_scores_csv(ranked_df: pd.DataFrame, path: str = 'raw_data/math_scores.csv') -> pd.DataFrame:
    """Export mathematical scores to CSV."""
    out = ranked_df[['zip', 'score']].rename(columns={'score': 'score_math'}).copy()
    out.to_csv(path, index=False)
    return out


@cache_data
def average_scores(math_df: pd.DataFrame, ai_df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """
    Blend mathematical and AI scores.

    Args:
        math_df: DataFrame with columns [zip, score_math]
        ai_df: DataFrame with columns [zip, ai_score]
        normalize: Whether to normalize before averaging

    Returns:
        DataFrame with columns [zip, final_score, score_math, ai_score]
    """
    merged = pd.merge(math_df, ai_df, on='zip', how='outer')
    if normalize:
        merged['math_n'] = norm01(merged['score_math'])
        merged['ai_n'] = norm01(merged['ai_score'])
        merged['final_score'] = merged[['math_n', 'ai_n']].mean(axis=1, skipna=True)
    else:
        merged['final_score'] = merged[['score_math', 'ai_score']].mean(axis=1, skipna=True)
    merged['final_score'] = merged['final_score'].fillna(0)
    return merged[['zip', 'final_score', 'score_math', 'ai_score']]


# =============================================================================
# NEW FLEXIBLE SCORING SYSTEM
# =============================================================================

def score_with_config(
    df: pd.DataFrame,
    config: ScoringConfig,
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Apply scoring using a flexible configuration.
    
    This is the new scoring function that can work with ANY column names.
    Users map their columns to standard scoring components via the config.
    
    Args:
        df: DataFrame with user's data (any column names)
        config: ScoringConfig with column mappings
        weights: Optional runtime weight overrides (from UI sliders)
    
    Returns:
        Scored DataFrame with 'score' and 'desert_flag' columns added
    """
    df = df.copy()
    
    # Build a lookup: component_name -> source_column
    col_map = {m.target_component: m.source_column for m in config.column_mappings}
    
    # Helper to get column value (returns None if not mapped)
    def get_col(component: str) -> Optional[pd.Series]:
        if component not in col_map:
            return None
        src_col = col_map[component]
        if src_col not in df.columns:
            return None
        return pd.to_numeric(df[src_col], errors='coerce')
    
    # Get effective weights (runtime override > config override > default)
    def get_weight(component: str) -> float:
        if weights and component in weights:
            return weights[component]
        return config.get_weight(component)
    
    # Track which components contribute to score
    score_components: List[Tuple[str, pd.Series, float]] = []
    
    # ---------------------------------------------------------------------------
    # SCARCITY SCORE (special handling - requires both pharmacy_count and population)
    # ---------------------------------------------------------------------------
    pharmacy_count = get_col("pharmacy_count")
    population = get_col("population")
    
    if pharmacy_count is not None and population is not None:
        # Calculate pharmacies per 10k population
        df['_pharmacies_per_10k'] = (pharmacy_count.fillna(0) / (population.fillna(0) + 1)) * 10000
        
        # Scarcity = inverse of pharmacies per 10k (fewer pharmacies = higher scarcity)
        df['_scarcity'] = 1 / (1 + df['_pharmacies_per_10k'])
        df['_scarcity_n'] = norm01(df['_scarcity'])
        
        w = get_weight("pharmacy_count")
        if w > 0:
            score_components.append(("pharmacy_count", df['_scarcity_n'], w))
        
        # Desert flag based on threshold
        df['pharm_per_10k'] = df['_pharmacies_per_10k']
        df['desert_flag'] = df['pharm_per_10k'] < config.desert_threshold
    else:
        df['pharm_per_10k'] = np.nan
        df['desert_flag'] = False
    
    # ---------------------------------------------------------------------------
    # STANDARD COMPONENTS (direct normalization based on direction)
    # ---------------------------------------------------------------------------
    standard_components = [
        "income", "health_burden", "pop_density", "education_low",
        "drive_time", "heat_vulnerability", "age_elderly_pct",
        "uninsured_pct", "vehicle_access", "chronic_disease_prevalence",
        "pharmacy_closures"
    ]
    
    for comp_name in standard_components:
        values = get_col(comp_name)
        if values is None:
            continue
            
        w = get_weight(comp_name)
        if w <= 0:
            continue
        
        comp_def = SCORING_COMPONENTS.get(comp_name)
        if comp_def is None:
            continue
        
        # Normalize to [0, 1]
        normalized = norm01(values)
        
        # If higher_is_better, invert so higher original = lower score contribution
        if comp_def.direction == ScoreDirection.HIGHER_IS_BETTER:
            normalized = 1 - normalized
        
        # Store for scoring
        score_col_name = f'_{comp_name}_n'
        df[score_col_name] = normalized.fillna(0)
        score_components.append((comp_name, df[score_col_name], w))
    
    # ---------------------------------------------------------------------------
    # POPULATION COMPONENT (special: higher population = more need)
    # ---------------------------------------------------------------------------
    pop_density = get_col("pop_density")
    if pop_density is not None:
        w = get_weight("pop_density")
        if w > 0 and "pop_density" not in [c[0] for c in score_components]:
            df['_pop_n'] = norm01(pop_density).fillna(0)
            score_components.append(("pop_density", df['_pop_n'], w))
    
    # Also handle raw population if mapped separately
    if population is not None:
        w = get_weight("population")
        if w > 0 and "population" not in [c[0] for c in score_components]:
            df['_population_n'] = norm01(population).fillna(0)
            score_components.append(("population", df['_population_n'], w))
    
    # ---------------------------------------------------------------------------
    # COMPUTE FINAL SCORE
    # ---------------------------------------------------------------------------
    if score_components:
        # Normalize weights to sum to 1
        total_weight = sum(w for _, _, w in score_components)
        
        if total_weight > 0:
            # Weighted sum
            df['score'] = sum(
                (w / total_weight) * series.fillna(0)
                for _, series, w in score_components
            )
        else:
            df['score'] = 0.0
        
        # Normalize final score to [0, 1]
        if config.normalize_scores:
            smin, smax = df['score'].min(), df['score'].max()
            if smax > smin:
                df['score'] = (df['score'] - smin) / (smax - smin)
    else:
        df['score'] = 0.0
    
    # Sort by desert flag (True first) then score (highest first)
    df = df.sort_values(['desert_flag', 'score'], ascending=[False, False])
    
    return df


def get_available_weights_for_dataset(
    df: pd.DataFrame,
    config: ScoringConfig
) -> Dict[str, Tuple[str, float]]:
    """
    Get the scoring components available for a dataset based on its config.
    
    Returns dict of component_name -> (display_name, default_weight)
    Used by the UI to show only relevant weight sliders.
    """
    col_map = {m.target_component: m.source_column for m in config.column_mappings}
    available = {}
    
    for comp_name, src_col in col_map.items():
        if src_col in df.columns and comp_name in SCORING_COMPONENTS:
            comp = SCORING_COMPONENTS[comp_name]
            if comp.default_weight > 0:  # Only scorable components
                available[comp_name] = (comp.display_name, config.get_weight(comp_name))
    
    return available


# =============================================================================
# LEGACY SCORING FUNCTION (backwards compatibility)
# =============================================================================

def score_candidates(
    df: pd.DataFrame,
    w_scarcity: float,
    w_health: float,
    w_income: float,
    w_pop: float,
    w_heat: float = 0.0,
    w_edu: float = 0.0,
    w_drive_time: float = 0.0
) -> pd.DataFrame:
    """
    Apply mathematical scoring model (LEGACY - for backwards compatibility).
    
    This function expects the original hardcoded column names.
    For new datasets with different columns, use score_with_config().

    Args:
        df: Preprocessed DataFrame with standard column names
        w_scarcity: Weight for pharmacy scarcity
        w_health: Weight for health burden
        w_income: Weight for income (inverted)
        w_pop: Weight for population density
        w_heat: Weight for heat vulnerability
        w_edu: Weight for education (low attainment)
        w_drive_time: Weight for driving time to nearest pharmacy

    Returns:
        Scored and sorted DataFrame
    """
    # Use the flexible system with default config and explicit weights
    config = get_default_scoring_config()
    
    weights = {
        "pharmacy_count": w_scarcity,
        "health_burden": w_health,
        "income": w_income,
        "pop_density": w_pop,
        "heat_vulnerability": w_heat,
        "education_low": w_edu,
        "drive_time": w_drive_time,
    }
    
    return score_with_config(df, config, weights)


def score_candidates_legacy(
    df: pd.DataFrame,
    w_scarcity: float,
    w_health: float,
    w_income: float,
    w_pop: float,
    w_heat: float = 0.0,
    w_edu: float = 0.0,
    w_drive_time: float = 0.0
) -> pd.DataFrame:
    """
    Original scoring implementation (preserved for reference/testing).
    
    This is the exact original logic, kept for comparison.
    """
    df = df.copy()
    df['median_income'] = pd.to_numeric(df['median_income'], errors='coerce')
    df['health_burden'] = pd.to_numeric(df['health_burden'], errors='coerce')
    df['pop_density'] = pd.to_numeric(df['pop_density'], errors='coerce').fillna(0)

    # Scarcity
    df['pharmacies_per_10k'] = (df['n_pharmacies'] / (df['population'] + 1)) * 10000
    df['scarcity'] = 1 / (1 + df['pharmacies_per_10k'])
    df['scarcity_n'] = norm01(df['scarcity'])

    # Health
    df['health_n'] = norm01(df['health_burden'])

    # Income
    df['income_inv'] = 1 - norm01(df['median_income'])
    df['pop_norm'] = norm01(df['pop_density'])

    if 'edu_hs_or_lower_pct' in df.columns:
        df['edu_low_norm'] = norm01(df['edu_hs_or_lower_pct'])
    else:
        df['edu_low_norm'] = 0.0
        w_edu = 0.0

    if 'zip_drive_time' in df.columns and df['zip_drive_time'].notna().any():
        df['drive_time_norm'] = norm01(df['zip_drive_time'])
        df['drive_time_norm'] = df['drive_time_norm'].fillna(df['drive_time_norm'].median(skipna=True))
    else:
        df['drive_time_norm'] = 0.0
        w_drive_time = 0.0

    if 'heat_hhb' in df.columns:
        df['heat_norm'] = norm01(df['heat_hhb'])
    else:
        df['heat_norm'] = 0.0
        w_heat = 0.0

    drive_time_score = df['drive_time_norm'].fillna(0) if w_drive_time > 0 else 0
    scarcity_score = df['scarcity_n'].fillna(0) if w_scarcity > 0 else 0
    health_score = df['health_n'].fillna(0) if w_health > 0 else 0
    income_score = df['income_inv'].fillna(0) if w_income > 0 else 0
    pop_score = df['pop_norm'].fillna(0) if w_pop > 0 else 0
    heat_score = df['heat_norm'] if w_heat > 0 else 0
    edu_score = df['edu_low_norm'].fillna(0) if w_edu > 0 else 0

    df['score'] = (w_drive_time * drive_time_score + w_scarcity * scarcity_score + w_health * health_score +
                   w_income * income_score + w_pop * pop_score + w_heat * heat_score + w_edu * edu_score)

    smin, smax = df['score'].min(), df['score'].max()
    if smax > smin:
        df['score'] = (df['score'] - smin) / (smax - smin)

    df['pharm_per_10k'] = (df['n_pharmacies'] / (df['population'] + 1)) * 10000
    df['desert_flag'] = df['pharm_per_10k'] < 2.0
    
    return df.sort_values(['desert_flag', 'score'], ascending=[False, False])
