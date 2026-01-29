# pharmacy_deserts/models/__init__.py
"""
Scoring and AI model modules for pharmacy desert analysis.
"""
from .scoring import (
    score_candidates, 
    score_with_config,
    average_scores, 
    export_math_scores_csv,
    get_available_weights_for_dataset,
)
from .ai_scores import read_ifae_csv
from .schema import (
    ScoringConfig,
    ScoringComponent,
    ScoreDirection,
    ColumnMapping,
    SCORING_COMPONENTS,
    REQUIRED_COMPONENTS,
    OPTIONAL_COMPONENTS,
    SCORED_COMPONENTS,
    COMPONENT_CATEGORIES,
    get_default_scoring_config,
    create_scoring_config_from_mappings,
)

__all__ = [
    # Scoring functions
    'score_candidates',
    'score_with_config',
    'average_scores',
    'export_math_scores_csv',
    'get_available_weights_for_dataset',
    'read_ifae_csv',
    # Schema classes
    'ScoringConfig',
    'ScoringComponent', 
    'ScoreDirection',
    'ColumnMapping',
    # Schema constants
    'SCORING_COMPONENTS',
    'REQUIRED_COMPONENTS',
    'OPTIONAL_COMPONENTS',
    'SCORED_COMPONENTS',
    'COMPONENT_CATEGORIES',
    # Config builders
    'get_default_scoring_config',
    'create_scoring_config_from_mappings',
]
