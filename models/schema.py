# models/schema.py
"""
Scoring schema definitions for flexible pharmacy desert analysis.

This module defines the standard scoring components that can be mapped
to arbitrary columns in uploaded datasets. Each component has:
- A standard name (used internally)
- A description (shown to users)
- A direction (higher_is_worse or higher_is_better)
- Whether it's required or optional
- A default weight
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ScoreDirection(Enum):
    """Direction for how a metric affects the final score."""
    HIGHER_IS_WORSE = "higher_is_worse"  # High values increase desert score
    HIGHER_IS_BETTER = "higher_is_better"  # High values decrease desert score (inverted)


@dataclass
class ScoringComponent:
    """Definition of a single scoring component."""
    name: str
    display_name: str
    description: str
    direction: ScoreDirection
    required: bool = False
    default_weight: float = 0.0
    category: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "direction": self.direction.value,
            "required": self.required,
            "default_weight": self.default_weight,
            "category": self.category,
        }


# =============================================================================
# STANDARD SCORING COMPONENTS
# =============================================================================
# These are the building blocks of the pharmacy desert score.
# Users map their data columns to these components.
# ALL COMPONENTS ARE OPTIONAL - the model works with whatever data is provided.

SCORING_COMPONENTS = {
    # ---------------------------------------------------------------------------
    # CORE METRICS (commonly used, but not required)
    # ---------------------------------------------------------------------------
    "population": ScoringComponent(
        name="population",
        display_name="Population",
        description="Total population in the ZIP/ZCTA area. Used to calculate per-capita metrics.",
        direction=ScoreDirection.HIGHER_IS_WORSE,  # More people = more need
        required=False,  # Optional - model works without it
        default_weight=0.10,
        category="demographics",
    ),
    
    "pharmacy_count": ScoringComponent(
        name="pharmacy_count",
        display_name="Pharmacy Count",
        description="Number of pharmacies in or serving this ZIP/ZCTA. Used for scarcity calculation.",
        direction=ScoreDirection.HIGHER_IS_BETTER,  # More pharmacies = less desert
        required=False,  # Optional - model works without it
        default_weight=0.20,
        category="access",
    ),
    
    # ---------------------------------------------------------------------------
    # VULNERABILITY INDICATORS
    # ---------------------------------------------------------------------------
    "income": ScoringComponent(
        name="income",
        display_name="Median Income",
        description="Median household income. Lower income areas have less pharmacy access options.",
        direction=ScoreDirection.HIGHER_IS_BETTER,  # Higher income = better access
        required=False,
        default_weight=0.20,
        category="socioeconomic",
    ),
    
    "health_burden": ScoringComponent(
        name="health_burden",
        display_name="Health Burden",
        description="Health burden metric (e.g., % in poor health, chronic disease prevalence). Higher = more need for pharmacies.",
        direction=ScoreDirection.HIGHER_IS_WORSE,
        required=False,
        default_weight=0.25,
        category="health",
    ),
    
    "pop_density": ScoringComponent(
        name="pop_density",
        display_name="Population Density",
        description="Population per square km/mile. Used to identify semi-urban areas with access gaps.",
        direction=ScoreDirection.HIGHER_IS_WORSE,  # Denser areas need more access points
        required=False,
        default_weight=0.10,
        category="demographics",
    ),
    
    # ---------------------------------------------------------------------------
    # OPTIONAL: Additional vulnerability factors
    # ---------------------------------------------------------------------------
    "education_low": ScoringComponent(
        name="education_low",
        display_name="Low Education Attainment",
        description="Percentage with high school education or lower. Correlates with health literacy barriers.",
        direction=ScoreDirection.HIGHER_IS_WORSE,
        required=False,
        default_weight=0.07,
        category="socioeconomic",
    ),
    
    "drive_time": ScoringComponent(
        name="drive_time",
        display_name="Drive Time to Pharmacy",
        description="Average/median drive time to nearest pharmacy in minutes.",
        direction=ScoreDirection.HIGHER_IS_WORSE,
        required=False,
        default_weight=0.15,
        category="access",
    ),
    
    "heat_vulnerability": ScoringComponent(
        name="heat_vulnerability",
        display_name="Heat Vulnerability",
        description="Heat health burden index. Heat-vulnerable populations need closer pharmacy access.",
        direction=ScoreDirection.HIGHER_IS_WORSE,
        required=False,
        default_weight=0.03,
        category="environmental",
    ),
    
    "age_elderly_pct": ScoringComponent(
        name="age_elderly_pct",
        display_name="Elderly Population %",
        description="Percentage of population 65+. Elderly have higher pharmacy needs.",
        direction=ScoreDirection.HIGHER_IS_WORSE,
        required=False,
        default_weight=0.05,
        category="demographics",
    ),
    
    "uninsured_pct": ScoringComponent(
        name="uninsured_pct",
        display_name="Uninsured Population %",
        description="Percentage without health insurance. May rely more on community pharmacies.",
        direction=ScoreDirection.HIGHER_IS_WORSE,
        required=False,
        default_weight=0.05,
        category="health",
    ),
    
    "vehicle_access": ScoringComponent(
        name="vehicle_access",
        display_name="Households Without Vehicle %",
        description="Percentage of households without vehicle access. Critical for pharmacy accessibility.",
        direction=ScoreDirection.HIGHER_IS_WORSE,
        required=False,
        default_weight=0.10,
        category="access",
    ),
    
    "chronic_disease_prevalence": ScoringComponent(
        name="chronic_disease_prevalence",
        display_name="Chronic Disease Prevalence",
        description="Rate of chronic conditions (diabetes, hypertension, etc.) requiring regular medication.",
        direction=ScoreDirection.HIGHER_IS_WORSE,
        required=False,
        default_weight=0.10,
        category="health",
    ),
    
    "pharmacy_closures": ScoringComponent(
        name="pharmacy_closures",
        display_name="Recent Pharmacy Closures",
        description="Number of pharmacy closures in past N years. Indicates declining access.",
        direction=ScoreDirection.HIGHER_IS_WORSE,
        required=False,
        default_weight=0.08,
        category="access",
    ),
    
    # ---------------------------------------------------------------------------
    # GEOGRAPHIC: Location data (not scored, but needed for mapping)
    # ---------------------------------------------------------------------------
    "latitude": ScoringComponent(
        name="latitude",
        display_name="Latitude",
        description="Geographic latitude for mapping. Not used in scoring.",
        direction=ScoreDirection.HIGHER_IS_WORSE,  # Not used
        required=False,
        default_weight=0.0,  # Not scored
        category="geography",
    ),
    
    "longitude": ScoringComponent(
        name="longitude",
        display_name="Longitude",
        description="Geographic longitude for mapping. Not used in scoring.",
        direction=ScoreDirection.HIGHER_IS_WORSE,  # Not used
        required=False,
        default_weight=0.0,  # Not scored
        category="geography",
    ),
}


# Convenience lists
# Note: All components are optional - the model works with whatever data is provided
REQUIRED_COMPONENTS = []  # Empty - nothing is required
OPTIONAL_COMPONENTS = [name for name, comp in SCORING_COMPONENTS.items()]  # All are optional
SCORED_COMPONENTS = [name for name, comp in SCORING_COMPONENTS.items() if comp.default_weight > 0]
GEOGRAPHIC_COMPONENTS = ["latitude", "longitude"]

# Categories for UI grouping
COMPONENT_CATEGORIES = {
    "demographics": "Demographics",
    "health": "Health Indicators", 
    "access": "Pharmacy Access",
    "socioeconomic": "Socioeconomic Factors",
    "environmental": "Environmental Factors",
    "geography": "Geographic Data",
}


@dataclass
class ColumnMapping:
    """Maps a user's data column to a scoring component."""
    source_column: str  # Column name in user's data
    target_component: str  # Standard component name
    transform: Optional[str] = None  # Optional transformation (e.g., "invert", "log")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_column": self.source_column,
            "target_component": self.target_component,
            "transform": self.transform,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColumnMapping":
        return cls(
            source_column=d["source_column"],
            target_component=d["target_component"],
            transform=d.get("transform"),
        )


@dataclass 
class ScoringConfig:
    """
    Complete scoring configuration for a dataset.
    
    This stores:
    - Column mappings (which user columns map to which components)
    - Weight overrides (custom weights for this dataset)
    - Any dataset-specific scoring parameters
    """
    column_mappings: List[ColumnMapping] = field(default_factory=list)
    weight_overrides: Dict[str, float] = field(default_factory=dict)
    desert_threshold: float = 2.0  # pharmacies per 10k for desert flag
    normalize_scores: bool = True
    
    def get_column_for_component(self, component_name: str) -> Optional[str]:
        """Get the source column mapped to a component."""
        for mapping in self.column_mappings:
            if mapping.target_component == component_name:
                return mapping.source_column
        return None
    
    def get_weight(self, component_name: str) -> float:
        """Get the weight for a component (override or default)."""
        if component_name in self.weight_overrides:
            return self.weight_overrides[component_name]
        if component_name in SCORING_COMPONENTS:
            return SCORING_COMPONENTS[component_name].default_weight
        return 0.0
    
    def get_mapped_components(self) -> List[str]:
        """Get list of component names that have mappings."""
        return [m.target_component for m in self.column_mappings]
    
    def validate(self) -> List[str]:
        """Validate the config and return list of warnings (no hard requirements)."""
        warnings = []
        
        # Check for unknown components (warning only)
        for mapping in self.column_mappings:
            if mapping.target_component not in SCORING_COMPONENTS:
                warnings.append(f"Unknown component: {mapping.target_component}")
        
        # Warn if no components are mapped
        if not self.column_mappings:
            warnings.append("No scoring components mapped - scores will be zero")
        
        return warnings
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_mappings": [m.to_dict() for m in self.column_mappings],
            "weight_overrides": self.weight_overrides,
            "desert_threshold": self.desert_threshold,
            "normalize_scores": self.normalize_scores,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScoringConfig":
        mappings = [ColumnMapping.from_dict(m) for m in d.get("column_mappings", [])]
        return cls(
            column_mappings=mappings,
            weight_overrides=d.get("weight_overrides", {}),
            desert_threshold=d.get("desert_threshold", 2.0),
            normalize_scores=d.get("normalize_scores", True),
        )


def get_default_scoring_config() -> ScoringConfig:
    """
    Get default scoring config that matches the original hardcoded behavior.
    
    This maps the original column names to the standard components.
    """
    return ScoringConfig(
        column_mappings=[
            ColumnMapping("population", "population"),
            ColumnMapping("n_pharmacies", "pharmacy_count"),
            ColumnMapping("median_income", "income"),
            ColumnMapping("health_burden", "health_burden"),
            ColumnMapping("pop_density", "pop_density"),
            ColumnMapping("edu_hs_or_lower_pct", "education_low"),
            ColumnMapping("zip_drive_time", "drive_time"),
            ColumnMapping("heat_hhb", "heat_vulnerability"),
            ColumnMapping("lat", "latitude"),
            ColumnMapping("lon", "longitude"),
        ],
        weight_overrides={},  # Use defaults
        desert_threshold=2.0,
    )


def create_scoring_config_from_mappings(
    user_mappings: Dict[str, str],  # source_col -> component_name
    weight_overrides: Optional[Dict[str, float]] = None
) -> ScoringConfig:
    """
    Create a ScoringConfig from a simple mapping dictionary.
    
    Args:
        user_mappings: Dict mapping user column names to component names
        weight_overrides: Optional weight overrides
    
    Returns:
        ScoringConfig instance
    """
    column_mappings = [
        ColumnMapping(source_column=src, target_component=target)
        for src, target in user_mappings.items()
    ]
    return ScoringConfig(
        column_mappings=column_mappings,
        weight_overrides=weight_overrides or {},
    )

