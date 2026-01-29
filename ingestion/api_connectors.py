"""
API Connectors for external data sources.

Provides pre-built connectors for:
- US Census Bureau (ACS 5-year estimates)
- Generic/Custom APIs (for future integrations like McKesson)

Each connector:
1. Fetches data from the API
2. Returns a pandas DataFrame with standardized columns
3. Stores configuration for reproducibility and refresh
"""
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Base Connector
# =============================================================================

@dataclass
class APISourceConfig:
    """Configuration for an API data source."""
    connector_type: str  # "census_acs", "custom", etc.
    name: str  # Human-readable name
    params: Dict[str, Any] = field(default_factory=dict)
    last_fetched: Optional[str] = None
    row_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "APISourceConfig":
        return cls(**d)


class BaseAPIConnector(ABC):
    """Base class for API connectors."""
    
    connector_type: str = "base"
    display_name: str = "Base Connector"
    description: str = "Base API connector"
    
    @abstractmethod
    def fetch(self, **params) -> pd.DataFrame:
        """Fetch data from the API and return as DataFrame."""
        pass
    
    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """Return schema for configuration parameters."""
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters. Returns list of error messages."""
        return []


# =============================================================================
# Census ACS Connector
# =============================================================================

# Pre-defined Census ACS variable sets
CENSUS_PRESETS = {
    "education": {
        "name": "Educational Attainment",
        "description": "Education levels for population 25+",
        "table": "S1501",
        "variables": {
            "S1501_C02_007E": "pct_less_9th_grade",
            "S1501_C02_008E": "pct_9to12_no_diploma",
            "S1501_C02_009E": "pct_hs_graduate",
            "S1501_C02_014E": "pct_hs_or_higher",
            "S1501_C02_015E": "pct_bachelors_or_higher",
        },
        "derived": {
            "edu_hs_or_lower_pct": "pct_less_9th_grade + pct_9to12_no_diploma + pct_hs_graduate",
            "edu_less_than_hs_pct": "100 - pct_hs_or_higher",
        }
    },
    "income": {
        "name": "Household Income",
        "description": "Median household income and poverty rates",
        "table": "S1901",
        "variables": {
            "S1901_C01_012E": "median_household_income",
        },
        "derived": {}
    },
    "poverty": {
        "name": "Poverty Status",
        "description": "Population below poverty level",
        "table": "S1701",
        "variables": {
            "S1701_C03_001E": "pct_below_poverty",
        },
        "derived": {}
    },
    "health_insurance": {
        "name": "Health Insurance Coverage",
        "description": "Uninsured population percentage",
        "table": "S2701",
        "variables": {
            "S2701_C05_001E": "pct_uninsured",
        },
        "derived": {}
    },
    "age_demographics": {
        "name": "Age Demographics",
        "description": "Population by age groups",
        "table": "S0101",
        "variables": {
            "S0101_C01_001E": "total_population",
            "S0101_C02_030E": "pct_65_and_over",
            "S0101_C02_026E": "pct_under_18",
        },
        "derived": {}
    },
    "disability": {
        "name": "Disability Status",
        "description": "Population with disabilities",
        "table": "S1810",
        "variables": {
            "S1810_C03_001E": "pct_with_disability",
        },
        "derived": {}
    },
}


class CensusACSConnector(BaseAPIConnector):
    """
    Connector for US Census Bureau American Community Survey (ACS) 5-year estimates.
    
    Supports:
    - Pre-defined presets (education, income, poverty, etc.)
    - Custom variable selection
    - Data by ZCTA (ZIP Code Tabulation Area)
    """
    
    connector_type = "census_acs"
    display_name = "US Census ACS"
    description = "American Community Survey 5-year estimates by ZIP/ZCTA"
    
    BASE_URL = "https://api.census.gov/data/{year}/acs/acs5/subject"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return schema for Census ACS configuration."""
        return {
            "year": {
                "type": "integer",
                "default": 2022,
                "min": 2015,
                "max": 2023,
                "description": "ACS 5-year data release year"
            },
            "preset": {
                "type": "select",
                "options": list(CENSUS_PRESETS.keys()),
                "default": "education",
                "description": "Pre-defined variable set"
            },
            "api_key": {
                "type": "string",
                "required": False,
                "description": "Census API key (optional but recommended)"
            },
            "custom_variables": {
                "type": "list",
                "required": False,
                "description": "Custom variable codes (advanced)"
            }
        }
    
    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate Census parameters."""
        errors = []
        
        year = params.get("year", 2022)
        if year < 2015 or year > 2023:
            errors.append(f"Year must be between 2015 and 2023, got {year}")
        
        preset = params.get("preset")
        custom_vars = params.get("custom_variables")
        
        if not preset and not custom_vars:
            errors.append("Must specify either a preset or custom variables")
        
        if preset and preset not in CENSUS_PRESETS:
            errors.append(f"Unknown preset: {preset}. Available: {list(CENSUS_PRESETS.keys())}")
        
        return errors
    
    def fetch(
        self,
        year: int = 2022,
        preset: Optional[str] = None,
        custom_variables: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        timeout: int = 120
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch Census ACS data.
        
        Args:
            year: ACS release year (e.g., 2022 for 2018-2022 5-year estimates)
            preset: Pre-defined variable set name
            custom_variables: Dict of {variable_code: column_name}
            api_key: Census API key
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        api_key = api_key or self.api_key
        
        # Determine variables to fetch
        if preset and preset in CENSUS_PRESETS:
            preset_config = CENSUS_PRESETS[preset]
            variables = preset_config["variables"]
            derived = preset_config.get("derived", {})
            source_name = preset_config["name"]
        elif custom_variables:
            variables = custom_variables
            derived = {}
            source_name = "Custom Census Variables"
        else:
            raise ValueError("Must specify either preset or custom_variables")
        
        # Build API request
        var_codes = ["NAME"] + list(variables.keys())
        url = self.BASE_URL.format(year=year)
        params = {
            "get": ",".join(var_codes),
            "for": "zip code tabulation area:*"
        }
        if api_key:
            params["key"] = api_key
        
        logger.info(f"Fetching Census ACS data: {url}")
        logger.info(f"Variables: {var_codes}")
        
        # Make request
        start_time = time.time()
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        fetch_time = time.time() - start_time
        
        # Parse response
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Rename columns
        df = df.rename(columns={"zip code tabulation area": "zcta5"})
        df = df.rename(columns=variables)
        
        # Convert numeric columns
        for col in variables.values():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Normalize ZCTA
        df["zcta5"] = df["zcta5"].astype(str).str.zfill(5)
        
        # Calculate derived columns
        for derived_col, formula in derived.items():
            try:
                # Simple formula evaluation (addition only for safety)
                if "+" in formula:
                    parts = [p.strip() for p in formula.split("+")]
                    df[derived_col] = sum(df[p] for p in parts if p in df.columns)
                elif "-" in formula and formula.startswith("100"):
                    # Handle "100 - column" pattern
                    parts = formula.split("-")
                    col_name = parts[1].strip()
                    if col_name in df.columns:
                        df[derived_col] = 100 - df[col_name]
            except Exception as e:
                logger.warning(f"Failed to compute derived column {derived_col}: {e}")
        
        # Keep only relevant columns
        keep_cols = ["zcta5"] + list(variables.values()) + list(derived.keys())
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]
        
        # Build metadata
        metadata = {
            "source_type": "api",
            "connector_type": self.connector_type,
            "source_name": source_name,
            "year": year,
            "preset": preset,
            "variables_fetched": list(variables.keys()),
            "row_count": len(df),
            "fetch_time_sec": round(fetch_time, 2),
            "fetched_at": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"Fetched {len(df)} rows in {fetch_time:.2f}s")
        
        return df, metadata


# =============================================================================
# Custom API Connector (for future APIs like McKesson)
# =============================================================================

class CustomAPIConnector(BaseAPIConnector):
    """
    Generic connector for custom REST APIs.
    
    Supports:
    - GET/POST requests
    - API key authentication (header or query param)
    - JSON response parsing
    - Configurable ZIP code column
    """
    
    connector_type = "custom"
    display_name = "Custom API"
    description = "Connect to any REST API that returns JSON data"
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "url": {
                "type": "string",
                "required": True,
                "description": "API endpoint URL"
            },
            "method": {
                "type": "select",
                "options": ["GET", "POST"],
                "default": "GET"
            },
            "auth_type": {
                "type": "select",
                "options": ["none", "api_key_header", "api_key_param", "bearer"],
                "default": "none"
            },
            "auth_key": {
                "type": "string",
                "required": False,
                "description": "API key or token"
            },
            "auth_header_name": {
                "type": "string",
                "default": "X-API-Key",
                "description": "Header name for API key"
            },
            "data_path": {
                "type": "string",
                "default": "",
                "description": "JSON path to data array (e.g., 'results.data')"
            },
            "zip_column": {
                "type": "string",
                "default": "zip",
                "description": "Column containing ZIP codes in response"
            }
        }
    
    def fetch(
        self,
        url: str,
        method: str = "GET",
        auth_type: str = "none",
        auth_key: Optional[str] = None,
        auth_header_name: str = "X-API-Key",
        data_path: str = "",
        zip_column: str = "zip",
        request_params: Optional[Dict] = None,
        request_body: Optional[Dict] = None,
        timeout: int = 60
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch data from a custom API.
        
        Args:
            url: API endpoint
            method: HTTP method (GET or POST)
            auth_type: Authentication type
            auth_key: API key or token
            auth_header_name: Header name for API key auth
            data_path: JSON path to data array (dot notation)
            zip_column: Column name containing ZIP codes
            request_params: Query parameters
            request_body: Request body for POST
            timeout: Request timeout
            
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        headers = {}
        params = request_params or {}
        
        # Set up authentication
        if auth_type == "api_key_header" and auth_key:
            headers[auth_header_name] = auth_key
        elif auth_type == "api_key_param" and auth_key:
            params["api_key"] = auth_key
        elif auth_type == "bearer" and auth_key:
            headers["Authorization"] = f"Bearer {auth_key}"
        
        # Make request
        logger.info(f"Fetching from custom API: {url}")
        start_time = time.time()
        
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, params=params, json=request_body, timeout=timeout)
        else:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
        
        response.raise_for_status()
        fetch_time = time.time() - start_time
        
        # Parse response
        data = response.json()
        
        # Navigate to data array if path specified
        if data_path:
            for key in data_path.split("."):
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    raise ValueError(f"Could not find '{key}' in response at path '{data_path}'")
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError(f"Expected list or dict, got {type(data)}")
        
        # Normalize ZIP column if present
        if zip_column in df.columns:
            df["zcta5"] = df[zip_column].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)
        
        metadata = {
            "source_type": "api",
            "connector_type": self.connector_type,
            "source_name": f"Custom API: {url}",
            "url": url,
            "row_count": len(df),
            "fetch_time_sec": round(fetch_time, 2),
            "fetched_at": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"Fetched {len(df)} rows in {fetch_time:.2f}s")
        
        return df, metadata


# =============================================================================
# HUD API Connector (Fair Market Rents, etc.)
# =============================================================================

# US State codes for HUD API
US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico",
}


class HUDConnector(BaseAPIConnector):
    """
    Connector for HUD (Housing and Urban Development) APIs.
    
    Requires free registration at: https://www.huduser.gov/hudapi/public/register
    
    Provides:
    - Fair Market Rents (FMR) by ZIP code
    - Small Area Fair Market Rents (SAFMR)
    - Income Limits
    
    Note: HUD API returns data by state, not bulk national data.
    """
    
    connector_type = "hud"
    display_name = "HUD (Housing & Urban Development)"
    description = "Fair Market Rents, Income Limits by ZIP code"
    
    BASE_URL = "https://www.huduser.gov/hudapi/public"
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "api_token": {
                "type": "string",
                "required": True,
                "description": "HUD API token (get free at huduser.gov/hudapi/public/register)"
            },
            "dataset": {
                "type": "select",
                "options": ["fmr", "il"],
                "default": "fmr",
                "description": "Dataset: Fair Market Rents (fmr) or Income Limits (il)"
            },
            "year": {
                "type": "integer",
                "default": 2024,
                "description": "Fiscal year for the data"
            },
            "states": {
                "type": "list",
                "required": True,
                "description": "List of state codes to fetch (e.g., ['CA', 'TX'])"
            }
        }
    
    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        errors = []
        if not params.get("api_token"):
            errors.append("HUD API token is required. Register free at huduser.gov/hudapi/public/register")
        if not params.get("states"):
            errors.append("Must specify at least one state code")
        return errors
    
    def _make_request(self, endpoint: str, token: str, timeout: int = 60) -> Dict:
        """Make authenticated request to HUD API."""
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        
        logger.info(f"HUD API request: {url}")
        response = requests.get(url, headers=headers, timeout=timeout)
        
        # Check for auth errors
        if response.status_code == 401:
            raise ValueError("Invalid HUD API token. Get a free token at huduser.gov/hudapi/public/register")
        elif response.status_code == 403:
            raise ValueError("HUD API access denied. Check your token permissions.")
        
        response.raise_for_status()
        
        # HUD API sometimes returns empty responses
        if not response.text:
            return {"data": []}
        
        return response.json()
    
    def fetch(
        self,
        api_token: str,
        dataset: str = "fmr",
        year: int = 2024,
        states: List[str] = None,
        timeout: int = 120
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch HUD data for specified states.
        
        Args:
            api_token: HUD API bearer token
            dataset: 'fmr' for Fair Market Rents, 'il' for Income Limits
            year: Fiscal year
            states: List of state codes (e.g., ['CA', 'NY'])
            timeout: Request timeout per state
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        if not api_token:
            raise ValueError("HUD API token required")
        
        states = states or ["CA"]  # Default to California for testing
        
        all_data = []
        errors = []
        start_time = time.time()
        
        for state_code in states:
            state_code = state_code.upper()
            if state_code not in US_STATES:
                errors.append(f"Unknown state code: {state_code}")
                continue
            
            try:
                # Endpoint varies by dataset
                if dataset == "fmr":
                    endpoint = f"fmr/statedata/{state_code}"
                elif dataset == "il":
                    endpoint = f"il/statedata/{state_code}"
                else:
                    raise ValueError(f"Unknown dataset: {dataset}")
                
                result = self._make_request(endpoint, api_token, timeout)
                
                # HUD returns data in different formats depending on endpoint
                if isinstance(result, dict):
                    if "data" in result:
                        data = result["data"]
                    else:
                        data = result
                elif isinstance(result, list):
                    data = result
                else:
                    continue
                
                # Handle nested data structure
                if isinstance(data, dict):
                    # FMR data is often nested by entity type
                    if "basicdata" in data:
                        records = data["basicdata"]
                        if isinstance(records, list):
                            for record in records:
                                record["state"] = state_code
                            all_data.extend(records)
                    elif "metroareas" in data or "counties" in data or "zipcodes" in data:
                        # Multiple entity types
                        for key in ["metroareas", "counties", "zipcodes"]:
                            if key in data and isinstance(data[key], list):
                                for record in data[key]:
                                    record["state"] = state_code
                                    record["entity_type"] = key
                                all_data.extend(data[key])
                    else:
                        data["state"] = state_code
                        all_data.append(data)
                elif isinstance(data, list):
                    for record in data:
                        if isinstance(record, dict):
                            record["state"] = state_code
                    all_data.extend(data)
                    
                logger.info(f"Fetched HUD {dataset} data for {state_code}: {len(data) if isinstance(data, list) else 1} records")
                
            except Exception as e:
                errors.append(f"{state_code}: {str(e)}")
                logger.error(f"Error fetching HUD data for {state_code}: {e}")
        
        fetch_time = time.time() - start_time
        
        if not all_data:
            error_msg = "; ".join(errors) if errors else "No data returned"
            raise ValueError(f"Failed to fetch HUD data: {error_msg}")
        
        df = pd.DataFrame(all_data)
        
        # Try to find and normalize ZIP column
        zip_columns = ["zip_code", "zipcode", "zip", "fips", "entity_id"]
        for col in zip_columns:
            if col in df.columns:
                df["zcta5"] = df[col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)
                break
        
        # Standardize some common column names
        rename_map = {
            "Efficiency": "fmr_efficiency",
            "One-Bedroom": "fmr_1br",
            "Two-Bedroom": "fmr_2br",
            "Three-Bedroom": "fmr_3br",
            "Four-Bedroom": "fmr_4br",
            "efficiency": "fmr_efficiency",
            "one_bedroom": "fmr_1br",
            "two_bedroom": "fmr_2br",
            "three_bedroom": "fmr_3br",
            "four_bedroom": "fmr_4br",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        metadata = {
            "source_type": "api",
            "connector_type": self.connector_type,
            "source_name": f"HUD {dataset.upper()} Data",
            "dataset": dataset,
            "year": year,
            "states": states,
            "row_count": len(df),
            "fetch_time_sec": round(fetch_time, 2),
            "fetched_at": datetime.utcnow().isoformat(),
            "errors": errors if errors else None,
        }
        
        logger.info(f"Fetched {len(df)} total HUD records in {fetch_time:.2f}s")
        
        return df, metadata


# =============================================================================
# Connector Registry
# =============================================================================

CONNECTORS = {
    "census_acs": CensusACSConnector,
    "hud": HUDConnector,
    "custom": CustomAPIConnector,
}


def get_connector(connector_type: str, **kwargs) -> BaseAPIConnector:
    """Get a connector instance by type."""
    if connector_type not in CONNECTORS:
        raise ValueError(f"Unknown connector type: {connector_type}. Available: {list(CONNECTORS.keys())}")
    return CONNECTORS[connector_type](**kwargs)


def list_connectors() -> List[Dict[str, str]]:
    """List available connectors with their info."""
    return [
        {
            "type": conn_cls.connector_type,
            "name": conn_cls.display_name,
            "description": conn_cls.description,
        }
        for conn_cls in CONNECTORS.values()
    ]


def list_census_presets() -> List[Dict[str, str]]:
    """List available Census ACS presets."""
    return [
        {
            "id": preset_id,
            "name": preset["name"],
            "description": preset["description"],
            "table": preset["table"],
            "variables": list(preset["variables"].keys()),
        }
        for preset_id, preset in CENSUS_PRESETS.items()
    ]

