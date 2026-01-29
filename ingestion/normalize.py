"""
ZIP code normalization utilities.
Handles various ZIP code formats and normalizes them to 5-digit strings.
"""
import re
import pandas as pd
from typing import Optional


def normalize_zip_already_5_digit(value) -> Optional[str]:
    """
    Normalize a value that is already a 5-digit ZIP code.
    Handles numeric values and strings, pads with leading zeros if needed.
    
    Args:
        value: ZIP code value (int, float, or string)
    
    Returns:
        5-digit ZIP string or None if invalid
    """
    if pd.isna(value):
        return None
    
    # Convert to string
    s = str(value).strip()
    
    # Remove any decimal points (e.g., "12345.0" -> "12345")
    if '.' in s:
        s = s.split('.')[0]
    
    # Check if it's a valid number
    if not s.isdigit():
        return None
    
    # Pad with leading zeros to 5 digits
    s = s.zfill(5)
    
    # Validate length
    if len(s) != 5:
        return None
    
    return s


def normalize_zip_extract_regex(value) -> Optional[str]:
    """
    Extract a 5-digit ZIP code from a string using regex.
    Useful for values like "ZIP: 12345" or "12345-6789".
    
    Args:
        value: String potentially containing a ZIP code
    
    Returns:
        5-digit ZIP string or None if not found
    """
    if pd.isna(value):
        return None
    
    s = str(value).strip()
    
    # Look for 5-digit sequence
    match = re.search(r'(\d{5})', s)
    if match:
        return match.group(1)
    
    return None


def normalize_zip_plus_4(value) -> Optional[str]:
    """
    Normalize a ZIP+4 format (e.g., "12345-6789") to just 5 digits.
    
    Args:
        value: ZIP+4 value
    
    Returns:
        5-digit ZIP string or None if invalid
    """
    if pd.isna(value):
        return None
    
    s = str(value).strip()
    
    # Match ZIP+4 pattern
    match = re.match(r'^(\d{5})[-\s]?\d{4}$', s)
    if match:
        return match.group(1)
    
    # Fall back to extract_regex if not ZIP+4 format
    return normalize_zip_extract_regex(s)


# Mapping of normalization mode names to functions
NORMALIZATION_MODES = {
    "already_5_digit": normalize_zip_already_5_digit,
    "extract_5_digit_regex": normalize_zip_extract_regex,
    "zip_plus_4": normalize_zip_plus_4,
}


def normalize_zip_column(df: pd.DataFrame, column: str, mode: str) -> pd.Series:
    """
    Normalize an entire column of ZIP codes.
    
    Args:
        df: DataFrame containing the column
        column: Column name to normalize
        mode: Normalization mode (one of NORMALIZATION_MODES keys)
    
    Returns:
        Series of normalized ZIP codes
    """
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode: {mode}. "
                        f"Valid modes: {list(NORMALIZATION_MODES.keys())}")
    
    normalize_func = NORMALIZATION_MODES[mode]
    return df[column].apply(normalize_func)


def detect_zip_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect the ZIP code column in a DataFrame.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Column name if found, None otherwise
    """
    # Common ZIP column names (case-insensitive)
    zip_patterns = [
        r'^zip$', r'^zipcode$', r'^zip_code$', r'^zip code$',
        r'^zcta$', r'^zcta5$', r'^postal$', r'^postalcode$',
        r'^postal_code$', r'^postal code$'
    ]
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for pattern in zip_patterns:
            if re.match(pattern, col_lower):
                return col
    
    return None


def suggest_normalization_mode(df: pd.DataFrame, column: str) -> str:
    """
    Analyze a column and suggest the best normalization mode.
    
    Args:
        df: DataFrame containing the column
        column: Column name to analyze
    
    Returns:
        Suggested normalization mode
    """
    sample = df[column].dropna().head(100)
    
    if sample.empty:
        return "already_5_digit"
    
    # Check for ZIP+4 patterns
    zip_plus_4_count = sample.astype(str).str.match(r'^\d{5}[-\s]?\d{4}$').sum()
    if zip_plus_4_count > len(sample) * 0.5:
        return "zip_plus_4"
    
    # Check for pure 5-digit (possibly with leading zeros removed)
    five_digit_count = sample.astype(str).str.match(r'^\d{1,5}(\.0)?$').sum()
    if five_digit_count > len(sample) * 0.7:
        return "already_5_digit"
    
    # Default to regex extraction
    return "extract_5_digit_regex"

