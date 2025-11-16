import pandas as pd
from pathlib import Path

# Try to import Streamlit cache, fall back to simple caching if not available
try:
    from streamlit import cache_data
except ImportError:
    def cache_data(func):
        """Fallback cache decorator if Streamlit not available"""
        _cache = {}
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in _cache:
                _cache[key] = func(*args, **kwargs)
            return _cache[key]
        return wrapper

@cache_data
def _load_data():
    """Helper function to load the health data Excel file once with caching."""
    # Try multiple possible paths
    possible_paths = [
        Path('data/health_data_updated.xlsm'),
        Path('health_data_updated.xlsm'),
        Path(__file__).parent / 'health_data_updated.xlsm',
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Loading health data from: {path}")
            df = pd.read_excel(path, sheet_name='health_data')
            print(f"Loaded {len(df):,} rows of health data")
            return df
    
    # If no file found, return empty dataframe
    print("Warning: health_data_updated.xlsm not found")
    return pd.DataFrame(columns=['Short_ZIP', 'Grand_total_with_cancer', 
                                  'Grand_total_without_cancer', 
                                  'Grand_total_without_cancer_insurace_paying'])


def get_health_stats(zip_code):
    """
    Get all health statistics for a given zip code.
    
    Args:
        zip_code: The zip code to search for (can be string or int)
    
    Returns:
        Dictionary with all health stats, or None if zip not found
    """
    df = _load_data()
    
    if df.empty:
        return None
    
    # Convert zip_code to match the format in the dataset
    zip_code = str(zip_code).strip().zfill(5)
    
    # Find matching row
    matching_rows = df[df['Short_ZIP'].astype(str).str.zfill(5) == zip_code]
    
    if not matching_rows.empty:
        row = matching_rows.iloc[0]
        return {
            'with_cancer': row.get('Grand_total_with_cancer'),
            'without_cancer': row.get('Grand_total_without_cancer'),
            'without_cancer_insurance': row.get('Grand_total_without_cancer_insurace_paying')
        }
    else:
        return None


def get_grand_total_with_cancer(zip_code):
    """
    Get the Grand_total_with_cancer value for a given zip code.
    
    Args:
        zip_code: The zip code to search for (can be string or int)
    
    Returns:
        The Grand_total_with_cancer value if found, None otherwise
    """
    stats = get_health_stats(zip_code)
    return stats['with_cancer'] if stats else None


def get_grand_total_without_cancer(zip_code):
    """
    Get the Grand_total_without_cancer value for a given zip code.
    
    Args:
        zip_code: The zip code to search for (can be string or int)
    
    Returns:
        The Grand_total_without_cancer value if found, None otherwise
    """
    stats = get_health_stats(zip_code)
    return stats['without_cancer'] if stats else None


def get_grand_total_without_cancer_insurance_paying(zip_code):
    """
    Get the Grand_total_without_cancer_insurace_paying value for a given zip code.
    
    Args:
        zip_code: The zip code to search for (can be string or int)
    
    Returns:
        The Grand_total_without_cancer_insurace_paying value if found, None otherwise
    """
    stats = get_health_stats(zip_code)
    return stats['without_cancer_insurance'] if stats else None


def format_health_stats_html(zip_code):
    """
    Format health statistics as HTML for display in map popups.
    
    Args:
        zip_code: The zip code to get stats for
    
    Returns:
        HTML string with formatted health statistics, or empty string if no data
    """
    stats = get_health_stats(zip_code)
    
    if not stats:
        return ""
    
    # Format currency values
    def fmt_currency(value):
        if value is None or pd.isna(value):
            return "N/A"
        try:
            return f"${float(value):,.0f}/year"
        except:
            return "N/A"
    
    html = f"""
    <div style="margin-top:10px; padding:8px; background:#f0f8ff; border-radius:4px;">
        <b style="color:#1f77b4;">💊 Potential Annual Revenue:</b><br>
        <span style="font-size:11px;">
        • <b>Without Cancer:</b> {fmt_currency(stats['without_cancer'])}<br>
        • <b>Insurance Paying:</b> {fmt_currency(stats['without_cancer_insurance'])}
        </span>
    </div>
    """
    
    return html


# Example usage (commented out):
if __name__ == "__main__":
    zip_code = "10001"  # Example zip code
    
    print(f"Zip Code: {zip_code}")
    print(f"Grand Total With Cancer: {get_grand_total_with_cancer(zip_code)}")
    print(f"Grand Total Without Cancer: {get_grand_total_without_cancer(zip_code)}")
    print(f"Grand Total Without Cancer Insurance Paying: {get_grand_total_without_cancer_insurance_paying(zip_code)}")

