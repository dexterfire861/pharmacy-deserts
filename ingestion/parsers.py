"""
File parsing utilities for various data formats.
Supports CSV, Excel (xlsx, xlsm), JSON, Parquet, and ZIP archives.
"""
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, BinaryIO
import pandas as pd


def get_file_type(filename: str) -> str:
    """
    Determine file type from filename extension.
    
    Args:
        filename: Name of the file
    
    Returns:
        File type string (csv, xlsx, xlsm, json, parquet, zip, unknown)
    """
    ext = Path(filename).suffix.lower()
    type_map = {
        '.csv': 'csv',
        '.xlsx': 'xlsx',
        '.xlsm': 'xlsm',
        '.xls': 'xlsx',
        '.json': 'json',
        '.parquet': 'parquet',
        '.pq': 'parquet',
        '.zip': 'zip',
    }
    return type_map.get(ext, 'unknown')


def get_excel_sheet_names(file_content: bytes) -> List[str]:
    """
    Get list of sheet names from an Excel file.
    
    Args:
        file_content: Raw bytes of the Excel file
    
    Returns:
        List of sheet names
    """
    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_content))
        return excel_file.sheet_names
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")


def parse_csv(
    file_content: bytes, 
    skip_rows: int = 0,
    encoding: str = 'utf-8',
    **kwargs
) -> pd.DataFrame:
    """
    Parse a CSV file into a DataFrame.
    
    Args:
        file_content: Raw bytes of the CSV file
        skip_rows: Number of rows to skip at the beginning
        encoding: File encoding
        **kwargs: Additional arguments to pass to pd.read_csv
    
    Returns:
        Parsed DataFrame
    """
    try:
        # Try with specified encoding first
        return pd.read_csv(
            io.BytesIO(file_content), 
            skiprows=skip_rows,
            encoding=encoding,
            low_memory=False,
            **kwargs
        )
    except UnicodeDecodeError:
        # Fall back to latin-1 encoding
        return pd.read_csv(
            io.BytesIO(file_content), 
            skiprows=skip_rows,
            encoding='latin-1',
            low_memory=False,
            **kwargs
        )


def parse_excel(
    file_content: bytes,
    sheet_name: Optional[str] = None,
    skip_rows: int = 0,
    **kwargs
) -> pd.DataFrame:
    """
    Parse an Excel file into a DataFrame.
    
    Args:
        file_content: Raw bytes of the Excel file
        sheet_name: Sheet name to read (first sheet if None)
        skip_rows: Number of rows to skip at the beginning
        **kwargs: Additional arguments to pass to pd.read_excel
    
    Returns:
        Parsed DataFrame
    """
    excel_file = pd.ExcelFile(io.BytesIO(file_content))
    
    if sheet_name is None:
        sheet_name = excel_file.sheet_names[0]
    
    return pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
        skiprows=skip_rows,
        **kwargs
    )


def parse_json(file_content: bytes, **kwargs) -> pd.DataFrame:
    """
    Parse a JSON file into a DataFrame.
    
    Args:
        file_content: Raw bytes of the JSON file
        **kwargs: Additional arguments to pass to pd.read_json
    
    Returns:
        Parsed DataFrame
    """
    # Try to detect if it's JSON lines or regular JSON
    content_str = file_content.decode('utf-8')
    
    try:
        # Try as regular JSON first
        data = json.loads(content_str)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check if it's a dict of lists (column-oriented)
            return pd.DataFrame(data)
    except json.JSONDecodeError:
        pass
    
    # Try as JSON lines
    return pd.read_json(io.BytesIO(file_content), lines=True, **kwargs)


def parse_parquet(file_content: bytes, **kwargs) -> pd.DataFrame:
    """
    Parse a Parquet file into a DataFrame.
    
    Args:
        file_content: Raw bytes of the Parquet file
        **kwargs: Additional arguments to pass to pd.read_parquet
    
    Returns:
        Parsed DataFrame
    """
    return pd.read_parquet(io.BytesIO(file_content), **kwargs)


def extract_zip_contents(file_content: bytes) -> Dict[str, bytes]:
    """
    Extract all files from a ZIP archive.
    
    Args:
        file_content: Raw bytes of the ZIP file
    
    Returns:
        Dictionary mapping filenames to file contents
    """
    contents = {}
    with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zf:
        for name in zf.namelist():
            # Skip directories and hidden files
            if name.endswith('/') or name.startswith('__MACOSX') or name.startswith('.'):
                continue
            contents[name] = zf.read(name)
    return contents


def parse_file(
    filename: str,
    file_content: bytes,
    file_type: Optional[str] = None,
    sheet_name: Optional[str] = None,
    skip_rows: int = 0,
    **kwargs
) -> pd.DataFrame:
    """
    Parse a file into a DataFrame based on its type.
    
    Args:
        filename: Name of the file
        file_content: Raw bytes of the file
        file_type: File type (auto-detected if None)
        sheet_name: Sheet name for Excel files
        skip_rows: Number of rows to skip
        **kwargs: Additional arguments for the parser
    
    Returns:
        Parsed DataFrame
    """
    if file_type is None:
        file_type = get_file_type(filename)
    
    parsers = {
        'csv': lambda: parse_csv(file_content, skip_rows=skip_rows, **kwargs),
        'xlsx': lambda: parse_excel(file_content, sheet_name=sheet_name, skip_rows=skip_rows, **kwargs),
        'xlsm': lambda: parse_excel(file_content, sheet_name=sheet_name, skip_rows=skip_rows, **kwargs),
        'json': lambda: parse_json(file_content, **kwargs),
        'parquet': lambda: parse_parquet(file_content, **kwargs),
    }
    
    if file_type not in parsers:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return parsers[file_type]()


def get_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get a summary of column types in a DataFrame.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary mapping column names to type descriptions
    """
    type_map = {}
    for col in df.columns:
        dtype = df[col].dtype
        
        # Check for more specific types
        if pd.api.types.is_numeric_dtype(dtype):
            if pd.api.types.is_integer_dtype(dtype):
                type_map[col] = "integer"
            else:
                type_map[col] = "float"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            type_map[col] = "datetime"
        elif pd.api.types.is_bool_dtype(dtype):
            type_map[col] = "boolean"
        else:
            type_map[col] = "string"
    
    return type_map


def get_column_stats(df: pd.DataFrame, column: str) -> Dict[str, any]:
    """
    Get basic statistics for a column.
    
    Args:
        df: DataFrame containing the column
        column: Column name
    
    Returns:
        Dictionary of statistics
    """
    series = df[column]
    stats = {
        "non_null_count": int(series.notna().sum()),
        "null_count": int(series.isna().sum()),
        "unique_count": int(series.nunique()),
    }
    
    # Add type-specific stats
    if pd.api.types.is_numeric_dtype(series):
        stats.update({
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
            "mean": float(series.mean()) if series.notna().any() else None,
        })
    else:
        # For non-numeric, show top values
        top_values = series.value_counts().head(5).to_dict()
        stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}
    
    return stats
