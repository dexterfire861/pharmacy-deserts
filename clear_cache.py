"""
Clear Streamlit cache to force reload of new pharmacy data
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Clear all caches
print("Clearing Streamlit cache...")
st.cache_data.clear()
print("✓ Cache cleared!")
print("\nNow run: streamlit run app/app.py")
