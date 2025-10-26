try:
    import streamlit as st
    cache_data = st.cache_data
except Exception:
    # No-op decorator if Streamlit isn't available (e.g., unit tests)
    def cache_data(func):
        return func