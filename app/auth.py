# pharmacy_deserts/app/auth.py
"""
Basic password authentication for the Streamlit app.
Provides a simple login form that must be passed before accessing the main app.
"""
import streamlit as st
import hashlib
import os
from typing import Optional

from app.config import get_config


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(password: str) -> bool:
    """
    Check if the provided password matches the configured password.
    
    Args:
        password: Password to check
        
    Returns:
        True if password matches, False otherwise
    """
    config = get_config()
    
    if not config.app_password:
        # No password configured, allow access
        return True
    
    # Check against configured password
    return password == config.app_password


def is_authenticated() -> bool:
    """
    Check if the current session is authenticated.
    
    Returns:
        True if authenticated or auth not required, False otherwise
    """
    config = get_config()
    
    # If auth is not required, always return True
    if not config.require_auth:
        return True
    
    # If no password is configured, allow access
    if not config.app_password:
        return True
    
    # Check session state for authentication
    return st.session_state.get('authenticated', False)


def login_form() -> bool:
    """
    Display a login form and handle authentication.
    
    Returns:
        True if authenticated, False otherwise
    """
    config = get_config()
    
    # If auth is not required, return True immediately
    if not config.require_auth:
        return True
    
    # If no password configured, return True
    if not config.app_password:
        return True
    
    # Check if already authenticated
    if st.session_state.get('authenticated', False):
        return True
    
    # Display login form
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    .login-title {
        color: white;
        text-align: center;
        font-size: 28px;
        margin-bottom: 30px;
        font-weight: 600;
    }
    .login-subtitle {
        color: rgba(255,255,255,0.8);
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🏥 Pharmacy Desert Explorer")
        st.markdown("Please enter the password to access the application.")
        
        with st.form("login_form"):
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if check_password(password):
                    st.session_state['authenticated'] = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Please try again.")
        
        st.markdown("---")
        st.caption("Contact your administrator if you need access.")
    
    return False


def logout():
    """Log out the current user by clearing authentication state."""
    if 'authenticated' in st.session_state:
        del st.session_state['authenticated']


def logout_button():
    """Display a logout button in the sidebar."""
    config = get_config()
    
    if config.require_auth and st.session_state.get('authenticated', False):
        if st.sidebar.button("🚪 Logout"):
            logout()
            st.rerun()


def require_auth(func):
    """
    Decorator to require authentication before running a function.
    
    Usage:
        @require_auth
        def main():
            # Your app code here
            pass
    """
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            login_form()
            st.stop()
        return func(*args, **kwargs)
    return wrapper

