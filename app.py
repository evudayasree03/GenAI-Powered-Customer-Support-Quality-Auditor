"""
SamiX - Quality Auditor Entry Point
Location: /app.py
"""
from __future__ import annotations

import os
import streamlit as st
from pathlib import Path
from PIL import Image

# Internal Imports
try:
    from src.auth.authenticator import AuthManager
    from src.db import get_db_engine
    from src.api_client import SamiXClient
    from src.ui.login_page import LoginPage
    from src.ui.dashboard import DashboardPage
    from src.ui.styles import inject_css
except ImportError as e:
    st.error(f"❌ Initialization Error: {e}")
    st.stop()

# 1. Page Configuration
st.set_page_config(
    page_title="SamiX · Quality Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def init_managers():
    """Initialize core engines once."""
    api_url = st.secrets.get("BACKEND_URL") or os.getenv("SAMIX_API_URL", "http://localhost:8000")
    return {
        "api": SamiXClient(base_url=api_url),
        "auth": AuthManager(get_db_engine())
    }

def render_sidebar_header(api: SamiXClient):
    """Renders the branding and backend health status."""
    with st.sidebar:
        # Logo fallback
        logo_path = Path("assets/images/logo.png")
        if logo_path.exists():
            st.image(Image.open(logo_path), width=100)
        else:
            st.markdown('<div style="width:48px;height:48px;background:linear-gradient(135deg,#6366F1,#4F46E5);border-radius:12px;margin:0 auto 1rem auto;display:flex;align-items:center;justify-content:center;"><span style="color:#fff;font-weight:800;font-size:1.2rem;">S</span></div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align:center;color:#F8FAFC;font-weight:800;font-size:1.3rem;">SamiX</div>', unsafe_allow_html=True)
        st.divider()

        # Health Check
        status = api.health()
        st.markdown('<div style="font-size:.65rem;font-weight:700;color:#64748B;">SERVER STATUS</div>', unsafe_allow_html=True)
        if status.get("status") == "healthy":
            st.success("● API Engine Online")
        else:
            st.warning("● Backend Waking Up...")

def main():
    inject_css()
    
    # Initialize state
    managers = init_managers()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Routing
    if not st.session_state.authenticated:
        LoginPage(managers["auth"]).render()
    else:
        render_sidebar_header(managers["api"])
        
        # Dashboard handles the internal navigation (Admin vs Agent)
        dashboard = DashboardPage(
            history_manager=managers["auth"].db, # Local SQLite for history
            kb_manager=managers["api"]           # API for RAG/Audits
        )
        dashboard.render()

if __name__ == "__main__":
    main()
