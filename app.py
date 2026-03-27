"""
SamiX - Quality Auditor Entry Point (Cloud Optimized)
Main Orchestrator connecting the UI components to the Render Backend.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from PIL import Image

# Load local environment variables
load_dotenv()

# --- PROJECT IMPORTS ---
try:
    from src.auth.authenticator import AuthManager
    from src.db import get_db
    from src.api_client import SamiXClient
    from src.ui.login_page import LoginPage
    from src.ui.dashboard import DashboardPage
    from src.ui.styles import inject_css
except ImportError as e:
    st.error(f"❌ Initialization Error: Check project structure. Details: {e}")
    st.stop()

# 1. Page Configuration
st.set_page_config(
    page_title="SamiX · Quality Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Global Resources (Cached to prevent re-init on every rerun)
@st.cache_resource
def get_api_client():
    """Points to the Render URL defined in Streamlit Secrets or ENV."""
    # Use 'BACKEND_URL' to match your existing secret name
    url = st.secrets.get("BACKEND_URL") or os.getenv("SAMIX_API_URL", "http://localhost:8000")
    return SamiXClient(base_url=url)

def initialize_session():
    """Syncs managers with the Streamlit session state."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = get_api_client()
    
    if "auth_manager" not in st.session_state:
        st.session_state.auth_manager = AuthManager(get_db())
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

# 3. Sidebar Branding & Health Status
def render_sidebar_header(api: SamiXClient):
    """Renders the logo and real-time backend health check in the sidebar."""
    with st.sidebar:
        # Logo Logic
        logo_path = Path("assets/images/logo.png")
        if logo_path.exists():
            st.image(Image.open(logo_path), width=100)
        else:
            st.markdown('<div style="width:48px;height:48px;background:linear-gradient(135deg,#6366F1,#4F46E5);border-radius:12px;margin:0 auto 1rem auto;display:flex;align-items:center;justify-content:center;box-shadow:0 10px 15px rgba(0,0,0,0.2);"><span style="color:#fff;font-weight:800;font-size:1.2rem;">S</span></div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align:center;color:#F8FAFC;font-weight:800;font-size:1.3rem;letter-spacing:-0.02em;">SamiX</div>', unsafe_allow_html=True)
        st.divider()

        # Dynamic Backend Health Check
        status = api.health()
        st.markdown('<div style="font-size:.65rem;font-weight:700;color:#64748B;letter-spacing:0.1em;margin-bottom:8px;">BACKEND INFRASTRUCTURE</div>', unsafe_allow_html=True)
        
        if status.get("status") in ["healthy", "online"]:
            st.success("● API Engine Online")
        else:
            st.warning("● Backend Waking Up...")
            st.caption("Note: Render Free Tier sleeps after 15m of inactivity.")

        st.divider()
        
        # User Info & Logout
        if st.session_state.get("authenticated"):
            user_data = st.session_state.get("user_data", {})
            st.markdown(f'<div style="font-size:.8rem; margin-bottom:10px;">User: <b>{user_data.get("name", "Staff")}</b></div>', unsafe_allow_html=True)
            if st.button("Log Out", use_container_width=True, type="secondary"):
                st.session_state.authenticated = False
                st.rerun()

# 4. Main Application Entry
def main() -> None:
    # Inject Custom Enterprise CSS
    inject_css()
    
    # Run Initializers
    initialize_session()
    
    api = st.session_state.api_client
    auth = st.session_state.auth_manager

    # --- ROUTING ---
    if not st.session_state.authenticated:
        # Show Auth Gateway
        LoginPage(auth).render()
    else:
        # Render Sidebar (Health + Brand)
        render_sidebar_header(api)

        # Launch DashboardPage (Internal Role Handling)
        dashboard = DashboardPage(
            history_manager=get_db(),
            kb_manager=api
        )
        dashboard.render()

if __name__ == "__main__":
    main()
