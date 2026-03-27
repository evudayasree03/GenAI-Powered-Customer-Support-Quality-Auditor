"""
SamiX - Quality Auditor Entry Point (Cloud Optimized)
Connects to the Render FastAPI backend for all heavy processing.
"""
from __future__ import annotations

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import streamlit as st
from PIL import Image

# --- BACKEND CONNECTION ---
# Point this to your Render URL in Streamlit Secrets
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

def check_backend_health():
    """Check if Render backend is awake."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None

# Configure the primary page settings.
st.set_page_config(
    page_title="SamiX · Quality Auditor",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import custom UI modules (Keep these light)
try:
    from src.auth.authenticator   import AuthManager
    from src.ui.admin_panel        import AdminPanel
    from src.ui.agent_panel        import AgentPanel
    from src.ui.login_page         import LoginPage
    from src.ui.styles             import inject_css
    # Import the GroqClient we updated to act as a Proxy
    from src.pipeline.groq_client  import GroqClient
except ImportError as e:
    st.error(f"❌ UI Import Error: {e}")
    st.stop()

# Apply custom CSS
inject_css()

@st.cache_resource
def _init_client():
    """ 
    Initialize components in 'Client Mode'.
    These now point to the BACKEND_URL instead of running locally.
    """
    return {
        "groq": GroqClient(api_base=BACKEND_URL),
        # Other managers will now be pass-throughs to API calls
    }

# Initialize resources
R = _init_client()
auth = AuthManager()
backend_status = check_backend_health()

def _sidebar_brand() -> None:
    """ Renders sidebar with dynamic health status from the Render API. """
    with st.sidebar:
        _render_logo()
        
        st.markdown(
            '<div style="text-align:center;margin-bottom:1rem;">'
            '<div style="font-size:1.45rem;color:#FFFFFF;font-weight:800;">SamiX</div>'
            '<div style="font-size:0.68rem;color:#93C5FD;text-transform:uppercase;">Support Quality Platform</div></div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # Dynamic System Health Status (Pulled from Render API)
        st.markdown('<div style="font-size:.7rem;font-weight:700;opacity:0.6;">SERVER STATUS</div>', unsafe_allow_html=True)
        
        if backend_status:
            services = [
                ("API Engine", "ONLINE", True),
                ("Groq Live", "LIVE" if backend_status.get("groq_live") else "OFF", backend_status.get("groq_live")),
                ("Deepgram", "READY" if backend_status.get("deepgram_live") else "OFF", backend_status.get("deepgram_live")),
                ("Vector DB", "ACTIVE" if backend_status.get("vector_enabled") else "OFF", True),
            ]
        else:
            services = [("Backend", "WAKING UP...", False)]

        for svc, label, ok in services:
            colour = "#10B981" if ok else "#F59E0B"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:.7rem;padding:4px 0;">'
                f'<span>{svc}</span>'
                f'<span style="color:{colour};font-weight:700;">● {label}</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown(f'<div style="font-size:.75rem;">Logged in as: <b>{auth.current_user_name}</b></div>', unsafe_allow_html=True)
        auth.render_logout()

def _render_logo():
    logo_path = Path("assets/images/logo.png")
    if logo_path.exists():
        st.image(Image.open(logo_path), width=100)
    else:
        st.markdown('<div style="width:45px;height:45px;background:linear-gradient(135deg,#152EAE,#2563EB);border-radius:12px;margin:0 auto 1rem auto;display:flex;align-items:center;justify-content:center;"><span style="color:#fff;font-weight:800;">S</span></div>', unsafe_allow_html=True)

def main() -> None:
    if not auth.is_authenticated():
        LoginPage(auth).render()
        return

    _sidebar_brand()
    
    # Simple Role Selection
    with st.sidebar:
        view = st.radio("Navigation", ["Agent Workspace", "Admin Workspace"], label_visibility="collapsed")

    if not backend_status:
        st.error("⚠️ The Backend Server is currently offline or waking up. Please refresh in 30 seconds.")
        st.info("Render Free Tier sleeps after inactivity. Accessing the URL wakes it up.")
        return

    if view == "Admin Workspace":
        AdminPanel(backend_url=BACKEND_URL).render()
    else:
        AgentPanel(backend_url=BACKEND_URL, groq=R["groq"]).render()

if __name__ == "__main__":
    main()
