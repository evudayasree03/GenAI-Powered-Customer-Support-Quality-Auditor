"""
SamiX - Quality Auditor Entry Point

This is the main driver for the SamiX application. It initializes the system components,
handles user authentication, and renders the appropriate UI (Agent or Admin) based on 
user roles.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Disable Hugging Face telemetry and legacy warnings for a cleaner console output.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load configuration
try:
    from config import Config
    Config.print_status()
except Exception as e:
    print(f"⚠ Warning: Could not load config module: {e}")
try:
    import audioop
except ImportError:
    import audioop_lts
    sys.modules["audioop"] = audioop_lts

import streamlit as st
from PIL import Image

# Configure the primary page settings.
try:
    st.set_page_config(
        page_title="SamiX · Quality Auditor",
        page_icon="👁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception as e:
    print(f"⚠ Warning: Could not set page config: {e}")

# Import our custom modules from the src directory with graceful fallback.
try:
    from src.auth.authenticator   import AuthManager
    from src.pipeline.alert_engine import AlertEngine
    from src.pipeline.groq_client  import GroqClient
    from src.pipeline.stt_processor import STTProcessor
    from src.ui.admin_panel        import AdminPanel
    from src.ui.agent_panel        import AgentPanel
    from src.ui.login_page         import LoginPage
    from src.ui.styles             import inject_css
    from src.utils.audio_processor import AudioProcessor
    from src.utils.cost_tracker    import CostTracker
    from src.utils.history_manager import HistoryManager
    from src.utils.kb_manager      import KBManager
except ImportError as e:
    st.error(f"❌ Failed to import required modules:\n\n`{e}`\n\n"
            f"**Solution:**\n"
            f"1. Ensure all files in `src/` directory exist\n"
            f"2. Run: `pip install -r requirements.txt`\n"
            f"3. Check file permissions and Python path")
    sys.exit(1)

# Apply custom CSS styles to match the premium dark-themed design.
try:
    inject_css()
except Exception as e:
    print(f"⚠ Warning: Could not inject CSS: {e}")


@st.cache_resource
def _init():
    """ Initialize and cache singleton instances of core system managers. """
    return {
        "history": HistoryManager(),
        "groq":    GroqClient(),
        "stt":     STTProcessor(),
        "audio":   AudioProcessor(),
        "cost":    CostTracker(),
        "alerts":  AlertEngine(),
        "kb":      KBManager(),
    }


# Global Resource container and Authentication manager.
R    = _init()
auth = AuthManager()
R["auth"] = auth


def _sidebar_brand() -> None:
    """ Renders the SamiX branding and system status indicators in the sidebar. """
    with st.sidebar:
        # Try to load custom logo from assets/images
        logo_path = Path("assets/images/logo.png")
        if logo_path.exists():
            try:
                logo = Image.open(logo_path)
                st.image(logo, width=100)
                st.markdown("<br>", unsafe_allow_html=True)
            except Exception as e:
                print(f"Could not load logo: {e}")
                _render_default_sidebar_logo()
        else:
            _render_default_sidebar_logo()
        
        # Main Title
        st.markdown(
            '<div style="text-align:center;margin-bottom:1rem;">'
            '<div style="font-size:1.45rem;color:#FFFFFF;letter-spacing:-0.03em;line-height:1;font-weight:800;">'
            'SamiX</div>'
            '<div style="font-size:0.68rem;font-weight:700;'
            'color:#93C5FD;letter-spacing:0.16em;margin-top:0.55rem;text-transform:uppercase;">'
            'Support Quality Platform</div></div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # Dynamic System Health Status
        statuses = [
            ("Groq API",    "LIVE"   if R["groq"].is_live         else "MOCK",    R["groq"].is_live),
            ("Deepgram",    "LIVE"   if R["stt"].has_deepgram      else "MOCK",    R["stt"].has_deepgram),
            ("Whisper",     "LOCAL", True),
            ("LangChain",   "READY", True),
            ("Milvus Lite", "VECTOR" if R["kb"].is_vector_enabled  else "KEYWORD", True),
            ("pydub",       "ACTIVE",True),
        ]
        st.markdown(
            '<div style="font-size:.7rem;font-weight:700;'
            'color:#FFFFFF;letter-spacing:.08em;margin-bottom:.8rem;opacity:0.6;">SYSTEM STATUS</div>',
            unsafe_allow_html=True,
        )
        for svc, label, ok in statuses:
            colour = "#10B981" if ok else "#F59E0B"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:.7rem;padding:6px 0;font-weight:500;'
                f'color:#94A3B8;">'
                f'<span>{svc}</span>'
                f'<span style="color:{colour};font-weight:700;display:flex;align-items:center;gap:4px;">'
                f'<span style="width:6px;height:6px;border-radius:50%;background:{colour};"></span>{label}</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()
        # User account info and Logout
        st.markdown(
            f'<div style="font-size:.75rem;'
            f'color:#94A3B8;margin-bottom:.8rem;font-weight:500;">'
            f'Logged in as<br>'
            f'<span style="color:#FFFFFF;font-weight:700;margin-top:0.25rem;display:block;">{auth.current_user_name}</span></div>',
            unsafe_allow_html=True,
        )
        auth.render_logout()


def _role() -> str:
    """ Sidebar toggle for switching between Agent and Admin views (if authorized). """
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        role = st.radio(
            "View as",
            ["Agent Workspace", "Admin Workspace"],
            index=0, horizontal=True,
            label_visibility="collapsed",
        )
    return "admin" if role == "Admin Workspace" else "agent"


def _render_default_sidebar_logo() -> None:
    """Fallback logo for sidebar: Modern gradient badge when no custom logo found."""
    st.markdown(
        '<div style="'
        'display:flex;align-items:center;gap:0.75rem;'
        'justify-content:center;margin-bottom:1.5rem;">'
        '<div style="'
        'width:45px;height:45px;'
        'background:linear-gradient(135deg,#152EAE,#2563EB);'
        'border-radius:12px;'
        'display:flex;align-items:center;justify-content:center;'
        'box-shadow:0 8px 16px rgba(21, 46, 174, 0.3);'
        '">'
        '<span style="font-size:1.4rem;color:#fff;font-weight:800;">S</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """ Main application flow: Auth -> Sidebar -> Panel Routing. """
    # Enforce login first.
    if not auth.is_authenticated():
        LoginPage(auth).render()
        return

    # Render branding and common sidebar elements.
    _sidebar_brand()
    role = _role()

    # Route to the appropriate panel based on selected role.
    if role == "admin":
        AdminPanel(
            history=R["history"],
            kb=R["kb"],
            alerts=R["alerts"],
        ).render()
    else:
        AgentPanel(
            history=R["history"],
            groq=R["groq"],
            stt=R["stt"],
            audio=R["audio"],
            cost=R["cost"],
            alerts=R["alerts"],
            kb=R["kb"],
        ).render()


if __name__ == "__main__":
    main()
