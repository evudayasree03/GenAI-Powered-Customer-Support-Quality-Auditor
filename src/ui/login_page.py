"""
SamiX Authentication Gateway
Handles brand storytelling and secure access with session persistence.
Location: src/ui/login_page.py
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image
import streamlit as st
from src.auth.authenticator import AuthManager

class LoginPage:
    def __init__(self, auth: AuthManager) -> None:
        self._auth = auth
        # Ensure session keys exist
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "user_data" not in st.session_state:
            st.session_state.user_data = None

    def render(self) -> None:
        """Main render loop for the login screen."""
        left, right = st.columns([1.2, 0.8], gap="large")
        
        with left:
            self._render_brand()
        
        with right:
            self._render_form()

    def _render_brand(self) -> None:
        """Renders the left-side marketing and tech-stack overview."""
        logo_path = Path("assets/images/logo.png")
        
        # Fallback to CSS logo if image is missing
        if logo_path.exists():
            try:
                st.image(Image.open(logo_path), width=100)
            except Exception:
                self._fallback_logo()
        else:
            self._fallback_logo()

        st.markdown(
            """
            <div style="margin-top: 2rem;">
                <h1 style="font-size: 2.5rem; font-weight: 800; line-height: 1.2;">
                    Review support calls with speed and grounded policy checks.
                </h1>
                <p style="font-size: 1.1rem; color: #94A3B8; margin-top: 1rem;">
                  SamiX orchestrates Groq-powered LLMs and Milvus RAG to automate 
                  compliance audits for high-volume support teams.
                </p>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 3rem;">
                  <div style="background: #1E293B; padding: 1rem; border-radius: 12px; border: 1px solid #334155;">
                    <div style="font-size: 0.7rem; color: #6366F1; font-weight: 700;">LLM ENGINE</div>
                    <div style="font-weight: 600; color: white;">Groq Llama 3</div>
                  </div>
                  <div style="background: #1E293B; padding: 1rem; border-radius: 12px; border: 1px solid #334155;">
                    <div style="font-size: 0.7rem; color: #6366F1; font-weight: 700;">VECTOR DB</div>
                    <div style="font-weight: 600; color: white;">Milvus Lite</div>
                  </div>
                  <div style="background: #1E293B; padding: 1rem; border-radius: 12px; border: 1px solid #334155;">
                    <div style="font-size: 0.7rem; color: #6366F1; font-weight: 700;">AUDIO STT</div>
                    <div style="font-weight: 600; color: white;">Deepgram</div>
                  </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _fallback_logo(self) -> None:
        """SaaS-style CSS logo."""
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:2rem;">
              <div style="width:48px;height:48px;border-radius:12px;background:linear-gradient(135deg,#6366F1,#4F46E5);display:flex;align-items:center;justify-content:center;">
                <span style="color:#fff;font-size:1.2rem;font-weight:800;">S</span>
              </div>
              <div style="font-size:1.4rem;font-weight:800;color:white;">SamiX</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_form(self) -> None:
        """Renders login/register tabs."""
        st.subheader("Access Workspace")
        st.caption("Sign in to review audits or manage policies.")

        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="admin@samix.ai")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Enter Dashboard", use_container_width=True)

                if submit:
                    if self._auth.login(email, password):
                        st.success("Authentication successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
            
            st.info("💡 Protip: Default admin is `admin@samix.ai` / `samix2026`")

        with tab_register:
            with st.form("register_form"):
                name = st.text_input("Full Name")
                reg_email = st.text_input("Work Email")
                reg_pass = st.text_input("Password", type="password")
                role = st.selectbox("Role", ["Agent", "Supervisor"])
                reg_submit = st.form_submit_button("Create Account", use_container_width=True)

                if reg_submit:
                    success = self._auth.register(reg_email, name, reg_pass, role=role.lower())
                    if success:
                        st.success("Account created! Please sign in.")
                    else:
                        st.error("Registration failed. Email may already be in use.")
