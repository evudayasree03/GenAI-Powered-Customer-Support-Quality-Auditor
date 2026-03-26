"""
SamiX login page.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
import streamlit as st

from src.auth.authenticator import AuthManager


class LoginPage:
    def __init__(self, auth: AuthManager) -> None:
        self._auth = auth

    def render(self) -> None:
        left, right = st.columns([1.15, 0.95], gap="large")
        with left:
            self._render_brand()
        with right:
            self._render_form()

    def _render_brand(self) -> None:
        logo_path = Path("assets/images/logo.png")
        st.markdown('<div class="samix-hero">', unsafe_allow_html=True)

        if logo_path.exists():
            try:
                st.image(Image.open(logo_path), width=104)
            except Exception:
                self._fallback_logo()
        else:
            self._fallback_logo()

        st.markdown(
            """
            <div class="samix-eyebrow">AI Quality Operations</div>
            <div class="samix-title">Review support calls with speed, consistency, and grounded policy checks.</div>
            <div class="samix-subtitle">
              SamiX combines transcription, LLM scoring, and LangChain + Milvus retrieval
              into one deployable Streamlit workspace for teams that need cleaner QA operations.
            </div>
            <div class="samix-kpi-grid">
              <div class="samix-kpi-card">
                <div class="samix-kpi-label">Scoring Engine</div>
                <div class="samix-kpi-value">Groq</div>
                <div class="samix-kpi-note">summary + audit scoring</div>
              </div>
              <div class="samix-kpi-card">
                <div class="samix-kpi-label">Transcription</div>
                <div class="samix-kpi-value">DG / Whisper</div>
                <div class="samix-kpi-note">cloud primary, local fallback</div>
              </div>
              <div class="samix-kpi-card">
                <div class="samix-kpi-label">RAG Stack</div>
                <div class="samix-kpi-value">Milvus Lite</div>
                <div class="samix-kpi-note">LangChain retrieval + fallback</div>
              </div>
              <div class="samix-kpi-card">
                <div class="samix-kpi-label">Persistence</div>
                <div class="samix-kpi-value">SQLite</div>
                <div class="samix-kpi-note">audits, users, API artifacts</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    def _fallback_logo(self) -> None:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:0.9rem;margin-bottom:1rem;">
              <div style="width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,#1d4ed8,#2563eb);display:flex;align-items:center;justify-content:center;box-shadow:0 14px 28px rgba(37,99,235,0.22);">
                <span style="color:#fff;font-size:1.4rem;font-weight:800;">S</span>
              </div>
              <div>
                <div style="font-size:1.4rem;font-weight:800;color:#0f172a;letter-spacing:-0.03em;">SamiX</div>
                <div style="font-size:0.75rem;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;">Customer Support QA</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_form(self) -> None:
        st.markdown(
            """
            <div class="samix-shell">
              <div class="samix-eyebrow">Secure Access</div>
              <div class="samix-title" style="font-size:1.65rem;">Sign in to SamiX</div>
              <div class="samix-subtitle" style="margin-bottom:1rem;">
                Use your workspace credentials to access audits, knowledge-base tools, and reports.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["Sign in", "Create account"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email", placeholder="name@company.com")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Sign in", use_container_width=True)

                if submit:
                    if not email or not password:
                        st.error("Enter both email and password.")
                    elif self._auth.login(email, password):
                        st.rerun()
                    else:
                        st.error("Incorrect email or password.")

            if self._auth.is_pending():
                st.caption("Default admin login: `admin@samix.ai` / `admin`")

        with tab_register:
            with st.form("register_form", clear_on_submit=True):
                name = st.text_input("Full name", placeholder="Jane Doe")
                email = st.text_input("Work email", placeholder="name@company.com")
                password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm = st.text_input("Confirm password", type="password", placeholder="Re-enter password")
                submit = st.form_submit_button("Create account", use_container_width=True)

                if submit:
                    if not name or not email or not password:
                        st.error("Fill in all fields.")
                    elif password != confirm:
                        st.error("Passwords do not match.")
                    elif self._auth.register(email, name, password):
                        st.success("Account created. You can sign in now.")
                    else:
                        st.error("That email is already registered or invalid.")
