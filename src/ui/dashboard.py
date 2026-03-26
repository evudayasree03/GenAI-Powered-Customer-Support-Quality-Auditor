from __future__ import annotations

import streamlit as st


class DashboardPage:
    def render(self) -> None:
        st.title("SamiX Dashboard")
        st.caption("Unified landing page scaffold for the milestone architecture.")
