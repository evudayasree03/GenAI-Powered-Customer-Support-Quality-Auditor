from __future__ import annotations

import streamlit as st


class ReportsPage:
    def render(self) -> None:
        st.title("Reports & Exports")
        st.caption("PDF and Excel export entry point scaffold.")
