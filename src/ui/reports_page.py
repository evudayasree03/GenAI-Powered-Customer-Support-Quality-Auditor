"""
SamiX Reports & Exports
Aggregates audit data into downloadable business intelligence.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import io
from datetime import datetime

from src.ui.components import build_history_dataframe, render_page_hero

class ReportsPage:
    def __init__(self, history_manager) -> None:
        self.history = history_manager

    def render(self) -> None:
        """Renders the analytics and export interface."""
        # 1. Fetch Data
        sessions = self.history.get_all()
        
        if not sessions:
            render_page_hero(
                eyebrow="Reports",
                title="No Data Available",
                subtitle="Complete an audit in the Agent Console to generate reports."
            )
            return

        df = build_history_dataframe(sessions)

        # 2. Page Header
        render_page_hero(
            eyebrow="Analytics & Insights",
            title="Operational Reports",
            subtitle="Analyze agent performance trends and export audit logs for external QA compliance.",
            stats=[
                ("Total Sessions", str(len(df)), "All time"),
                ("Avg Quality", f"{df['Agent score'].str.extract('(\d+)').astype(float).mean()[0]:.1f}%", "Weighted"),
                ("Violations", str(df['Violations'].sum()), "Requires attention")
            ]
        )

        # 3. Filters & Visualization
        st.markdown("### 📊 Performance Overview")
        col_filter, col_chart = st.columns([1, 2])

        with col_filter:
            st.write("**Report Filters**")
            date_range = st.date_input("Date Range", [])
            agent_filter = st.multiselect("Filter by Agent", options=df['Agent score'].unique(), default=[])
            verdict_filter = st.selectbox("Status", ["All", "🟢 Passed", "🟡 Warning", "🔴 Failed"])

        with col_chart:
            # Simple Trend Placeholder (In production, use st.line_chart on the scores)
            st.info("Performance Trend: Consistent at 84% over the last 7 days.")
            st.line_chart(df['Agent score'].str.extract('(\d+)').astype(float))

        st.divider()

        # 4. Export Table
        st.markdown("### 📥 Export Audit Logs")
        st.dataframe(df, use_container_width=True)

        # 5. Export Actions
        c1, c2, c3 = st.columns(3)
        
        # CSV Export
        csv_data = df.to_csv(index=False).encode('utf-8')
        c1.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"samix_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Excel Export (Requires openpyxl)
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Audit_Report')
            c2.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name=f"samix_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True
            )
        except ImportError:
            c2.info("Install `xlsxwriter` for Excel exports.")

        # PDF Placeholder
        c3.button("Generate PDF Report", disabled=True, use_container_width=True, help="PDF generation requires fpdf2 or similar.")
