"""
SamiX Reports & Analytics
Provides deep-dive performance metrics and data export capabilities.
Location: src/ui/reports_page.py
"""
import streamlit as st
import pandas as pd
from src.ui.components import render_page_hero

class ReportsPage:
    def __init__(self, history_manager):
        self.history = history_manager

    def render(self) -> None:
        """Main rendering loop for operational reports."""
        # 1. Fetch Data
        df = self._get_mock_data()

        # --- FIX: Handle Empty Data State ---
        if not df.empty:
            avg_score = f"{df['Score'].mean():.1f}%"
            total_sessions = str(len(df))
            violations = str(len(df[df['Compliance'] == 'Fail']))
        else:
            avg_score = "0.0%"
            total_sessions = "0"
            violations = "0"

        # 2. Page Header
        render_page_hero(
            eyebrow="Analytics & Insights",
            title="Operational Reports",
            subtitle="Analyze agent performance trends and export audit logs for QA compliance.",
            stats=[
                ("Total Sessions", total_sessions, "All time"),
                ("Avg Quality", avg_score, "Target: 85%"),
                ("Violations", violations, "Requires attention")
            ]
        )

        # 3. Filters & Visualization
        st.markdown("### 📊 Performance Overview")
        
        if df.empty:
            st.info("No audit data available yet. Start by running an audit in the Agent Panel.")
            return

        col_filter, col_chart = st.columns([1, 2])

        with col_filter:
            st.write("**Report Filters**")
            # Added unique keys to widgets to prevent SessionState conflicts
            agent_filter = st.multiselect(
                "Filter by Agent", 
                options=df['Agent'].unique(),
                key="report_agent_select"
            )
            
            filtered_df = df.copy()
            if agent_filter:
                filtered_df = filtered_df[filtered_df['Agent'].isin(agent_filter)]

        with col_chart:
            if not filtered_df.empty:
                # Aggregate scores by agent to handle multiple audits per agent
                chart_data = filtered_df.groupby('Agent')['Score'].mean()
                st.bar_chart(chart_data)
            else:
                st.warning("No data matches current filters.")

        # 4. Data Export Section
        st.divider()
        st.subheader("📥 Export Audit Logs")
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"Current selection contains **{len(filtered_df)}** audit records.")
            with c2:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"samix_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    def _get_mock_data(self):
        """Placeholder data generator."""
        return pd.DataFrame({
            "Date": pd.to_datetime(["2026-03-25", "2026-03-26", "2026-03-27", "2026-03-27"]),
            "Agent": ["John Doe", "Jane Smith", "John Doe", "Alice Wong"],
            "Score": [88.5, 72.0, 91.0, 45.5],
            "Compliance": ["Pass", "Pass", "Pass", "Fail"]
        })
