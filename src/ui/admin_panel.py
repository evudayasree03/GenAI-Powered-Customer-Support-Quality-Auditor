"""
SamiX - Admin Console
Performance Metrics & User Management
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime

class AdminPanel:
    def __init__(self, history_manager):
        self.db = history_manager # Local SQLite Session

    def render(self):
        st.header("🔑 Admin Control Tower")
        
        tabs = st.tabs(["📈 System Analytics", "👥 User Management", "⚙️ Backend Config"])

        with tabs[0]:
            self._render_analytics()

        with tabs[1]:
            st.subheader("Platform Users")
            # Query users from the local DB
            users = self.db.query("SELECT id, username, role, created_at FROM users")
            if users:
                df_users = pd.DataFrame(users, columns=["ID", "Username", "Role", "Joined"])
                st.dataframe(df_users, use_container_width=True)
            else:
                st.info("No users registered yet.")

        with tabs[2]:
            st.subheader("API & Connection Settings")
            st.json({
                "Backend URL": st.secrets.get("BACKEND_URL", "Not Set"),
                "Database Path": "data/samix.db",
                "Environment": "Production (2026)"
            })

    def _render_analytics(self):
        """Displays high-level KPIs from audit history."""
        st.subheader("Quality Performance Metrics")
        
        # Pull stats from database
        stats = self.db.query("SELECT COUNT(*), AVG(score) FROM audit_sessions")
        total_audits = stats[0][0] if stats else 0
        avg_score = round(stats[0][1], 2) if stats and stats[0][1] else 0

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Audits", total_audits)
        kpi2.metric("Average Quality Score", f"{avg_score}%")
        kpi3.metric("System Health", "Optimal", delta="Stable")

        # Trend Chart
        st.markdown("### Audit Volume Trend")
        history = self.db.query("SELECT timestamp, score FROM audit_sessions ORDER BY timestamp ASC")
        if history:
            df = pd.DataFrame(history, columns=["Date", "Score"])
            df["Date"] = pd.to_datetime(df["Date"])
            st.line_chart(df.set_index("Date"))
        else:
            st.caption("Insufficient data for trending.")
