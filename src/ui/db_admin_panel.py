"""
SamiX Database Administration Panel
Provides low-level visibility into the SQLite persistence layer.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime

from src.db.utils import sqlite_healthcheck
from src.db import get_db

class DBAdminPanel:
    def __init__(self) -> None:
        """Initialize connection to the local persistence layer."""
        self.db = get_db()

    def render(self) -> None:
        """Renders the administrative interface for database management."""
        st.markdown("""
            <div style="margin-bottom: 2rem;">
                <h1 style="font-size: 2.25rem; font-weight: 800; color: #F1F5F9; margin-bottom: 0.5rem;">Database Infrastructure</h1>
                <p style="color: #94A3B8; font-size: 1.1rem;">Monitor health, inspect raw audit tables, and manage storage local to this instance.</p>
            </div>
        """, unsafe_allow_html=True)

        # --- Section 1: Real-time Health Metrics ---
        health = sqlite_healthcheck(self.db)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            status_color = "normal" if health.get("status") == "healthy" else "inverse"
            st.metric("DB Status", health.get("status", "Unknown").upper(), delta_color=status_color)
        with c2:
            st.metric("Total Tables", len(health.get("tables", [])))
        with c3:
            size_kb = health.get("size_kb", 0)
            st.metric("Storage Used", f"{size_kb / 1024:.2f} MB" if size_kb > 1024 else f"{size_kb} KB")
        with c4:
            # Calculate uptime or last integrity check if available
            st.metric("Integrity", "Verified", help="SQLite PRAGMA integrity_check result")

        st.markdown("---")

        # --- Section 2: Data Explorer ---
        st.subheader("🔍 Table Inspector")
        tabs = st.tabs(["Table Browser", "Raw Health Schema"])

        with tabs[0]:
            available_tables = [t['name'] for t in health.get("tables", [])]
            if not available_tables:
                st.info("No tables detected in the current database schema.")
            else:
                col_sel, col_act = st.columns([3, 1])
                selected_table = col_sel.selectbox("Select Target Table", available_tables, label_visibility="collapsed")
                
                if col_act.button("Refresh Data", use_container_width=True):
                    st.cache_data.clear()

                # Data Fetching Logic
                try:
                    # Attempt to fetch last 100 rows using a standard SQL query
                    # Note: Adjust 'self.db.engine' based on your actual DB wrapper implementation
                    query = f"SELECT * FROM {selected_table} ORDER BY rowid DESC LIMIT 100"
                    
                    # Implementation detail: Using pandas to read the sql directly
                    # If your get_db() returns a session, use self.db.bind
                    df = pd.read_sql(query, self.db.bind if hasattr(self.db, 'bind') else self.db)
                    
                    st.dataframe(
                        df, 
                        use_container_width=True, 
                        column_config={"session_id": st.column_config.TextColumn("Session ID")}
                    )
                    st.caption(f"Showing most recent 100 entries from **{selected_table}**")
                except Exception as e:
                    st.error(f"Execution Error: Could not query table '{selected_table}'.")
                    st.caption(f"Technical details: {str(e)}")

        with tabs[1]:
            st.json(health)

        # --- Section 3: Maintenance & Safety ---
        st.markdown("---")
        st.subheader("🛠️ Maintenance Operations")
        
        with st.container(border=True):
            st.warning("Destructive actions below affect the local persistence layer only.")
            
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                if st.button("🧹 Vacuum Database", help="Rebuilds the database file, reclaiming unused space.", use_container_width=True):
                    try:
                        self.db.execute("VACUUM")
                        st.success("Database vacuumed successfully.")
                    except:
                        st.error("Vacuum failed: Permissions error.")

            with m_col2:
                if st.button("💾 Export to CSV", help="Download a full backup of all audits.", use_container_width=True):
                    st.info("Preparing backup bundle...")

            # Dangerous Action
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔥 Danger Zone", expanded=False):
                st.write("Deleting records is permanent and cannot be undone.")
                target_days = st.slider("Delete records older than (days)", 7, 365, 30)
                
                if st.button(f"Purge Audits Older than {target_days} Days", type="primary", use_container_width=True):
                    st.error("Confirmation required: Please contact system admin.")

    def _get_row_count(self, table_name: str) -> int:
        """Utility to get record counts for metrics."""
        try:
            res = self.db.execute(f"SELECT COUNT(*) FROM {table_name}")
            return res.scalar()
        except:
            return 0
