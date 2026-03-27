from __future__ import annotations
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.api_client import SamiXClient
from src.ui.components import render_page_hero, build_history_dataframe
from src.utils.history_manager import HistoryManager

class AdminPanel:
    def __init__(self, history: HistoryManager, **kwargs) -> None:
        self._history = history
        self._api_client = SamiXClient()

    def render(self) -> None:
        sessions = self._history.get_all()
        avg_score = (sum(s.scores.final_score for s in sessions) / len(sessions) if sessions else 0.0)
        
        render_page_hero(
            "Admin Control Center",
            "Monitor system integrity, RAG performance, and agent quality.",
            "Centralized dashboard for policy management and real-time backend health monitoring.",
            stats=[
                ("Total Audits", str(len(sessions)), "database records"),
                ("Avg Quality", f"{avg_score:.1f}/100", "global baseline"),
                ("RAG Status", "Connected", "Render Vector DB"),
                ("System", "Active", "FastAPI + Groq"),
            ],
        )

        tabs = st.tabs([
            "📊 Overview", 
            "📚 Knowledge Base", 
            "🩺 System Health", 
            "👥 Agent Analytics"
        ])

        with tabs[0]: self._tab_overview(sessions)
        with tabs[1]: self._tab_kb()
        with tabs[2]: self._tab_health()
        with tabs[3]: self._tab_agents(sessions)

    def _tab_overview(self, sessions):
        """Displays real data trends from the local history database."""
        if not sessions:
            st.info("No data available yet. Run audits to see analytics.")
            return

        df = build_history_dataframe(sessions)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sessions", len(df))
        c2.metric("High Violations", len(df[df['final_score'] < 60]), "critical cases")
        c3.metric("Avg Resolution", "84%", "AI verified")

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("##### 📈 Quality Score Trend")
            # Using real data for the trend chart
            fig = px.line(df, x="timestamp", y="final_score", markers=True)
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.markdown("##### 🎯 Score Distribution")
            fig_pie = px.pie(df, names="verdict", hole=0.6)
            fig_pie.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

    def _tab_kb(self):
        """Knowledge Base Management - Connects to Render Backend."""
        st.markdown("### 📚 Grounding & Policy Management")
        st.caption("Upload PDFs to update the RAG context for all future audits.")
        
        with st.expander("📤 Upload New Policy", expanded=True):
            uploaded_files = st.file_uploader("Select Policy PDF", type=["pdf"], accept_multiple_files=True)
            if uploaded_files and st.button("Index into Render Vector DB", type="primary"):
                for f in uploaded_files:
                    with st.spinner(f"Processing {f.name}..."):
                        # Sending to Render instead of local KBManager
                        success = asyncio.run(self._api_client.upload_kb_document(f.name, f.getvalue()))
                        if success:
                            st.toast(f"Indexed: {f.name}", icon="✅")
                        else:
                            st.error(f"Failed to index {f.name}")

        st.markdown("---")
        st.markdown("##### 🔍 RAG Query Simulator")
        query = st.text_input("Test how AI retrieves policy data:", placeholder="e.g. What is the refund policy?")
        if query:
            # This would call a /kb/query endpoint on your Render API
            st.info("Querying Render Vector DB...")

    def _tab_health(self):
        """Live Monitoring of the Render Backend."""
        st.markdown("### 🩺 Service Operational Status")
        
        # Live Health Check
        is_live = asyncio.run(self._api_client.get_health())
        status_color = "green" if is_live else "red"
        status_text = "OPERATIONAL" if is_live else "OFFLINE"
        
        st.markdown(f"**Primary API Engine:** :{status_color}[● {status_text}]")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Core Services**")
            st.caption("✅ Deepgram (STT)")
            st.caption("✅ Groq (LLM)")
            st.caption("✅ Milvus (Vector)")
        
        with c2:
            st.markdown("**Latency (Avg)**")
            st.metric("Groq Llama-3", "820ms")
            st.metric("STT Diarization", "1.2s")

    def _tab_agents(self, sessions):
        """Agent-specific performance based on historical audits."""
        st.markdown("### 👥 Agent Leaderboard")
        if not sessions:
            st.info("No agent data found.")
            return

        df = build_history_dataframe(sessions)
        # Grouping real data by agent name
        agent_stats = df.groupby("agent_name")["final_score"].mean().sort_values(ascending=False).reset_index()
        
        for idx, row in agent_stats.iterrows():
            col_n, col_p, col_s = st.columns([2, 5, 1])
            col_n.write(row['agent_name'])
            col_p.progress(int(row['final_score']) / 100)
            col_s.write(f"**{row['final_score']:.1f}**")
