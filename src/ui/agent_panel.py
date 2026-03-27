from __future__ import annotations
import asyncio
import os
import time
import streamlit as st
import plotly.graph_objects as go

# Centralized API Client
from src.api_client import SamiXClient
# Core Data Models
from src.utils.history_manager import (
    AuditSession, AuditScores, HistoryManager, 
    WrongTurn, EngineAResult, EngineBResult, EngineBClaim, EngineCResult
)
# UI Visual Components
from src.ui.components import (
    render_page_hero, build_history_dataframe, render_cost_card, 
    render_dual_score_chart, render_filename_badge, render_three_gauges, 
    render_transcript, render_wrong_turns
)

class AgentPanel:
    def __init__(self, history: HistoryManager, **kwargs):
        self._history = history
        # SamiXClient communicates with the Render FastAPI backend
        self._api_client = SamiXClient() 

    def render(self) -> None:
        """Renders the main Agent Workspace with API-driven tabs."""
        render_page_hero(
            "Agent Workspace",
            "SamiX AI-Powered Quality Auditing",
            "Upload conversations to receive instant QA scoring, compliance checks, and RAG-verified feedback.",
            stats=[
                ("Sessions", str(len(self._history.get_all())), "records in local DB"),
                ("AI Engine", "Groq Llama-3", "High-speed Scoring"),
                ("Status", "Connected", "Backend Live")
            ],
        )
        
        tabs = st.tabs(["🚀 New Audit", "📜 Session History", "🔍 Deep-Dive Analysis", "📊 My Performance"])
        
        with tabs[0]: self._new_audit_tab()
        with tabs[1]: self._history_tab()
        with tabs[2]: self._session_detail_tab()
        with tabs[3]: self._performance_tab()

    def _new_audit_tab(self):
        """Ingestion Tab: Sends files to Render for processing."""
        st.markdown("### 📤 Upload New Call")
        uploaded = st.file_uploader(
            "Drop audio (WAV/MP3) or JSON transcript", 
            type=["wav", "mp3", "m4a", "json", "txt"]
        )
        
        if uploaded:
            st.info(f"File **{uploaded.name}** is ready for AI analysis.")
            if st.button("🔍 Start SamiX AI Audit", use_container_width=True, type="primary"):
                # Run the async API call within Streamlit
                asyncio.run(self._run_backend_audit(uploaded))

    async def _run_backend_audit(self, uploaded_file):
        """Streams file to the Render backend and parses the complex multi-engine response."""
        with st.status("📡 Connecting to SamiX AI Engine...", expanded=True) as status:
            try:
                status.write("📤 Uploading bytes to Render server...")
                # Call the central API client
                result = await self._api_client.run_audit(uploaded_file.name, uploaded_file.getvalue())
                
                status.write("🧠 AI is executing multi-stage scoring (Groq + RAG)...")
                session = self._create_session_from_api(uploaded_file.name, result)
                
                # Persistence: Save the result to local SQLite so it survives refreshes
                self._history.save(session)
                st.session_state["active_session_id"] = session.session_id
                
                status.update(label="✅ Audit Complete!", state="complete")
                st.toast(f"Scored: {session.scores.final_score}/100", icon="🎉")
                st.rerun() # Refresh to show data in the Detail tab
                
            except Exception as e:
                st.error(f"Backend Processing Error: {str(e)}")

    def _create_session_from_api(self, filename: str, data: dict) -> AuditSession:
        """
        Maps the FastAPI JSON response back to the SamiX AuditSession object.
        Ensures all 6 dimensions and 3 engines are correctly populated.
        """
        session = AuditSession.new(filename, mode="upload", agent_name=st.session_state.get("username", "Agent"))
        
        # 1. Map Core Scores
        s = data.get("scores", {})
        session.scores = AuditScores(
            final_score=s.get("final_score", 0),
            empathy=s.get("empathy", 0),
            professionalism=s.get("professionalism", 0),
            compliance=s.get("compliance", 0),
            resolution=s.get("resolution", 0),
            communication=s.get("communication", 0),
            integrity=s.get("integrity", 0),
            verdict=s.get("verdict", "N/A"),
            customer_overall=s.get("customer_overall", 0)
        )
        
        # 2. Map Transcript & AI Summary
        session.transcript = data.get("transcript", [])
        session.summary = data.get("summary", "")
        session.violations = len(data.get("violations", []))
        session.token_count = data.get("token_count", 0)
        session.cost_usd = data.get("cost_usd", 0.0)
        
        # 3. Map Wrong Turns (RAG-verified errors)
        session.wrong_turns = [
            WrongTurn(**wt) if isinstance(wt, dict) else wt 
            for wt in data.get("wrong_turns", [])
        ]
        
        # 4. Map Advanced Engine Verification (A, B, C)
        session.engine_a = EngineAResult(**data.get("engine_a", {}))
        
        # Handle Engine B (Factual claims)
        eb_data = data.get("engine_b", {})
        claims = [EngineBClaim(**c) for c in eb_data.get("claims", [])]
        session.engine_b = EngineBResult(claims=claims)
        
        session.engine_c = EngineCResult(**data.get("engine_c", {}))
        
        return session

    def _history_tab(self):
        """History Tab: View and select previous audits from local DB."""
        st.markdown("### 📜 Session Records")
        sessions = self._history.get_all()
        if not sessions:
            st.info("No audit records found. Head to 'New Audit' to begin.")
            return

        df = build_history_dataframe(sessions)
        
        # User selection logic
        selected = st.dataframe(
            df.drop(columns=["session_id"]), 
            use_container_width=True, 
            hide_index=True, 
            on_select="rerun", 
            selection_mode="single-row"
        )
        
        if selected.selection.rows:
            idx = selected.selection.rows[0]
            st.session_state["active_session_id"] = df.iloc[idx]["session_id"]
            st.toast(f"Loaded: {df.iloc[idx]['filename']}", icon="📂")

    def _session_detail_tab(self):
        """Deep-Dive Tab: Detailed visualization of a specific session."""
        sid = st.session_state.get("active_session_id")
        session = self._history.get_by_id(sid) if sid else None
        
        if not session:
            st.warning("No session selected. Please pick one from 'History' or run a 'New Audit'.")
            return

        # Header Badge & Metrics
        render_filename_badge(session.filename, session.session_id)
        
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Overall Quality", f"{session.scores.final_score}/100")
        with m2: render_three_gauges(session.scores) # Compliance, Empathy, Integrity
        with m3: render_cost_card(session.token_count, session.cost_usd)

        st.divider()
        
        # Split view for Transcript and Wrong Turns
        col_tx, col_err = st.columns([2, 1])
        with col_tx:
            st.markdown("#### 💬 Verified Transcript")
            render_transcript(session.transcript)
        
        with col_err:
            st.markdown("#### 🚩 Policy Critical Moments")
            if session.wrong_turns:
                render_wrong_turns(session.wrong_turns)
            else:
                st.success("Compliant Session: No policy breaches detected.")

    def _performance_tab(self):
        """Analytics Tab: User performance over time."""
        st.markdown("### 📈 Performance Trends")
        sessions = self._history.get_all()
        if not sessions:
            st.info("Insufficient data for analytics.")
            return
            
        # Example trend chart
        scores = [s.scores.final_score for s in reversed(sessions[-10:])]
        st.line_chart(scores)
        st.caption("Visualizing QA Score trends over the last 10 sessions.")
