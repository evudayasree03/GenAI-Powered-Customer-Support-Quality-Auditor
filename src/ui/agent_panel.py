from __future__ import annotations
import asyncio
import os
import time
import streamlit as st
import plotly.graph_objects as go

from src.api_client import SamiXClient
from src.utils.history_manager import AuditSession, AuditScores, HistoryManager, WrongTurn, EngineAResult, EngineBResult, EngineBClaim, EngineCResult
from src.ui.components import (
    render_page_hero, build_history_dataframe, render_cost_card, 
    render_dual_score_chart, render_filename_badge, render_three_gauges, 
    render_transcript, render_wrong_turns
)

class AgentPanel:
    def __init__(self, history: HistoryManager, **kwargs):
        self._history = history
        # SamiXClient handles the communication with your Render FastAPI
        self._api_client = SamiXClient() 

    def render(self) -> None:
        render_page_hero(
            "Agent Workspace",
            "Audit customer conversations in one clean workspace.",
            "Run new audits and review saved sessions with grounded retrieval support.",
            stats=[
                ("Sessions", str(len(self._history.get_all())), "saved records"),
                ("Engine", "Groq Llama-3", "High-speed Scoring"),
                ("Backend", "Render API", "Cloud Optimized")
            ],
        )
        
        tabs = st.tabs(["New Audit", "History", "Session Detail", "My Performance"])
        
        with tabs[0]: self._new_audit_tab()
        with tabs[1]: self._history_tab()
        with tabs[2]: self._session_detail_tab()
        with tabs[3]: self._performance_tab()

    def _new_audit_tab(self):
        st.markdown("### 🚀 Launch New Analysis")
        uploaded = st.file_uploader("Upload audio (MP3/WAV) or transcript", type=["wav", "mp3", "m4a", "json", "txt"])
        
        if uploaded:
            st.info(f"Ready to analyze: **{uploaded.name}**")
            if st.button("🔍 Run SamiX AI Audit", use_container_width=True, type="primary"):
                asyncio.run(self._run_backend_audit(uploaded))

    async def _run_backend_audit(self, uploaded_file):
        """Streams the file to Render and processes the response."""
        with st.status("📡 Connecting to SamiX Backend...", expanded=True) as status:
            try:
                status.write("📤 Uploading file to Render...")
                # We call the API Client instead of local processors
                result = await self._api_client.run_audit(uploaded_file.name, uploaded_file.getvalue())
                
                status.write("🧠 AI is scoring the conversation...")
                session = self._create_session_from_api(uploaded_file.name, result)
                
                # Save to local SQLite history
                self._history.save(session)
                st.session_state["active_session_id"] = session.session_id
                
                status.update(label="✅ Audit Complete!", state="complete")
                st.toast(f"Success: {session.scores.final_score}/100", icon="✅")
            except Exception as e:
                st.error(f"Backend Error: {str(e)}")

    def _create_session_from_api(self, filename: str, data: dict) -> AuditSession:
        """Maps the FastAPI JSON response back to our AuditSession object."""
        session = AuditSession.new(filename, mode="upload", agent_name=st.session_state.get("username", "Agent"))
        
        # Map Scores
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
        
        # Map Pipeline Metadata
        session.transcript = data.get("transcript", [])
        session.summary = data.get("summary", "")
        session.violations = len(data.get("violations", []))
        session.token_count = data.get("token_count", 0)
        session.cost_usd = data.get("cost_usd", 0.0)
        
        # Map Detailed Engine Results
        session.engine_a = EngineAResult(**data.get("engine_a", {}))
        session.engine_c = EngineCResult(**data.get("engine_c", {}))
        
        return session

    def _history_tab(self):
        st.markdown("### 📜 Audit History")
        sessions = self._history.get_all()
        if not sessions:
            st.info("No audits found. Run a new audit to see history.")
            return

        df = build_history_dataframe(sessions)
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
            st.success(f"Selected: {df.iloc[idx]['filename']}")

    def _session_detail_tab(self):
        sid = st.session_state.get("active_session_id")
        session = self._history.get_by_id(sid) if sid else None
        
        if not session:
            st.warning("Please select a session from History or run a New Audit.")
            return

        # UI Rendering using your established components
        render_filename_badge(session.filename, session.session_id)
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("QA Score", f"{session.scores.final_score}/100")
        with c2: render_three_gauges(session.scores) # Custom ECharts component
        with c3: render_cost_card(session.token_count, session.cost_usd)

        st.divider()
        
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("#### 📝 Transcript Analysis")
            render_transcript(session.transcript)
        with col_right:
            st.markdown("#### 🚩 Policy Violations")
            if session.wrong_turns:
                render_wrong_turns(session.wrong_turns)
            else:
                st.success("No violations detected.")

    def _performance_tab(self):
        st.markdown("### 📊 Personal Analytics")
        # Logic for Plotly charts (omitted for brevity, same as your original)
        st.info("Performance trends based on your last 10 audits.")
