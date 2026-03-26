"""
SamiX Agent Intelligence Workspace

This module implements the primary interface for quality auditors and agents.
It manages the end-to-end audit lifecycle:
1. Ingestion: Support for audio (MP3/WAV) and structured text (JSON/CSV).
2. Processing: Multi-stage pipeline (pydub -> Deepgram -> Groq -> Milvus).
3. Inspection: Deep-dive analysis of sessions with multi-dimensional scoring.
4. Analytics: Individual performance tracking and trend analysis.
"""
from __future__ import annotations

import asyncio
import io
import os
import time
from typing import Optional, Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipeline.alert_engine     import AlertEngine
from src.pipeline.groq_client      import GroqClient
from src.pipeline.stt_processor    import STTProcessor, transcript_to_text
from src.utils.audio_processor     import AudioProcessor
from src.utils.cost_tracker        import CostTracker
from src.utils.history_manager     import (
    AuditSession, AuditScores, HistoryManager,
    WrongTurn, EngineAResult, EngineBResult, EngineBClaim, EngineCResult,
)
from src.utils.kb_manager          import KBManager
from src.utils.report_generator    import ReportGenerator
from src.api_client                import SamiXClient
from audio_recorder_streamlit import audio_recorder
from src.ui.components import (
    render_page_hero,
    build_history_dataframe, render_cost_card, render_dual_score_chart,
    render_filename_badge, render_three_gauges, render_transcript,
    render_wrong_turns,
)


class AgentPanel:
    """
    The main UI controller for the SamiX Agent Workspace.
    
    Coordinates a complex array of pipeline services to provide agents with 
    instant, AI-driven feedback on their customer interactions.
    """
    def __init__(
        self,
        history:  HistoryManager,
        groq:     GroqClient,
        stt:      STTProcessor,
        audio:    AudioProcessor,
        cost:     CostTracker,
        alerts:   AlertEngine,
        kb:       KBManager,
    ) -> None:
        self._history  = history
        self._groq     = groq
        self._stt      = stt
        self._audio    = audio
        self._cost     = cost
        self._alerts   = alerts
        self._kb       = kb
        self._reports  = ReportGenerator()
        self._api_client = SamiXClient()

    # Entry 
    def render(self) -> None:
        """
        Renders the multi-tabbed agent workspace.
        Organizes the workflow into Audit, History, Detail, and Personal Analytics.
        """
        render_page_hero(
            "Agent Workspace",
            "Audit customer conversations in one clean workspace.",
            "Run new audits, inspect wrong turns, and review saved sessions with grounded retrieval support.",
            stats=[
                ("Sessions", str(len(self._history.get_all())), "saved audit records"),
                ("Scoring", "Groq", "summary and QA scoring"),
                ("STT", "DG / Whisper", "cloud primary with fallback"),
                ("RAG", "Milvus Lite", "LangChain retrieval"),
            ],
        )
        tabs = st.tabs([
            "New Audit",
            "History",
            "Session Detail",
            "My Scores",
        ])
        with tabs[0]: self._new_audit()
        with tabs[1]: self._history_tab()
        with tabs[2]: self._session_detail()
        with tabs[3]: self._my_scores()

    # TAB 1 — New audit
    def _new_audit(self) -> None:
        """
        The Ingestion Tab.
        Handles file uploads and triggers the SamiX analysis pipeline.
        """
        st.markdown('<div class="section-header">New Audit</div>', unsafe_allow_html=True)

        col_up, col_info = st.columns([3, 2], gap="large")

        with col_up:
            uploaded = st.file_uploader(
                "Drop transcript or audio",
                type=["csv","json","txt","mp3","wav","m4a","flac","ogg"],
                help="Filename is stored exactly as uploaded — no renaming.",
                label_visibility="visible",
            )
            if uploaded and len(uploaded.getvalue()) > 0:
                st.markdown(
                    f'<div class="mono-badge">📄 {uploaded.name} · '
                    f'{len(uploaded.getvalue())/1024:.1f} KB</div>',
                    unsafe_allow_html=True,
                )
                self._pipeline_strip()
                st.markdown("")
                
                # Workspace Switcher
                mode = st.radio("Audit Mode", ["📂 File Upload", "🎤 Live Call"], horizontal=True)

                if mode == "📂 File Upload":
                    if st.button("🔍  Analyse with SamiX", use_container_width=True, type="primary"):
                        asyncio.run(self._run_audit(uploaded.name, uploaded.getvalue()))
                else:
                    self._live_call_workspace()
            else:
                # Even without upload, allow Live Call
                mode = st.radio("Audit Mode", ["📂 File Upload", "🎤 Live Call"], horizontal=True, index=1)
                if mode == "🎤 Live Call":
                    self._live_call_workspace()
                else:
                    st.info("Please upload a file to begin.")

        with col_info:
            st.markdown("##### What SamiX analyses")
            rows = [
                ("Speaker separation", "AGENT + CUSTOMER"),
                ("Filename preserved", "always"),
                ("pydub conversion",   "WAV 16 kHz mono"),
                ("Quantity of logs",   f"{len(self._history.get_all())} sessions"),
                ("Groundedness check", "LangChain + Milvus"),
                ("Groq dual-call",     "summarise + score"),
                ("Dual scoring",       "agent 0-100 + cust 0-10"),
                ("Wrong turns",        "exact quote + fix"),
            ]
            for label, val in rows:
                c1, c2 = st.columns([3, 2])
                c1.caption(label)
                c2.markdown(f'<div class="mono-badge">{val}</div>',
                            unsafe_allow_html=True)

            # Last session quick-view
            sessions = self._history.get_all()
            if sessions:
                last = sessions[0]
                st.markdown("---")
                st.markdown("##### Last audit")
                col = "verdict-fail" if "Fail" in last.scores.verdict else "verdict-good"
                st.markdown(
                    f'<div class="glass-card">'
                    f'<div style="font-size:.72rem;color:#64748B;font-family:Sora,sans-serif;">'
                    f'{last.upload_time}</div>'
                    f'<div style="font-size:.95rem;font-weight:600;color:#212529;margin:4px 0;">{last.filename}</div>'
                    f'<div class="{col}" style="font-family:Sora,sans-serif;font-weight:500;font-size:.82rem;">'
                    f'{last.scores.final_score:.0f}/100 · {last.scores.verdict}</div>'
                    f'</div>', unsafe_allow_html=True
                )
                if st.button("Open last session →", key="open_last"):
                    st.session_state["active_session_id"] = last.session_id

    # TAB 2 — History
    def _history_tab(self) -> None:
        st.markdown('<div class="section-header">Session History</div>',
                    unsafe_allow_html=True)

        sessions = self._history.get_all()
        if not sessions:
            st.info("No audits yet — upload a file in the 'New audit' tab.")
            return

        # Live analytics strip
        total   = len(sessions)
        avg_s   = sum(s.scores.final_score for s in sessions) / total
        total_v = sum(s.violations for s in sessions)
        fails   = sum(1 for s in sessions if s.scores.final_score < 60)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Sessions assessed", str(total))
        c2.metric("Avg QA score",      f"{avg_s:.1f}/100")
        c3.metric("Total violations",  str(total_v))
        c4.metric("Below threshold",   str(fails), delta=f"{fails/total*100:.0f}%",
                  delta_color="inverse")

        st.markdown("---")

        # Search
        query = st.text_input("🔍 Search by filename or agent", "",
                               placeholder="billing_chat.csv")
        if query:
            sessions = self._history.search(query)
            if not sessions:
                st.warning("No sessions match that query.")
                return

        df  = build_history_dataframe(sessions)
        vis = [c for c in df.columns if c != "session_id"]

        st.markdown(
            '<div style="font-family:Sora,sans-serif;font-size:.72rem;color:#64748B;margin-bottom:6px;font-weight:500;">'
            '✓ Uploaded filename = Stored history name (guaranteed identical)</div>',
            unsafe_allow_html=True,
        )

        sel = st.dataframe(
            df[vis], use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
        )
        if sel and sel.selection and sel.selection.rows:
            idx = sel.selection.rows[0]
            sid = df.iloc[idx]["session_id"]
            session = self._history.get_by_id(sid)
            if session:
                st.session_state["active_session_id"] = sid
                st.toast(f"Opened: {session.filename}", icon="📂")

    # TAB 3 — Session detail
    def _session_detail(self) -> None:
        """
        The Deep-Dive Session Review Tab.
        Displays detailed audit results, including audio playback, transcripts,
        multi-dimensional scoring, and RAG-verified 'wrong turns'.
        """
        sid = st.session_state.get("active_session_id")
        session = self._history.get_by_id(sid) if sid else None
        if not session:
            sessions = self._history.get_all()
            session  = sessions[0] if sessions else None
        if not session:
            st.info("No session selected. Run an audit or pick one from History.")
            return

        s = session.scores
        verdict_colour = (
            "#10B981" if s.final_score >= 80
            else "#EB643E" if s.final_score >= 70
            else "#F59E0B" if s.final_score >= 60
            else "#EF4444"
        )
        # Header bar
        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;'
            f'border-radius:14px;padding:22px 28px;margin-bottom:1.5rem;box-shadow:var(--shadow-sm);">'
            f'<div style="font-family:Inter,sans-serif;font-weight:700;font-size:0.75rem;color:#152EAE;margin-bottom:8px;letter-spacing:0.05em;text-transform:uppercase;">'
            f'Analysis Session · {session.upload_time}</div>'
            f'<div style="font-size:1.6rem;font-weight:800;color:#0F172A;margin-bottom:12px;letter-spacing:-0.5px;">'
            f'{session.filename}</div>'
            f'<div style="display:flex;gap:28px;flex-wrap:wrap;'
            f'font-family:Inter,sans-serif;font-size:.85rem;color:#475569;font-weight:600;">'
            f'<span>Performance: <span style="color:#152EAE;font-weight:800;">'
            f'{s.final_score:.0f}/100</span></span>'
            f'<span>Sensitivity: <span style="color:#10B981;font-weight:800;">'
            f'{s.customer_overall:.1f}/10</span></span>'
            f'<span>Violations: <span style="color:#EF4444;font-weight:800;">{session.violations}</span></span>'
            f'<span>Turns: <span style="color:#0F172A;">{len(session.transcript)}</span></span>'
            f'<span>Duration: <span style="color:#0F172A;">{self._audio.duration_label(session.duration_sec)}</span></span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        render_filename_badge(session.filename, session.stored_name)

        # Audio player (if audio file) 
        audio_path = os.path.join("data", "uploads", session.filename)
        if os.path.exists(audio_path):
            st.markdown("##### Audio playback")
            with open(audio_path, "rb") as fh:
                st.audio(fh.read())

        # pydub Smart Summary 
        col_a, col_b = st.columns([4, 1])
        with col_a:
            display_text = session.summary_customer_query if session.summary_customer_query else (session.summary or "Run an audit to generate the summary.")
            
            if session.summary_customer_query:
                subs = "".join([f"<li style='margin-bottom:4px;'>{sq}</li>" for sq in session.summary_sub_queries])
                html_summary = f'''
                <div class="glass-card" style="font-size:.9rem;color:#475569;line-height:1.7;">
                    <div style="margin-bottom:12px;"><strong style="color:#0F172A;font-weight:700;">Primary Query:</strong> {session.summary_customer_query}</div>
                    <div style="margin-bottom:12px;"><strong style="color:#0F172A;font-weight:700;">Expectation:</strong> {session.summary_customer_expectation}</div>
                    <div style="margin-bottom:4px;"><strong style="color:#0F172A;font-weight:700;">Key Topics:</strong></div>
                    <ul style="margin:4px 0 0 20px;padding:0;color:#64748B;">{subs}</ul>
                </div>
                '''
            else:
                html_summary = f'<div class="glass-card" style="font-size:.9rem;color:#475569;line-height:1.7;">{display_text}</div>'
                
            st.markdown(html_summary, unsafe_allow_html=True)

        with col_b:
            if st.button("▶ Smart Summary", use_container_width=True, key="smart_sum"):
                with st.spinner("Generating audio via pydub…"):
                    aud_text = session.summary_customer_query if session.summary_customer_query else (session.summary or "")
                    text = self._audio.generate_text_summary(
                        transcript_summary=aud_text,
                        key_moments=[wt.what_went_wrong[:80] for wt in session.wrong_turns[:3]],
                        scores={"final_score": s.final_score,
                                "verdict": s.verdict,
                                "integrity": s.integrity},
                    )
                    audio_bytes = self._audio.synthesise_audio(text)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        with st.expander("Text summary"):
                            st.write(text)

        # Sub-tabs 
        sub = st.tabs([
            "🗒 Transcript",
            "📈 Dual scoring",
            "🔴 Where it went wrong",
            "🔗 RAG groundedness",
            "💬 Feedback",
            "⬇ Download",
        ])
        with sub[0]: self._tab_transcript(session)
        with sub[1]: self._tab_scoring(session)
        with sub[2]: self._tab_wrong(session)
        with sub[3]: self._tab_rag(session)
        with sub[4]: self._tab_feedback(session)
        with sub[5]: self._tab_download(session)

    # Sub-tab: Transcript 
    def _tab_transcript(self, session: AuditSession) -> None:
        c1, c2, c3 = st.columns([2,2,3])
        c1.markdown('<span class="mono-badge">● AGENT</span>',
                    unsafe_allow_html=True)
        c2.markdown('<span class="mono-badge" style="color:#64748B;border-color:#E2E8F0;">○ CUSTOMER</span>',
                    unsafe_allow_html=True)
        c3.markdown('<span style="font-size:.68rem;color:#94A3B8;font-family:Sora,sans-serif;">'
                    'Deepgram diarization · conf 0.94</span>', unsafe_allow_html=True)
        st.markdown("")
        render_transcript(session.transcript, session.wrong_turns)

    # Sub-tab: Dual scoring 
    def _tab_scoring(self, session: AuditSession) -> None:
        s = session.scores
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Agent final",         f"{s.final_score:.0f}/100")
        c2.metric("Customer sentiment",  f"{s.customer_overall:.1f}/10")
        c3.metric("Critical moments",    str(len(session.wrong_turns)))
        c4.metric("Phase bonus",         f"{s.phase_bonus:+.1f} pts")

        st.markdown("##### Compliance · Empathy · Professionalism (ECharts gauges)")
        render_three_gauges(s)

        st.markdown("##### Agent quality + Customer sentiment — turn by turn")
        st.caption("Red shading = below 40% threshold")
        render_dual_score_chart(s)

        st.markdown("##### All 6 dimensions")
        dims = [
            ("EMPATHY",        s.empathy,        0.20),
            ("PROFESSIONALISM", s.professionalism, 0.15),
            ("COMPLIANCE",     s.compliance,      0.25),
            ("RESOLUTION",     s.resolution,      0.20),
            ("COMMUNICATION",  s.communication,   0.05),
            ("INTEGRITY",      s.integrity,       0.15),
        ]
        for name, val, weight in dims:
            dn, db, dv, dw = st.columns([2,5,1,1])
            dn.caption(name)
            db.progress(val / 10.0)
            dv.markdown(f'<div style="font-size:.9rem;font-weight:500;">{val:.1f}</div>',
                        unsafe_allow_html=True)
            dw.caption(f"{int(weight*100)}%")

        st.markdown("##### Cost evaluation")
        render_cost_card(session.token_count, session.cost_usd)

        st.markdown("---")
        st.markdown("##### Dual Engines Verification")
        
        ce_a, ce_b, ce_c = st.columns(3)
        with ce_a:
            st.markdown("###### A: Query Resolution")
            resolved = session.engine_a.resolution_state
            fc_color = "#EF4444" if session.engine_a.is_fake_close else "#10B981"
            st.markdown(f'<div class="glass-card" style="font-size:0.82rem; height: 160px; overflow-y: auto;">'
                        f'<div style="color:#64748B; margin-bottom:4px;">Primary Query Handled: <br><b style="color:{"#10B981" if session.engine_a.primary_query_answered else "#EF4444"}">{session.engine_a.primary_query_answered}</b></div>'
                        f'<div style="color:#64748B; margin-bottom:4px;">Sub-Queries Handled: <br><b style="color:{"#10B981" if session.engine_a.sub_queries_addressed else "#EF4444"}">{session.engine_a.sub_queries_addressed}</b></div>'
                        f'<div style="color:#64748B; margin-bottom:4px;">Fake Close Detected: <br><b style="color:{fc_color}">{session.engine_a.is_fake_close}</b></div>'
                        f'<div style="margin-top:8px;padding:5px;border-radius:6px;background:#F8FAFC;color:#212529;font-size:0.75rem;border:1px solid #E2E8F0;">State: {resolved}</div>'
                        f'</div>', unsafe_allow_html=True)
                        
        with ce_b:
            st.markdown("###### B: Hallucinations")
            claims = session.engine_b.claims
            if claims:
                claims_html = ""
                for c in claims:
                    warn = "#EF4444" if (c.is_unverifiable or c.is_impossible_promise or c.is_contradiction) else "#10B981"
                    status = "Failed" if (warn == "#EF4444") else "Verified"
                    claims_html += f'<div style="padding:8px 0; border-bottom:1px solid #F1F5F9;">'
                    claims_html += f'<div style="color:#212529; line-height: 1.3; font-weight:500;">"{c.claim}"</div>'
                    claims_html += f'<div style="color:{warn}; font-size:0.72rem; font-family:Sora,sans-serif; margin-top:4px; font-weight:600;">[{status}] Conf: {c.confidence_score:.2f}</div>'
                    claims_html += f'</div>'
                st.markdown(f'<div class="glass-card" style="font-size:0.8rem; height: 160px; overflow-y: auto;">{claims_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="glass-card" style="font-size:0.8rem; color:#10B981; height: 160px; overflow-y: auto;">No factual claims evaluated.</div>', unsafe_allow_html=True)

        with ce_c:
            st.markdown("###### C: False Close")
            rushed = "#EF4444" if session.engine_c.agent_rushed else "#10B981"
            frustrated = "#EF4444" if session.engine_c.customer_frustrated_but_ok else "#10B981"
            confirmed = "#10B981" if session.engine_c.resolution_confirmed_by_customer else "#F59E0B"
            st.markdown(f'<div class="glass-card" style="font-size:0.8rem; height: 160px; overflow-y: auto;">'
                        f'<div style="color:#94A3B8; margin-bottom:8px;">Agent Rushed Close: <br><b style="color:{rushed}">{session.engine_c.agent_rushed}</b></div>'
                        f'<div style="color:#94A3B8; margin-bottom:8px;">Cust OK but Frustrated: <br><b style="color:{frustrated}">{session.engine_c.customer_frustrated_but_ok}</b></div>'
                        f'<div style="color:#94A3B8; margin-bottom:8px;">Resolution Confirmed: <br><b style="color:{confirmed}">{session.engine_c.resolution_confirmed_by_customer}</b></div>'
                        f'</div>', unsafe_allow_html=True)

    # Sub-tab: Where it went wrong 
    def _tab_wrong(self, session: AuditSession) -> None:
        st.caption(
            "Exact turn · verbatim quote · specific factual error · "
            "RAG-verified correct fact · exact phrase the agent should have used"
        )
        render_wrong_turns(session.wrong_turns)

    # Sub-tab: RAG groundedness 
    def _tab_rag(self, session: AuditSession) -> None:
        st.markdown("##### Context-aware policy audit")
        st.caption("Query the RAG knowledge base against any agent statement")

        query = st.text_input(
            "Enter statement to check against KB",
            placeholder='e.g. "Refund takes 2 days"',
            key="rag_query",
        )
        if query and st.button("Run RAG audit", key="run_rag"):
            with st.spinner("Retrieving from Milvus Lite…"):
                result = asyncio.run(
                    self._kb.audit_chain(
                        agent_statement=query,
                        context_question=query,
                    )
                )
            breach_colour = "#EF4444" if result["policy_breach"] else "#10B981"
            breach_label  = "POLICY BREACH DETECTED" if result["policy_breach"] else "COMPLIANT"
            st.markdown(
                f'<div style="border-left:3px solid {breach_colour};'
                f'padding:8px 12px;background:rgba(0,0,0,.2);border-radius:0 6px 6px 0;">'
                f'<span style="color:{breach_colour};font-family:Sora,sans-serif;font-weight:700;font-size:.8rem;">'
                f'{breach_label}</span><br>'
                f'<span style="font-size:.75rem;color:#64748B;">'
                f'Groundedness: {result["groundedness"]:.2f} · '
                f'Top source: {result.get("top_source","—")}</span>'
                f'</div>', unsafe_allow_html=True
            )
            if result.get("citations"):
                st.markdown("**Citations:**")
                for cite in result["citations"]:
                    st.markdown(f"- `{cite}`")

        # Show existing wrong-turn RAG citations
        if session.wrong_turns:
            st.markdown("##### RAG citations from this session")
            for wt in session.wrong_turns:
                st.markdown(
                    f'<div class="glass-card" style="font-size:.82rem;">'
                    f'<b>T{wt.turn_number}</b> · {wt.rag_source} · '
                    f'conf <b>{wt.rag_confidence:.2f}</b><br>'
                    f'<span style="color:#94A3B8;">{wt.what_went_wrong[:120]}</span>'
                    f'</div>', unsafe_allow_html=True
                )

    # Sub-tab: Feedback 
    def _tab_feedback(self, session: AuditSession) -> None:
        st.markdown("##### Rate this audit")
        rating = st.select_slider(
            "Scoring accuracy",
            options=[1,2,3,4,5],
            value=4,
            format_func=lambda x: "★"*x + "☆"*(5-x),
        )
        note = st.text_area("Feedback note", height=80,
                             placeholder="Comments on scoring accuracy…")
        if st.button("Submit feedback", key="fbsub"):
            session.feedback.append({
                "rating":    rating,
                "note":      note,
                "reviewer":  "Supervisor",
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            })
            self._history.save(session)
            st.toast("Feedback saved", icon="✅")

        if session.feedback:
            st.markdown("##### Online feedback — this session")
            for fb in session.feedback:
                stars = "★" * fb.get("rating",3) + "☆"*(5-fb.get("rating",3))
                st.markdown(
                    f'<div class="glass-card">'
                    f'<div style="font-size:.72rem;color:#64748B;font-family:Sora,sans-serif;font-weight:600;">'
                    f'{fb.get("reviewer","—")} · {fb.get("timestamp","—")} · {stars}</div>'
                    f'<div style="font-size:.85rem;color:#94A3B8;margin-top:4px;">'
                    f'{fb.get("note","—")}</div></div>',
                    unsafe_allow_html=True,
                )

    # Sub-tab: Download 
    def _tab_download(self, session: AuditSession) -> None:
        st.markdown("##### Export report")
        st.caption(
            f"All exports use the original filename: `{session.filename}` "
            f"(matches stored history name exactly)"
        )

        base = session.filename.rsplit(".", 1)[0]
        c1, c2, c3, c4, c5 = st.columns(5)

        # PDF
        with st.spinner(""):
            pdf_bytes = self._reports.to_pdf(session)
        c1.download_button(
            "⬇ PDF", data=pdf_bytes,
            file_name=f"{base}_audit.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        # Excel
        xlsx_bytes = self._reports.to_excel(session)
        c2.download_button(
            "⬇ Excel", data=xlsx_bytes,
            file_name=f"{base}_audit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # JSON
        import json, dataclasses
        c3.download_button(
            "⬇ JSON",
            data=json.dumps(dataclasses.asdict(session), indent=2, default=str).encode(),
            file_name=f"{base}_audit.json",
            mime="application/json",
            use_container_width=True,
        )

        # TXT
        c4.download_button(
            "⬇ TXT",
            data=self._build_txt(session).encode(),
            file_name=f"{base}_audit.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # CSV scores
        rows = "\n".join([
            "dimension,score,weight",
            f"Empathy,{session.scores.empathy:.1f},20",
            f"Professionalism,{session.scores.professionalism:.1f},15",
            f"Compliance,{session.scores.compliance:.1f},25",
            f"Resolution,{session.scores.resolution:.1f},20",
            f"Communication,{session.scores.communication:.1f},5",
            f"Integrity,{session.scores.integrity:.1f},15",
            f"Final,{session.scores.final_score:.1f},100",
        ])
        c5.download_button(
            "⬇ CSV",
            data=rows.encode(),
            file_name=f"{base}_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        if st.button("📧 Email report to supervisor"):
            to = st.text_input("Recipient email", "anandkumarkambar@gmail.com", key="rep_email")
            if to:
                self._alerts.send_custom(
                    to=to,
                    subject=f"SamiX Audit Report — {session.filename}",
                    body=self._build_txt(session)[:2000],
                )

    # TAB 4 — My scores
    def _my_scores(self) -> None:
        st.markdown('<div class="section-header">MY SCORES</div>', unsafe_allow_html=True)
        sessions = self._history.get_all()
        if not sessions:
            st.info("No sessions yet.")
            return

        avg  = sum(s.scores.final_score for s in sessions) / len(sessions)
        cust = sum(s.scores.customer_overall for s in sessions) / len(sessions)
        viol = sum(s.violations for s in sessions)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg QA score",       f"{avg:.1f}/100")
        c2.metric("Avg cust sentiment", f"{cust:.1f}/10")
        c3.metric("Total violations",   str(viol))
        c4.metric("Sessions",           str(len(sessions)))

        times  = [s.upload_time[:16] for s in reversed(sessions)]
        values = [s.scores.final_score for s in reversed(sessions)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=values, mode="lines+markers",
            line=dict(color="#152EAE", width=3),
            marker=dict(size=8, color=[
                "#EF4444" if v < 60 else "#F59E0B" if v < 75 else "#10B981"
                for v in values
            ]),
            fill="tozeroy", fillcolor="rgba(21, 46, 174, 0.04)",
        ))
        fig.add_hline(y=60, line_dash="dash",
                      line_color="rgba(239,68,68,.3)",
                      annotation_text="Fail threshold")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", color="#64748B", size=10),
            margin=dict(l=0, r=0, t=10, b=0), height=210,
            yaxis=dict(range=[0,105], gridcolor="#F1F5F9"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Helpers 
    def _pipeline_strip(self) -> None:
        steps = [
            ("01", "pydub",    "WAV 16 kHz"),
            ("02", "Deepgram", "STT+diarize"),
            ("03", "Groq x2",  "score+detect"),
            ("04", "Milvus",   "RAG policy"),
            ("05", "Store",    "same filename"),
        ]
        cols = st.columns(len(steps))
        for col, (num, name, sub) in zip(cols, steps):
            col.markdown(
                f'<div style="border:1px solid rgba(235,100,62,0.15);border-radius:8px;'
                f'padding:8px;text-align:center;background:#FFFFFF;box-shadow:var(--shadow);">'
                f'<div style="font-family:Sora,sans-serif;font-size:.6rem;color:#94A3B8;font-weight:600;">{num}</div>'
                f'<div style="font-size:.8rem;font-weight:700;color:#EB643E;margin:2px 0;">{name}</div>'
                f'<div style="font-size:.65rem;color:#64748B;font-weight:500;">{sub}</div>'
                f'</div>', unsafe_allow_html=True
            )

    def _live_call_workspace(self) -> None:
        """
        The Real-time Live Call Workspace.
        Integrates audio recording, live Whisper transcription, and RAG suggestions.
        """
        st.markdown('<div class="glass-card" style="padding:20px;">', unsafe_allow_html=True)
        st.markdown("##### 🎤 Live Assistant")
        st.caption("Recording is processed in chunks for real-time RAG suggestions.")

        # Initialize session state for live call
        if "live_transcript" not in st.session_state:
            st.session_state.live_transcript = []
        if "live_suggestions" not in st.session_state:
            st.session_state.live_suggestions = []
        if "is_recording" not in st.session_state:
            st.session_state.is_recording = False

        c1, c2 = st.columns([1, 2])
        
        with c1:
            audio_bytes = audio_recorder(
                text="Click to Record",
                recording_color="#e74c3c",
                neutral_color="#95a5a6",
                icon_name="microphone",
                icon_size="2x",
            )
            
            if audio_bytes:
                st.session_state.is_recording = True
                with st.spinner("Processing turn..."):
                    # Process the chunk asynchronously
                    turns = asyncio.run(self._stt.process_chunk(audio_bytes))
                    if turns:
                        for t in turns:
                            st.session_state.live_transcript.append(t)
                            # Get real-time suggestions
                            suggs = asyncio.run(self._kb.get_live_suggestions(t.text))
                            if suggs:
                                st.session_state.live_suggestions = suggs

            if st.button("🛑 End Call & Audit", use_container_width=True, type="primary"):
                if st.session_state.live_transcript and len(st.session_state.live_transcript) > 0:
                    # Validate transcript turns have required fields
                    try:
                        for turn in st.session_state.live_transcript:
                            if not hasattr(turn, 'speaker') or not hasattr(turn, 'text'):
                                st.error("Invalid transcript structure. Please try again.")
                                return
                        # Finalize the session
                        filename = f"live_call_{int(time.time())}.wav"
                        # For live call, we treat the transcript as the source
                        asyncio.run(self._run_audit(filename, b"", transcript_override=st.session_state.live_transcript))
                        # Reset state
                        st.session_state.live_transcript = []
                        st.session_state.live_suggestions = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error finalizing live call: {e}")
                else:
                    st.warning("No transcript recorded. Please record some audio first.")

        with c2:
            st.markdown("###### Active Suggestions")
            if st.session_state.live_suggestions:
                for s in st.session_state.live_suggestions:
                    st.info(s)
            else:
                st.caption("Suggestions will appear here based on the conversation.")

        # Display growing transcript
        if st.session_state.live_transcript:
            st.markdown("---")
            st.markdown("###### Live Transcript")
            for t in st.session_state.live_transcript[-5:]:
                st.markdown(f"**{t.speaker}:** {t.text}")

        st.markdown('</div>', unsafe_allow_html=True)

    async def _run_audit(self, filename: str, file_bytes: bytes, transcript_override: list = None) -> None:
        """
        [ASYNC] Core SamiX Pipeline orchestrator.
        Uses FastAPI backend if SAMIX_API_URL is configured, else runs locally.
        """
        # ── FastAPI Backend Path ──
        if self._api_client.is_available and transcript_override is None:
            with st.status(f"Analysing {filename} via API…", expanded=True) as status:
                try:
                    st.write("📡 Sending to SamiX API backend…")
                    result = await self._api_client.run_audit(filename, file_bytes)

                    # Build session from API response
                    session = AuditSession.new(filename, mode="upload", agent_name="Alex K.")
                    s = result.get("scores", {})
                    session.scores = AuditScores(
                        empathy=s.get("empathy", 5.0),
                        professionalism=s.get("professionalism", 5.0),
                        compliance=s.get("compliance", 5.0),
                        resolution=s.get("resolution", 5.0),
                        communication=s.get("communication", 5.0),
                        integrity=s.get("integrity", 5.0),
                        opening=s.get("opening", 5.0),
                        middle=s.get("middle", 5.0),
                        closing=s.get("closing", 5.0),
                        phase_bonus=s.get("phase_bonus", 0.0),
                        final_score=s.get("final_score", 50.0),
                        verdict=s.get("verdict", "Needs work"),
                        customer_sentiment=s.get("customer_sentiment", []),
                        customer_overall=s.get("customer_overall", 5.0),
                        agent_by_turn=s.get("agent_by_turn", []),
                    )
                    sm = result.get("summary", {})
                    session.summary_customer_query = sm.get("customer_query", "")
                    session.summary_sub_queries = sm.get("sub_queries", [])
                    session.summary_query_category = sm.get("query_category", "")
                    session.summary_customer_expectation = sm.get("customer_expectation", "")
                    session.summary_phases = sm.get("phases", {})
                    session.summary_key_moments = sm.get("key_moments", [])
                    session.violations = len(result.get("violations", []))
                    session.wrong_turns = [
                        WrongTurn(**wt) for wt in result.get("wrong_turns", [])
                    ]
                    session.token_count = result.get("token_count", 0)
                    session.cost_usd = result.get("cost_usd", 0.0)
                    session.duration_sec = result.get("duration_sec", 0)
                    session.engine_a = EngineAResult(
                        primary_query_answered=result.get("engine_a", {}).get("primary_query_answered", False),
                        sub_queries_addressed=result.get("engine_a", {}).get("sub_queries_addressed", False),
                        is_fake_close=result.get("engine_a", {}).get("is_fake_close", False),
                        resolution_state=result.get("engine_a", {}).get("resolution_state", "Unknown"),
                    )
                    eb_data = result.get("engine_b", {})
                    session.engine_b = EngineBResult(claims=[
                        EngineBClaim(**c) for c in eb_data.get("claims", [])
                    ])
                    ec_data = result.get("engine_c", {})
                    session.engine_c = EngineCResult(
                        customer_frustrated_but_ok=ec_data.get("customer_frustrated_but_ok", False),
                        agent_rushed=ec_data.get("agent_rushed", False),
                        resolution_confirmed_by_customer=ec_data.get("resolution_confirmed_by_customer", False),
                    )

                    self._history.save(session)
                    st.session_state["active_session_id"] = session.session_id

                    status.update(
                        label=f"✅ Audit complete — {session.scores.verdict}",
                        state="complete",
                    )
                    st.toast(
                        f"{filename} · {session.scores.final_score:.0f}/100 · {session.scores.verdict}",
                        icon="✅" if session.scores.final_score >= 60 else "⚠️",
                    )
                    return
                except Exception as exc:
                    st.warning(f"API call failed ({exc}), falling back to local processing…")

        # ── Local Processing Path (original) ──
        """
        [ASYNC] The core SamiX Pipeline orchestrator.
        Uses asyncio.gather for high-speed concurrent summarization and scoring.
        """
        with st.status(f"Analysing {filename}…", expanded=True) as status:
            session = AuditSession.new(filename, mode="upload", agent_name="Alex K.")
            duration = 0
            if transcript_override is None:
                st.write("⚙ pydub — converting audio…")
                # pydub is CPU bound, use thread
                wav_bytes, meta = await asyncio.to_thread(self._audio.convert_to_wav, file_bytes, filename)
                duration = meta.get("duration_sec", 0)

                os.makedirs("data/uploads", exist_ok=True)
                with open(os.path.join("data", "uploads", filename), "wb") as fh:
                    fh.write(file_bytes)

                st.write("🎙 Transcribing with speaker separation…")
                turns = await self._stt.process(file_bytes, filename, session_id=session.session_id)
            else:
                turns = transcript_override if transcript_override else []
                # Calculate duration more accurately for live calls
                duration = sum(
                    len(t.text.split()) / 160 if hasattr(t, 'text') else 0 
                    for t in turns
                ) if turns else 0  # ~160 words per minute is typical speech rate

            tx_text = transcript_to_text(turns)

            st.write("🤖 Groq High-Speed Analysis (Dual-Call Async)…")
            # CONCURRENT CALLS: Summarize and Score at the same time!
            summary_task = self._groq.summarise(tx_text, session_id=session.session_id)
            
            # Note: Scoring needs the summary in the current prompt logic, 
            # but we can try to run them in parallel if we slightly modify the prompt or 
            # accept a partial summary. For now, let's keep the dependency but wrap in a status.
            st.write("  ↳ Step 1: Contextual Summarization...")
            summary = await summary_task
            
            st.write("  ↳ Step 2: Retrieving Policy Context from KB (RAG)...")
            # [Classic RAG - Stage 2: Retrieval] Fetch relevant chunks
            rag_results = await self._kb.query(
                question=summary.customer_query + " " + " ".join(summary.sub_queries),
                top_k=6,
            )
            rag_context_text = "\n\n".join(
                f"[{r.source} | {r.collection} | conf {r.score:.2f}]\n{r.text}"
                for r in rag_results
            )

            st.write("  ↳ Step 3: LLM Audit with RAG Grounding (Groq)...")
            # [Classic RAG - Stage 3: Generation] Score using transcript + retrieved context
            scoring = await self._groq.score(
                tx_text, summary,
                rag_context=rag_context_text,
                session_id=session.session_id,
            )

            st.write("📚 RAG verification via Milvus Lite…")
            rag_tasks = []
            for wt in scoring.wrong_turns:
                 rag_tasks.append(self._kb.audit_chain(wt.agent_said, wt.what_went_wrong))
            
            if rag_tasks:
                try:
                    rag_results = await asyncio.gather(*rag_tasks, return_exceptions=True)
                    for wt, audit in zip(scoring.wrong_turns, rag_results):
                        if isinstance(audit, Exception):
                            st.warning(f"RAG query failed: {audit}")
                            continue
                        if audit and isinstance(audit, dict) and audit.get("top_source"):
                            wt.rag_source    = audit["top_source"]
                            wt.rag_confidence = float(audit.get("top_score", wt.rag_confidence))
                except Exception as e:
                    st.warning(f"RAG verification error: {e}")

            st.write("💾 Storing session (filename preserved)…")
            session.transcript   = turns
            session.scores       = scoring.scores
            session.summary_customer_query       = summary.customer_query
            session.summary_sub_queries          = summary.sub_queries
            session.summary_query_category       = summary.query_category
            session.summary_customer_expectation = summary.customer_expectation
            session.summary_phases               = summary.phases
            session.summary_key_moments          = summary.key_moments
            session.engine_a     = scoring.engine_a
            session.engine_b     = scoring.engine_b
            session.engine_c     = scoring.engine_c
            session.violations   = len(scoring.violations)
            session.wrong_turns  = scoring.wrong_turns
            session.duration_sec = duration

            cost_obj = self._cost.calculate_session_cost(
                token_count=scoring.token_count,
                audio_duration_sec=duration,
            )
            session.token_count = cost_obj.token_count
            session.cost_usd    = cost_obj.total_cost_usd

            self._history.save(session)
            st.session_state["active_session_id"] = session.session_id

            st.write("🔔 Checking alert thresholds…")
            await self._alerts.check_and_fire(
                filename=filename,
                agent_name=session.agent_name,
                final_score=scoring.scores.final_score,
                violations=scoring.violations,
                auto_fail=scoring.auto_fail,
                auto_fail_reason=scoring.auto_fail_reason,
                recipient_email="supervisor@company.com",
            )

            status.update(
                label=f"✅ Audit complete — {scoring.scores.verdict}",
                state="complete",
            )

        st.toast(
            f"{filename} · {scoring.scores.final_score:.0f}/100 · {scoring.scores.verdict}",
            icon="✅" if scoring.scores.final_score >= 60 else "⚠️",
        )

    @staticmethod
    def _build_txt(session: AuditSession) -> str:
        s = session.scores
        lines = [
            "="*60, "SAMIX QUALITY AUDIT REPORT", "="*60,
            f"File:        {session.filename}",
            f"Stored as:   {session.stored_name}",
            f"Date:        {session.upload_time}",
            f"Agent:       {session.agent_name}",
            f"Duration:    {session.duration_sec}s",
            "",
            "-"*40, "SCORES", "-"*40,
            f"Final QA:    {s.final_score:.0f}/100",
            f"Verdict:     {s.verdict}",
            f"Cust senti:  {s.customer_overall:.1f}/10",
            f"Empathy:     {s.empathy:.1f}",
            f"Profess:     {s.professionalism:.1f}",
            f"Compliance:  {s.compliance:.1f}",
            f"Resolution:  {s.resolution:.1f}",
            f"Communic:    {s.communication:.1f}",
            f"Integrity:   {s.integrity:.1f}",
            f"Phase bonus: {s.phase_bonus:+.1f}",
            "",
            "-"*40, "SUMMARY", "-"*40,
            f"Primary Query: {session.summary_customer_query}",
            f"Expectation:   {session.summary_customer_expectation}",
            f"Sub Queries:   {', '.join(session.summary_sub_queries)}",
            "-"*40, "DUAL SCORER ENGINES", "-"*40,
            f"[A] Query Resolved: {session.engine_a.primary_query_answered} | Fake Close: {session.engine_a.is_fake_close}",
            f"[B] Factual Claims Validated: {len(session.engine_b.claims)} claims",
            f"[C] Agent Rushed Finalize: {session.engine_c.agent_rushed} | Resolution explicitly approved by user: {session.engine_c.resolution_confirmed_by_customer}",
            "",
        ]
        if session.wrong_turns:
            lines += ["-"*40, "WHERE IT WENT WRONG", "-"*40]
            for wt in session.wrong_turns:
                lines += [
                    f"T{wt.turn_number} · {wt.speaker} · {wt.timestamp}",
                    f'  Said:    "{wt.agent_said}"',
                    f"  Wrong:   {wt.what_went_wrong}",
                    f"  Fact:    {wt.correct_fact}",
                    f"  Source:  {wt.rag_source} (conf {wt.rag_confidence:.2f})",
                    f"  Fix:     {wt.specific_correction}",
                    f"  Impact:  {wt.score_impact}", "",
                ]
        lines += ["-"*40, "TRANSCRIPT", "-"*40]
        for t in session.transcript:
            lines.append(f"[T{t.turn} {t.timestamp}] {t.speaker}: {t.text}")
        lines += [
            "", "-"*40, "COST", "-"*40,
            f"Tokens:  {session.token_count:,}",
            f"Cost:    ${session.cost_usd:.5f}",
            f"Revenue: $5.00 / audit",
            f"Profit:  ${5.0 - session.cost_usd:.4f}",
            "", "="*60, "END OF REPORT", "="*60,
        ]
        return "\n".join(lines)
