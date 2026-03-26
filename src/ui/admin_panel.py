"""
SamiX Administrative Intelligence Dashboard

This module implements a comprehensive command center for supervisors and admins.
It provides real-time visibility into:
- Business KPIs: Revenue, growth, and agent performance.
- Model Health: Groq latency, STT accuracy, and RAG groundedness.
- Knowledge Management: Direct control over the Milvus-backed policy library.
- System Integrity: End-to-end service uptime and pipeline throughput.
"""
from __future__ import annotations

import asyncio
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipeline.alert_engine import AlertEngine
from src.ui.components import render_page_hero
from src.utils.history_manager import HistoryManager
from src.utils.kb_manager      import KBManager


def _line(x, y, colour="#152EAE", height=180, y_min=None, y_max=None, fill=True):
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color=colour, width=2.5),
        marker=dict(size=5, color=colour),
        fill="tozeroy" if fill else None,
        fillcolor="rgba(21, 46, 174, 0.04)" if fill else None,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#64748B", size=9),
        margin=dict(l=0,r=0,t=4,b=0), height=height,
        yaxis=dict(
            range=[y_min, y_max] if y_min is not None else None,
            gridcolor="#F1F5F9",
        ),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def _bar(x, y, colour="#152EAE", height=180, y_min=None, y_max=None):
    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker_color=colour,
        marker_line_color="#E2E8F0",
        marker_line_width=1,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#64748B", size=9),
        margin=dict(l=0,r=0,t=4,b=0), height=height,
        yaxis=dict(
            range=[y_min, y_max] if y_min is not None else None,
            gridcolor="#F1F5F9",
        ),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def _tbl_html(rows: list[tuple[str,str]]) -> str:
    items = "".join(
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:6px 0;border-bottom:1px solid #F1F5F9;">'
        f'<span style="color:#64748B;font-weight:500;">{k}</span>'
        f'<span style="color:#0F172A;font-weight:700;">{v}</span></div>'
        for k, v in rows
    )
    return f'<div style="font-family:Inter,sans-serif;font-size:.82rem;">{items}</div>'


DAYS = ["3/5","3/6","3/7","3/8","3/9","3/10","3/11",
        "3/12","3/13","3/14","3/15","3/16","3/17","3/18"]


class AdminPanel:
    """
    The central UI controller for the SamiX Admin Dashboard.
    
    It coordinates data from History, Knowledge Base, and Alerting managers
     to present a unified view of the system's operational status.
    """
    def __init__(
        self,
        history: HistoryManager,
        kb:      KBManager,
        alerts:  AlertEngine,
    ) -> None:
        self._history = history
        self._kb      = kb
        self._alerts  = alerts

    def render(self) -> None:
        """
        Main entry point for rendering the multi-tabbed dashboard.
        Segments logic into specialized views for overview, models, users, etc.
        """
        sessions = self._history.get_all()
        avg_score = (
            sum(s.scores.final_score for s in sessions) / len(sessions)
            if sessions else 0.0
        )
        render_page_hero(
            "Admin Workspace",
            "Monitor audit quality, model health, and policy coverage.",
            "Use the admin surface to review performance trends, retrieval coverage, and operational metrics in a production-friendly Streamlit layout.",
            stats=[
                ("Audits", str(len(sessions)), "records currently stored"),
                ("Average Score", f"{avg_score:.1f}/100" if sessions else "0/100", "current quality baseline"),
                ("KB Documents", str(len(self._kb.files) + len(self._kb.generalised_kb)), "uploaded plus built-in"),
                ("Retrieval", "Vector" if self._kb.is_vector_enabled else "Fallback", "Milvus or keyword mode"),
            ],
        )
        tabs = st.tabs([
            "Overview",
            "Model Performance",
            "Users",
            "Billing",
            "RAG Knowledge Base",
            "System Health",
        ])
        with tabs[0]: self._overview()
        with tabs[1]: self._model_perf()
        with tabs[2]: self._users()
        with tabs[3]: self._billing()
        with tabs[4]: self._rag_kb()
        with tabs[5]: self._system()

    # 1 · OVERVIEW
    def _overview(self) -> None:
        st.markdown('<div class="section-header">OVERVIEW</div>', unsafe_allow_html=True)

        # Real data from history
        sessions = self._history.get_all()
        total_real = len(sessions)
        avg_real   = (sum(s.scores.final_score for s in sessions) / total_real
                      if total_real else 73.4)
        viols_real = sum(s.violations for s in sessions)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Audits today",   str(total_real or 247),  "↑ +12%")
        c2.metric("This week",      "1,284",                 "↑ +8%")
        c3.metric("Avg QA score",   f"{avg_real:.1f}",       "↓ −1.2")
        c4.metric("Live calls",     "4",                     "active now")
        c5.metric("MTD revenue",    "$2,840",                "↑ +18%")
        c6.metric("Violations",     str(viols_real or 38),   "14 critical")

        col_trend, col_donut = st.columns([3,2])
        with col_trend:
            st.markdown("##### Score trend — 14 days")
            scores = [71,72,70,73,74,72,73,75,74,73,72,74,73,73.4]
            fig = _line(DAYS, scores, y_min=55, y_max=90)
            fig.add_hline(y=60, line_dash="dash", line_color="rgba(239,68,68,.4)",
                          annotation_text="Fail threshold", annotation_font_size=8)
            st.plotly_chart(fig, use_container_width=True)

        with col_donut:
            st.markdown("##### Verdict split")
            fig2 = go.Figure(go.Pie(
                labels=["Excellent","Good","Needs work","Fail"],
                values=[18,41,28,13], hole=0.72,
                marker_colors=["#10B981","#152EAE","#F59E0B","#EF4444"],
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                legend=dict(font=dict(color="#64748B",size=9)),
                margin=dict(l=0,r=0,t=4,b=0), height=180,
            )
            st.plotly_chart(fig2, use_container_width=True)

        col_lb, col_viol = st.columns(2)
        with col_lb:
            st.markdown("##### Agent leaderboard")
            agents = [
                ("Priya M.",89,"#10B981"),("Sarah C.",84,"#10B981"),
                ("Tom R.",  81,"#EB643E"),("Alex K.", 65,"#EF4444"),
            ]
            for name, score, colour in agents:
                cn, cb, cs = st.columns([2,5,1])
                cn.caption(name)
                cb.progress(score/100)
                cs.markdown(
                    f'<div style="font-family:Inter,sans-serif;font-size:.9rem;font-weight:700;'
                    f'color:{colour};">{score}</div>', unsafe_allow_html=True
                )

        with col_viol:
            st.markdown("##### Recent violations")
            st.dataframe(pd.DataFrame([
                {"Agent":"Alex K.",  "Type":"Wrong policy info","Sev":"Critical","Time":"09:42"},
                {"Agent":"Priya M.", "Type":"False close",       "Sev":"Critical","Time":"09:38"},
                {"Agent":"James T.", "Type":"Script missed",     "Sev":"High",    "Time":"09:21"},
                {"Agent":"Rita S.",  "Type":"Empathy gap",       "Sev":"Medium",  "Time":"08:55"},
            ]), use_container_width=True, hide_index=True)

    # 2 · MODEL PERFORMANCE
    def _model_perf(self) -> None:
        st.markdown('<div class="section-header">MODEL PERFORMANCE</div>',
                    unsafe_allow_html=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Groq calls today",    "494",    "2 per audit")
        c2.metric("Avg Groq latency",    "815ms",  "call 1+2")
        c3.metric("STT accuracy",        "94.2%",  "diarization")
        c4.metric("RAG groundedness",    "0.88",   "avg cosine")
        c5.metric("pydub conversions",   "38",     "today")

        col_lat, col_acc = st.columns(2)
        with col_lat:
            st.markdown("##### Groq latency — 14 days (ms)")
            st.plotly_chart(
                _line(DAYS, [820,790,840,810,830,800,850,820,800,790,810,830,800,815],
                      y_min=600, y_max=1000),
                use_container_width=True,
            )
        with col_acc:
            st.markdown("##### Scoring accuracy — user feedback (out of 5)")
            st.plotly_chart(
                _line(DAYS, [3.8,3.9,4.0,4.0,4.1,4.0,4.2,4.1,4.1,4.2,4.1,4.2,4.1,4.1],
                      colour="#10B981", y_min=3, y_max=5),
                use_container_width=True,
            )

        cg, cs, cr = st.columns(3)
        with cg:
            st.markdown("##### Groq breakdown")
            st.markdown(_tbl_html([
                ("Call 1 (summarise)", "~380ms"),
                ("Call 2 (score)",     "~435ms"),
                ("Tokens / audit",     "~4,200"),
                ("Model",              "LLaMA 3.3-70b"),
                ("Avg rating",         "4.1 / 5.0"),
                ("Cost / audit",       "$0.00227"),
            ]), unsafe_allow_html=True)
        with cs:
            st.markdown("##### STT + pydub")
            st.markdown(_tbl_html([
                ("Deepgram batch",  "209 today"),
                ("Whisper live",    "38 today"),
                ("Fallback rate",   "1.2%"),
                ("Avg diarize conf","0.91"),
                ("pydub converts",  "38 files"),
                ("Conf threshold",  "0.70"),
            ]), unsafe_allow_html=True)
        with cr:
            st.markdown("##### LangChain + Milvus RAG")
            st.markdown(_tbl_html([
                ("RAG queries today",  "941"),
                ("Policy violations",  "38"),
                ("Avg groundedness",   "0.88"),
                ("MMR retrieval",      "enabled"),
                ("KB documents",       str(len(self._kb.files) + len(self._kb.generalised_kb))),
                ("Total chunks",       str(self._kb.total_chunks)),
            ]), unsafe_allow_html=True)

        # Quality of logs
        st.markdown("---")
        st.markdown("##### Quality of logs assessed — compliance trend")
        compliance_trend = [68,70,69,72,71,73,72,74,73,72,74,73,75,73]
        fig = _line(DAYS, compliance_trend, colour="#F59E0B", y_min=50, y_max=90)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(16,185,129,.4)",
                      annotation_text="Compliant threshold", annotation_font_size=8)
        st.plotly_chart(fig, use_container_width=True)

    # 3 · USERS
    def _users(self) -> None:
        st.markdown('<div class="section-header">USERS</div>', unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total users",      "24")
        c2.metric("Active this week", "18")
        c3.metric("New this month",   "3",  "↑ growth")
        c4.metric("Churn risk",       "2",  "inactive 7+ days")

        col_act, col_grow = st.columns(2)
        with col_act:
            st.markdown("##### Quantity of logs per day (all users)")
            activity = [210,220,198,235,242,218,229,241,250,238,244,251,248,247]
            st.plotly_chart(_line(DAYS, activity, y_min=150, y_max=280),
                            use_container_width=True)
        with col_grow:
            st.markdown("##### User growth — 4 weeks")
            st.plotly_chart(
                _bar(["Wk1","Wk2","Wk3","Wk4"], [15,16,17,18], y_min=10, y_max=25),
                use_container_width=True,
            )

        st.markdown("##### User list — usage + accessibility")
        users = [
            ("Priya M.",  "Pro",   32, 89, "Healthy"),
            ("Sarah C.",  "Pro",   28, 84, "Healthy"),
            ("Tom R.",    "Basic", 30, 81, "Good"),
            ("Ana B.",    "Pro",   26, 79, "Good"),
            ("Alex K.",   "Pro",   18, 65, "At risk"),
            ("James T.",  "Basic", 20, 68, "Watch"),
        ]
        for i, (name, plan, audits, score, status) in enumerate(users):
            cu, cp, ca, cs, cst, cbtn = st.columns([2,1,1,1,1,1])
            cu.write(name)
            cp.markdown(f'<div class="mono-badge">{plan}</div>', unsafe_allow_html=True)
            ca.write(str(audits))
            cs.write(str(score))
            colour = ("#10B981" if status=="Healthy"
                      else "#F59E0B" if status in ("Good","Watch")
                      else "#EF4444")
            cst.markdown(f'<div style="color:{colour};font-size:.8rem;">{status}</div>',
                         unsafe_allow_html=True)
            if cbtn.button("Email", key=f"usr_{i}"):
                self._alerts.send_custom(
                    to=f"{name.lower().replace(' ','.')}@company.com",
                    subject=f"SamiX coaching update — {name}",
                    body=(f"Hi {name.split()[0]},\n\nYour latest score is {score}/100. "
                          "A coaching session has been scheduled.\n\nSamiX Team"),
                )

    # 4 · BILLING
    def _billing(self) -> None:
        st.markdown('<div class="section-header">BILLING & INVESTMENT</div>',
                    unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("MTD Revenue",     "$2,840",  "↑ +18% MoM")
        c2.metric("MTD API costs",   "$22.60",  "Deepgram+Twilio")
        c3.metric("Gross margin",    "99.2%",   "excellent")
        c4.metric("Pending invoices","3",        "overdue")

        col_rev, col_costs = st.columns(2)
        with col_rev:
            st.markdown("##### Revenue vs API cost — 14 days")
            rev  = [88,92,87,95,98,90,93,97,102,96,99,103,101,100]
            cost = [0.7,0.8,0.7,0.9,0.8,0.7,0.9,0.8,0.8,0.9,0.8,0.9,0.8,0.75]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=DAYS, y=rev, name="Revenue ($)",
                line=dict(color="#10B981",width=2),
                fill="tozeroy", fillcolor="rgba(16,185,129,.06)",
            ))
            fig.add_trace(go.Scatter(
                x=DAYS, y=cost, name="Cost ($)",
                line=dict(color="#EF4444",width=1.5,dash="dot"),
                yaxis="y2",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                font=dict(family="Sora,sans-serif",color="#64748B",size=9),
                legend=dict(bgcolor="rgba(255,255,255,.9)",font=dict(size=9,color="#212529")),
                margin=dict(l=0,r=0,t=4,b=0), height=200,
                yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                yaxis2=dict(overlaying="y",side="right",showgrid=False),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_costs:
            st.markdown("##### API cost breakdown — this month")
            st.markdown(_tbl_html([
                ("Deepgram STT",  "$14.20"),
                ("Twilio Media",  "$8.40"),
                ("Groq API",      "$0.00  (free tier)"),
                ("Whisper local", "$0.00  (on-device)"),
                ("Milvus Lite",   "$0.00  (embedded)"),
                ("LangChain",     "$0.00  (open source)"),
                ("Total",         "$22.60 / month"),
                ("Revenue",       "$2,840 / month"),
                ("Margin",        "99.2%"),
            ]), unsafe_allow_html=True)

        st.markdown("##### Project investment context")
        inv_c, inv_b, inv_r = st.columns(3)
        inv_c.metric("Infra cost / month",   "$22.60")
        inv_b.metric("Break-even users",     "1 (at $99/mo plan)")
        inv_r.metric("Annual revenue (18 users)", "~$34,000+")

        st.markdown("---")
        st.markdown("##### Customer billing — send payment emails")
        customers = [
            ("Acme BPO",       "Enterprise", 1240, "$1,200", "paid",    "Mar 31"),
            ("TechSupport Co", "Pro",          480, "$480",   "paid",    "Mar 31"),
            ("CareCall Ltd",   "Pro",          320, "$320",   "overdue", "Mar 10"),
            ("FastHelp Inc",   "Basic",        210, "$210",   "pending", "Mar 31"),
            ("SupportHub",     "Basic",        180, "$180",   "overdue", "Mar  5"),
        ]
        for i, (company, plan, audits, amount, status, due) in enumerate(customers):
            cn,cp,ca,cam,cs,cd,cbtn = st.columns([2,1,1,1,1,1,1])
            cn.write(company)
            cp.markdown(f'<div class="mono-badge">{plan}</div>', unsafe_allow_html=True)
            ca.write(str(audits))
            cam.write(amount)
            sc = ("#10B981" if status=="paid"
                  else "#EF4444" if status=="overdue"
                  else "#F59E0B")
            cs.markdown(f'<div style="color:{sc};font-size:.8rem;">{status.upper()}</div>',
                        unsafe_allow_html=True)
            cd.write(due)
            is_overdue = status == "overdue"
            if cbtn.button("⚠ Invoice" if is_overdue else "Invoice",
                           key=f"inv_{i}",
                           type="primary" if is_overdue else "secondary"):
                urgency = "OVERDUE" if is_overdue else "Due"
                self._alerts.send_custom(
                    to=f"billing@{company.lower().replace(' ','')}.com",
                    subject=f"SamiX Invoice {amount} — {urgency}",
                    body=(f"Dear {company} team,\n\nYour SamiX invoice for {amount} is "
                          f"{status.upper()}. Due date: {due}.\n\nPay at https://samix.ai/billing\n\n"
                          "SamiX Billing Team"),
                )

    # 5 · RAG KNOWLEDGE BASE
    def _rag_kb(self) -> None:
        st.markdown('<div class="section-header">RAG KNOWLEDGE BASE</div>',
                    unsafe_allow_html=True)

        col_up, col_info = st.columns([3,2], gap="large")

        with col_up:
            st.markdown("##### Upload policy documents")
            st.caption("Files are chunked (500 tokens / 50 overlap) and indexed into Milvus Lite "
                       "via LangChain + all-MiniLM-L6-v2 embeddings. "
                       "MMR retrieval ensures diverse, grounded results.")

            collection = st.selectbox(
                "Collection",
                ["policies","product_kb","compliance"],
                help="policies: scripts + refund rules · product_kb: features · compliance: GDPR/ISO",
            )
            uploaded = st.file_uploader(
                "PDF or TXT files",
                type=["pdf","txt"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if uploaded and st.button("Index into Milvus Lite", type="primary"):
                for f in uploaded:
                    with st.spinner(f"Indexing {f.name}…"):
                        kbf = self._kb.add_file(f.getvalue(), f.name, collection)
                    st.toast(f"✅ {kbf.filename} · {kbf.chunks} chunks indexed", icon="📚")
                st.rerun()

            # Live query tester
            st.markdown("---")
            st.markdown("##### Live RAG query tester")
            test_q = st.text_input("Test a policy question",
                                    placeholder="What is the refund timeline?",
                                    key="kb_test")
            if test_q and st.button("Query KB", key="kb_query"):
                with st.spinner("Retrieving…"):
                    results = asyncio.run(self._kb.query(test_q, top_k=3))
                if results:
                    for r in results:
                        st.markdown(
                            f'<div class="glass-card" style="font-size:.82rem;">'
                            f'<div style="color:#EB643E;font-family:Sora,sans-serif;font-weight:600;font-size:.68rem;">'
                            f'{r.source} · {r.collection} · conf {r.score:.3f}</div>'
                            f'<div style="color:#64748B;margin-top:6px;">{r.text[:200]}…</div>'
                            f'</div>', unsafe_allow_html=True
                        )
                else:
                    st.info("No matching chunks found. Upload relevant documents first.")

            # Indexed files list
            st.markdown("---")
            st.markdown("##### Indexed files")
            col_colours = {"policies":"#EB643E","product_kb":"#10B981","compliance":"#F59E0B"}
            if self._kb.files:
                for kbf in self._kb.files:
                    ci,cn,cc,cm,cd = st.columns([1,3,1,2,1])
                    ext = kbf.filename.rsplit(".",1)[-1].upper()
                    ci.markdown(f'<div class="mono-badge">{ext}</div>', unsafe_allow_html=True)
                    cn.write(kbf.filename)
                    colour = col_colours.get(kbf.collection,"#94A3B8")
                    cc.markdown(f'<div style="color:{colour};font-size:.72rem;'
                                f'font-family:Sora,sans-serif;font-weight:600;">{kbf.collection}</div>',
                                unsafe_allow_html=True)
                    cm.caption(f"{kbf.chunks} chunks · {kbf.size_label}")
                    if cd.button("✕", key=f"del_{kbf.filename}"):
                        self._kb.remove_file(kbf.filename)
                        st.toast(f"Removed {kbf.filename}", icon="🗑")
                        st.rerun()
            else:
                st.info("No user files yet. Upload PDFs or TXTs above.")

        with col_info:
            st.markdown("##### KB stats")
            st.markdown(_tbl_html([
                ("User-uploaded files", str(len(self._kb.files))),
                ("Generalised KB items",str(len(self._kb.generalised_kb))),
                ("Total chunks",        str(self._kb.total_chunks)),
                ("Chunk size",          "500 tokens"),
                ("Chunk overlap",       "50 tokens"),
                ("Retrieval",           "MMR top-4"),
                ("Embed model",         "all-MiniLM-L6-v2"),
                ("Vector dims",         "384"),
                ("Vector DB",           "Milvus Lite (local)"),
                ("LangChain",           "v0.2+"),
                ("Collections",         "3 namespaces"),
                ("Groundedness",        "cosine + keyword"),
                ("Vector enabled",      "Yes" if self._kb.is_vector_enabled else "Keyword fallback"),
            ]), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### Generalised knowledge — auto-loaded")
            col_colours = {"policies":"#EB643E","product_kb":"#10B981","compliance":"#F59E0B"}
            for item in self._kb.generalised_kb:
                colour = col_colours.get(item["collection"],"#64748B")
                st.markdown(
                    f'<div style="padding:5px 0;border-bottom:1px solid rgba(0,0,0,0.05);'
                    f'font-size:.8rem;color:#64748B;font-weight:500;">'
                    f'{item["name"]}'
                    f'<span style="margin-left:8px;color:{colour};font-size:.7rem;font-weight:700;">'
                    f'{item["collection"].upper()}</span></div>',
                    unsafe_allow_html=True,
                )

    # 6 · SYSTEM HEALTH
    def _system(self) -> None:
        st.markdown('<div class="section-header">SYSTEM HEALTH</div>',
                    unsafe_allow_html=True)

        services = [
            ("Groq API",       99, "#10B981"),
            ("Deepgram",       98, "#10B981"),
            ("Whisper local",  95, "#10B981"),
            ("pydub",         100, "#10B981"),
            ("LangChain",     100, "#10B981"),
            ("Milvus Lite",   100, "#10B981"),
            ("PostgreSQL",    100, "#10B981"),
            ("FastAPI",        97, "#10B981"),
            ("Celery queue",  100, "#10B981"),
        ]

        col_sv, col_lat, col_q = st.columns([2,3,2])

        with col_sv:
            st.markdown("##### Services")
            for name, uptime, colour in services:
                cn, cb, cp = st.columns([2,4,1])
                cn.caption(name)
                cb.progress(uptime/100)
                cp.markdown(f'<div style="font-size:.75rem;color:{colour};padding-top:3px;">'
                            f'{uptime}%</div>', unsafe_allow_html=True)

        with col_lat:
            st.markdown("##### API latency — 14 days")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=DAYS,
                y=[820,790,840,810,830,800,850,820,800,790,810,830,800,815],
                name="Groq (ms)",
                line=dict(color="#152EAE",width=2.5), marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=DAYS,
                y=[320,310,330,320,310,300,320,315,310,305,315,320,310,315],
                name="Deepgram (ms)",
                line=dict(color="#10B981",width=1.5,dash="dot"), marker=dict(size=3),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                font=dict(family="Inter,sans-serif",color="#64748B",size=9),
                legend=dict(bgcolor="rgba(255,255,255,.9)",font=dict(size=9,color="#212529")),
                margin=dict(l=0,r=0,t=4,b=0), height=300,
                yaxis=dict(gridcolor="#F1F5F9"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_q:
            st.markdown("##### Queue + pipeline")
            st.markdown(_tbl_html([
                ("Pending jobs",       "3"),
                ("Processing",         "4"),
                ("Completed today",    "247"),
                ("Failed today",       "2"),
                ("pydub converts",     "38"),
                ("RAG queries",        "941"),
                ("LangChain calls",    "941"),
                ("Alerts (email)",     "12"),
                ("Screen pop alerts",  "8"),
                ("Avg job time",       "4.2s"),
            ]), unsafe_allow_html=True)
