"""
SamiX UI Component Library - Cloud Optimized
A premium design system inspired by AirCover and modern SaaS interfaces.
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Any
from datetime import datetime

# Core Data Models
from src.utils.history_manager import AuditScores, TranscriptTurn, WrongTurn

# ----------------------------------------------------------------------------
# 1. FIXED IMPORT SECTION (The Bridges)
# ----------------------------------------------------------------------------

def render_hero_section():
    """Fixes the import error for hero section"""
    render_page_hero(
        eyebrow="AI AUDITOR",
        title="SamiX Quality Suite",
        subtitle="GenAI-Powered Customer Support Quality Auditor"
    )

def render_metrics_showcase(stats: list[tuple[str, str, str]]):
    """Fixes the 'no module named render_metrics_showcase' error"""
    cols = st.columns(len(stats))
    for i, (label, value, note) in enumerate(stats):
        with cols[i]:
            st.markdown(f"""
                <div style="background: rgba(30, 41, 49, 0.5); border: 1px solid rgba(226, 232, 240, 0.1); padding: 1.25rem; border-radius: 12px;">
                    <div style="color: #94A3B8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">{label}</div>
                    <div style="color: #F8FAFC; font-size: 1.5rem; font-weight: 700;">{value}</div>
                    <div style="color: #6366F1; font-size: 0.7rem;">{note}</div>
                </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 2. Page Foundations
# ----------------------------------------------------------------------------

def render_page_hero(
    eyebrow: str,
    title: str,
    subtitle: str,
    stats: Optional[list[tuple[str, str, str]]] = None,
) -> None:
    st.markdown(f"""
        <div style="padding: 2rem 0 1rem 0;">
            <div style="color: #6366F1; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">{eyebrow}</div>
            <h1 style="font-size: 2.75rem; font-weight: 800; color: #F1F5F9;">{title}</h1>
            <p style="font-size: 1.15rem; color: #94A3B8;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if stats:
        render_metrics_showcase(stats)

# ----------------------------------------------------------------------------
# 3. Visualization Engines
# ----------------------------------------------------------------------------

def render_gauge(value: float, title: str, max_val: float = 10.0) -> None:
    try:
        from streamlit_echarts import st_echarts
    except ImportError:
        st.metric(title, f"{value:.1f}/{max_val}")
        return

    pct = min(100.0, max(0.0, (value / max_val) * 100))
    color = "#10B981" if pct >= 80 else "#F59E0B" if pct >= 60 else "#EF4444"

    option = {
        "backgroundColor": "transparent",
        "series": [{
            "type": "gauge",
            "startAngle": 210, "endAngle": -30,
            "min": 0, "max": 100,
            "itemStyle": {"color": color},
            "progress": {"show": True, "width": 12},
            "pointer": {"show": False},
            "axisLine": {"lineStyle": {"width": 12, "color": [[1, "rgba(255,255,255,0.05)"]]}},
            "axisTick": {"show": False}, "splitLine": {"show": False}, "axisLabel": {"show": False},
            "title": {"offsetCenter": [0, "80%"], "fontSize": 12, "color": "#94A3B8", "fontWeight": "600"},
            "detail": {"offsetCenter": [0, "10%"], "fontSize": 24, "fontWeight": "800", "color": "#F8FAFC", "formatter": f"{value:.1f}"},
            "data": [{"value": pct, "name": title.upper()}]
        }]
    }
    st_echarts(options=option, height="180px", key=f"gauge_{title}_{value}")

def render_three_gauges(scores: Any) -> None:
    def get_v(attr): 
        return getattr(scores, attr, scores.get(attr, 0)) if hasattr(scores, 'get') or hasattr(scores, attr) else 0

    c1, c2, c3 = st.columns(3)
    with c1: render_gauge(get_v("empathy"), "Empathy")
    with c2: render_gauge(get_v("professionalism"), "Professionalism")
    with c3: render_gauge(get_v("compliance"), "Compliance")

def render_dual_score_chart(scores: Any) -> None:
    agent_data = getattr(scores, 'agent_by_turn', scores.get('agent_by_turn', []))
    cust_data = getattr(scores, 'customer_sentiment', scores.get('customer_sentiment', []))
    
    n = max(len(agent_data), len(cust_data), 1)
    x_labels = [f"Turn {i+1}" for i in range(n)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=agent_data, name="Agent Quality", line=dict(color="#6366F1", width=3), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=x_labels, y=cust_data, name="Customer Sentiment", line=dict(color="#10B981", width=2, dash='dot'), mode='lines+markers'))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.3)",
        font=dict(color="#94A3B8", size=10), height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# 4. Session Details
# ----------------------------------------------------------------------------

def render_transcript(turns: list[Any], wrong_turns: Optional[list[Any]] = None) -> None:
    wrong_map = {}
    if wrong_turns:
        for wt in wrong_turns:
            t_num = getattr(wt, 'turn_number', wt.get('turn_number'))
            wrong_map[t_num] = wt

    for turn in turns:
        speaker = getattr(turn, 'speaker', turn.get('speaker', 'Unknown'))
        text = getattr(turn, 'text', turn.get('text', ''))
        t_id = getattr(turn, 'turn', turn.get('turn', 0))
        
        is_agent = speaker.upper() == "AGENT"
        bg = "rgba(99, 102, 241, 0.1)" if is_agent else "rgba(148, 163, 184, 0.1)"
        border = "rgba(99, 102, 241, 0.3)" if is_agent else "rgba(148, 163, 184, 0.2)"
        
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: {'flex-start' if is_agent else 'flex-end'}; margin-bottom: 1rem;">
                <div style="font-size: 0.65rem; font-weight: 700; color: #94A3B8; margin-bottom: 0.25rem;">{speaker}</div>
                <div style="background: {bg}; border: 1px solid {border}; padding: 0.75rem 1rem; border-radius: 12px; max-width: 85%; color: #E2E8F0; font-size: 0.95rem; line-height: 1.5;">
                    {text}
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_wrong_turns(wrong_turns: list[Any]) -> None:
    if not wrong_turns:
        st.success("✅ No policy violations detected.")
        return

    for wt in wrong_turns:
        t_num = getattr(wt, 'turn_number', wt.get('turn_number'))
        impact = getattr(wt, 'score_impact', wt.get('score_impact', 'Unknown'))
        said = getattr(wt, 'agent_said', wt.get('agent_said', ''))
        fact = getattr(wt, 'correct_fact', wt.get('correct_fact', ''))
        source = getattr(wt, 'rag_source', wt.get('rag_source', 'General KB'))

        with st.expander(f"🚩 Turn {t_num}: {impact}", expanded=False):
            st.markdown(f"**Agent Said:**\n> {said}")
            st.markdown(f"**Correct Fact (RAG):**\n> :green[{fact}]")
            st.caption(f"📚 Source: {source}")

# ----------------------------------------------------------------------------
# 5. Utilities
# ----------------------------------------------------------------------------

def render_cost_card(token_count: int, cost_usd: float) -> None:
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(16, 185, 129, 0.1)); border: 1px solid rgba(255,255,255,0.05); padding: 1.25rem; border-radius: 12px;">
            <div style="color: #94A3B8; font-size: 0.7rem; font-weight: 700; text-transform: uppercase;">Cloud Processing Cost</div>
            <div style="color: #F8FAFC; font-size: 1.5rem; font-weight: 800; margin: 0.25rem 0;">${cost_usd:.5f}</div>
            <div style="color: #6366F1; font-size: 0.75rem; font-weight: 500;">{token_count:,} tokens consumed</div>
        </div>
    """, unsafe_allow_html=True)

def render_filename_badge(filename: str, session_id: str) -> None:
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem;">
            <span style="background: #6366F1; color: white; padding: 3px 10px; border-radius: 6px; font-size: 0.7rem; font-weight: 800;">AUDIT</span>
            <span style="color: #F1F5F9; font-weight: 700; font-size: 1.1rem;">{filename}</span>
            <span style="color: #64748B; font-family: monospace; font-size: 0.8rem;">ID: {session_id[:8]}</span>
        </div>
    """, unsafe_allow_html=True)

def build_history_dataframe(sessions: list[Any]) -> pd.DataFrame:
    rows = []
    for s in sessions:
        rows.append({
            "session_id": s.session_id,
            "Timestamp": s.timestamp.strftime("%b %d, %H:%M") if isinstance(s.timestamp, datetime) else str(s.timestamp),
            "File Name": s.filename,
            "Agent": s.agent_name,
            "Score": f"{s.scores.final_score}/100" if hasattr(s.scores, 'final_score') else f"{s.scores.get('final_score', 0)}/100",
            "Verdict": s.scores.verdict if hasattr(s.scores, 'verdict') else s.scores.get('verdict', 'N/A'),
            "Violations": s.violations
        })
    return pd.DataFrame(rows)
