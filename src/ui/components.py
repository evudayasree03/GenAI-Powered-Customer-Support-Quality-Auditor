"""
SamiX UI Components
Reusable glassmorphism elements and analysis charts.
Location: src/ui/components.py
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Any
from datetime import datetime

# --- 1. LANDING & HERO COMPONENTS ---

def render_hero_section():
    """Main landing hero for the unauthenticated state."""
    st.markdown('<div style="text-align:center;padding:3rem 0;"><h1>SamiX AI Auditor</h1><p>GenAI-Powered Quality Assurance</p></div>', unsafe_allow_html=True)

def render_page_hero(eyebrow: str, title: str, subtitle: str, stats: list = None):
    """
    Standardized hero section for all app pages.
    Includes glassmorphism-styled typography and metrics.
    """
    st.markdown(f'<div style="color: #6366f1; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.85rem; margin-bottom: 0.5rem;">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem; color: #f8fafc;">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin-bottom: 2rem;">{subtitle}</p>', unsafe_allow_html=True)
    
    if stats:
        cols = st.columns(len(stats))
        for i, (label, value, note) in enumerate(stats):
            with cols[i]:
                st.metric(label, value, note)

def render_feature_cards():
    """Renders the main value propositions."""
    col1, col2, col3 = st.columns(3)
    with col1: 
        st.info("**Audit Analysis**\n\nAutomated scoring and insight generation.")
    with col2: 
        st.success("**Policy Check**\n\nVerify compliance with company guidelines.")
    with col3: 
        st.warning("**Trend Tracking**\n\nMonitor agent performance over time.")

def render_testimonial(text: str, author: str, role: str):
    """
    FIX: Added missing function to resolve Initialization Error.
    Renders a stylized quote for the Login Page sidebar.
    """
    st.markdown(f"""
        <div style="margin-top: 2rem; padding: 1.5rem; border-left: 3px solid #6366f1; background: rgba(99, 102, 241, 0.05); border-radius: 0 12px 12px 0;">
            <p style="font-style: italic; color: #e2e8f0; font-size: 1.1rem; line-height: 1.6;">"{text}"</p>
            <p style="margin-top: 1rem; font-weight: 600; color: #f8fafc; margin-bottom: 0;">{author}</p>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">{role}</p>
        </div>
    """, unsafe_allow_html=True)

# --- 2. ANALYSIS & CHART COMPONENTS ---

def render_gauge(value: float, title: str, max_val: float = 10.0):
    """Simple metric-based gauge."""
    st.metric(title, f"{value}/{max_val}")

def render_three_gauges(scores: Any):
    """Displays key performance indicators in three columns."""
    c1, c2, c3 = st.columns(3)
    # Helper to handle both Dict and Object types from LLM output
    get_v = lambda a: getattr(scores, a, scores.get(a, 0)) if hasattr(scores, 'get') or hasattr(scores, a) else 0
    
    with c1: render_gauge(get_v("empathy"), "Empathy")
    with c2: render_gauge(get_v("professionalism"), "Professionalism")
    with c3: render_gauge(get_v("compliance"), "Compliance")

def render_dual_score_chart(scores: Any):
    """Renders a Plotly line chart comparing agent performance to customer sentiment."""
    agent_data = getattr(scores, 'agent_by_turn', scores.get('agent_by_turn', []))
    cust_data = getattr(scores, 'customer_sentiment', scores.get('customer_sentiment', []))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=agent_data, name="Agent Score", line=dict(color='#6366f1', width=3)))
    fig.add_trace(go.Scatter(y=cust_data, name="Customer Sentiment", line=dict(color='#10b981', width=3)))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=0, r=0, t=30, b=0),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 3. TRANSCRIPT & UTILITIES ---

def render_transcript(turns: list):
    """Renders a chat-like interface for the call transcript."""
    for turn in turns:
        speaker = getattr(turn, 'speaker', turn.get('speaker', 'User'))
        text = getattr(turn, 'text', turn.get('text', ''))
        st.chat_message(speaker.lower()).write(text)

def render_wrong_turns(wrong_turns: list):
    """Highlights specific violations or areas for improvement."""
    if not wrong_turns:
        st.success("No compliance violations detected in this session.")
        return
    for wt in wrong_turns:
        with st.expander(f"⚠️ Violation: {getattr(wt, 'score_impact', 'Compliance Alert')}"):
            st.write(f"**Agent Said:** {getattr(wt, 'agent_said', 'N/A')}")
            st.info(f"**Feedback:** {getattr(wt, 'reasoning', 'No specific feedback provided.')}")

def render_cost_card(token_count: int, cost_usd: float):
    """Displays API usage cost in the sidebar."""
    st.sidebar.metric("Analysis Cost", f"${cost_usd:.4f}", f"{token_count} tokens")

def render_filename_badge(filename: str, session_id: str):
    """Renders a small metadata badge for the current file."""
    st.caption(f"📄 {filename} | ID: {session_id[:8]}")

def build_history_dataframe(sessions: list) -> pd.DataFrame:
    """Converts raw session data into a displayable DataFrame for history logs."""
    return pd.DataFrame([
        {
            "Session ID": s.session_id[:8],
            "Date": s.created_at.strftime("%Y-%m-%d %H:%M"),
            "Agent": s.agent_name,
            "File": s.filename,
            "Score": f"{s.score}/10"
        } for s in sessions
    ])
