import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Any
from datetime import datetime

# 1. LANDING & HERO COMPONENTS
def render_hero_section():
    st.markdown('<div style="text-align:center;padding:3rem 0;"><h1>SamiX AI Auditor</h1><p>GenAI-Powered Quality Assurance</p></div>', unsafe_allow_html=True)

def render_page_hero(eyebrow, title, subtitle, stats=None):
    st.markdown(f"## {title}\n*{subtitle}*")
    if stats: render_metrics_showcase(stats)

def render_feature_cards():
    """This fixes the current crash"""
    col1, col2, col3 = st.columns(3)
    with col1: st.info("**Audit Analysis**\n\nAutomated scoring and insight generation.")
    with col2: st.success("**Policy Check**\n\nVerify compliance with company guidelines.")
    with col3: st.warning("**Trend Tracking**\n\nMonitor agent performance over time.")

def render_metrics_showcase(stats):
    cols = st.columns(len(stats))
    for i, (label, value, note) in enumerate(stats):
        cols[i].metric(label, value, note)

# 2. ANALYSIS & CHART COMPONENTS
def render_gauge(value, title, max_val=10.0):
    st.metric(title, f"{value}/{max_val}")

def render_three_gauges(scores):
    c1, c2, c3 = st.columns(3)
    get_v = lambda a: getattr(scores, a, scores.get(a, 0)) if hasattr(scores, 'get') or hasattr(scores, a) else 0
    with c1: render_gauge(get_v("empathy"), "Empathy")
    with c2: render_gauge(get_v("professionalism"), "Professionalism")
    with c3: render_gauge(get_v("compliance"), "Compliance")

def render_dual_score_chart(scores):
    agent_data = getattr(scores, 'agent_by_turn', scores.get('agent_by_turn', []))
    cust_data = getattr(scores, 'customer_sentiment', scores.get('customer_sentiment', []))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=agent_data, name="Agent"))
    fig.add_trace(go.Scatter(y=cust_data, name="Customer"))
    st.plotly_chart(fig, use_container_width=True)

# 3. TRANSCRIPT & UTILITIES
def render_transcript(turns, wrong_turns=None):
    for turn in turns:
        speaker = getattr(turn, 'speaker', turn.get('speaker', 'User'))
        st.chat_message(speaker.lower()).write(getattr(turn, 'text', turn.get('text', '')))

def render_wrong_turns(wrong_turns):
    if not wrong_turns:
        st.success("No violations detected.")
        return
    for wt in wrong_turns:
        with st.expander(f"Violation: {getattr(wt, 'score_impact', 'Alert')}"):
            st.write(getattr(wt, 'agent_said', ''))

def render_cost_card(token_count, cost_usd):
    st.sidebar.metric("Cost", f"${cost_usd:.4f}", f"{token_count} tokens")

def render_filename_badge(filename, session_id):
    st.caption(f"📄 {filename} | ID: {session_id[:8]}")

def build_history_dataframe(sessions):
    return pd.DataFrame([{"session_id": s.session_id, "File": s.filename} for s in sessions])
