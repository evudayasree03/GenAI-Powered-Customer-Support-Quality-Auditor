"""
SamiX UI Component Library

This module contains reusable Streamlit components for data visualization
and session reporting. It includes:
- ECharts Gauges: For high-impact metric visualization.
- Plotly Charts: For detailed turn-by-turn sentiment and quality analysis.
- Specialized Renderers: For speaker-separated transcripts and RAG-verified 'wrong turns'.
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

from src.utils.history_manager import (
    AuditScores, TranscriptTurn, WrongTurn
)


def render_page_hero(
    eyebrow: str,
    title: str,
    subtitle: str,
    stats: Optional[list[tuple[str, str, str]]] = None,
) -> None:
    """Reusable page hero for production-friendly Streamlit screens."""
    html = [
        '<div class="samix-hero">',
        f'<div class="samix-eyebrow">{eyebrow}</div>',
        f'<div class="samix-title">{title}</div>',
        f'<div class="samix-subtitle">{subtitle}</div>',
    ]
    if stats:
        html.append('<div class="samix-kpi-grid">')
        for label, value, note in stats:
            html.append(
                '<div class="samix-kpi-card">'
                f'<div class="samix-kpi-label">{label}</div>'
                f'<div class="samix-kpi-value">{value}</div>'
                f'<div class="samix-kpi-note">{note}</div>'
                '</div>'
            )
        html.append('</div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


# ECharts Gauges 

def render_gauge(value: float, title: str, max_val: float = 10.0) -> None:
    """
    Renders a sophisticated half-ring ECharts gauge.
    Automatically applies semantic coloring (Green/Amber/Red) based on the score.
    Enhanced with dark mode styling and premium animations.
    """
    try:
        from streamlit_echarts import st_echarts
    except ImportError:
        # Fallback to standard Streamlit metrics if the ECharts component is missing.
        st.metric(title, f"{value:.1f}/{max_val:.0f}")
        st.progress(value / max_val)
        return

    # Normalize value to a percentage for the gauge progress bar.
    pct     = min(100.0, max(0.0, value / max_val * 100))
    colour  = "#10B981" if pct >= 70 else "#F59E0B" if pct >= 50 else "#EF4444"

    option = {
        "backgroundColor": "transparent",
        "series": [{
            "type": "gauge",
            "startAngle": 200, "endAngle": -20,
            "min": 0, "max": 100,
            "splitNumber": 5,
            "itemStyle": {
                "color": colour,
                "shadowColor": colour,
                "shadowBlur": 12,
                "shadowOffsetX": 0,
                "shadowOffsetY": 2,
            },
            "progress": {"show": True, "width": 16, "itemStyle": {"borderRadius": [10, 0]}},
            "pointer":  {"show": False},
            "axisLine": {
                "lineStyle": {
                    "width": 16,
                    "color": [[1, "#F1F5F9"]],
                    "borderRadius": 10,
                }
            },
            "axisTick":  {"show": False},
            "splitLine": {"show": False},
            "axisLabel": {"show": False},
            "title": {
                "show": True,
                "offsetCenter": [0, "75%"],
                "fontSize": 12,
                "color": "#64748B",
                "fontFamily": "'Inter', sans-serif",
                "fontWeight": "600",
            },
            "detail": {
                "show": True,
                "offsetCenter": [0, "15%"],
                "formatter": f"{value:.1f}",
                "fontSize": 32,
                "fontWeight": "800",
                "color": colour,
                "fontFamily": "'Inter', sans-serif",
                "letterSpacing": -1,
            },
            "data": [{"value": pct, "name": title.upper()}],
        }]
    }
    st_echarts(options=option, height="180px", key=f"gauge_{title}_{id(value)}")


def render_three_gauges(scores: AuditScores) -> None:
    """ Renders the top 3 quality dimensions side-by-side. """
    c1, c2, c3 = st.columns(3)
    with c1:
        render_gauge(scores.empathy,        "Empathy")
    with c2:
        render_gauge(scores.professionalism, "Professionalism")
    with c3:
        render_gauge(scores.compliance,      "Compliance")


# Dual Score Chart 

def render_dual_score_chart(scores: AuditScores) -> None:
    """
    Renders a Plotly line chart comparing Agent Quality vs. Customer Sentiment.
    Highlights 'danger zones' (scores < 40%) with red background shading.
    """
    n_agent = len(scores.agent_by_turn)
    n_cust  = len(scores.customer_sentiment)
    n       = max(n_agent, n_cust, 1)

    # Pad data to ensure both lines have the same length.
    agent_data = scores.agent_by_turn + [5.0] * (n - n_agent)
    cust_data  = scores.customer_sentiment + [5.0] * (n - n_cust)
    x_labels   = [f"T{i+1}" for i in range(n)]

    fig = go.Figure()

    # Highlight turns where performance or sentiment dipped below the failure threshold.
    for i, (a, c) in enumerate(zip(agent_data, cust_data)):
        if a < 4.0 or c < 4.0:
            fig.add_vrect(
                x0=i - 0.4, x1=i + 0.4,
                fillcolor="rgba(239,68,68,0.12)",
                layer="below", line_width=0,
            )

    # Primary Line: Agent Quality
    fig.add_trace(go.Scatter(
        x=x_labels, y=agent_data,
        mode="lines+markers",
        name="Agent quality",
        line=dict(color="#3B82F6", width=3),
        marker=dict(
            size=7,
            color=["#241010" if v < 4 else "#F59E0B" if v < 7 else "#10B981" for v in agent_data],
            line=dict(width=1, color="#0F172A"),
        ),
        hovertemplate="<b>%{x}</b><br>Agent: %{y:.1f}/10<extra></extra>",
    ))

    # Secondary Line: Customer Sentiment (Dashed)
    fig.add_trace(go.Scatter(
        x=x_labels, y=cust_data,
        mode="lines+markers",
        name="Customer sentiment",
        line=dict(color="#749CE2", width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.1f}/10<extra></extra>",
        marker=dict(size=5, color="#749CE2")
    ))

    # Static fail threshold marker.
    fig.add_hline(y=4.0, line_dash="dash", line_color="rgba(239,68,68,0.4)",
                  annotation_text="40% threshold", annotation_font_size=9,
                  annotation_font_color="#EF4444")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFC",
        font=dict(family="'Inter', sans-serif", color="#475569", size=11),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            font=dict(size=11, color="#0F172A"),
            x=0.01,
            y=0.99,
        ),
        xaxis=dict(
            gridcolor="#E2E8F0",
            zerolinecolor="#E2E8F0",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 10.5],
            dtick=2,
            gridcolor="#E2E8F0",
            zerolinecolor="#E2E8F0",
            showgrid=True,
        ),
        margin=dict(l=50, r=20, t=20, b=40),
        height=300,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


# Transcript Renderer 

def render_transcript(
    turns: list[TranscriptTurn],
    wrong_turns: Optional[list[WrongTurn]] = None,
) -> None:
    """
    Renders a color-coded chat interface with professional dark mode styling.
    Agent and Customer turns are visually distinct. Critical 'wrong turns'
    are flagged with warning banners directly beneath the offending turn.
    """
    wrong_map: dict[int, WrongTurn] = {}
    if wrong_turns:
        for wt in wrong_turns:
            wrong_map[wt.turn_number] = wt

    for turn in turns:
        spk_colour = "#3B82F6" if turn.speaker == "AGENT" else "#64748B"
        bg_rgb     = "59,130,246"  if turn.speaker == "AGENT" else "107,114,128"
        bg_alpha   = "0.08"        if turn.speaker == "AGENT" else "0.05"

        col_spk, col_text = st.columns([0.95, 4.95])
        with col_spk:
            st.markdown(
                f'<div style="text-align:right;padding-top:8px;">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;'
                f'color:{spk_colour};letter-spacing:0.1em;font-weight:700;">'
                f'{turn.speaker}</span><br>'
                f'<span style="font-size:0.65rem;color:#64748B;margin-top:3px;display:block;">{turn.timestamp}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_text:
            st.markdown(
                f'<div class="{"transcript-agent" if turn.speaker == "AGENT" else "transcript-customer"}">'
                f'<span style="color:var(--text-primary);">{turn.text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # If this turn was flagged as a violation, render the audit warning.
        if turn.turn in wrong_map:
            wt = wrong_map[turn.turn]
            st.markdown(
                f'<div style="background:rgba(239,68,68,0.12);border-left:3px solid rgba(239,68,68,0.6);'
                f'border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0 12px 0;'
                f'color:#FCA5A5;font-size:0.85rem;line-height:1.5;">'
                f'⚠ <strong>Turn {wt.turn_number}</strong> · {wt.score_impact}<br>'
                f'{wt.what_went_wrong[:140]}{"…" if len(wt.what_went_wrong) > 140 else ""}<br>'
                f'<span style="opacity:0.8;font-size:0.75rem;margin-top:6px;display:block;">📚 RAG: {wt.rag_source}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# Fact-Verification UI 

def render_wrong_turns(
    wrong_turns: list[WrongTurn],
    specific_corrections: Optional[dict[int, str]] = None,
) -> None:
    """
    Deep-dive view for factual and policy errors with superior dark mode design.
    Shows the offending quote, the RAG-verified fact, and a correction field.
    """
    if not wrong_turns:
        st.markdown(
            '<div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.4);'
            'border-radius:10px;padding:14px 16px;color:#6EE7B7;font-size:0.9rem;">'
            '✓ No critical failures detected in this session.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for wt in wrong_turns:
        with st.expander(
            f"🔴 Turn {wt.turn_number} — {wt.speaker}  ·  {wt.score_impact}",
            expanded=False,
        ):
            # Section: What was said
            st.markdown(
                '<span style="font-size:0.85rem;font-weight:600;color:#475569;'
                'text-transform:uppercase;letter-spacing:0.5px;">📝 What was said</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-style:italic;background:#F8FAFC;'
                f'border-left:3px solid #64748B;padding:12px 16px;border-radius:0 8px 8px 0;'
                f'color:#0F172A;font-size:0.88rem;margin:8px 0 16px 0;line-height:1.6;">'
                f'"{wt.agent_said}"'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Section: What went wrong
            st.markdown(
                '<span style="font-size:0.85rem;font-weight:600;color:#475569;'
                'text-transform:uppercase;letter-spacing:0.5px;">⚠️ What went wrong</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#FEF2F2;border-left:3px solid #EF4444;'
                f'border-radius:0 8px 8px 0;padding:12px 16px;color:#991B1B;'
                f'font-size:0.88rem;margin:8px 0 16px 0;line-height:1.6;">'
                f'{wt.what_went_wrong}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Section: Correct fact
            st.markdown(
                '<span style="font-size:0.85rem;font-weight:600;color:#475569;'
                'text-transform:uppercase;letter-spacing:0.5px;">✓ Correct fact (RAG verified)</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#F0FDF4;border-left:3px solid #10B981;'
                f'border-radius:0 8px 8px 0;padding:12px 16px;color:#065F46;'
                f'font-size:0.88rem;margin:8px 0 16px 0;line-height:1.6;">'
                f'{wt.correct_fact}<br>'
                f'<span style="opacity:0.7;font-size:0.75rem;margin-top:8px;display:block;">'
                f'📚 Source: <strong>{wt.rag_source}</strong> · Confidence: <strong>{wt.rag_confidence:.0%}</strong></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Section: Suggested correction
            st.markdown(
                '<span style="font-size:0.85rem;font-weight:600;color:#CBD5E1;'
                'text-transform:uppercase;letter-spacing:0.5px;margin-top:16px;display:block;">✏️ Suggested correction</span>',
                unsafe_allow_html=True,
            )
            key = f"correction_{wt.turn_number}"
            default = specific_corrections.get(wt.turn_number, wt.specific_correction) \
                if specific_corrections else wt.specific_correction
            st.text_area(
                label="Edit correction",
                value=default,
                height=100,
                key=key,
                label_visibility="collapsed",
                placeholder="Enter the correction that should have been made...",
            )


# Financial Reporting 

def render_cost_card(
    token_count: int,
    cost_usd: float,
    revenue_per_call: float = 5.0,
) -> None:
    """ Renders a premium metric card summarizing API costs vs. revenue profit with dark mode styling. """
    profit = revenue_per_call - cost_usd
    margin = (profit / revenue_per_call * 100) if revenue_per_call else 0.0

    # Create a professional container
    st.markdown(
        '<div style="background:#FFFFFF;'
        'border:1px solid #E2E8F0;border-radius:12px;padding:24px;'
        'box-shadow:var(--shadow-sm);">'
        '<span style="font-size:0.85rem;font-weight:700;color:#152EAE;'
        'text-transform:uppercase;letter-spacing:0.8px;">Cost Analysis</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        c1.metric("Tokens used", f"{token_count:,}", help="Total tokens processed")
    with c2:
        c2.metric("API cost", f"${cost_usd:.5f}", help="Total API cost for this audit")
    with c3:
        c3.metric("Revenue", f"${revenue_per_call:.2f}", help="Revenue per audit")
    with c4:
        margin_color = "off" if margin < 20 else "normal"
        c4.metric(
            "Profit",
            f"${profit:.4f}",
            delta=f"{margin:.1f}% margin",
            delta_color=margin_color,
            help="Net profit per audit",
        )


# Metadata Components 

def render_filename_badge(uploaded_name: str, stored_name: str) -> None:
    """ Displays a badge verifying that the stored filename matches the upload with premium styling. """
    match = uploaded_name == stored_name
    icon  = "✓" if match else "✗"
    colour = "#6EE7B7" if match else "#FCA5A5"
    bg_alpha = "0.12" if match else "0.12"
    border_colour = "rgba(16,185,129,0.4)" if match else "rgba(239,68,68,0.4)"
    
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;'
        f'color:{colour};padding:8px 12px;'
        f'background:rgba({colour},0.08);border:1px solid {border_colour};'
        f'border-radius:6px;display:inline-block;">'
        f'{icon} <strong>Stored as:</strong> <code style="color:{colour};background:transparent;">{stored_name}</code>'
        f'{"" if not match else "&nbsp;✓ Matches"}</div>',
        unsafe_allow_html=True,
    )


def build_history_dataframe(sessions: list) -> pd.DataFrame:
    """ Converts a list of AuditSession objects into a Pandas DataFrame for UI display. """
    rows = []
    for s in sessions:
        verdict_emoji = (
            "🟢" if s.scores.final_score >= 80 else
            "🟡" if s.scores.final_score >= 60 else "🔴"
        )
        rows.append({
            "Uploaded filename":     s.filename,
            "Stored history name":   s.stored_name,
            "Date / time":           s.upload_time,
            "Mode":                  s.mode.capitalize(),
            "Agent score":           f"{s.scores.final_score:.0f}/100",
            "Cust sentiment":        f"{s.scores.customer_overall:.1f}/10",
            "Verdict":               f"{verdict_emoji} {s.scores.verdict}",
            "Violations":            s.violations,
            "session_id":            s.session_id,
        })
    df = pd.DataFrame(rows)
    return df


# ============================================================================
# AirCover-Inspired Professional Components
# ============================================================================

def render_hero_section(
    headline: str,
    subheadline: str,
    cta_label: str = "Get Started",
    cta_key: str = "hero_cta",
) -> bool:
    """
    Renders a professional hero section with headline, subheadline, and CTA.
    Inspired by AirCover's modern SaaS design.
    
    Returns: True if CTA button was clicked, False otherwise.
    """
    st.markdown(f'<div style="margin:50px 0 60px 0;">', unsafe_allow_html=True)
    
    # Headline
    st.markdown(
        f'<h1 style="font-size:48px;font-weight:800;color:#F1F5F9;'
        f'letter-spacing:-1px;margin-bottom:16px;line-height:1.2;">'
        f'{headline}</h1>',
        unsafe_allow_html=True,
    )
    
    # Subheadline
    st.markdown(
        f'<p style="font-size:18px;color:#CBD5E1;line-height:1.6;'
        f'max-width:600px;margin-bottom:32px;">'
        f'{subheadline}</p>',
        unsafe_allow_html=True,
    )
    
    # CTA Button (using Streamlit's native button)
    col1, col2, col3 = st.columns([1.2, 2, 2])
    with col1:
        clicked = st.button(cta_label, key=cta_key, use_container_width=True)
    
    st.markdown(f'</div>', unsafe_allow_html=True)
    return clicked


def render_metrics_showcase(metrics: dict[str, str | int]) -> None:
    """
    Renders large impact metrics in a grid layout.
    Inspired by AirCover's "50% Reduction in Ramp Time" style metrics.
    
    Args:
        metrics: Dictionary of {metric_label: value}
        Example: {"Accuracy": "98%", "Calls Analyzed": "2,500", "Time Saved": "45hrs"}
    """
    st.markdown('<div style="margin:60px 0;">', unsafe_allow_html=True)
    
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(
                f'<div style="text-align:center;'
                f'background:#FFFFFF;border:1px solid #E2E8F0;'
                f'border-radius:14px;padding:34px 16px;box-shadow:var(--shadow);">'
                f'<div style="font-size:42px;font-weight:800;color:#152EAE;'
                f'margin-bottom:12px;letter-spacing:-1px;">{value}</div>'
                f'<div style="font-size:14px;color:#64748B;font-weight:600;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_feature_cards(
    features: list[dict[str, str]],
    columns: int = 3,
) -> None:
    """
    Renders a grid of feature cards in AirCover style.
    
    Args:
        features: List of dicts with keys: 'title', 'description', 'icon' (optional)
        columns: Number of columns (default 3)
        Example:
            [
                {'title': 'Before Call', 'description': 'Context from conversations', 'icon': '📋'},
                {'title': 'During Call', 'description': 'Real-time guidance', 'icon': '🎯'},
                {'title': 'After Call', 'description': 'Automated insights', 'icon': '✅'},
            ]
    """
    st.markdown('<div style="margin:40px 0;">', unsafe_allow_html=True)
    
    cols = st.columns(columns)
    for idx, feature in enumerate(features):
        col = cols[idx % columns]
        with col:
            icon = feature.get('icon', '✨')
            title = feature.get('title', 'Feature')
            description = feature.get('description', '')
            
            st.markdown(
                f'<div style="background:linear-gradient(135deg,'
                f'rgba(30,41,59,0.7),rgba(26,31,53,0.5));'
                f'border:1px solid rgba(59,130,246,0.3);'
                f'border-radius:12px;padding:24px;'
                f'backdrop-filter:blur(12px);transition:all 0.3s ease;'
                f'cursor:pointer;">'
                f'<div style="font-size:32px;margin-bottom:12px;">{icon}</div>'
                f'<h3 style="font-size:18px;font-weight:700;color:#F1F5F9;'
                f'margin:0 0 12px 0;">{title}</h3>'
                f'<p style="font-size:14px;color:#CBD5E1;margin:0;line-height:1.5;">'
                f'{description}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_testimonial(
    quote: str,
    author: str,
    title: str = "",
    company: str = "",
) -> None:
    """
    Renders a professional testimonial card in AirCover style.
    
    Args:
        quote: The testimonial text
        author: Person's name
        title: Job title (optional)
        company: Company name (optional)
    """
    author_info = f"{author}"
    if title:
        author_info += f", {title}"
    if company:
        author_info += f" at {company}"
    
    st.markdown(
        f'<div style="background:rgba(30,41,59,0.6);'
        f'border-left:3px solid #3B82F6;border-radius:8px;'
        f'padding:24px;margin:16px 0;">'
        f'<p style="font-size:15px;color:#CBD5E1;font-style:italic;'
        f'margin:0 0 16px 0;line-height:1.6;">"{quote}"</p>'
        f'<p style="font-size:13px;color:#94A3B8;margin:0;font-weight:600;">'
        f'— {author_info}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_professional_divider() -> None:
    """ Renders a professional divider section. """
    st.markdown(
        '<div style="height:2px;background:linear-gradient(90deg,'
        'rgba(59,130,246,0),rgba(59,130,246,0.5),rgba(59,130,246,0));'
        'margin:60px 0;"></div>',
        unsafe_allow_html=True,
    )
