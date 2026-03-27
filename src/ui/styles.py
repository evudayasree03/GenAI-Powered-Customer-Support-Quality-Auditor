"""
SamiX UI theme - Refactored for Dark Mode & Glassmorphism
Optimized for 2026 SaaS trends and high-impact AI visualization.
"""

CSS = """
<style>
:root {
  --bg: #0F172A; /* Deep Slate */
  --surface: #1E293B; /* Lighter Slate */
  --surface-muted: rgba(30, 41, 59, 0.5);
  --sidebar: #020617; /* Near Black */
  --text: #F1F5F9;
  --text-soft: #94A3B8;
  --text-muted: #64748B;
  --border: rgba(226, 232, 240, 0.1);
  --primary: #6366F1; /* Indigo */
  --primary-soft: rgba(99, 102, 241, 0.15);
  --success: #10B981;
  --warning: #F59E0B;
  --danger: #EF4444;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.5);
  --shadow-md: 0 12px 30px rgba(0, 0, 0, 0.3);
  --radius: 16px;
  --radius-sm: 12px;
}

/* Base App Layout */
html, body, .stApp {
  background: 
    radial-gradient(circle at top left, rgba(99, 102, 241, 0.15), transparent 35%),
    var(--bg) !important;
  color: var(--text) !important;
  font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
}

[data-testid="stAppViewContainer"] {
  background: transparent !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
  background: var(--sidebar) !important;
  border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] * {
  color: var(--text-soft) !important;
}

/* Glassmorphism Shells */
.samix-shell {
  background: rgba(30, 41, 59, 0.7);
  border: 1px solid var(--border);
  box-shadow: var(--shadow-md);
  backdrop-filter: blur(12px);
  border-radius: 24px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.samix-hero {
  background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 2rem;
  box-shadow: var(--shadow-sm);
  margin-bottom: 1.5rem;
}

.samix-eyebrow {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--primary);
  margin-bottom: 0.5rem;
}

.samix-title {
  font-size: 2.25rem;
  line-height: 1.1;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #FFFFFF;
}

.samix-kpi-card {
  background: rgba(15, 23, 42, 0.5);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.25rem;
  transition: transform 0.2s ease;
}

.samix-kpi-card:hover {
  transform: translateY(-2px);
  border-color: var(--primary);
}

/* Metric Containers */
div[data-testid="metric-container"] {
  background: rgba(30, 41, 59, 0.4) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  gap: 10px !important;
}

.stTabs [data-baseweb="tab"] {
  background: rgba(255, 255, 255, 0.05) !important;
  border-radius: 8px !important;
  color: var(--text-soft) !important;
}

.stTabs [aria-selected="true"] {
  background: var(--primary-soft) !important;
  color: var(--primary) !important;
  border: 1px solid var(--primary) !important;
}

/* Buttons */
.stButton > button {
  background: var(--primary) !important;
  border-radius: 10px !important;
  border: none !important;
  font-weight: 600 !important;
  transition: all 0.3s ease !important;
}

.stButton > button:hover {
  box-shadow: 0 0 20px rgba(99, 102, 241, 0.4) !important;
  transform: scale(1.02) !important;
}

/* Inputs */
input, textarea, select {
  background: rgba(15, 23, 42, 0.6) !important;
  color: white !important;
  border: 1px solid var(--border) !important;
}
</style>
"""

def inject_css() -> None:
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)
