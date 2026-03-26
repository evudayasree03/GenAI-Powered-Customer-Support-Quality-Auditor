"""
SamiX UI theme

Production-oriented Streamlit styling with a clean, lightweight enterprise look.
"""

CSS = """
<style>
:root {
  --bg: #f4f7fb;
  --surface: #ffffff;
  --surface-muted: #f8fafc;
  --sidebar: #0f172a;
  --text: #0f172a;
  --text-soft: #475569;
  --text-muted: #64748b;
  --border: #e2e8f0;
  --primary: #1d4ed8;
  --primary-soft: rgba(29, 78, 216, 0.08);
  --success: #059669;
  --warning: #d97706;
  --danger: #dc2626;
  --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
  --shadow-md: 0 12px 30px rgba(15, 23, 42, 0.08);
  --radius: 16px;
  --radius-sm: 12px;
}

html, body, .stApp {
  background:
    radial-gradient(circle at top left, rgba(29,78,216,0.06), transparent 28%),
    linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%) !important;
  color: var(--text) !important;
  font-family: "Segoe UI", Inter, system-ui, sans-serif !important;
}

[data-testid="stAppViewContainer"] {
  background: transparent !important;
}

section[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0)),
    var(--sidebar) !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
}

section[data-testid="stSidebar"] * {
  color: rgba(255,255,255,0.78) !important;
}

.stMainBlockContainer {
  max-width: 1240px !important;
  padding-top: 1.25rem !important;
  padding-bottom: 3rem !important;
}

.samix-shell {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(226,232,240,0.9);
  box-shadow: var(--shadow-md);
  backdrop-filter: blur(8px);
  border-radius: 24px;
  padding: 1.25rem 1.25rem 1rem 1.25rem;
  margin-bottom: 1.25rem;
}

.samix-hero {
  background:
    radial-gradient(circle at top right, rgba(29,78,216,0.14), transparent 32%),
    linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 1.4rem 1.5rem;
  box-shadow: var(--shadow-sm);
  margin-bottom: 1rem;
}

.samix-eyebrow {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--primary);
  margin-bottom: 0.45rem;
}

.samix-title {
  font-size: 2rem;
  line-height: 1.05;
  font-weight: 800;
  letter-spacing: -0.04em;
  color: var(--text);
  margin-bottom: 0.45rem;
}

.samix-subtitle {
  color: var(--text-soft);
  font-size: 0.98rem;
  line-height: 1.6;
  max-width: 72ch;
}

.samix-kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.85rem;
  margin-top: 1rem;
}

.samix-kpi-card {
  background: rgba(255,255,255,0.9);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1rem;
  box-shadow: var(--shadow-sm);
}

.samix-kpi-label {
  color: var(--text-muted);
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.samix-kpi-value {
  color: var(--text);
  font-size: 1.75rem;
  font-weight: 800;
  letter-spacing: -0.05em;
  margin-top: 0.3rem;
}

.samix-kpi-note {
  color: var(--text-soft);
  font-size: 0.82rem;
  margin-top: 0.25rem;
}

.section-header {
  font-size: 1.45rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.03em;
  margin-bottom: 1rem;
}

.samix-card, .glass-card {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-sm) !important;
  padding: 1rem !important;
}

div[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.15rem !important;
  box-shadow: var(--shadow-sm) !important;
}

div[data-testid="metric-container"] label {
  color: var(--text-muted) !important;
  font-size: 0.8rem !important;
  font-weight: 700 !important;
}

div[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-weight: 800 !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.55rem !important;
  border-bottom: 1px solid var(--border) !important;
  padding-bottom: 0.7rem !important;
}

.stTabs [data-baseweb="tab"] {
  height: 42px !important;
  background: rgba(255,255,255,0.68) !important;
  border: 1px solid transparent !important;
  border-radius: 999px !important;
  padding: 0.45rem 1rem !important;
  color: var(--text-soft) !important;
  font-weight: 700 !important;
}

.stTabs [aria-selected="true"] {
  background: #ffffff !important;
  border-color: var(--border) !important;
  color: var(--primary) !important;
  box-shadow: var(--shadow-sm) !important;
}

.stButton > button, .stDownloadButton > button {
  background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  padding: 0.7rem 1rem !important;
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.2) !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
  filter: brightness(1.04) !important;
  transform: translateY(-1px) !important;
}

.stTextInput input, .stTextArea textarea, .stSelectbox select, .stFileUploader {
  border-radius: 12px !important;
}

input, textarea, select {
  background: #ffffff !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}

input:focus, textarea:focus, select:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.12) !important;
}

.mono-badge {
  background: var(--surface-muted) !important;
  color: var(--primary) !important;
  border: 1px solid var(--border) !important;
  padding: 0.35rem 0.65rem !important;
  border-radius: 999px !important;
  font-size: 0.75rem !important;
  font-weight: 700 !important;
  display: inline-block !important;
}

.verdict-good {
  color: var(--success) !important;
  background: rgba(5,150,105,0.1) !important;
  padding: 0.35rem 0.65rem !important;
  border-radius: 999px !important;
  font-weight: 700 !important;
  display: inline-block !important;
}

.verdict-fail {
  color: var(--danger) !important;
  background: rgba(220,38,38,0.1) !important;
  padding: 0.35rem 0.65rem !important;
  border-radius: 999px !important;
  font-weight: 700 !important;
  display: inline-block !important;
}

.transcript-agent,
.transcript-customer {
  background: #ffffff !important;
  border: 1px solid var(--border) !important;
  border-left-width: 4px !important;
  padding: 0.95rem !important;
  border-radius: 0 14px 14px 0 !important;
  margin-bottom: 0.75rem !important;
  box-shadow: var(--shadow-sm) !important;
}

.transcript-agent {
  border-left-color: var(--primary) !important;
}

.transcript-customer {
  border-left-color: #94a3b8 !important;
}

@media (max-width: 980px) {
  .samix-kpi-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
"""


def inject_css() -> None:
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)
