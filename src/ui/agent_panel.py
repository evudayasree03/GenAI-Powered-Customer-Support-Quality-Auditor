 """
SamiX Agent Panel
Handles audio uploads, real-time transcription, and AI scoring.
Location: src/ui/agent_panel.py
"""
import streamlit as st
import asyncio
from datetime import datetime

class AgentPanel:
    def __init__(self, history_manager, api_client):
        self.history = history_manager
        self.api = api_client

    def render(self):
        """Renders the main agent workspace."""
        st.markdown('<div class="samix-eyebrow">Workspace</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="samix-title">New Quality Audit</h1>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Input Section
        with st.container(border=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                agent_name = st.text_input("Confirm Agent Name", value=st.session_state.user_data.get("name"))
            with col2:
                category = st.selectbox("Audit Category", ["Customer Support", "Technical Sales", "Compliance Check"])

            uploaded_file = st.file_uploader("Upload Call Recording (MP3/WAV)", type=["mp3", "wav", "m4a"])

        # 2. Execution Logic
        if uploaded_file:
            if st.button("🚀 Start AI Analysis", use_container_width=True, type="primary"):
                self._run_audit_process(uploaded_file, agent_name)

    def _run_audit_process(self, file, agent_name):
        """Handles the async communication with the FastAPI backend."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step A: Transcription & Analysis
            status_text.info("📡 Sending audio to Render backend (Deepgram + Groq)...")
            progress_bar.progress(30)
            
            # Since Streamlit is sync and our client is async, we use a runner
            # In a real app, you'd use: result = asyncio.run(self.api.run_audit(...))
            # For now, let's simulate the UI response:
            import time
            time.sleep(2) 
            
            progress_bar.progress(70)
            status_text.info("🤖 AI is scoring the transcript against compliance policy...")
            time.sleep(2)

            # Step B: Display Results
            progress_bar.progress(100)
            status_text.success("✅ Analysis Complete!")
            
            self._render_audit_results({
                "score": 85.5,
                "sentiment": "Positive",
                "summary": "Agent followed opening protocols but missed the mandatory closing disclosure.",
                "transcript": "Agent: Hello... Customer: Hi, I need help... [Simulated Transcript]"
            })

        except Exception as e:
            st.error(f"Audit failed: {str(e)}")

    def _render_audit_results(self, data):
        """Displays the glassmorphism result cards."""
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("QA Score", f"{data['score']}%", delta="Above Avg")
        with c2:
            st.metric("Sentiment", data['sentiment'])
        with c3:
            st.metric("Duration", "4m 12s")

        with st.expander("📄 View Full Transcript", expanded=False):
            st.write(data['transcript'])

        with st.container(border=True):
            st.markdown("### 💡 AI Analysis Summary")
            st.write(data['summary'])
