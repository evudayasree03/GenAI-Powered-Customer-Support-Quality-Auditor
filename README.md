# SamiX - GenAI-Powered Customer Support Quality Auditor

SamiX is a modern, AI-driven platform for auditing customer support interactions (audio and transcripts). It automates quality assurance by transcribing calls, evaluating agent performance against complex QA rubrics, and verifying factual statements against a dynamic knowledge base.

SamiX has recently been upgraded to a decoupled architecture:
- **FastAPI Backend**: Handles heavy AI processing (STT, LLM inference, RAG Retrieval).
- **Streamlit Frontend**: A lightweight, lightning-fast UI client for agents and admins.

---

## 🏗 Architecture

### 1. Stage 1: Indexing (Knowledge Base)
- **Documents**: PDF/TXT policies
- **Chunking**: `RecursiveCharacterTextSplitter`
- **Embeddings**: `BAAI/bge-small-en-v1.5`
- **Vector DB**: `Milvus Lite` (with automatic BM25 keyword fallback)

### 2. Stage 2: Retrieval (Classic RAG)
- **Query**: Agent statements or customer questions
- **Search**: Hybrid (Semantic similarity via Milvus + BM25 keyword overlap)
- **Fusion**: Reciprocal Rank Fusion (RRF)
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` ensures incredibly high precision.

### 3. Stage 3: Generation (Audit & Scoring)
- **Prompt Construction**: Injects retrieved policy context directly into the prompt.
- **LLM**: `Groq` (`llama-3.3-70b-versatile`)
- **Output**: Structured JSON report grading empathy, compliance, resolution, and factual integrity.

### 4. Stage 4: Transcription (STT)
- **Primary**: `Deepgram Nova-3`
- **Fallback**: Local `openai-whisper`

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- `Python 3.11+`
- `FFmpeg` installed and available on `PATH` (required for audio conversion)
- Internet access (for downloading embedding/reranker models on first run)

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
# AI Models
GROQ_API_KEY=gsk_your_key_here
DEEPGRAM_API_KEY=your_deepgram_key_here

# Optional: HuggingFace Token for higher rate limits on model downloads
HF_TOKEN=hf_your_token_here

# Database
SQLITE_DB_PATH=samix.db

# Optional: If you want to force Streamlit to use the FastAPI backend locally
# SAMIX_API_URL=http://localhost:8000
```

### 3. Installation

**Windows:**
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts\init_db.py
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/init_db.py
```

### 4. Running the Application

SamiX can run as a standalone Streamlit app, or as a decoupled Client/Server setup.

**Option A: Unified Standalone (Easiest for local Dev)**
Just run the Streamlit app. It will process audio, LLM requests, and RAG locally within the Streamlit session.
```bash
streamlit run app.py
```

**Option B: Client / Server Architecture (Recommended for Production)**
1. Open Terminal 1 and start the FastAPI Backend:
   ```bash
   uvicorn api.main:app --reload
   ```
2. Open Terminal 2, ensure `SAMIX_API_URL=http://localhost:8000` is set in your `.env`, and start the Frontend:
   ```bash
   streamlit run app.py
   ```

**Default Login:**
- **Email:** `admin@samix.ai`
- **Password:** `admin`

*(To change passwords, run `python generate_hash.py`, paste your new password, and copy the hash into `.streamlit/secrets.toml`)*

---

## ☁️ Free Cloud Deployment Guide

You can host SamiX entirely for free using Render.com and Streamlit Community Cloud.

### Step 1: Deploy FastAPI Backend to Render.com
1. Create a free account at [Render.com](https://render.com).
2. Click **New +** → **Blueprint**.
3. Connect your GitHub repository containing the SamiX codebase.
4. Render will read the `render.yaml` file and automatically provision a Web Service.
5. Once created, go to the Service **Environment** tab and explicitly add your secret keys:
   - `GROQ_API_KEY`
   - `DEEPGRAM_API_KEY`
   - `HF_TOKEN` (optional)
6. Copy the deployed URL (e.g., `https://samix-api.onrender.com`).

*(Note: Render's free tier spins down after 15 minutes of inactivity. The first request after sleeping will take ~30-50 seconds as the server wakes up).*

### Step 2: Deploy Streamlit UI to Streamlit Community Cloud
1. Create a free account at [Streamlit Cloud](https://share.streamlit.io).
2. Click **New app** and connect your GitHub repository.
3. Set the Main file path to `app.py`.
4. Click **Advanced settings** and configure your secrets in TOML format:

```toml
[auth]
hashed_password   = "$2b$12$YOUR_BCRYPT_HASH_HERE"
cookie_key        = "samix_super_secret_cookie_key_change_this"

# Configure Streamlit to use your new Render FastAPI backend:
SAMIX_API_URL = "https://samix-api.onrender.com"
```

5. Click **Deploy**.

The app is now fully decoupled and ready for production usage. Streamlit handles the lightweight UI, while Render handles the heavy STT/RAG computations asynchronously!

---

## 🛠 Project Structure

```text
samix/
├── api/                   # FastAPI Backend
│   ├── main.py            # API Endpoints (/audit, /rag/query, /health)
│   ├── schemas.py         # Pydantic Request/Response models
│   ├── deps.py            # Dependency Injection
│   └── requirements.txt   # Backend-only dependencies
├── src/                   # Core Logic & Frontend
│   ├── api_client.py      # HTTPX wrapper connecting Streamlit to FastAPI
│   ├── auth/              # Authentication protocols
│   ├── db/                # SQLite Data Layer
│   ├── pipeline/          # Groq LLM and Deepgram STT Processors
│   ├── ui/                # Streamlit UI Components & Workspaces
│   └── utils/             # KBManager, HistoryManager, CostTracking
├── data/                  # Local Storage (DB, uploads, KB chunks, artifacts)
├── app.py                 # Streamlit Entry Point
├── config.py              # Environment configuration loader
├── render.yaml            # Render.com IaC Deployment Blueprint
└── requirements.txt       # Global Dependencies
```
