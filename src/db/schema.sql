-- SamiX Database Schema
-- Updated for SQLite Compatibility & Render.com Deployment

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'agent',
    is_active INTEGER DEFAULT 1, -- Using INTEGER for SQLite Boolean
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Transcriptions Table
CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    audio_path TEXT,
    transcription_text TEXT,
    language TEXT DEFAULT 'en',
    duration_seconds REAL,
    confidence_score REAL,
    processor_used TEXT,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. API Responses Table
CREATE TABLE IF NOT EXISTS api_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    api_name TEXT NOT NULL,
    endpoint TEXT,
    request_hash TEXT,
    response_json TEXT, -- Stored as JSON string
    status_code INTEGER,
    processing_time_ms REAL,
    tokens_used INTEGER,
    cost_usd REAL,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Audit Sessions Table
CREATE TABLE IF NOT EXISTS audit_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL, -- UNIQUE is required for UPSERT logic
    filename TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mode TEXT,
    transcript_text TEXT,
    transcript_word_count INTEGER DEFAULT 0,
    empathy_score REAL,
    compliance_score REAL,
    resolution_score REAL,
    overall_score REAL,
    summary TEXT,
    violations INTEGER DEFAULT 0,
    key_moments TEXT, -- Stored as JSON string
    token_count INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    processing_time_seconds REAL,
    is_flagged INTEGER DEFAULT 0,
    flag_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Compliance Alerts Table
CREATE TABLE IF NOT EXISTS compliance_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT,
    message TEXT NOT NULL,
    is_resolved INTEGER DEFAULT 0,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. API Cost Tracking Table
CREATE TABLE IF NOT EXISTS api_costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    service TEXT,
    cost_usd REAL,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Audit Logs Table
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action TEXT,
    resource TEXT,
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- ── Indexes for Performance ──────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_transcriptions_session ON transcriptions(session_id);
CREATE INDEX IF NOT EXISTS idx_transcriptions_status ON transcriptions(status);
CREATE INDEX IF NOT EXISTS idx_api_responses_session ON api_responses(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_sessions_agent ON audit_sessions(agent_name);
CREATE INDEX IF NOT EXISTS idx_audit_sessions_score ON audit_sessions(overall_score);
CREATE INDEX IF NOT EXISTS idx_compliance_alerts_session ON compliance_alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
