"""
SamiX Database Utilities
Thread Safety Enabled - Optimized for Render & Streamlit Cloud
  
"""
import os
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.db.models import User # Ensure this exists in models.py

def get_db_path():
    """Determines the persistent path for the SQLite database."""
    # Check for Render persistent disk first
    if os.path.exists("/data"):
        return "/data/samix.db"
    
    # Local fallback logic: Move up 3 levels from src/db/utils.py to root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    return os.path.join(data_dir, "samix.db")

def get_db_engine():
    """Creates the SQLAlchemy engine with thread safety for SQLite."""
    db_path = f"sqlite:///{get_db_path()}"
    return create_engine(
        db_path, 
        connect_args={"check_same_thread": False} 
    )

def get_db() -> Session:
    """Helper to get a standalone database session."""
    engine = get_db_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

# --- AUTH HELPER FUNCTIONS (Required by AuthManager) ---

def get_user_by_email(session: Session, email: str):
    """Fetch a single user record by email."""
    return session.query(User).filter(User.email == email).first()

def create_user(session: Session, email: str, hashed_pw: str, name: str, role: str):
    """Inserts a new user into the database."""
    new_user = User(
        email=email,
        hashed_password=hashed_pw,
        full_name=name,
        role=role
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user

def sqlite_healthcheck() -> dict:
    """Returns database metadata for the UI status indicators."""
    try:
        db_file = get_db_path()
        size_kb = os.path.getsize(db_file) // 1024 if os.path.exists(db_file) else 0
        
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [{"name": row[0]} for row in cursor.fetchall()]
        conn.close()

        return {
            "status": "healthy",
            "path": db_file,
            "size_kb": size_kb,
            "tables": tables
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
