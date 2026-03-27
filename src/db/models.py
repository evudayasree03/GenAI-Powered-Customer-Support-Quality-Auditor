"""
SamiX Database Models
"""
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from src.db.utils import get_db_engine

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String, default="agent") # 'admin' or 'agent'
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AuditSession(Base):
    __tablename__ = "audit_sessions"
    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String)
    score = Column(Float)
    sentiment = Column(String)
    transcript = Column(Text)
    duration = Column(Integer)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

def init_tables():
    """Creates tables if they don't exist."""
    engine = get_db_engine()
    Base.metadata.create_all(bind=engine)
