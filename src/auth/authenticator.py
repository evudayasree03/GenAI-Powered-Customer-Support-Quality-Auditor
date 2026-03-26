"""
SamiX Authentication Manager

This module provides a local, file-based authentication system using YAML for storage
and bcrypt for secure password hashing. It handles user login, registration, and
session state management for the Streamlit application.
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import bcrypt

from src.db import get_db
from src.utils.validators import is_valid_email

class AuthManager:
    """
    Manages user credentials and session authentication status.
    
    Credentials are stored in `data/auth/users.yaml`. The manager supports
    email-based usernames and secure password verification.
    """

    _USERS_YAML: str  = "data/auth/users.yaml"

    def __init__(self) -> None:
        """ Initialize the manager and load the user database. """
        self._db = get_db()
        self._load_config()

    def _load_config(self) -> None:
        """
        Loads the YAML configuration from disk.
        If the file doesn't exist, a default 'admin' account is created.
        """
        yaml_path = Path(self._USERS_YAML)
        
        if not yaml_path.exists():
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            # Default admin account info (Password: 'admin')
            default_config = {
                "credentials": {
                    "admin@samix.ai": {
                        "name": "Admin",
                        "password": "$2b$12$R.S/XfIay1V/2n3bF/O51uA3Vw0sSBYh4Cih.qInMFRc/VXZiJgGW" 
                    }
                }
            }
            with open(yaml_path, "w", encoding="utf-8") as file:
                yaml.dump(default_config, file, default_flow_style=False)

        # Parse the YAML file using a SafeLoader.
        with open(yaml_path, "r", encoding="utf-8") as file:
            self._config = yaml.load(file, Loader=SafeLoader)
            if "credentials" not in self._config:
                self._config["credentials"] = {}
        self.migrate_yaml_to_sqlite()

    def save_config(self) -> None:
        """ Persists the current in-memory user database back to the YAML file. """
        with open(self._USERS_YAML, "w", encoding="utf-8") as file:
            yaml.dump(self._config, file, default_flow_style=False)

    def migrate_yaml_to_sqlite(self) -> None:
        """ Ensures legacy YAML credentials are mirrored into SQLite. """
        creds = self._config.get("credentials", {})
        for email, user in creds.items():
            self._db.upsert_user(
                email=email,
                name=user.get("name", "User"),
                password_hash=user.get("password", ""),
                role=user.get("role", "admin" if email == "admin@samix.ai" else "agent"),
                is_active=user.get("is_active", True),
            )

    def login(self, email: str, password: str) -> bool:
        """
        Verifies user credentials.
        Updates Streamlit session state on successful authentication.
        """
        email = email.lower().strip()
        db_user = self._db.get_user_by_email(email)
        if db_user:
            hashed_pw = db_user.get("password_hash", "")
            # Securely compare the provided password with the stored hash.
            if self._check_password(password, hashed_pw):
                st.session_state["authentication_status"] = True
                st.session_state["name"] = db_user.get("name", "User")
                st.session_state["email"] = email
                st.session_state["role"] = db_user.get("role", "agent")
                self._db.update_last_login(email)
                return True
        else:
            creds = self._config.get("credentials", {})
            if email in creds:
                user = creds[email]
                hashed_pw = user.get("password", "")
                if self._check_password(password, hashed_pw):
                    st.session_state["authentication_status"] = True
                    st.session_state["name"] = user.get("name", "User")
                    st.session_state["email"] = email
                    st.session_state["role"] = user.get("role", "agent")
                    self._db.upsert_user(
                        email=email,
                        name=user.get("name", "User"),
                        password_hash=hashed_pw,
                        role=user.get("role", "agent"),
                    )
                    self._db.update_last_login(email)
                    return True
                
        st.session_state["authentication_status"] = False
        return False

    def register(self, email: str, name: str, password: str) -> bool:
        """
        Registers a new user with a hashed password.
        Returns False if the email is already in use.
        """
        email = email.lower().strip()
        if not is_valid_email(email):
            return False
        creds = self._config.setdefault("credentials", {})
        
        if email in creds or self._db.get_user_by_email(email):
            return False # Prevent duplicate registration.
            
        hashed_pw = self._hash_password(password)
        creds[email] = {
            "name": name.strip() or "User",
            "password": hashed_pw
        }
        self.save_config()
        self._db.upsert_user(
            email=email,
            name=name.strip() or "User",
            password_hash=hashed_pw,
            role="agent",
        )
        return True

    def _hash_password(self, raw: str) -> str:
        """ Generates a secure bcrypt hash for a raw password string. """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(raw.encode('utf-8'), salt).decode('utf-8')

    def _check_password(self, raw: str, hashed: str) -> bool:
        """ Validates a raw password against a bcrypt hash. """
        try:
            return bcrypt.checkpw(raw.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False

    def is_authenticated(self) -> bool:
        """ Returns True if the current session is successfully authenticated. """
        return st.session_state.get("authentication_status") is True

    def is_failed(self) -> bool:
        """ Returns True if the last login attempt failed. """
        return st.session_state.get("authentication_status") is False

    def is_pending(self) -> bool:
        """ Returns True if no authentication attempt has been made yet. """
        return st.session_state.get("authentication_status") is None

    def render_logout(self) -> None:
        """ Clears session state and triggers a page refresh to log the user out. """
        if st.sidebar.button("Sign Out", key="sidebar_logout_btn"):
            st.session_state["authentication_status"] = None
            st.session_state["name"] = None
            st.session_state["email"] = None
            st.session_state["role"] = None
            st.rerun()

    @property
    def current_user_name(self) -> str:
        """ Returns the display name of the logged-in user. """
        return st.session_state.get("name", "User")

    @property
    def current_user_email(self) -> str:
        """ Returns the email address of the logged-in user. """
        return st.session_state.get("email", "")

    @property
    def current_user_role(self) -> str:
        return st.session_state.get("role", "agent")
