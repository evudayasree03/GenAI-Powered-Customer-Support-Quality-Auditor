from __future__ import annotations

import streamlit as st

from src.db.utils import sqlite_healthcheck
from src.db import get_db


class DBAdminPanel:
    def render(self) -> None:
        st.title("Database Admin")
        st.json(sqlite_healthcheck(get_db()))
