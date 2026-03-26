"""
SamiX alert and notification engine.
"""
from __future__ import annotations

import asyncio
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import streamlit as st


def _safe_console(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="replace").decode("ascii"))


class AlertEngine:
    """Dispatches UI toasts and optional SMTP alerts."""

    SCORE_THRESHOLD: float = 60.0

    def __init__(self) -> None:
        self._email_cfg = self._load_email_cfg()

    def _load_email_cfg(self) -> Optional[dict]:
        try:
            cfg = st.secrets.get("email", {})
            if cfg and "REPLACE" not in cfg.get("sender_address", "REPLACE"):
                return dict(cfg)
        except Exception:
            pass
        return None

    async def check_and_fire(
        self,
        filename: str,
        agent_name: str,
        final_score: float,
        violations: int,
        auto_fail: bool,
        auto_fail_reason: str,
        recipient_email: str = "",
    ) -> list[str]:
        triggered: list[str] = []
        violation_count = violations if isinstance(violations, int) else len(violations) if violations else 0

        if auto_fail:
            msg = f"AUTO-FAIL - {filename} | {agent_name} | Reason: {auto_fail_reason}"
            self._toast(msg, icon="🚨")
            if recipient_email:
                await self._email(recipient_email, "SamiX AUTO-FAIL Alert", msg)
            triggered.append(msg)

        if final_score < self.SCORE_THRESHOLD:
            msg = (
                f"LOW SCORE - {filename} | {agent_name} | "
                f"Score: {final_score:.0f}/100 (threshold {self.SCORE_THRESHOLD:.0f})"
            )
            self._toast(msg, icon="⚠️")
            if recipient_email:
                await self._email(recipient_email, "SamiX Low Score Alert", msg)
            triggered.append(msg)

        if violation_count > 2:
            msg = (
                f"CRITICAL VIOLATIONS - {filename} | {agent_name} | "
                f"{violation_count} violations detected"
            )
            self._toast(msg, icon="🔴")
            if recipient_email:
                await self._email(recipient_email, "SamiX Critical Violation", msg)
            triggered.append(msg)

        return triggered

    async def send_custom(self, to: str, subject: str, body: str) -> bool:
        success = await self._email(to, subject, body)
        if success:
            st.toast(f"Email sent to {to}", icon="✅")
        else:
            st.toast(f"Email queued (mock) for {to}", icon="📬")
        return success

    @staticmethod
    def _toast(message: str, icon: str = "⚠️") -> None:
        st.toast(message, icon=icon)

    async def _email(self, recipient: str, subject: str, body: str) -> bool:
        if not self._email_cfg or not recipient:
            self._mock_log(recipient, subject, body)
            return False
        return await asyncio.to_thread(self._sync_email, recipient, subject, body)

    def _sync_email(self, recipient: str, subject: str, body: str) -> bool:
        try:
            host = self._email_cfg["smtp_host"]
            port = int(self._email_cfg.get("smtp_port", 587))
            sender = self._email_cfg["sender_address"]
            password = self._email_cfg["sender_password"]

            msg = MIMEMultipart("alternative")
            msg["From"] = sender
            msg["To"] = recipient
            msg["Subject"] = f"[SamiX] {subject}"

            html_body = f"""
            <html><body style="font-family:Arial,sans-serif;background:#0F172A;color:#E2E8F0;padding:24px;">
            <h2 style="color:#60A5FA;">SamiX Quality Auditor Alert</h2>
            <p>{body.replace(chr(10), '<br>')}</p>
            <hr style="border-color:#334155;"/>
            <p style="color:#94A3B8;font-size:12px;">
              Sent by SamiX | {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </p>
            </body></html>
            """
            msg.attach(MIMEText(body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            context = ssl.create_default_context()
            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(sender, password)
                server.sendmail(sender, recipient, msg.as_string())
            return True
        except Exception as exc:
            self._mock_log(recipient, subject, f"{body}\n\n[SMTP Error: {exc}]")
            return False

    @staticmethod
    def _mock_log(to: str, subject: str, body: str) -> None:
        _safe_console(
            "\n"
            + "-" * 60
            + "\n[SamiX MOCK EMAIL]\n"
            + f"To:      {to or '(no recipient)'}\n"
            + f"Subject: {subject}\n"
            + f"Body:\n{body}\n"
            + "-" * 60
            + "\n"
        )
