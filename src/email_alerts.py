from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def _setting(name: str, secrets=None, default: str = "") -> str:
    """Read configuration from Streamlit secrets, then environment variables."""
    if secrets is not None:
        try:
            value = secrets.get(name)
            if value:
                return str(value)
        except Exception:
            pass
    return os.getenv(name, default)


def send_anomaly_alert(subject: str, body: str, secrets=None) -> tuple[bool, str]:
    """Send an email using SMTP credentials stored outside source code."""
    config = {
        "host": _setting("SMTP_HOST", secrets, "smtp.gmail.com"),
        "port": _setting("SMTP_PORT", secrets, "587"),
        "username": _setting("SMTP_USERNAME", secrets),
        "password": _setting("SMTP_PASSWORD", secrets),
        "sender": _setting("ALERT_FROM", secrets),
        "recipient": _setting("ALERT_TO", secrets),
    }
    if not all(config.values()):
        return False, (
            "Email is not configured. Add SMTP_HOST, SMTP_PORT, SMTP_USERNAME, "
            "SMTP_PASSWORD, ALERT_FROM, and ALERT_TO to .streamlit/secrets.toml. "
            "Never put passwords in source code or commit secrets to GitHub."
        )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["sender"]
    message["To"] = config["recipient"]
    message.set_content(body)
    try:
        with smtplib.SMTP(config["host"], int(config["port"]), timeout=20) as server:
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(message)
        return True, f"Email alert sent to {config['recipient']}."
    except Exception as exc:
        return False, f"Email could not be sent ({type(exc).__name__}). Check SMTP settings and app-password permissions."
