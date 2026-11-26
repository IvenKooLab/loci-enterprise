"""Notifications: a detected problem has to reach a person.

Monitoring that only writes to a log is a diary. This module delivers alerts to
the places teams actually watch, with the payload shape each product expects:

    slack    {"text": "..."}
    feishu   {"msg_type": "text", "content": {"text": "..."}}
    wecom    {"msgtype": "text", "text": {"content": "..."}}
    generic  {"title": ..., "text": ..., "severity": ...}

Design rules:

* **Severity threshold, not everything.** A notifier configured for `critical`
  must not page at 3am because a warning fired.
* **One failing channel must not block the others.** A list of notifiers is
  attempted independently and failures are returned, not raised.
* **Secrets stay out of the payload.** The URL is the credential; nothing else
  is added.
* **Testable without a network**: ``LogNotifier`` keeps the last messages, and
  the webhook notifier accepts an injectable opener.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

SEVERITY_ORDER = {"info": 1, "warning": 2, "critical": 3}


@dataclass
class Message:
    title: str
    text: str
    severity: str = "info"
    source: str = "loci-enterprise"

    def as_dict(self) -> dict:
        return {"title": self.title, "text": self.text,
                "severity": self.severity, "source": self.source}


class Notifier:
    name = "notifier"

    def send(self, message: Message) -> bool:
        raise NotImplementedError


class LogNotifier(Notifier):
    """Keeps messages in memory and writes one line to stderr."""

    name = "log"

    def __init__(self, stream=None):
        self.messages: list[Message] = []
        self.stream = stream

    def send(self, message: Message) -> bool:
        self.messages.append(message)
        if self.stream is not None:
            self.stream.write(f"[{message.severity}] {message.title}: {message.text}\n")
        return True


class WebhookNotifier(Notifier):
    name = "webhook"

    def __init__(self, url: str, *, fmt: str = "slack", timeout: float = 10.0,
                 opener=None):
        if fmt not in ("slack", "feishu", "wecom", "generic"):
            raise ValueError(f"unknown webhook format {fmt!r}")
        self.url = url
        self.fmt = fmt
        self.timeout = timeout
        self._open = opener or urllib.request.urlopen

    def payload(self, message: Message) -> dict:
        text = f"[{message.severity.upper()}] {message.title}\n{message.text}"
        if self.fmt == "slack":
            return {"text": text}
        if self.fmt == "feishu":
            return {"msg_type": "text", "content": {"text": text}}
        if self.fmt == "wecom":
            return {"msgtype": "text", "text": {"content": text}}
        return message.as_dict()

    def send(self, message: Message) -> bool:
        data = json.dumps(self.payload(message), ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data,
            headers={"Content-Type": "application/json; charset=utf-8"})
        with self._open(req, timeout=self.timeout) as resp:
            status = getattr(resp, "status", 200)
        return status is None or 200 <= status < 300


def at_least(message: Message, minimum: str) -> bool:
    return SEVERITY_ORDER.get(message.severity, 0) >= SEVERITY_ORDER.get(minimum, 0)


def broadcast(notifiers: list[Notifier], message: Message, *,
              minimum: str = "warning") -> dict:
    """Send to every notifier independently; report per-channel outcomes."""
    if not at_least(message, minimum):
        return {"sent": [], "failed": [], "skipped": "below threshold"}
    sent, failed = [], []
    for n in notifiers:
        try:
            ok = n.send(message)
        except Exception as exc:                  # noqa: BLE001 - reported per channel
            failed.append({"notifier": n.name, "error": str(exc)})
            continue
        (sent if ok else failed).append({"notifier": n.name})
    return {"sent": sent, "failed": failed}


def from_monitor_alerts(alerts: list[dict], *, source: str = "retrieval-monitor") -> list[Message]:
    """Convert monitor.Alert dicts into messages, preserving severity."""
    out = []
    for alert in alerts:
        severity = "critical" if alert.get("severity") == "critical" else "warning"
        lib = alert.get("library", "?")
        out.append(Message(title=f"[{lib}] {alert.get('kind', 'alert')}",
                           text=alert.get("message", ""), severity=severity,
                           source=source))
    return out


def from_doctor(report) -> list[Message]:
    """Convert doctor findings into messages (warnings and errors only)."""
    out = []
    for f in getattr(report, "findings", []):
        if f.severity not in ("warn", "error"):
            continue
        out.append(Message(title=f"doctor: {f.check}",
                           text=f.message,
                           severity="critical" if f.severity == "error" else "warning",
                           source="doctor"))
    return out
