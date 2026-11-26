"""Notification payload shapes, thresholds and failure isolation."""
from __future__ import annotations

import json

import pytest

from loci_enterprise.notify import (LogNotifier, Message, WebhookNotifier,
                                   at_least, broadcast, from_doctor,
                                   from_monitor_alerts)


def msg(sev="warning"):
    return Message("t", "body", sev)


def test_severity_ordering():
    assert at_least(msg("critical"), "warning")
    assert not at_least(msg("info"), "warning")
    assert at_least(msg("info"), "info")


def test_log_notifier_records():
    n = LogNotifier()
    n.send(msg())
    assert n.messages[0].title == "t"


@pytest.mark.parametrize("fmt,key", [
    ("slack", "text"), ("feishu", "msg_type"), ("wecom", "msgtype"),
    ("generic", "severity")])
def test_webhook_payload_shapes(fmt, key):
    n = WebhookNotifier("https://example.invalid/hook", fmt=fmt)
    payload = n.payload(msg())
    assert key in payload


def test_unknown_format_is_refused():
    with pytest.raises(ValueError):
        WebhookNotifier("https://x.invalid", fmt="carrier-pigeon")


def test_broadcast_skips_below_threshold():
    log = LogNotifier()
    result = broadcast([log], msg("info"), minimum="warning")
    assert result["skipped"] and log.messages == []


def test_broadcast_isolates_a_failing_channel():
    class Broken:
        name = "broken"

        def send(self, m):
            raise RuntimeError("down")

    log = LogNotifier()
    result = broadcast([Broken(), log], msg("critical"))
    assert result["failed"][0]["notifier"] == "broken"
    assert result["sent"][0]["notifier"] == "log"


def test_from_monitor_alerts_maps_severity():
    msgs = from_monitor_alerts([
        {"severity": "critical", "kind": "stale_truth", "library": "hr",
         "message": "answers got worse"},
        {"severity": "warning", "kind": "zero_hit_spike", "library": "hr",
         "message": "zero hits"}])
    assert [m.severity for m in msgs] == ["critical", "warning"]
    assert "stale_truth" in msgs[0].title


def test_from_doctor_keeps_only_actionable_findings():
    from loci_enterprise.doctor import ERROR, INFO, WARN, Report
    r = Report()
    r.add("x", INFO, "fine")
    r.add("y", WARN, "watch")
    r.add("z", ERROR, "broken")
    msgs = from_doctor(r)
    assert {m.severity for m in msgs} == {"warning", "critical"}


def test_webhook_send_posts_json():
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout):
        captured["body"] = json.loads(req.data)
        captured["ctype"] = req.headers.get("Content-type") or req.headers.get("Content-Type")
        return FakeResponse()

    n = WebhookNotifier("https://example.invalid/h", fmt="wecom", opener=opener)
    assert n.send(msg("critical"))
    assert captured["body"]["msgtype"] == "text"
    assert "CRITICAL" in captured["body"]["text"]["content"]
