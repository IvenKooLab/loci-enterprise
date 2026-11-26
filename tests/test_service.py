"""HTTP surface: auth, authorisation, and the round-trip of each route."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from loci_enterprise.access import AccessControl
from loci_enterprise.service import serve
from loci_enterprise.store import Store


@pytest.fixture
def api():
    store = Store(":memory:")
    ac = AccessControl(store)
    keys = {
        "admin": ac.issue_key("t1", "admin", role="admin",
                              scopes=["read", "write", "admin", "delete"]).token,
        "editor": ac.issue_key("t1", "editor", role="editor",
                               scopes=["read", "write"]).token,
        "viewer": ac.issue_key("t1", "viewer", role="viewer", scopes=["read"]).token,
    }
    server = serve(store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, keys, store
    server.shutdown()


def get(base, path, token=None):
    req = urllib.request.Request(base + path)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def post(base, path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_health_needs_no_key(api):
    base, _, _ = api
    assert get(base, "/health") == (200, {"status": "ok", "version": "0.1.0"})


def test_missing_key_is_401(api):
    base, _, _ = api
    status, body = get(base, "/stats")
    assert status == 401


def test_unknown_key_is_401(api):
    base, _, _ = api
    assert get(base, "/stats", "lek_dead_beef")[0] == 401


def test_viewer_can_read_stats(api):
    base, keys, _ = api
    status, body = get(base, "/stats", keys["viewer"])
    assert status == 200 and "documents" in body


def test_viewer_cannot_write(api):
    base, keys, _ = api
    status, _ = post(base, "/documents", {"path": "a.md", "body": "x"},
                     keys["viewer"])
    assert status == 403


def test_editor_can_create_and_search(api):
    base, keys, _ = api
    status, body = post(base, "/documents",
                        {"path": "policy.md", "title": "差旅",
                         "body": "# 差旅\n\n住宿上限 800 元", "library": "hr"},
                        keys["editor"])
    assert status == 200 and body["version"] == 1
    status, found = post(base, "/search", {"query": "住宿上限", "library": "hr"},
                         keys["viewer"])
    assert status == 200 and found["hits"][0]["path"] == "policy.md"


def test_supersede_hides_the_old_document(api):
    base, keys, _ = api
    post(base, "/documents", {"path": "v1.md", "body": "# P\n\n500", "library": "hr"},
         keys["editor"])
    post(base, "/documents", {"path": "v2.md", "body": "# P\n\n800", "library": "hr"},
         keys["editor"])
    status, _ = post(base, "/supersede",
                     {"old": "v1.md", "new": "v2.md", "library": "hr"},
                     keys["editor"])
    assert status == 200
    _, found = post(base, "/search", {"query": "# P", "library": "hr"}, keys["viewer"])
    assert [h["path"] for h in found["hits"]] == ["v2.md"]


def test_document_read_and_404(api):
    base, keys, _ = api
    post(base, "/documents", {"path": "a.md", "body": "hello", "library": "lib"},
         keys["editor"])
    status, body = get(base, "/document?path=a.md&library=lib", keys["viewer"])
    assert status == 200 and body["body"] == "hello"
    assert get(base, "/document?path=missing.md&library=lib", keys["viewer"])[0] == 404


def test_memory_round_trip(api):
    base, keys, _ = api
    post(base, "/memory", {"subject_id": "u1", "key": "budget", "value": "500"},
         keys["editor"])
    post(base, "/memory", {"subject_id": "u1", "key": "budget", "value": "800"},
         keys["editor"])
    status, body = get(base, "/memory?subject_id=u1", keys["viewer"])
    assert status == 200 and [i["value"] for i in body["items"]] == ["800"]


def test_governance_lists_candidates(api):
    base, keys, _ = api
    post(base, "/documents", {"path": "old.md", "title": "P", "body": "旧",
                              "library": "hr"}, keys["editor"])
    post(base, "/documents", {"path": "new.md", "title": "P", "body": "新",
                              "library": "hr"}, keys["editor"])
    post(base, "/supersede", {"old": "old.md", "new": "new.md", "library": "hr"},
         keys["editor"])
    status, body = get(base, "/governance?library=hr", keys["viewer"])
    assert status == 200
    assert any(c["path"] == "old.md" for c in body["retirement"])


def test_audit_is_admin_only_and_records_actions(api):
    base, keys, _ = api
    post(base, "/documents", {"path": "a.md", "body": "x", "library": "lib"},
         keys["editor"])
    assert get(base, "/audit", keys["editor"])[0] == 403
    status, body = get(base, "/audit", keys["admin"])
    assert status == 200 and any("create" in row["action"] for row in body["audit"])


def test_invalid_json_body_is_400(api):
    base, keys, _ = api
    req = urllib.request.Request(base + "/search", data=b"{oops",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {keys['viewer']}"})
    try:
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_unknown_route_is_404(api):
    base, keys, _ = api
    assert get(base, "/nope", keys["viewer"])[0] == 404


def test_document_scoped_to_a_role_is_not_visible_to_viewers(api):
    base, keys, _ = api
    post(base, "/documents", {"path": "admin.md", "title": "Admin",
                              "body": "# Admin\n\nsecret", "library": "lib",
                              "acl": ["role:admin"]}, keys["admin"])
    _, viewer_hits = post(base, "/search", {"query": "secret", "library": "lib"},
                          keys["viewer"])
    assert viewer_hits["hits"] == []
    _, admin_hits = post(base, "/search", {"query": "secret", "library": "lib"},
                         keys["admin"])
    assert admin_hits["hits"][0]["path"] == "admin.md"
