"""Tenant export/import round-trip."""
from __future__ import annotations

import json

import pytest

from loci_enterprise.lifecycle import supersede
from loci_enterprise.memory import MemoryStore
from loci_enterprise.portability import (FORMAT, ImportError_, export_tenant,
                                         import_tenant, load_export, save_export)
from loci_enterprise.store import Document, Store
from loci_enterprise.wiki import WikiBuilder


def build():
    s = Store(":memory:")
    s.put_document(Document(id="", tenant_id="src", library="hr", path="v1.md",
                            title="政策", body="# 政策\n\n旧", acl=["public"],
                            written_by="t"))
    s.put_document(Document(id="", tenant_id="src", library="hr", path="v2.md",
                            title="政策", body="# 政策\n\n新", acl=["public"],
                            written_by="t"))
    supersede(s, "src", "v1.md", "v2.md", library="hr", actor="t")
    MemoryStore(s).remember("src", "u1", "budget", "800")
    WikiBuilder(s).build("src", "hr")
    return s


def test_export_has_the_documented_format():
    data = export_tenant(build(), "src")
    assert data["format"] == FORMAT
    assert len(data["documents"]) == 2
    assert data["document_versions"]


def test_round_trip_into_a_fresh_store_preserves_documents_and_status():
    data = export_tenant(build(), "src")
    dst = Store(":memory:")
    report = import_tenant(dst, "dst", data)
    assert report.documents == 2
    docs = {d.path: d for d in dst.list_documents("dst")}
    assert docs["v1.md"].status == "superseded"
    assert docs["v2.md"].status == "active"
    assert "新" in docs["v2.md"].body


def test_round_trip_preserves_memory_and_wiki():
    data = export_tenant(build(), "src")
    dst = Store(":memory:")
    report = import_tenant(dst, "dst", data)
    assert report.memories == 1
    from loci_enterprise.memory import MemoryStore as MS
    assert MS(dst).recall("dst", "u1", "budget")[0].value == "800"
    from loci_enterprise.wiki import WikiStore
    assert WikiStore(dst).page("dst", "hr", "政策") is not None


def test_save_and_load_file(tmp_path):
    path = tmp_path / "tenant.json"
    info = save_export(build(), "src", path)
    assert info["documents"] == 2
    assert load_export(path)["tenant_id"] == "src"


def test_importing_into_the_same_tenant_is_additive_not_destructive():
    s = build()
    data = export_tenant(s, "src")
    before = len(s.list_documents("src"))
    import_tenant(s, "src", data)
    assert len(s.list_documents("src")) == before


def test_wrong_format_is_refused():
    with pytest.raises(ImportError_):
        import_tenant(Store(":memory:"), "t", {"format": "something-else"})


def test_export_is_json_serialisable():
    data = export_tenant(build(), "src")
    json.dumps(data, ensure_ascii=False)


def test_import_writes_an_audit_entry():
    dst = Store(":memory:")
    import_tenant(dst, "dst", export_tenant(build(), "src"), actor="ops")
    actions = [a["action"] for a in dst.audit_tail("dst", 20)]
    assert "tenant:import" in actions
