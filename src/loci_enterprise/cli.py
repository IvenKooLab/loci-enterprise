"""`le` — the operator surface for the governance layer.

Deliberately small. Every command answers one question an on-call engineer
actually asks at 2am:

    le scan      what in this library can no longer be trusted?
    le retire    take it out (dry-run by default)
    le supersede declare that A was replaced by B
    le chain     what is in force for this topic, and since when
    le eval      re-score the pinned question set
    le monitor   has quality moved since the baseline?
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import access as access_mod
from . import feedback as feedback_mod
from . import agent as agent_mod
from . import mcp as mcp_mod
from . import scheduler as scheduler_mod
from . import storage as storage_mod
from . import service as service_mod
from . import backends as backends_mod
from . import connectors as connectors_mod
from . import doctor as doctor_mod
from . import portability as portability_mod
from . import evalmetrics, governance, lifecycle, memory as memory_mod, monitor as monitor_mod
from . import providers as providers_mod
from . import queue as queue_mod
from . import wiki as wiki_mod
from .retrieval import RetrievalConfig, Retriever
from .store import Document, Store


def _store(args) -> Store:
    return Store(args.db)


def _memory(args) -> memory_mod.MemoryStore:
    return memory_mod.MemoryStore(_store(args))


def cmd_memory(args) -> int:
    mem = _memory(args)
    cmd = args.memory_cmd
    if cmd == "remember":
        res = mem.remember(args.tenant, args.subject, args.key, args.value,
                           source=args.source, ttl_days=args.ttl_days,
                           actor=args.actor, requires_review=args.review)
        print(json.dumps({"status": res.status, "id": res.item.id if res.item else None,
                          "detail": res.detail}, ensure_ascii=False))
        return 0
    if cmd == "recall":
        for item in mem.recall(args.tenant, args.subject, args.key or None):
            print(f"{item.key:<18} {item.value:<40} {item.source:<10} {item.updated_at}")
        return 0
    if cmd == "pending":
        for item in mem.pending(args.tenant):
            print(f"{item.id}  {item.subject_id:<12} {item.key:<16} {item.value:<38} {item.source}")
        return 0
    if cmd == "approve":
        item = mem.approve(args.memory_id, actor=args.actor)
        print(f"approved {args.memory_id}" if item else "not pending")
        return 0
    if cmd == "reject":
        print("rejected" if mem.reject(args.memory_id, actor=args.actor,
                                       reason=args.reason) else "not pending")
        return 0
    if cmd == "history":
        for item in mem.history(args.tenant, args.subject, args.key):
            print(f"{item.status:<11} {item.value:<40} {item.created_at} "
                  f"{'(by ' + str(item.superseded_by)[:8] + ')' if item.superseded_by else ''}")
        return 0
    if cmd == "forget":
        print(json.dumps(mem.forget(args.tenant, args.subject, key=args.key or None,
                                    actor=args.actor), ensure_ascii=False))
        return 0
    if cmd == "capacity":
        print(json.dumps(mem.capacity(args.tenant, args.subject), ensure_ascii=False))
        return 0
    raise SystemExit(f"unknown memory command {cmd}")


def _access(args) -> access_mod.AccessControl:
    return access_mod.AccessControl(_store(args))


def cmd_member(args) -> int:
    ac = _access(args)
    if args.member_cmd == "grant":
        ac.grant(args.tenant, args.user, args.role, actor=args.actor)
        print(f"granted {args.user} {args.role} in {args.tenant}")
    elif args.member_cmd == "revoke":
        print("revoked" if ac.revoke(args.tenant, args.user, actor=args.actor)
              else "no such member")
    else:
        for row in ac.members(args.tenant):
            print(f"{row['user_id']:<24} {row['role']:<8} since {row['created_at']}")
    return 0


def cmd_key(args) -> int:
    ac = _access(args)
    if args.key_cmd == "issue":
        issued = ac.issue_key(args.tenant, args.name, role=args.role,
                              scopes=[s for s in args.scopes.split(",") if s],
                              expires_in_days=args.expires_in_days, actor=args.actor)
        print("Store this token now — it is not shown again:")
        print(issued.token)
    elif args.key_cmd == "revoke":
        print("revoked" if ac.revoke_key(args.tenant, args.key_id, actor=args.actor)
              else "no such key")
    else:
        for row in ac.keys(args.tenant):
            state = "revoked" if row["revoked_at"] else "active"
            print(f"{row['id']:<18} {row['name']:<16} {row['role']:<8} "
                  f"{row['scopes']:<14} {state}")
    return 0


def cmd_perm(args) -> int:
    matrix = _access(args).matrix()
    roles = list(matrix)
    actions = list(next(iter(matrix.values())))
    print("role".ljust(10) + "".join(a.ljust(9) for a in actions))
    for role in roles:
        print(role.ljust(10) + "".join(
            ("yes" if matrix[role][a] else "-").ljust(9) for a in actions))
    return 0


def cmd_put(args) -> int:
    store = _store(args)
    doc = Document(id="", tenant_id=args.tenant, library=args.library, path=args.path,
                   title=args.title or Path(args.path).stem,
                   body=Path(args.file).read_text(encoding="utf-8") if args.file
                   else (args.body or ""),
                   effective_from=args.effective, owner=args.owner,
                   authority=args.authority, written_by=args.actor,
                   acl=args.acl.split(","))
    gov = governance.Governor(store)
    res = gov.write(doc, on_conflict=args.on_conflict)
    print(json.dumps({"status": res.status, "version": res.document.version
                      if res.document else None, "detail": res.detail}))
    return 0


def cmd_supersede(args) -> int:
    store = _store(args)
    old, new = lifecycle.supersede(store, args.tenant, args.old, args.new,
                                   library=args.library, actor=args.actor,
                                   note=args.note)
    print(json.dumps({"superseded": old.path, "by": new.path}))
    return 0


def cmd_chain(args) -> int:
    store = _store(args)
    items = lifecycle.chain(store, args.tenant, args.library, args.path)
    cur = lifecycle.current(store, args.tenant, args.library, args.path)
    for item in items:
        mark = "*" if cur and item["path"] == cur["path"] else " "
        print(f"{mark} {item['status']:<11} {item['path']:<32} "
              f"{item.get('effective_from') or '-':<12} {item['reason']}")
    return 0


def cmd_export(args) -> int:
    info = portability_mod.save_export(_store(args), args.tenant, args.out)
    print(json.dumps(info, ensure_ascii=False))
    return 0


def cmd_import(args) -> int:
    report = portability_mod.import_tenant(
        _store(args), args.to_tenant, portability_mod.load_export(args.file),
        actor=args.actor)
    print(json.dumps(report.as_dict(), ensure_ascii=False))
    return 0


def cmd_doctor(args) -> int:
    report = doctor_mod.run(_store(args), tenant_id=args.tenant,
                            stale_connector_days=args.stale_days,
                            stuck_job_minutes=args.stuck_minutes)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(doctor_mod.render(report))
    return report.exit_code()


def cmd_blob(args) -> int:
    store_dir = Path(args.blob_root)
    blobs = storage_mod.LocalBlobStore(store_dir)
    if args.blob_cmd == "put":
        blob, _ = storage_mod.store_file(blobs, args.file)
        print(json.dumps({"digest": blob.digest, "size": blob.size,
                          "media_type": blob.media_type}, ensure_ascii=False))
        return 0
    if args.blob_cmd == "get":
        data = blobs.get(args.digest)
        out = Path(args.out) if args.out else Path(args.digest[:12] + ".bin")
        out.write_bytes(data)
        print(str(out))
        return 0
    if args.blob_cmd == "stats":
        print(json.dumps(blobs.stats(), ensure_ascii=False))
        return 0
    raise SystemExit(f"unknown blob command {args.blob_cmd}")


def cmd_schedule(args) -> int:
    store = _store(args)
    sc = scheduler_mod.Schedules(store)
    cmd = args.schedule_cmd
    if cmd == "add":
        payload = json.loads(args.payload) if args.payload else {}
        sc.add(args.tenant, args.name, args.kind,
               interval_seconds=args.every, payload=payload,
               first_run_in=args.first_run_in)
        print(f"scheduled {args.name}: {args.kind} every {args.every:.0f}s")
        return 0
    if cmd == "list":
        for row in sc.list(args.tenant):
            state = "on" if row.enabled else "off"
            nxt = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(row.next_run))
            print(f"{row.name:<20} {row.kind:<16} every {row.interval_s:>7.0f}s  "
                  f"{state}  next {nxt}")
        return 0
    if cmd == "remove":
        print("removed" if sc.remove(args.tenant, args.name) else "no such schedule")
        return 0
    if cmd in ("enable", "disable"):
        ok = sc.set_enabled(args.tenant, args.name, cmd == "enable")
        print(f"{cmd}d" if ok else "no such schedule")
        return 0
    if cmd == "tick":
        fired = sc.tick(queue_mod.Queue(store), limit=args.limit)
        for row in fired:
            print(f"{row['schedule']:<20} -> job {row['job_id'][:8]} ({row['kind']})")
        if not fired:
            print("nothing due")
        return 0
    raise SystemExit(f"unknown schedule command {cmd}")


def cmd_serve(args) -> int:
    """Run the HTTP API until interrupted."""
    store = _store(args)
    server = service_mod.serve(store, args.host, args.port,
                               allow_anonymous=args.allow_anonymous)
    print(f"loci-enterprise listening on http://{args.host}:"
          f"{server.server_address[1]}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


def cmd_mcp(args) -> int:
    """Run the MCP server over stdio (one JSON-RPC message per line)."""
    mcp_mod.MCPServer(_store(args), tenant_id=args.tenant,
                      principals=[args.principal]).serve()
    return 0


def cmd_agent(args) -> int:
    """Answer a question with a tool-using loop over the knowledge base."""
    store = _store(args)
    retriever = Retriever(store, config=RetrievalConfig(top_k=5))

    def search(query, principals, library):
        hits = retriever.search(query, tenant_id=args.tenant,
                                principals=principals, library=library)
        return [{"path": h.path, "text": h.text} for h in hits]

    def get_doc(path):
        doc = store.get_document(args.tenant, args.library, path)
        return doc.body if doc else f"(no document at {path})"

    registry = agent_mod.knowledge_tools(search, get_doc)
    try:
        provider = providers_mod.deepseek(model=args.model or None)
    except providers_mod.ProviderError as exc:
        print(f"{exc}. Set DEEPSEEK_API_KEY (or use --model with an "
              f"OpenAI-compatible endpoint via the library).", file=sys.stderr)
        return 2
    tracer = None
    if args.trace:
        from .tracing import Tracer
        tracer = Tracer(path=args.trace)
        tracer.start_trace("agent", question=args.question[:80])
    result = agent_mod.Agent(provider, registry, max_steps=args.max_steps,
                             tracer=tracer).run(
        args.question,
        agent_mod.AgentContext(tenant_id=args.tenant, principals=[args.principal],
                               library=args.library, read_only=True))
    print(result.answer or "(no answer)")
    if result.citations:
        print("\nsources: " + ", ".join(result.citations))
    if args.verbose:
        for step in result.steps:
            print(f"  {step.kind:<7} {step.tool:<18} {step.ms:7.1f} ms "
                  f"{step.error or step.observation[:80]}")
    if tracer:
        tracer.end_trace()
    return 0 if result.stopped == "answer" else 1


def cmd_backend(args) -> int:
    """Rebuild or query a pluggable retrieval backend from the document store."""
    be = (backends_mod.MemoryBackend() if args.kind == "memory"
          else backends_mod.SQLiteBackend(args.path))
    if args.backend_cmd == "rebuild":
        from .retrieval import HashEmbedder
        store = _store(args)
        embedder = HashEmbedder()
        records = []
        for chunk in store.visible_chunks(args.tenant, [args.principal]):
            records.append(backends_mod.IndexRecord(
                chunk_id=chunk["id"], doc_id=chunk["doc_id"],
                tenant_id=args.tenant, library=chunk["library"],
                text=chunk["text"], vector=embedder.embed(chunk["text"]),
                acl=[args.principal], status=chunk["status"]))
        be.upsert(records)
        print(json.dumps(be.stats(args.tenant), ensure_ascii=False))
        return 0
    if args.backend_cmd == "search":
        hits = be.search(backends_mod.Query(
            text=args.query, tenant_id=args.tenant, principals=[args.principal],
            vector=None, top_k=args.top_k))
        for h in hits:
            print(f"{h.score:.5f}  {h.chunk_id}  ({h.doc_id})")
        return 0
    print(json.dumps(be.stats(args.tenant), ensure_ascii=False))
    return 0


def _handlers(store, tenant):
    """Built-in job kinds, so the queue is useful without a daemon."""
    def sync_local(job):
        conn = connectors_mod.LocalDirConnector(job.payload["path"],
                                               name=job.payload.get("name", "local"))
        rep = connectors_mod.SyncEngine(store).sync(
            conn, tenant_id=job.tenant_id, library=job.payload.get("library", "default"),
            actor="queue")
        return json.dumps(rep.as_dict(), ensure_ascii=False)

    def wiki_build(job):
        rep = wiki_mod.WikiBuilder(store).build(
            job.tenant_id, job.payload.get("library", "default"), note="queued build")
        return json.dumps(rep.as_dict(), ensure_ascii=False)

    def memory_expire(job):
        return f"expired {memory_mod.MemoryStore(store).expire_due(tenant_id=job.tenant_id)}"

    return {"sync.local": sync_local, "wiki.build": wiki_build,
            "memory.expire": memory_expire}


def cmd_queue(args) -> int:
    store = _store(args)
    qq = queue_mod.Queue(store)
    cmd = args.queue_cmd
    if cmd == "enqueue":
        payload = json.loads(args.payload) if args.payload else {}
        job = qq.enqueue(args.tenant, args.kind, payload, priority=args.priority)
        print(job.id)
        return 0
    if cmd == "run":
        done = qq.drain(_handlers(store, args.tenant), tenant_id=args.tenant,
                        limit=args.limit)
        if not done:
            print("queue empty")
        for job in done:
            print(f"{job.kind:<14} {job.status:<10} {job.result[:60] or job.error[:60]}")
        return 0 if all(j.status == queue_mod.SUCCEEDED for j in done) else 1
    if cmd == "stats":
        print(json.dumps(qq.stats(args.tenant), ensure_ascii=False))
        return 0
    if cmd == "list":
        for job in qq.list(args.tenant, status=args.status or None):
            print(f"{job.id[:8]}  {job.kind:<14} {job.status:<10} "
                  f"try {job.attempts}/{job.max_attempts}  {job.error[:40]}")
        return 0
    if cmd == "cancel":
        print("cancelled" if qq.cancel(args.tenant, args.job_id) else "not cancellable")
        return 0
    if cmd == "reclaim":
        print(f"reclaimed {qq.reclaim(older_than_seconds=args.older_than)} job(s)")
        return 0
    raise SystemExit(f"unknown queue command {cmd}")


def cmd_wiki(args) -> int:
    store = _store(args)
    w = wiki_mod.WikiStore(store)
    cmd = args.wiki_cmd
    if cmd == "build":
        report = wiki_mod.WikiBuilder(store, w).build(args.tenant, args.library)
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0
    if cmd == "pages":
        for page in w.pages(args.tenant, args.library):
            print(f"{page.slug:<30} v{page.version:<3} {len(page.citations)} src  "
                  f"{page.title}")
        return 0
    if cmd == "show":
        page = w.page(args.tenant, args.library, args.slug)
        if page is None:
            print(f"no such page: {args.slug}", file=sys.stderr)
            return 2
        print(page.body)
        print(f"\n-- v{page.version} · sources: {', '.join(page.citations) or '-'}")
        return 0
    if cmd == "history":
        for row in w.revisions(args.tenant, args.library, args.slug):
            print(f"v{row['version']:<3} {row['created_at']}  {row['chars']:>6} chars  "
                  f"{row['note']}")
        return 0
    if cmd == "rollback":
        page = w.rollback(args.tenant, args.library, args.slug, args.version,
                          actor=args.actor)
        print(f"restored {args.slug} from v{args.version} as v{page.version}")
        return 0
    if cmd == "links":
        dead = w.dead_links(args.tenant, args.library)
        if not dead:
            print("no dead links")
            return 0
        for row in dead:
            print(f"{row['from']} -> [[{row['to']}]]")
        return 0
    raise SystemExit(f"unknown wiki command {cmd}")


def cmd_sync(args) -> int:
    """Pull one source into a library; ids, updates and removals are reconciled."""
    store = _store(args)
    kind = args.kind
    if kind == "local":
        conn = connectors_mod.LocalDirConnector(args.target, name=args.name or "local")
    elif kind == "rss":
        conn = connectors_mod.RssConnector(args.target, name=args.name or "rss")
    elif kind == "web":
        conn = connectors_mod.WebPageConnector([args.target], name=args.name or "web")
    elif kind == "sitemap":
        conn = connectors_mod.SitemapConnector(args.target, name=args.name or "sitemap")
    elif kind == "api":
        headers = {}
        for pair in args.header or []:
            k, _, v = pair.partition(":")
            headers[k.strip()] = v.strip()
        conn = connectors_mod.JsonApiConnector(
            args.target, name=args.name or "api", items_path=args.items_path,
            id_field=args.id_field, title_field=args.title_field,
            body_field=args.body_field, headers=headers)
    else:
        raise SystemExit(f"unknown source kind {kind}")
    report = connectors_mod.SyncEngine(store).sync(
        conn, tenant_id=args.tenant, library=args.library,
        acl=[a for a in args.acl.split(",") if a], actor=f"sync:{kind}")
    print(json.dumps(report.as_dict(), ensure_ascii=False))
    return 0 if not report.errors else 1


def cmd_sources(args) -> int:
    state = connectors_mod.SyncEngine(_store(args)).state()
    if not state:
        print("(no source has been synced yet)")
        return 0
    print(f"{'source':<40} {'library':<12} {'items':>6}  last sync")
    for row in state:
        print(f"{row['source_key'][:39]:<40} {row['library']:<12} "
              f"{row['items']:>6}  {row['last_sync']}")
    return 0


def cmd_scan(args) -> int:
    store = _store(args)
    ledger = feedback_mod.FeedbackLedger(store)
    cands = lifecycle.retirement_candidates(
        store, args.tenant, library=args.library,
        zero_citation=ledger.cited_documents(args.tenant),
        corrected=ledger.corrected_documents(args.tenant))
    conflicts = lifecycle.conflict_scan(store, args.tenant, args.library)
    if args.json:
        print(json.dumps({"retirement": [c.__dict__ for c in cands],
                          "conflicts": conflicts}, ensure_ascii=False, indent=2))
        return 0
    if not cands and not conflicts:
        print("nothing to retire; corpus is consistent")
        return 0
    for c in cands:
        print(f"[{c.severity:<6}] {c.path:<34} {c.reason}")
        print(f"         -> {c.suggested_action} {c.detail}")
    for c in conflicts:
        print(f"[conflict] {c['title']}: {', '.join(c['documents'])}")
        print(f"           {c['hint']}")
    return 0


def cmd_retire(args) -> int:
    store = _store(args)
    doc = store.get_document(args.tenant, args.library, args.path)
    if doc is None:
        print(f"no such document: {args.path}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"would retire {args.path} (v{doc.version}); pass --apply to do it")
        return 0
    store.set_status(doc.id, "retired", actor=args.actor, reason=args.reason)
    print(f"retired {args.path}")
    return 0


def cmd_evaluate(args) -> int:
    """IR-metric evaluation against doc-id relevance, with an optional gate."""
    store = _store(args)
    retriever = Retriever(store, config=RetrievalConfig(top_k=max(args.k, 10)))

    def search(query, principals, library):
        hits = retriever.search(query, tenant_id=args.tenant,
                                principals=principals, library=library)
        seen, out = set(), []
        for h in hits:                     # one vote per document, not per chunk
            if h.path not in seen:
                seen.add(h.path)
                out.append(h.path)
        return out

    cases = evalmetrics.load_cases(args.cases)
    report = evalmetrics.evaluate(cases, search, k=args.k)
    if args.save:
        report.save(args.save)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        for row in report.per_case:
            print(f"{row['query'][:34]:<36} ranked={row['ranked']}")
        print()
        for name, value in report.metrics.items():
            print(f"  {name:<16} {value:.4f}")
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        verdict = evalmetrics.gate(report, baseline, tolerance=args.tolerance)
        for f in verdict["failures"]:
            print(f"REGRESSION {f['metric']}: {f['baseline']:.4f} -> {f['current']:.4f}",
                  file=sys.stderr)
        for w in verdict["warnings"]:
            print(f"warning {w['metric']}: {w['baseline']:.4f} -> {w['current']:.4f}",
                  file=sys.stderr)
        return 0 if verdict["passed"] else 1
    return 0


def cmd_monitor(args) -> int:
    store = _store(args)
    mon = monitor_mod.Monitor(path=args.log)
    alerts = mon.check(args.tenant, args.library)
    report = mon.stats(args.tenant, args.library).as_dict()
    print(json.dumps({"stats": report,
                      "alerts": [a.as_dict() for a in alerts],
                      "actions": [a.__dict__ for a in
                                  mon.propose_actions(args.tenant, args.library)]},
                     ensure_ascii=False, indent=2))
    if alerts:
        return 1
    return 0


def cmd_eval(args) -> int:
    """Score a pinned question set: top-1 must contain the expected fact."""
    store = _store(args)
    retriever = Retriever(store, config=RetrievalConfig(top_k=args.top_k))
    cases = [json.loads(l) for l in Path(args.cases).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    ok, rows = 0, []
    for case in cases:
        hits = retriever.search(case["query"], tenant_id=args.tenant,
                                principals=case.get("principals", ["public"]),
                                library=case.get("library"))
        text = " ".join(h.text for h in hits[: args.top_k])
        good = all(str(f) in text for f in case.get("expect_facts", []))
        ok += good
        rows.append((case["query"], good, hits[0].path if hits else "—"))
    acc = ok / len(cases) if cases else 0.0
    if args.json:
        print(json.dumps({"accuracy": acc, "n": len(cases)}))
    else:
        for query, good, path in rows:
            print(f"{'✓' if good else '✗'} {query:<30} {path}")
        print(f"\naccuracy {acc:.1%} over {len(cases)} cases")
    if args.baseline and Path(args.baseline).exists():
        base = json.loads(Path(args.baseline).read_text())["accuracy"]
        if acc < base - args.tolerance:
            print(f"REGRESSION: {base:.1%} -> {acc:.1%}", file=sys.stderr)
            return 1
    return 0


def cmd_demo(args) -> int:
    from . import demo
    demo.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="le", description="loci-enterprise governance CLI")
    p.add_argument("--db", default="./le.db")
    p.add_argument("--tenant", default="default")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("memory", help="memory items: remember | recall | pending | approve | reject | history | forget | capacity")
    q.add_argument("memory_cmd",
                   choices=["remember", "recall", "pending", "approve", "reject",
                            "history", "forget", "capacity"])
    q.add_argument("subject", nargs="?", default="")
    q.add_argument("key", nargs="?", default="")
    q.add_argument("value", nargs="?", default="")
    q.add_argument("--memory-id", dest="memory_id", default="")
    q.add_argument("--source", default="agent")
    q.add_argument("--ttl-days", type=float, default=None)
    q.add_argument("--review", action="store_true",
                   help="hold as pending until approved")
    q.add_argument("--reason", default="")
    q.add_argument("--actor", default="cli")
    q.set_defaults(func=cmd_memory)

    q = sub.add_parser("member", help="tenant membership: grant | revoke | list")
    q.add_argument("member_cmd", choices=["grant", "revoke", "list"])
    q.add_argument("user", nargs="?", default="")
    q.add_argument("role", nargs="?", default="viewer", choices=list(access_mod.ROLE_LEVELS))
    q.add_argument("--actor", default="cli")
    q.set_defaults(func=cmd_member)

    q = sub.add_parser("key", help="API keys: issue | revoke | list")
    q.add_argument("key_cmd", choices=["issue", "revoke", "list"])
    q.add_argument("name", nargs="?", default="")
    q.add_argument("--role", default="viewer", choices=list(access_mod.ROLE_LEVELS))
    q.add_argument("--scopes", default="read", help="comma-separated: read,write,admin,delete")
    q.add_argument("--expires-in-days", type=float, default=None)
    q.add_argument("--key-id", default="")
    q.add_argument("--actor", default="cli")
    q.set_defaults(func=cmd_key)

    q = sub.add_parser("perm", help="print the role x action matrix")
    q.set_defaults(func=cmd_perm)

    q = sub.add_parser("put", help="write a document (creates or updates)")
    q.add_argument("path")
    q.add_argument("--library", default="default")
    q.add_argument("--title")
    q.add_argument("--body")
    q.add_argument("--file")
    q.add_argument("--effective")
    q.add_argument("--owner", default="")
    q.add_argument("--authority", type=float, default=1.0)
    q.add_argument("--actor", default="cli")
    q.add_argument("--acl", default="public")
    q.add_argument("--on-conflict", default="reject",
                   choices=["reject", "last_write_wins", "merge"])
    q.set_defaults(func=cmd_put)

    q = sub.add_parser("supersede", help="declare old replaced by new")
    q.add_argument("old")
    q.add_argument("new")
    q.add_argument("--library", default="default")
    q.add_argument("--actor", default="cli")
    q.add_argument("--note", default="")
    q.set_defaults(func=cmd_supersede)

    q = sub.add_parser("chain", help="what is in force for a topic")
    q.add_argument("path")
    q.add_argument("--library", default="default")
    q.set_defaults(func=cmd_chain)

    q = sub.add_parser("export", help="export one tenant's data as JSON")
    q.add_argument("out")
    q.set_defaults(func=cmd_export)

    q = sub.add_parser("import", help="import a tenant export into a tenant")
    q.add_argument("file")
    q.add_argument("--to-tenant", dest="to_tenant", default="default")
    q.add_argument("--actor", default="cli")
    q.set_defaults(func=cmd_import)

    q = sub.add_parser("doctor", help="self-check: integrity, orphans, ACL, jobs, connectors")
    q.add_argument("--json", action="store_true")
    q.add_argument("--stale-days", dest="stale_days", type=float, default=7.0)
    q.add_argument("--stuck-minutes", dest="stuck_minutes", type=float, default=30.0)
    q.set_defaults(func=cmd_doctor)

    q = sub.add_parser("blob", help="content-addressed blob store: put | get | stats")
    q.add_argument("blob_cmd", choices=["put", "get", "stats"])
    q.add_argument("file", nargs="?", default="")
    q.add_argument("digest", nargs="?", default="")
    q.add_argument("--blob-root", dest="blob_root", default="./blobs")
    q.add_argument("--out", default="")
    q.set_defaults(func=cmd_blob)

    q = sub.add_parser("schedule", help="periodic jobs: add | list | remove | enable | disable | tick")
    q.add_argument("schedule_cmd",
                   choices=["add", "list", "remove", "enable", "disable", "tick"])
    q.add_argument("name", nargs="?", default="")
    q.add_argument("kind", nargs="?", default="")
    q.add_argument("--every", type=float, default=3600.0)
    q.add_argument("--payload", default="")
    q.add_argument("--first-run-in", dest="first_run_in", type=float, default=None)
    q.add_argument("--limit", type=int, default=100)
    q.set_defaults(func=cmd_schedule)

    q = sub.add_parser("serve", help="serve the HTTP API")
    q.add_argument("--host", default="127.0.0.1")
    q.add_argument("--port", type=int, default=8971)
    q.add_argument("--allow-anonymous", action="store_true",
                   help="loopback-only convenience: treat requests without a key as viewers")
    q.set_defaults(func=cmd_serve)

    q = sub.add_parser("mcp", help="serve the knowledge base over MCP (stdio)")
    q.add_argument("--principal", default="public")
    q.set_defaults(func=cmd_mcp)

    q = sub.add_parser("agent", help="answer a question with a tool-using loop")
    q.add_argument("question")
    q.add_argument("--library", default="default")
    q.add_argument("--principal", default="public")
    q.add_argument("--model", default="")
    q.add_argument("--max-steps", type=int, default=8)
    q.add_argument("--trace", default="")
    q.add_argument("--verbose", action="store_true")
    q.set_defaults(func=cmd_agent)

    q = sub.add_parser("backend", help="pluggable retrieval backend: rebuild | stats | search")
    q.add_argument("backend_cmd", choices=["rebuild", "stats", "search"])
    q.add_argument("--kind", choices=["memory", "sqlite"], default="sqlite")
    q.add_argument("--path", default="./vectors.db")
    q.add_argument("--principal", default="public")
    q.add_argument("--query", default="")
    q.add_argument("--top-k", type=int, default=5)
    q.set_defaults(func=cmd_backend)

    q = sub.add_parser("queue", help="jobs: enqueue | run | stats | list | cancel | reclaim")
    q.add_argument("queue_cmd",
                   choices=["enqueue", "run", "stats", "list", "cancel", "reclaim"])
    q.add_argument("kind", nargs="?", default="")
    q.add_argument("--payload", default="")
    q.add_argument("--priority", type=int, default=0)
    q.add_argument("--status", default="")
    q.add_argument("--job-id", dest="job_id", default="")
    q.add_argument("--limit", type=int, default=100)
    q.add_argument("--older-than", dest="older_than", type=float, default=300)
    q.set_defaults(func=cmd_queue)

    q = sub.add_parser("wiki", help="compiled wiki: build | pages | show | history | rollback | links")
    q.add_argument("wiki_cmd",
                   choices=["build", "pages", "show", "history", "rollback", "links"])
    q.add_argument("slug", nargs="?", default="")
    q.add_argument("version", nargs="?", type=int, default=0)
    q.add_argument("--library", default="default")
    q.add_argument("--actor", default="cli")
    q.set_defaults(func=cmd_wiki)

    q = sub.add_parser("sync", help="sync a data source into a library")
    q.add_argument("kind", choices=["local", "rss", "web", "sitemap", "api"])
    q.add_argument("target")
    q.add_argument("--library", default="default")
    q.add_argument("--name", default="")
    q.add_argument("--acl", default="public")
    q.add_argument("--items-path", dest="items_path", default="items")
    q.add_argument("--id-field", dest="id_field", default="id")
    q.add_argument("--title-field", dest="title_field", default="title")
    q.add_argument("--body-field", dest="body_field", default="body")
    q.add_argument("--header", action="append", default=[],
                   help="extra request header, e.g. --header 'Authorization: Bearer x'")
    q.set_defaults(func=cmd_sync)

    q = sub.add_parser("sources", help="show connector watermarks")
    q.set_defaults(func=cmd_sources)

    q = sub.add_parser("scan", help="find documents that can no longer be trusted")
    q.add_argument("--library")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_scan)

    q = sub.add_parser("retire", help="take a document out of the active index")
    q.add_argument("path")
    q.add_argument("--library", default="default")
    q.add_argument("--actor", default="cli")
    q.add_argument("--reason", default="")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--apply", dest="dry_run", action="store_false")
    q.set_defaults(func=cmd_retire, dry_run=True)

    q = sub.add_parser("evaluate",
                       help="IR metrics (P@k, nDCG, MRR, MAP) with a regression gate")
    q.add_argument("cases")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("--json", action="store_true")
    q.add_argument("--save", default="")
    q.add_argument("--baseline", default="")
    q.add_argument("--tolerance", type=float, default=0.02)
    q.set_defaults(func=cmd_evaluate)

    q = sub.add_parser("monitor", help="has retrieval quality moved?")
    q.add_argument("--library", default="default")
    q.add_argument("--log", default="monitor.jsonl")
    q.set_defaults(func=cmd_monitor)

    q = sub.add_parser("eval", help="score a pinned question set")
    q.add_argument("cases")
    q.add_argument("--library")
    q.add_argument("--top-k", type=int, default=3)
    q.add_argument("--json", action="store_true")
    q.add_argument("--baseline")
    q.add_argument("--tolerance", type=float, default=0.02)
    q.set_defaults(func=cmd_eval)

    q = sub.add_parser("demo", help="reproduce the 85%% -> 60%% incident")
    q.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
