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
from pathlib import Path

from . import access as access_mod
from . import feedback as feedback_mod
from . import governance, lifecycle, monitor as monitor_mod
from .retrieval import RetrievalConfig, Retriever
from .store import Document, Store


def _store(args) -> Store:
    return Store(args.db)


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
