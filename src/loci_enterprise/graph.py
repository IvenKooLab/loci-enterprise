"""Knowledge graph: the links the corpus already contains, made queryable.

Every document already carries three kinds of relationship, and all of them
are invisible inside a flat index:

    [[wikilinks]]          what a note points at
    supersedes             which policy replaced which
    same title             two active documents claiming one topic

The graph is *derived*, never authored: rebuilding it from the corpus produces
the same edges, so a diff means the documents changed, not that a model felt
creative. Unresolved links become explicit stub nodes instead of being dropped
— a broken link is information, and hiding it is how a wiki rots.

Queries are deliberately small: neighbours, a bounded path, a subgraph, and
stats. That covers "what does this depend on" and "how are these two
connected" without pretending to be a graph database.
"""
from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field

from . import lifecycle
from .store import Store
from .wiki import slugify

SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    tenant_id TEXT NOT NULL,
    library   TEXT NOT NULL,
    node_id   TEXT NOT NULL,
    kind      TEXT NOT NULL DEFAULT 'document',
    label     TEXT NOT NULL DEFAULT '',
    status    TEXT NOT NULL DEFAULT 'active',
    meta      TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (tenant_id, library, node_id)
);
CREATE TABLE IF NOT EXISTS graph_edges (
    tenant_id TEXT NOT NULL,
    library   TEXT NOT NULL,
    src       TEXT NOT NULL,
    dst       TEXT NOT NULL,
    kind      TEXT NOT NULL,
    weight    REAL NOT NULL DEFAULT 1.0,
    meta      TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (tenant_id, library, src, dst, kind)
);
CREATE INDEX IF NOT EXISTS graph_edges_src ON graph_edges(tenant_id, library, src);
CREATE INDEX IF NOT EXISTS graph_edges_dst ON graph_edges(tenant_id, library, dst);
"""

_LINK = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
_EDGE_KINDS = ("links", "supersedes", "duplicate")


@dataclass
class Node:
    node_id: str
    kind: str
    label: str
    status: str = "active"
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Edge:
    src: str
    dst: str
    kind: str
    weight: float = 1.0
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class BuildReport:
    nodes: int = 0
    edges: int = 0
    stubs: int = 0
    broken: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges, "stubs": self.stubs,
                "broken": self.broken}


class GraphStore:
    def __init__(self, store: Store):
        self.store = store
        self.store.conn.executescript(SCHEMA)

    # ---------------------------------------------------------------- build
    def build(self, tenant_id: str, library: str) -> BuildReport:
        report = BuildReport()
        docs = [d for d in self.store.list_documents(tenant_id, library)]
        self.store.conn.execute(
            "DELETE FROM graph_edges WHERE tenant_id=? AND library=?",
            (tenant_id, library))
        self.store.conn.execute(
            "DELETE FROM graph_nodes WHERE tenant_id=? AND library=?",
            (tenant_id, library))

        by_slug: dict[str, str] = {}
        by_path = {d.path: d.path for d in docs}
        for d in docs:
            slug = slugify(d.title or d.path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
            by_slug.setdefault(slug, d.path)
        titles: dict[str, list[str]] = {}
        for d in docs:
            titles.setdefault((d.title or "").strip().lower(), []).append(d.path)

        for d in docs:
            verdict = lifecycle.evaluate(d)
            node = Node(node_id=d.path, kind="document", label=d.title or d.path,
                        status=verdict.status,
                        meta={"library": d.library, "version": d.version,
                              "path": d.path})
            self._node(tenant_id, library, node)
            report.nodes += 1

            for match in _LINK.finditer(d.body or ""):
                target_slug = slugify(match.group(1))
                target = by_slug.get(target_slug)
                if target:
                    self._edge(tenant_id, library, Edge(d.path, target, "links"))
                    report.edges += 1
                else:
                    stub_id = f"stub:{target_slug}"
                    self._node(tenant_id, library,
                               Node(stub_id, "stub", match.group(1).strip()))
                    self._edge(tenant_id, library,
                               Edge(d.path, stub_id, "links", meta={"unresolved": True}))
                    report.stubs += 1
                    report.broken.append({"from": d.path, "to": target_slug})

            if d.supersedes:
                # lineage stores a path, not a title; fall back to a slug for
                # hand-written frontmatter that used a note name
                target = by_path.get(d.supersedes) or by_slug.get(slugify(d.supersedes))
                if target:
                    self._edge(tenant_id, library, Edge(d.path, target, "supersedes"))

        for title, paths in titles.items():
            if not title or len(paths) < 2:
                continue
            for i, a in enumerate(paths):
                for b in paths[i + 1:]:
                    self._edge(tenant_id, library,
                               Edge(a, b, "duplicate", meta={"title": title}))
        report.edges = self.store.conn.execute(
            "SELECT COUNT(*) n FROM graph_edges WHERE tenant_id=? AND library=?",
            (tenant_id, library)).fetchone()["n"]
        return report

    def _node(self, tenant_id: str, library: str, node: Node) -> None:
        self.store.conn.execute(
            "INSERT OR REPLACE INTO graph_nodes(tenant_id, library, node_id, kind,"
            " label, status, meta) VALUES (?,?,?,?,?,?,?)",
            (tenant_id, library, node.node_id, node.kind, node.label,
             node.status, json.dumps(node.meta, ensure_ascii=False)))

    def _edge(self, tenant_id: str, library: str, edge: Edge) -> None:
        self.store.conn.execute(
            "INSERT OR REPLACE INTO graph_edges(tenant_id, library, src, dst, kind,"
            " weight, meta) VALUES (?,?,?,?,?,?,?)",
            (tenant_id, library, edge.src, edge.dst, edge.kind, edge.weight,
             json.dumps(edge.meta, ensure_ascii=False)))

    # --------------------------------------------------------------- reads
    def nodes(self, tenant_id: str, library: str, kind: str | None = None) -> list[Node]:
        sql = "SELECT * FROM graph_nodes WHERE tenant_id=? AND library=?"
        args: list = [tenant_id, library]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        return [Node(r["node_id"], r["kind"], r["label"], r["status"],
                     json.loads(r["meta"] or "{}"))
                for r in self.store.conn.execute(sql, args).fetchall()]

    def edges(self, tenant_id: str, library: str,
              kind: str | None = None) -> list[Edge]:
        sql = "SELECT * FROM graph_edges WHERE tenant_id=? AND library=?"
        args: list = [tenant_id, library]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        return [Edge(r["src"], r["dst"], r["kind"], r["weight"],
                     json.loads(r["meta"] or "{}"))
                for r in self.store.conn.execute(sql, args).fetchall()]

    def neighbors(self, tenant_id: str, library: str, node_id: str, *,
                  direction: str = "both", kind: str | None = None) -> list[dict]:
        if direction not in ("in", "out", "both"):
            raise ValueError("direction must be in|out|both")
        out: list[dict] = []
        for e in self.edges(tenant_id, library, kind):
            if e.src == node_id and direction in ("out", "both"):
                out.append({"direction": "out", "kind": e.kind, "node": e.dst})
            if e.dst == node_id and direction in ("in", "both"):
                out.append({"direction": "in", "kind": e.kind, "node": e.src})
        return out

    def subgraph(self, tenant_id: str, library: str, node_id: str, *,
                 depth: int = 1) -> dict:
        seen = {node_id}
        frontier = [node_id]
        nodes_out = []
        for _ in range(max(0, depth)):
            nxt = []
            for current in frontier:
                for nb in self.neighbors(tenant_id, library, current):
                    if nb["node"] not in seen:
                        seen.add(nb["node"])
                        nxt.append(nb["node"])
            frontier = nxt
        node_map = {n.node_id: n for n in self.nodes(tenant_id, library)}
        for nid in seen:
            if nid in node_map:
                nodes_out.append(node_map[nid].as_dict())
        edges_out = [e.as_dict() for e in self.edges(tenant_id, library)
                     if e.src in seen and e.dst in seen]
        return {"nodes": nodes_out, "edges": edges_out}

    def path(self, tenant_id: str, library: str, start: str, goal: str,
             *, max_depth: int = 4) -> list[str] | None:
        if start == goal:
            return [start]
        prev: dict[str, str] = {start: ""}
        queue = deque([(start, 0)])
        adjacency: dict[str, list[str]] = {}
        for e in self.edges(tenant_id, library):
            adjacency.setdefault(e.src, []).append(e.dst)
            adjacency.setdefault(e.dst, []).append(e.src)
        while queue:
            node, dist = queue.popleft()
            if dist >= max_depth:
                continue
            for nxt in adjacency.get(node, []):
                if nxt in prev:
                    continue
                prev[nxt] = node
                if nxt == goal:
                    out = [goal]
                    while out[-1] != start:
                        out.append(prev[out[-1]])
                    return list(reversed(out))
                queue.append((nxt, dist + 1))
        return None

    def stats(self, tenant_id: str, library: str) -> dict:
        by_kind: dict[str, int] = {}
        for e in self.edges(tenant_id, library):
            by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
        nodes = self.nodes(tenant_id, library)
        stubs = [n for n in nodes if n.kind == "stub"]
        isolated = []
        if nodes:
            connected = {e.src for e in self.edges(tenant_id, library)} | \
                        {e.dst for e in self.edges(tenant_id, library)}
            isolated = [n.node_id for n in nodes
                        if n.kind == "document" and n.node_id not in connected]
        return {"nodes": len(nodes), "documents": len(nodes) - len(stubs),
                "stubs": len(stubs), "edges": sum(by_kind.values()),
                "by_kind": by_kind, "isolated": isolated}

    def export(self, tenant_id: str, library: str) -> dict:
        return {"nodes": [n.as_dict() for n in self.nodes(tenant_id, library)],
                "edges": [e.as_dict() for e in self.edges(tenant_id, library)]}
