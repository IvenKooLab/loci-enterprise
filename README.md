# loci-enterprise

**A knowledge base that measures itself.** An MCP-first knowledge base with
retrieval-quality governance built in: document lifecycle, multi-writer safety,
drift monitoring, feedback, plus the plumbing a real deployment needs —
connectors, parsers, wiki, queue, scheduler, tracing, HTTP API and an agent
loop. Standard-library Python; no runtime dependencies.

> Status: alpha. Zero runtime dependencies is a hard constraint: a governance
> layer that cannot be installed into an air-gapped build is not a governance
> layer. Everything below is implemented and covered by tests.

---

## The problem this exists to solve

> A RAG system goes live at 85% answer accuracy. Three months later it is at
> 60%. The retrieved documents "did not change", the model was not swapped, the
> prompt was not edited. **Will adding more documents help?**

No. It makes it worse — and this repository demonstrates that with a runnable
test rather than an opinion:

```
$ le demo

  healthy baseline                            85.7%
  policy changed, only added docs             64.3%
  added even more documents                   50.0%
  retired superseded documents                85.7%
```

**Why adding documents fails.** Retrieval did not degrade — *the truth moved*.
Four policies changed; the pages describing them did not. The index still holds
last quarter's rule, the retriever faithfully finds it, the model faithfully
quotes it, and the answer is now confidently wrong. Add the new policy without
retiring the old one and you now have **two equally similar documents with
different numbers**. Cosine similarity has no concept of "still in force", so
it is a coin flip — and worse, the stale page usually wins, because it has been
cited for a year and carries more authority.

**Why systems get more accurate over time.** Not because they accumulate more
documents, but because they accumulate *judgement*: what was true when, which
page replaced which, and which sources actually produced correct answers. That
is lifecycle plus feedback, not corpus growth.

---

## The four mechanisms

### 1. Lifecycle — retire instead of accumulate

| Field | Meaning |
|---|---|
| `effective_from` / `effective_to` | when the document is true (machine-checkable) |
| `supersedes` / `superseded_by` | what replaced what, written both ways |
| `stale_after_days` | when to stop trusting silence |
| `authority` | how much this source has earned (decays, transfers on supersede) |

`superseded` and `expired` documents are weighted out **before** fusion, not
filtered after it. Filtering after top-k is how a stale policy keeps a slot a
current one should have had.

```python
lifecycle.supersede(store, "acme", "policy-v3.md", "policy-v4.md",
                    library="hr", actor="hr-admin")
lifecycle.retirement_candidates(store, "acme")   # proposes, never deletes
```

### 2. Multi-writer governance

Enterprise reality: a support engineer pastes a fix, a manager corrects a
number, a sync job reconciles both — inside the same minute. Three failure
modes, each with an explicit answer:

| Failure | Answer |
|---|---|
| lost update | optimistic `expected_version`; the loser gets a `Conflict` |
| duplicate truth | unique `(tenant, library, path)`; identical writes are no-ops |
| ghost write | `on_conflict="merge"` keeps both edits, audited — never silent |

Default is `reject`. `last_write_wins` exists because some writers (a job
replaying a system of record) genuinely want it, but it must be chosen and it
is audited as such.

### 3. Retrieval quality monitoring

Every request records hits, top score, latency, judge score and user feedback.
Per library, a rolling window is compared against a pinned baseline with a
two-proportion z-test. Three alerts matter:

- `accuracy_drop` — accepted-answer rate fell
- `zero_hit_spike` — corpus gap or over-tight ACL
- `stale_truth` — **top scores unchanged, answers worse.** This is the
  signature of the 85→60 incident, and the one nobody instruments.

Mitigation is proposed, not applied blindly: quarantine superseded, freeze the
baseline, route through review. Every action is reversible and audited.

### 4. Feedback loop

Two loops that must not be confused:

- **fast** — thumbs-down demotes the chunks actually used (bounded, asymmetric:
  a false negative is cheaper than a false positive that keeps a wrong citation)
- **slow** — accepted pairs become eval cases, so a change is measured against
  a pinned baseline instead of argued about

The fast loop alone drifts whichever way the loudest user pushes it, which is
why the slow loop is not optional.

---

## What is in the box

Every module is independent and tested on its own; `le` is the operator surface
over all of them.

| Module | What it does | Why it exists |
|---|---|---|
| `lifecycle` | effective windows, supersede chains, staleness decay, retirement | the corpus drifts away from the truth |
| `monitor` | per-library rolling window vs pinned baseline; `stale_truth` alert | nobody notices until users complain |
| `feedback` | bounded chunk re-weighting + accepted pairs become eval cases | a wrong answer must not repeat |
| `access` | four roles, tenant isolation, scoped API keys, audit | a role that leaks across tenants is not a role |
| `memory` | active / superseded / archived / pending, expiry, review, forget | two contradictory facts must not coexist |
| `governance` | duplicate titles, dangling supersedes, retirement candidates | the silent failure modes |
| `parsers` | docx / pptx / xlsx / csv / html / json / epub, stdlib only | companies do not store knowledge in .md |
| `connectors` | local, RSS, web, sitemap; incremental with prune | a deleted source must leave the index |
| `wiki` | deterministic pages, citations, dead links, rollback | a wiki that invents facts is worse than none |
| `queue` | durable jobs, retries, backoff, crash recovery | ingestion outlives the request |
| `scheduler` | periodic jobs without burst on recovery | "nightly" should not need a human |
| `tracing` | span trees, token accounting, wired into retrieval | averages hide the regressing stage |
| `evalmetrics` | P@k, nDCG, MRR, MAP, BLEU, ROUGE-L, gates | an eval that cannot fail a build is a dashboard |
| `backends` | one retrieval contract, memory + SQLite | swapping storage should be bounded |
| `agent` | bounded tool loop with read-only policy | the model must not invent a tool |
| `mcp` | eight tools over stdio JSON-RPC | MCP is the primary interface |
| `service` | HTTP API with per-route authorisation | everything else needs to call it |
| `storage` | content-addressed blobs, dedup, integrity | originals outlive extracted text |
| `doctor` | integrity, orphans, missing ACL, stuck jobs, stale connectors | these failures are silent |
| `portability` | per-tenant export/import | data belongs to the tenant |
| `deploy` | Dockerfile, compose, Helm chart | deployment is a feature |

### CLI at a glance

```
le demo                      reproduce the 85% -> 60% incident
le put / supersede / chain    write and inspect document lineage
le scan / retire              lifecycle findings and explicit retirement
le monitor / evaluate         live drift, offline IR metrics with a gate
le sync <kind> <target>       local / rss / web / sitemap, incremental
le wiki build|pages|show      compiled pages, history, rollback
le queue / schedule           durable jobs and periodic work
le memory remember|recall     facts with supersede, expiry and review
le member / key / perm        roles, API keys, permission matrix
le serve / mcp                HTTP API or MCP stdio
le agent "question"           bounded tool-using loop
le backend / blob             pluggable retrieval backend, blob storage
le doctor / export / import   self-check and tenant portability
```

## Permission and truth are part of the query

```python
hits = retriever.search("住宿上限", tenant_id="acme", principals=["public"])
```

Order of operations is the whole design:

```
ACL join -> lifecycle gate -> BM25 over THAT corpus -> vector -> RRF
         -> lifecycle weight -> authority -> feedback weight -> top-k
```

Two consequences worth stating:

- **IDF is computed inside the permission domain.** Another team's documents
  cannot change your term weights. (Test: `test_idf_is_computed_within_the_permission_domain`.)
- **There is no "all tenants" code path.** `tenant_id` and `principals` are
  mandatory arguments, and unauthorised chunks are never candidates — so they
  cannot occupy a top-k slot or reach the reranker.

---

## Models: pluggable, and deliberately separable

```python
from loci_enterprise.providers import deepseek
from loci_enterprise.judge import verify

v = verify(deepseek(model="deepseek-flash"), question, answer, hits)
v.faithfulness, v.cross_file, v.acceptable
```

- **Judge** runs on a sample of live traffic. Any OpenAI-compatible endpoint
  works (`OpenAICompatProvider`); the DeepSeek default is `deepseek-flash`.
  `StubProvider` keeps tests and demos offline and deterministic.
- **Embeddings are a separate provider.** DeepSeek's API serves chat
  completions, not `/embeddings`; point `[embed]` at an embedding-capable
  provider or a local model. `HashEmbedder` is the offline fallback (stable,
  not semantic — it exists so the ranking plumbing is testable without a
  network).
- The judge never reorders results. A verifier that silently reranks is a
  second hidden retriever; scores go to the monitor, not to the ranking.

### Why DeepSeek is a good default here (and where it stops)

- **Open weights under MIT** (DeepSeek-V3.2 / V3.2-Exp). For a governance layer
  this matters more than benchmarks: the *judge* sees answers and source
  excerpts, which is exactly the data an enterprise does not want to ship
  elsewhere. MIT weights mean the judge can run on-prem, on the same network
  boundary as the corpus.
- **Model names move; pin them.** As of 2026-11 the API catalogue is
  `deepseek-flash` (V4.1-Flash, 1M context, JSON output) and `deepseek-v4-pro`
  (larger, ~10x the output price). `deepseek-chat` / `deepseek-reasoner` are
  retired aliases. `deepseek-flash` is the judge default here — judging is
  short-context, high-volume work; the pro model earns its price only when
  adjudicating a hard cross-file inference.
- **No embeddings endpoint.** The API serves `/chat/completions` only. Pair it
  with a local embedding model or another provider; keep `[embed]` independent
  so a chat-model swap never silently invalidates the vector space.

The judge checks two different failures: unsupported claims, and **claims that
are individually quoted correctly but wrong when joined across files**. The
second one survives ordinary citation checking and is exactly what happens when
an answer blends an old policy with a new one.

---

## Quick start

```bash
python -m pip install -e ".[dev]"
le demo                       # the 85% -> 60% incident, reproduced

le --db ./le.db --tenant acme put policy-v3.md --library hr \
   --file policy.md --effective 2026-01-01 --authority 1.3
le --db ./le.db --tenant acme supersede policy-v3.md policy-v4.md --library hr
le --db ./le.db --tenant acme scan                # what can no longer be trusted
le --db ./le.db --tenant acme retire policy-v3.md --apply
le --db ./le.db --tenant acme chain policy-v4.md  # what is in force, and since when
le --db ./le.db --tenant acme eval cases.jsonl --baseline baseline.json
```

`scan` proposes; a human or an explicit policy retires. Auto-deleting on a
heuristic is how a governance tool loses someone's work.

---

## What this is not

- **Not a vector database.** `HashEmbedder` is a placeholder for tests; plug in
  a real one. The governance layer is the point.
- **Not every enterprise connector.** Local, RSS, web and sitemap ship; Confluence/
  Jira/飞书 need per-vendor APIs and are separate adapters.
- **Not multi-tenant SaaS.** Tenant isolation is enforced in every query, but
  provisioning, SSO and quotas are out of scope.
- **Not a judge you should trust blindly.** Faithfulness scores are a signal
  for the monitor, sampled and aggregated — never a gate on a single answer.

---

## Design notes worth arguing about

1. **Lifecycle status belongs to the corpus, not the prompt.** Telling the model
   "ignore outdated documents" is asking it to do what the index should have
   done — and it cannot see the documents it was never shown.
2. **Detection and mitigation are separate.** Detection must be cheap and
   trustworthy; mitigation must be reversible and attributable.
3. **A governance tool that reports 100% is reporting on itself.** The demo
   keeps two questions that never worked, on purpose.
4. **Adding documents is not a quality strategy.** It is how you get two
   equally confident answers, one of which is last quarter's.

## License

Apache-2.0.
