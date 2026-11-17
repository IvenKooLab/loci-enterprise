"""loci-enterprise — retrieval-quality-governed knowledge base.

The research memo (`../loci-enterprise-research.md`) landed on a specific
claim: enterprise RAG does not fail because retrieval is weak, it fails
because the *corpus drifts away from the truth* and nobody measures it.
This package is the smallest thing that addresses that claim end to end:

    store      documents + chunks + ACL + version history, one SQLite file
    lifecycle  validity windows, supersede chains, staleness, retirement
    retrieval  hybrid BM25/vector fused with RRF, ACL pushed into the query
    monitor    per-library quality metrics, drift detection, live mitigation
    feedback   thumbs/citations -> chunk quality -> ranking -> eval cases
    judge      pluggable faithfulness verifier (DeepSeek, or offline stub)

Deliberate non-goals: connectors, SSO, a web UI. They are integration work;
the governance layer here is the part that has to be right first.
"""

__version__ = "0.1.0"
