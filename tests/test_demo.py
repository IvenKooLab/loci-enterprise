"""The incident, asserted. If this test fails, the thesis is wrong."""
from __future__ import annotations

from loci_enterprise import demo


def test_adding_documents_does_not_fix_a_drifting_corpus():
    r = demo.run(verbose=False)
    healthy = r["1_healthy"]
    after_change = r["2_added_without_retiring"]
    after_more = r["3_added_more_documents"]
    after_retire = r["4_retired_superseded"]

    # a believable baseline, not a perfect one
    assert 0.75 <= healthy <= 0.95
    # the incident: the world changed, the corpus did not
    assert after_change <= healthy - 0.15
    # adding more documents does not recover it, and usually hurts
    assert after_more <= after_change + 0.02
    # retiring what was superseded does recover it
    assert after_retire >= healthy - 0.05
