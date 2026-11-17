#!/usr/bin/env python3
"""CI gate: the incident must still reproduce.

`tests/test_demo.py` asserts this on the source tree. This script asserts the
same thing against an *installed* package, which is what the packaging job
needs — a wheel that installs but ships a stale demo would otherwise pass.

Exit codes: 0 = the story holds, 1 = a threshold moved.
"""
from __future__ import annotations

import sys

from loci_enterprise import __version__, demo


def main() -> int:
    r = demo.run(verbose=False)
    healthy = r["1_healthy"]
    after_change = r["2_added_without_retiring"]
    after_more = r["3_added_more_documents"]
    after_retire = r["4_retired_superseded"]

    # If a change moves these, it changed the thesis, not just the numbers.
    rules = [
        ("healthy baseline", healthy, healthy >= 0.80,
         "believable baseline (>= 80%), not a perfect score"),
        ("policy changed, only added docs", after_change, after_change <= healthy - 0.15,
         "a policy change without retirement must cost >= 15 points"),
        ("added even more documents", after_more, after_more <= after_change + 0.02,
         "adding documents must not recover accuracy"),
        ("retired superseded documents", after_retire, after_retire >= healthy - 0.05,
         "retiring must recover to within 5 points of baseline"),
    ]

    print(f"loci-enterprise {__version__} — incident reproduction")
    failed = 0
    for label, value, ok, why in rules:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<36} {value:6.1%}   {why}")
        failed += not ok

    if failed:
        print(f"\n{failed} rule(s) violated — the demo no longer shows what the "
              f"README claims.", file=sys.stderr)
        return 1
    print("\nadding documents does not fix a drifting corpus; retiring does")
    return 0


if __name__ == "__main__":
    sys.exit(main())
