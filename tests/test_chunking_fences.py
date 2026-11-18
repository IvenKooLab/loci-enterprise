"""Fenced code keeps its text and citation section (ported from loci #25/#26).

A naive any-``` toggle lets an inner triple-backtick example close an outer
four-backtick fence; without fence tracking at all, even a plain block's
`# comment` hijacks the section trail. These pin the CommonMark rules:
matching delimiter type, closing run at least as long, empty info string.
"""
from __future__ import annotations

import pytest

from loci_enterprise.chunking import chunk_document


FENCES = [
    ("```python", "print(1)", "```"),
    ("````markdown", "```python\n# literal heading\n```", "````"),
    ("~~~markdown", "```python\n# literal heading\n```", "~~~"),
    ("~~~~text", "~~~\n# literal heading", "~~~~"),
    ("```text", "```not-a-closing-fence\n# literal heading", "```"),
    ("```text", "    ```\n# literal heading", "```"),
    ("  ````text", "```\n# literal heading", "   `````\t"),
    ("~~~text", "~~~~not-a-closing-fence\n# literal heading", "~~~~"),
]


@pytest.mark.parametrize("opening,interior,closing", FENCES)
def test_fence_boundaries_preserve_code_and_sections(opening, interior, closing):
    code = "\n".join([opening, interior, "payload " * 30, closing])
    chunks = chunk_document(f"# Guide\n\n{code}\n\n## Next\nfinish")
    assert len([c for c in chunks if code.strip() in c]) == 1
    assert any(c.startswith("Guide\n") and code.strip() in c for c in chunks)
    assert any(c.startswith("Guide > Next\n") and "finish" in c for c in chunks)
    assert not any("literal heading" in c.splitlines()[0] for c in chunks)


def test_plain_block_comment_is_not_a_section():
    chunks = chunk_document("# Guide\n\n```python\n# setup config\nx = 1\n```\n\n## Next\nfinish")
    assert [c.splitlines()[0] for c in chunks] == ["Guide", "Guide > Next"]
    assert "# setup config" in chunks[0]


def test_unclosed_fence_keeps_remaining_headings_as_code():
    code = "````markdown\n```\n# literal heading\n" + "payload " * 30
    chunks = chunk_document(f"# Guide\n\n{code}")
    assert all(c.startswith("Guide") for c in chunks)
    assert any("# literal heading" in c for c in chunks)


def test_backticks_in_info_string_are_not_an_opening_fence():
    chunks = chunk_document("# Guide\n```invalid`info\n## Real\nbody")
    assert [c.splitlines()[0] for c in chunks] == ["Guide", "Guide > Real"]


def test_tilde_line_inside_backtick_fence_stays_code():
    chunks = chunk_document("# Guide\n\n```text\n~~~\n# literal\n```\n\n## Next\nfin")
    assert [c.splitlines()[0] for c in chunks] == ["Guide", "Guide > Next"]
    assert "# literal" in chunks[0]
