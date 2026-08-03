"""Pull the two root documents into the site without copying them.

`README.md` and `RESEARCH.md` have to stay at the repository root — that is
where GitHub renders them and where anyone who clones the project looks first.
They also have to appear in the built site. Copying them into `docs/` would
give every measured figure two homes, and this project has already spent a full
audit reconciling documents that drifted apart. One source, read at build time.

The links then need fixing, because the same relative path cannot be correct in
both places. `README.md` says `docs/02-leakage.md`, which is right from the
repository root and wrong from inside `docs/`. The rewrites below are the whole
of the difference:

    README.md      docs/<page>.md  ->  <page>.md        RESEARCH.md -> research.md
    RESEARCH.md    docs/<page>.md  ->  <page>.md
    docs/*.md      ../RESEARCH.md  ->  research.md

`mkdocs build --strict` fails on a link that resolves to nothing, so a rule
that stops matching is a build failure rather than a quiet 404.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Which built page is fed by which root document.
SOURCES = {"index.md": "README.md", "research.md": "RESEARCH.md"}

_DOCS_PREFIX = re.compile(r"\]\(docs/([^)]+)\)")
_RESEARCH_FROM_ROOT = re.compile(r"\]\(RESEARCH\.md([^)]*)\)")
_RESEARCH_FROM_DOCS = re.compile(r"\]\(\.\./RESEARCH\.md([^)]*)\)")


def _rewrite_from_root(text: str) -> str:
    """Links written relative to the repo root, read from inside `docs/`."""
    text = _DOCS_PREFIX.sub(r"](\1)", text)
    return _RESEARCH_FROM_ROOT.sub(r"](research.md\1)", text)


def on_page_markdown(markdown: str, page, config, files) -> str:
    source = SOURCES.get(page.file.src_uri)
    if source:
        return _rewrite_from_root((REPO / source).read_text(encoding="utf-8"))
    # An ordinary page under docs/ pointing back at the root research report.
    return _RESEARCH_FROM_DOCS.sub(r"](research.md\1)", markdown)
