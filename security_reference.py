"""
Vectorless RAG: OWASP guidance for Security-category findings.

Why vectorless
--------------
The knowledge source is a single 315-page PDF with a fixed table of contents.
There is no corpus to embed, no need for nearest-neighbour search, and nothing
to re-index when it changes. So this module uses the structure the document
already provides:

    OWASP PDF
        -> pdftotext (already installed; no new Python dependency)
        -> table of contents parsed into 45 cheat sheets with page ranges
        -> a finding is classified into a vulnerability type by keyword rules
        -> candidate sheets for that type are scored by keyword overlap
        -> the best-scoring sheet's own text supplies the summary
        -> reference attached to the Security finding

No embeddings, no vector store, no LLM in the retrieval path. The only
dependency is the `pdftotext` binary from poppler-utils, which ships with the
machine already; `extract_pages()` shells out to it rather than adding a PDF
library.

Fidelity rules
--------------
1. The PDF is the source of truth. Every word of the `summary` field is copied
   from the document. Nothing is generated, paraphrased, or invented.
2. No confident match -> the `reference` field is omitted entirely. A missing
   reference is the correct outcome for an unsupported category; an irrelevant
   one is worse than nothing because it reads as authoritative.
3. Only Security findings are enriched. A Bug/Performance/Quality finding never
   receives an OWASP reference no matter what it says.

A note on `url`
---------------
The PDF is a 2015 LaTeX build ("Created with LaTeX with hyperref package") and
its text contains no canonical cheat-sheet page URLs -- only incidental links
such as owasp.org wiki pages. The `url` is therefore *derived* from OWASP's
official Cheat Sheet Series naming convention
(cheatsheetseries.owasp.org/cheatsheets/<Sheet_Name>.html), and is emitted only
for sheets that genuinely exist in this PDF. That URL points at today's live
sheet, which may have been revised since 2015; `summary` is always the 2015 PDF
text, never the live page. `url_source` records which it is.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF = REPO_ROOT / "OWASP_Cheatsheets_Book.pdf"

# Page range of the table of contents within the PDF. Verified: the TOC lives on
# PDF pages 2-10, and a TOC page number equals the PDF page number.
TOC_PAGES = (2, 10)

# The TOC's own trailing note, which is not a cheat sheet.
_NOT_A_SHEET = re.compile(r"^(contents|these cheat sheets|all the articles)", re.I)


class ReferenceUnavailable(RuntimeError):
    """Raised when the PDF cannot be read at all (missing binary/file)."""


# --------------------------------------------------------------------------
# text extraction
# --------------------------------------------------------------------------

def _pdftotext_binary() -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        raise ReferenceUnavailable(
            "pdftotext not found. Install poppler-utils "
            "(Debian/Ubuntu: apt install poppler-utils)."
        )
    return exe


def extract_pages(pdf_path: str | Path = DEFAULT_PDF,
                  cache_path: str | Path | None = None) -> list[str]:
    """Return the PDF's text as a list of page strings, index 0 == page 1.

    pdftotext separates pages with form feeds, which is all we need to address
    pages by number. Results are cached on disk because the pipeline is a
    long-lived API process and re-extracting 315 pages per request would be
    wasteful.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise ReferenceUnavailable(f"OWASP PDF not found: {pdf_path}")

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and cache_path.stat().st_mtime >= pdf_path.stat().st_mtime:
            return cache_path.read_text(encoding="utf-8").split("\f")

    try:
        proc = subprocess.run([_pdftotext_binary(), str(pdf_path), "-"],
                              capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise ReferenceUnavailable("pdftotext timed out") from exc

    if proc.returncode != 0:
        raise ReferenceUnavailable(f"pdftotext failed: {proc.stderr.strip()[:200]}")

    pages = proc.stdout.split("\f")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(proc.stdout, encoding="utf-8")
    return pages


# --------------------------------------------------------------------------
# cheat-sheet index
# --------------------------------------------------------------------------

@dataclass
class CheatSheet:
    number: int
    title: str
    start_page: int          # 1-based, inclusive
    end_page: int            # 1-based, inclusive
    _pages: list[str] = field(default_factory=list, repr=False)

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", self.title).strip("_")

    @property
    def url(self) -> str:
        return f"https://cheatsheetseries.owasp.org/cheatsheets/{self.slug}.html"

    @property
    def text(self) -> str:
        """The sheet's own text, running headers and page furniture removed."""
        if not self._pages:
            return ""
        return _clean_sheet_text("\n".join(self._pages), self.title, self.number)

    def section_lines(self) -> list[tuple[str, str]]:
        """(heading, body) pairs, using the numbered x.y headings as separators."""
        out: list[tuple[str, str]] = []
        heading: str | None = None
        buf: list[str] = []

        header_variants = {self.title, f"{self.number}. {self.title}"}
        for raw in self.text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line in header_variants or line.rstrip(".").strip() in header_variants:
                continue
            # Numbered subsection headings, e.g. "20.3. Parameterized Queries".
            if re.match(r"^\d+\.\d+\.?\s+\S", line):
                if heading is not None and buf:
                    out.append((heading, "\n".join(buf)))
                heading, buf = line, []
                continue
            buf.append(line)
        if heading is not None and buf:
            out.append((heading, "\n".join(buf)))
        return out


def _clean_sheet_text(text: str, title: str, number: int | None = None) -> str:
    """Drop the repeated running header and bare page numbers.

    The running header is printed as "<n>. <Sheet Title>", so it must be matched
    both with and without the leading sheet number. Missing the numbered form
    let the header survive inside `CheatSheet.text`, which made `text` and
    `section_lines()` disagree about what the sheet actually says.
    """
    variants = {title, f"{title} ", title.replace(" Cheat Sheet", "")}
    if number is not None:
        variants |= {f"{number}. {title}", f"{number}. {title.replace(' Cheat Sheet', '')}"}
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if s.isdigit() and len(s) <= 3:
            continue
        if any(s == v.strip() or s.rstrip(".") == v.strip().rstrip(".") for v in variants):
            continue
        if re.match(r"^Last revision \(mm/dd/yy\):", s):
            continue
        kept.append(s)
    return "\n".join(kept)


def _parse_toc(toc_text: str) -> list[tuple[int, str, int]]:
    """Pull (number, title, start_page) for each top-level cheat sheet.

    TOC lines look like:
        20 SQL Injection Prevention Cheat Sheet . . . . . . 139
        20.1. Introduction . . . . . . . . . . . . . . . . . . . 139
    Only the top-level `N Title` entries matter, and the page number appears
    on its own line after the heading/subheading block.
    """
    lines = toc_text.splitlines()
    cleaned = [re.sub(r"\s*\.\s*\.\s*\.\s*\d+\s*$", "", ln).strip() for ln in lines]

    entries: list[tuple[int, str, int]] = []
    pending: tuple[int, str] | None = None
    for line in cleaned:
        if not line or _NOT_A_SHEET.match(line):
            continue
        # `13 .NET Security Cheat Sheet` -- the leading dot is part of the name.
        m = re.match(r"^(\d{1,3})\s+(\.?[A-Z(].+)$", line)
        if m and not re.match(r"^\d+\.\d", line):
            pending = (int(m.group(1)), m.group(2).strip())
            continue
        if pending and re.fullmatch(r"\d{1,3}", line):
            entries.append((pending[0], pending[1], int(line)))
            pending = None
    return entries


def build_index(pdf_path: str | Path = DEFAULT_PDF) -> dict[int, CheatSheet]:
    """Map cheat-sheet number -> CheatSheet with resolved page ranges."""
    pages = extract_pages(pdf_path)
    toc = "\n".join(pages[TOC_PAGES[0] - 1: TOC_PAGES[1]])
    entries = _parse_toc(toc)
    if not entries:
        raise ReferenceUnavailable("could not parse the OWASP table of contents")

    index: dict[int, CheatSheet] = {}
    for i, (num, title, start) in enumerate(entries):
        # A sheet runs until the next sheet begins.
        end = entries[i + 1][2] - 1 if i + 1 < len(entries) else len(pages)
        end = max(end, start)
        index[num] = CheatSheet(
            number=num, title=title, start_page=start, end_page=end,
            _pages=pages[start - 1: end],
        )
    return index


# --------------------------------------------------------------------------
# vulnerability taxonomy
# --------------------------------------------------------------------------
#
# Maps a vulnerability type to the cheat sheets that actually exist in this PDF.
# Deliberately conservative: a type with no genuine sheet is left unmapped so
# `lookup()` returns None rather than attaching an unrelated reference.
#
# Keywords come in two strengths:
#   strong - a distinctive vulnerability name or signature term. One hit is
#            enough on its own, because "sql injection" or "csrf" is diagnostic.
#   weak   - a generic supporting term ("password", "certificate"). These need
#            corroboration so they cannot classify a finding on their own.
#
# This split exists because a single flat threshold was wrong in both
# directions: it missed "SQL injection in query" (one strong, unambiguous
# phrase) while being one word away from matching generic terms.
#
# `tool_ids` are static-analyzer identifiers that pin a finding to a type
# outright (Bandit B608 is always SQL injection), weighted highest.

TAXONOMY: dict[str, dict] = {
    "sql_injection": {
        "sheets": [20, 16],
        "strong": ["sql injection", "sqli", "prepared statement",
                   "parameterized quer", "parameterised quer"],
        "weak": ["sql query", "query string", "hardcoded_sql", "string concatenation"],
        "tool_ids": {"B608", "S608"},
    },
    "xss": {
        "sheets": [25, 7],
        "strong": ["xss", "cross site scripting", "cross-site scripting",
                   "html injection", "script injection", "dom based xss"],
        "weak": ["output encoding", "unescaped", "innerhtml", "reflected"],
        "tool_ids": {"B702"},
    },
    "csrf": {
        "sheets": [5],
        "strong": ["csrf", "cross site request forgery",
                   "cross-site request forgery", "xsrf", "anti-forgery",
                   "anti-forgery token"],
        "weak": ["same-site", "state changing", "token"],
        "tool_ids": set(),
    },
    "weak_cryptography": {
        "sheets": [6, 43],
        "strong": ["cryptographic storage", "insecure cipher", "weak cipher",
                   "hardcoded encryption key", "hardcoded key", "ecb mode",
                   "unauthenticated encryption"],
        "weak": ["encryption key", "key management", "cipher", "plaintext",
                 "salt", "iv ", "aes", "des"],
        "tool_ids": set(),
    },
    "password_storage": {
        "sheets": [14],
        "strong": ["password storage", "password hash", "insecure password hash",
                   "weak hash", "md5", "sha1 password"],
        "weak": ["bcrypt", "argon2", "pbkdf2", "scrypt", "work factor",
                 "credential storage", "salt"],
        "tool_ids": set(),
    },
    "authentication": {
        "sheets": [1],
        "strong": ["authentication bypass", "missing authentication",
                   "no authentication", "hardcoded credential",
                   "weak password", "credential stuffing"],
        "weak": ["authentication", "login", "credential", "password",
                 "multi-factor", "mfa", "brute force", "session id"],
        "tool_ids": set(),
    },
    "session_management": {
        "sheets": [19],
        "strong": ["session fixation", "session management", "insecure session",
                   "predictable session"],
        "weak": ["session token", "session id", "httponly", "secure cookie",
                 "session"],
        "tool_ids": set(),
    },
    "input_validation": {
        "sheets": [10],
        "strong": ["input validation", "improper input validation",
                   "allowlist", "whitelist"],
        "weak": ["sanitize", "sanitise", "validate input", "unexpected input"],
        "tool_ids": set(),
    },
    "path_traversal": {
        # NOTE: this PDF has no path-traversal cheat sheet. Input Validation
        # (#10) is the only candidate, and the sheet-side evidence test below
        # still has to pass for anything to be cited.
        "sheets": [10],
        "strong": ["path traversal", "directory traversal", "zip slip",
                   "arbitrary file read", "arbitrary file write"],
        "weak": ["file path", "filename", "../", "canonical", "file name"],
        "tool_ids": set(),
    },
    "command_injection": {
        # No OS Command Injection sheet in this PDF.
        "sheets": [31],
        "strong": ["command injection", "shell injection", "os command injection",
                   "shell=true"],
        "weak": ["subprocess", "system()", "shell", "command execution", "popen"],
        "tool_ids": {"B602", "B605", "S602", "S605", "S607"},
    },
    "insecure_deserialization": {
        # No Deserialization sheet in this PDF. Listed so the type is
        # recognised (and therefore deliberately not cited) rather than being
        # silently misfiled under something else.
        "sheets": [],
        "strong": ["insecure deserialization", "insecure deserialisation",
                   "unsafe deserialization", "pickle load", "yaml.load"],
        "weak": ["deserialization", "deserialisation", "pickle", "marshal"],
        "tool_ids": set(),
    },
    "logging": {
        "sheets": [12],
        "strong": ["log injection", "log forging", "crlf injection",
                   "audit log"],
        "weak": ["logging", "log entry", "log file"],
        "tool_ids": {"B113"},
    },
    "transport": {
        "sheets": [21],
        "strong": ["transport layer protection", "cleartext transmission",
                   "missing tls", "disabled certificate validation",
                   "hostname verification"],
        "weak": ["tls", "ssl", "certificate", "https", "cipher suite"],
        "tool_ids": set(),
    },
    "idor": {
        "sheets": [44],
        "strong": ["insecure direct object reference", "idor",
                   "missing authorization check", "broken access control"],
        "weak": ["authorization", "access control", "object reference"],
        "tool_ids": set(),
    },
    "redirect": {
        "sheets": [22],
        "strong": ["unvalidated redirect", "open redirect", "unvalidated forward"],
        "weak": ["redirect", "forward"],
        "tool_ids": set(),
    },
    "clickjacking": {
        "sheets": [3],
        "strong": ["clickjacking", "x-frame-options", "frame-ancestors",
                   "ui redress"],
        "weak": ["iframe", "overlay"],
        "tool_ids": set(),
    },
    "content_security_policy": {
        "sheets": [45],
        "strong": ["content security policy", "frame-ancestors"],
        "weak": ["csp", "script-src"],
        "tool_ids": set(),
    },
    "secrets_exposure": {
        "sheets": [10, 37],
        "strong": ["hardcoded password", "hardcoded secret", "api key in source",
                   "secret in source", "credential in source"],
        "weak": ["hardcoded", "secret", "api key", "sensitive data"],
        "tool_ids": {"B105", "B106", "B107", "S105", "S106", "S107"},
    },
    "xss_filter_evasion": {
        "sheets": [27],
        "strong": ["xss filter", "filter evasion", "blacklist filter",
                   "blacklist-based"],
        "weak": ["evasion", "payload", "bypass"],
        "tool_ids": set(),
    },
}

# Words too common to carry signal when scoring.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "be", "been", "for", "on", "with", "that", "this", "it", "as", "at", "by",
    "from", "if", "not", "no", "but", "can", "may", "will", "should", "must",
    "use", "using", "used", "do", "does", "all", "any", "when", "which", "than",
    "then", "there", "their", "they", "you", "your", "we", "our", "has", "have",
    "had", "been", "also", "more", "most", "other", "such", "only", "own", "same",
    "so", "too", "very", "just", "about", "into", "over", "under", "out", "up",
}


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9_]{2,}", (text or "").lower())
            if w not in _STOPWORDS]


# Single words are matched on word boundaries; multi-word phrases are matched as
# substrings. Without this, short weak keywords match inside unrelated words --
# "des" was found inside "description", "designed" and "loaded", which
# inflated weak_cryptography's apparent support and let tangential sheets look
# like good matches.
_MULTIWORD = re.compile(r"\s")


def _term_present(keyword: str, low_text: str) -> bool:
    """True when `keyword` occurs in already-lowercased `low_text`."""
    kw = keyword.strip()
    if _MULTIWORD.search(kw):
        return kw in low_text
    return re.search(r"\b" + re.escape(kw) + r"\b", low_text) is not None


def _finding_text(finding) -> str:
    """All free text on a finding, used for both classification and scoring."""
    parts = [
        getattr(finding, "problem", "") or "",
        getattr(finding, "explanation", "") or "",
        getattr(finding, "suggested_fix", "") or "",
    ]
    return " ".join(parts)


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------

@dataclass
class SecurityReference:
    """The `reference` payload attached to a finding. Every field is
    traceable to a specific place in the PDF."""

    cheat_sheet: str
    section: str | None
    summary: str
    url: str | None
    pdf_page: int
    url_source: str = "derived:cheatsheetseries.owasp.org"

    def to_dict(self) -> dict:
        return {
            "cheat_sheet": self.cheat_sheet,
            "section": self.section,
            "summary": self.summary,
            "url": self.url,
            "pdf_page": self.pdf_page,
            "url_source": self.url_source,
        }


# A tool id is unambiguous, so it satisfies the evidence bar on its own.
_TOOL_ID_SCORE = 4
# One strong (distinctive) keyword is sufficient; otherwise corroborate.
_STRONG_SCORE = 2
_WEAK_SCORE = 1
# Weak terms need this much corroboration before they classify anything.
MIN_WEAK_HITS = 3

# The sheet-side evidence bar is deliberately lower than the finding-side bar
# (`_STRONG_SCORE`). A whole sheet about Authentication obviously discusses
# "authentication"/"login"/"credential", so generic terms are real evidence
# there, whereas a one-sentence finding needs a distinctive term.
# Measured scores in this PDF: every tangential or unsupported pairing scores 0
# (path_traversal->Input Validation, command_injection->Virtual Patching,
# secrets_exposure->both candidates), while the correct Authentication sheet
# scores 1. So a bar of 1 admits only genuine matches.
MIN_SHEET_SCORE = 1
SUMMARY_MAX_CHARS = 700


def _keyword_score(text: str, spec: dict, tool_ref: str | None = None) -> int:
    """Score how strongly `text` matches one taxonomy entry.

    Combines distinctive terms (strong), generic corroborating terms (weak) and
    an exact static-analyzer id, which is the most reliable signal available.
    """
    low = text.lower()
    score = 0
    if any(_term_present(kw, low) for kw in spec["strong"]):
        score += _STRONG_SCORE
    if sum(1 for kw in spec["weak"] if _term_present(kw, low)) >= MIN_WEAK_HITS:
        score += _WEAK_SCORE
    if tool_ref and tool_ref.upper() in spec["tool_ids"]:
        score += _TOOL_ID_SCORE
    return score


class OWASPReferenceIndex:
    """Loads the PDF once, then answers reference lookups deterministically."""

    def __init__(self, pdf_path: str | Path = DEFAULT_PDF,
                 cache_path: str | Path | None = None):
        self.pdf_path = Path(pdf_path)
        self._cache_path = cache_path
        self._sheets: dict[int, CheatSheet] = {}
        self._lock = threading.Lock()
        self._loaded = False
        self.load_error: str | None = None
        # title lookup so a category name can name a sheet directly
        self._by_title: dict[str, CheatSheet] = {}

    # -- loading ---------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                sheets = build_index(self.pdf_path)
            except ReferenceUnavailable as exc:
                self.load_error = str(exc)
                return
            self._sheets = sheets
            self._by_title = {s.title.lower(): s for s in sheets.values()}
            self._loaded = True

    @property
    def available(self) -> bool:
        self.load()
        return self._loaded

    def sheet_count(self) -> int:
        self.load()
        return len(self._sheets)

    def sheet_titles(self) -> list[str]:
        self.load()
        return [s.title for s in sorted(self._sheets.values(), key=lambda s: s.number)]

    def sheet_by_title(self, title: str) -> CheatSheet | None:
        self.load()
        return self._by_title.get(title.strip().lower())

    # -- classification --------------------------------------------------
    def classify(self, finding) -> tuple[str | None, int]:
        """Return (vulnerability_type, score) for a finding, or (None, 0)."""
        text = _finding_text(finding)
        ref = (getattr(finding, "tool_ref", None) or "").upper()

        best_type: str | None = None
        best_score = 0
        for vtype, spec in TAXONOMY.items():
            score = _keyword_score(text, spec, ref)
            if score > best_score:
                best_type, best_score = vtype, score
        if best_type is None or best_score < _STRONG_SCORE:
            return None, 0
        return best_type, best_score

    # -- lookup ----------------------------------------------------------
    def lookup(self, finding) -> SecurityReference | None:
        """Best OWASP reference for a Security finding, or None.

        Returns None -- leaving the `reference` field off the finding -- when
        the category is not Security, the PDF is unavailable, no type matches,
        no cheat sheet exists for the type, or no candidate sheet's own text
        supports the match.
        """
        # Rule 3: only Security findings are ever enriched.
        if (getattr(finding, "category", "") or "").lower() != "security":
            return None

        self.load()
        if not self._loaded:
            return None

        vtype, score = self.classify(finding)
        if vtype is None:
            return None

        spec = TAXONOMY[vtype]
        if not spec["sheets"]:
            # Recognised vulnerability, but this PDF carries no sheet for it.
            return None

        text = _finding_text(finding)
        finding_tokens = set(_tokens(text))

        best: tuple[int, CheatSheet, list[tuple[str, str]]] | None = None
        for num in spec["sheets"]:
            sheet = self._sheets.get(num)
            if sheet is None:
                continue  # not present in this PDF
            sections = sheet.section_lines()
            if not sections:
                continue
            # The sheet's own text must support the classification, otherwise a
            # summary pulled from it would be about the wrong subject.
            sheet_score = _keyword_score(sheet.text, spec)
            if sheet_score < MIN_SHEET_SCORE:
                continue
            # Prefer the sheet whose content overlaps the finding most.
            overlap = len(finding_tokens & set(_tokens(sheet.text)))
            total = sheet_score * 10 + score + overlap
            if best is None or total > best[0]:
                best = (total, sheet, sections)

        if best is None:
            return None

        _, sheet, sections = best
        passage = _best_passage(sections, finding_tokens, spec)
        if not passage:
            return None
        heading, text_body = passage

        return SecurityReference(
            cheat_sheet=sheet.title,
            section=heading or None,
            summary=_condense(text_body, SUMMARY_MAX_CHARS),
            url=sheet.url,
            pdf_page=sheet.start_page,
        )


# Heading preferences. Citing "Introduction" is technically on-topic but
# useless as guidance, so those are demoted in favour of the sections that
# actually state defences and rules.
_INTRO_HEADINGS = ("introduction", "goals", "abstract", "about", "background",
                   "overview of", "purpose", "scope",
                   # Credits and bibliography are never useful as guidance.
                   "authors", "primary editors", "references",
                   "related articles", "see also")
_GUIDANCE_HEADINGS = ("defense", "defence", "prevention", "rule", "how to",
                      "countermeasure", "mitigation", "best practice",
                      "guideline", "protect", "avoid", "prevent", "secure",
                      "control", "recommend", "implement", "do not", "verify")
_INTRO_PENALTY = 6
_GUIDANCE_BONUS = 4


def _heading_score(heading: str) -> int:
    low = (heading or "").lower()
    score = 0
    if any(w in low for w in _INTRO_HEADINGS):
        score -= _INTRO_PENALTY
    if any(w in low for w in _GUIDANCE_HEADINGS):
        score += _GUIDANCE_BONUS
    return score


def _best_passage(sections: list[tuple[str, str]], finding_tokens: set[str],
                  spec: dict) -> tuple[str | None, str] | None:
    """Pick the subsection whose text best matches the finding.

    Extractive only: the returned body is verbatim PDF text, trimmed.
    """
    best: tuple[int, str, str] | None = None
    for heading, body in sections:
        body_tokens = set(_tokens(body))
        score = _keyword_score(body, spec) * 10
        score += len(finding_tokens & body_tokens)
        score += _heading_score(heading)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, heading, body)
    if best is None:
        return None
    return best[1], best[2]


def _condense(text: str, limit: int) -> str:
    """Trim verbatim PDF prose to a readable length without rewording it.

    The returned string never exceeds `limit` characters, and is always either a
    verbatim prefix of the source or that prefix followed by an ellipsis.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text

    # Only consider a sentence end that lands at or before the limit, otherwise
    # the "trim" could return far more text than requested.
    window = text[: limit + 1]
    best: int | None = None
    for stop in (". ", "! ", "? "):
        pos = window.rfind(stop)
        if pos == -1:
            continue
        end = pos + 1
        if end >= limit * 0.5:
            best = end if best is None else max(best, end)
    if best is not None:
        return window[:best].strip()

    # No usable sentence boundary: cut on a word boundary and mark the elision.
    tail = " ..."
    return text[: max(1, limit - len(tail))].rsplit(" ", 1)[0].rstrip(",;:") + tail


# --------------------------------------------------------------------------
# module-level singleton + enrichment entry point
# --------------------------------------------------------------------------

_INDEX: OWASPReferenceIndex | None = None
_INDEX_LOCK = threading.Lock()


def get_index(pdf_path: str | Path | None = None) -> OWASPReferenceIndex:
    global _INDEX
    if _INDEX is None:
        with _INDEX_LOCK:
            if _INDEX is None:
                pdf = Path(pdf_path) if pdf_path else DEFAULT_PDF
                _INDEX = OWASPReferenceIndex(pdf, cache_path=_default_cache(pdf))
    return _INDEX


def _default_cache(pdf: Path) -> Path:
    return REPO_ROOT / ".cache" / f"{pdf.stem}.txt"


def enrich_findings(findings: list, pdf_path: str | Path | None = None) -> list:
    """Attach OWASP `reference` to Security findings in place.

    Returns the same list for convenience. Non-Security findings are left
    untouched -- they never gain a `reference` attribute. Failures here are
    swallowed: OWASP enrichment is an add-on and must never fail a review.
    """
    index = get_index(pdf_path)
    if not index.available:
        return findings
    for f in findings:
        try:
            ref = index.lookup(f)
        except Exception:
            ref = None
        if ref is not None:
            f.reference = ref.to_dict()
    return findings


def reset_index() -> None:
    """Test hook: drop the cached index."""
    global _INDEX
    with _INDEX_LOCK:
        _INDEX = None
