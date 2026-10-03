#!/usr/bin/env python3
"""Checks for the vectorless OWASP cheat-sheet reference layer.

Run:  python test_security_reference.py

Requires OWASP_Cheatsheets_Book.pdf and the `pdftotext` binary. Skips (rather
than fails) if either is unavailable, so this suite stays runnable elsewhere.

What is asserted here is deliberately narrow: that the feature is honest. It
must cite a genuinely relevant sheet when one exists, and stay silent otherwise,
rather than attaching an unrelated reference to look useful.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

from review_core import Finding
from security_reference import (
    DEFAULT_PDF,
    MIN_SHEET_SCORE,
    TAXONOMY,
    get_index,
)

CHECKS: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    CHECKS.append((name, bool(cond)))


def sec(problem: str, explanation: str = "", tool_ref: str | None = None,
        severity: str = "High") -> Finding:
    return Finding(category="Security", severity=severity, problem=problem,
                   explanation=explanation, tool_ref=tool_ref)


# -- availability -----------------------------------------------------------
pdf_text = DEFAULT_PDF.exists()
if not pdf_text:
    print(f"SKIP: {DEFAULT_PDF.name} not present")
    print("SKIP: OWASP reference tests not run")
    sys.exit(0)

idx = get_index()
idx.load()
if not idx.available:
    print(f"SKIP: PDF text unavailable ({idx.load_error})")
    print("SKIP: OWASP reference tests not run")
    sys.exit(0)


# -- index integrity --------------------------------------------------------
check("PDF text extracted via pdftotext", bool(idx._sheets) and idx.load_error is None)
check("all 45 cheat sheets indexed", idx.sheet_count() == 45)
check("sheet numbers are 1..45", sorted(idx._sheets) == list(range(1, 46)))
check("every sheet has a title", all(s.title for s in idx._sheets.values()))
check("every sheet has a start page", all(s.start_page for s in idx._sheets.values()))
check("sheet pages are sane", all(1 < s.start_page < 316 for s in idx._sheets.values()))
check("index reports as available", idx.available is True)

# Four sheets in this PDF genuinely have no numbered subsections: #32 and #39
# are stubs pointing at the website, #41 is prose-only, #43 is a stub. They are
# never cited, so the meaningful invariant is that any sheet which *could* be
# cited can also be summarised.
STUB_SHEETS = {32, 39, 41, 43}
check("only the known stubs lack sections",
      {n for n, s in idx._sheets.items() if not s.section_lines()} == STUB_SHEETS)

from security_reference import _keyword_score  # noqa: E402

for _vtype, _spec in TAXONOMY.items():
    for _num in _spec["sheets"]:
        _sheet = idx._sheets.get(_num)
        if _sheet is None:
            continue
        _citable = _keyword_score(_sheet.text, _spec) >= MIN_SHEET_SCORE
        if _citable:
            check(f"citable sheet #{_num} has extractable sections",
                  bool(_sheet.section_lines()))


# -- supported categories must be cited correctly ---------------------------
EXPECT = {
    "sql_injection":    ("sql injection in query built by concatenation", "", "SQL Injection Prevention"),
    "xss":              ("reflected cross site scripting from unescaped input", "", "XSS"),
    "csrf":             ("missing CSRF token on a state changing form", "", "Cross-Site Request Forgery"),
    "weak_cryptography": ("weak cipher using ECB mode to encrypt", "", "Cryptographic Storage"),
    "password_storage": ("weak password hash for a stored credential", "", "Password Storage"),
    "authentication":   ("authentication bypass, credential check skipped", "", "Authentication"),
    "session_management": ("session fixation through a predictable session id", "", "Session Management"),
    "redirect":         ("unvalidated redirect to a user supplied url", "", "Unvalidated Redirects"),
    "clickjacking":     ("missing frame-ancestors permits clickjacking", "", "Clickjacking"),
    "logging":          ("log injection through CRLF in a username", "", "Logging"),
    "idor":             ("insecure direct object reference on a user id", "", "Insecure Direct Object Reference"),
}
for vtype, (problem, expl, expect_sheet) in EXPECT.items():
    ref = idx.lookup(sec(problem, expl))
    check(f"{vtype}: cited", ref is not None)
    if ref:
        check(f"{vtype}: correct sheet ({ref.cheat_sheet})",
              expect_sheet.lower() in ref.cheat_sheet.lower())
        check(f"{vtype}: has summary", len(ref.summary) > 80)
        check(f"{vtype}: summary bounded", len(ref.summary) <= 800)
        check(f"{vtype}: has canonical url",
              ref.url.startswith("https://cheatsheetseries.owasp.org/cheatsheets/")
              and ref.url.endswith(".html"))
        check(f"{vtype}: has a real page", 1 < ref.pdf_page < 316)

# -- the section chosen must be guidance, not an introduction ---------------
for vtype in ("sql_injection", "xss", "authentication"):
    ref = idx.lookup(sec(*EXPECT[vtype][:2]))
    check(f"{vtype}: avoids Introduction section",
          ref is not None and "introduction" not in (ref.section or "").lower())


# -- unsupported categories must stay silent (honesty over coverage) -------
UNSUPPORTED = [
    ("command injection", "OS command injection through a shell", "subprocess shell=True", "B605"),
    ("path traversal", "path traversal, user input builds the file path", "filename joined without canonicalization", None),
    ("insecure deserialization", "insecure deserialization of untrusted data", "pickle load", None),
    ("hardcoded secret", "hardcoded API key committed to source", "api key in source", None),
]
for name, problem, expl, ref_id in UNSUPPORTED:
    got = idx.lookup(sec(problem, expl, tool_ref=ref_id))
    check(f"no sheet exists for {name}: stays silent", got is None)

# Types with no sheet mapped must never cite anything.
for vtype in ("insecure_deserialization",):
    check(f"{vtype} maps to no sheets", TAXONOMY[vtype]["sheets"] == [])


# -- non-Security findings are never enriched ------------------------------
for cat in ("Bug", "Performance", "Quality", "Documentation", "Testing", "Style"):
    f = Finding(category=cat, severity="High",
                problem="SQL injection in query construction",
                explanation="sql injection via string concatenation")
    check(f"{cat} finding never gets a reference", idx.lookup(f) is None)

# -- degenerate inputs ------------------------------------------------------
check("empty Security finding stays silent",
      idx.lookup(Finding(category="Security", severity="High", problem="")) is None)
check("whitespace-only finding stays silent",
      idx.lookup(sec("   ", "  \n ")) is None)
check("case-insensitive category gate",
      idx.lookup(Finding(category="security", severity="High",
                         problem="sql injection")) is not None)
check("unknown category stays silent",
      idx.lookup(Finding(category="Nonsense", severity="High",
                         problem="sql injection")) is None)


# -- tool ids are authoritative --------------------------------------------
check("B608 pins sql injection",
      idx.classify(sec("hardcoded query construction", "no text match at all",
                       tool_ref="B608"))[0] == "sql_injection")
check("B605 pins command injection",
      idx.classify(sec("x", "y", tool_ref="B605"))[0] == "command_injection")
check("B113 pins logging",
      idx.classify(sec("x", "y", tool_ref="B113"))[0] == "logging")
check("B105 pins secrets exposure",
      idx.classify(sec("x", "y", tool_ref="B105"))[0] == "secrets_exposure")


# -- single-word keywords must not match inside other words ---------------
from security_reference import _term_present  # noqa: E402

for word, decoy in [("des", "description"), ("des", "designed"), ("des", "loaded"),
                    ("mfa", "mformat"), ("tls", "tlsdownloader")]:
    check(f"'{word}' does not match inside '{decoy}'",
          not _term_present(word, decoy))
    check(f"'{word}' still matches as a whole word", _term_present(word, f"a {word} b"))
check("phrase 'sql injection' matches as substring",
      _term_present("sql injection", "possible sql injection vulnerability"))
check("multiword needs full phrase",
      not _term_present("cross site request forgery", "cross site script"))


# -- sheet-side evidence gate ----------------------------------------------
check("MIN_SHEET_SCORE is 1 (measured from PDF)", MIN_SHEET_SCORE == 1)
for vtype in ("path_traversal", "command_injection", "secrets_exposure"):
    for num in TAXONOMY[vtype]["sheets"]:
        sheet = idx._sheets.get(num)
        if sheet:
            check(f"{vtype}: sheet #{num} fails the evidence gate",
                  _keyword_score(sheet.text, TAXONOMY[vtype]) < MIN_SHEET_SCORE)


# -- summaries must be verbatim PDF text, never generated ------------------
#
# Two layers are checked separately, because each can fabricate text:
#   1. the summary must be a literal substring of the sheet text it came from
#   2. the sheet text must itself be literal PDF prose
# Comparing against raw `pdftotext` output is not sufficient on its own: the raw
# stream interleaves page numbers and running headers ("166", "22. Unvalidated
# Redirects and Forwards Cheat Sheet") into the middle of the prose, which the
# indexer deliberately removes. So the reference below is the raw PDF with that
# page furniture stripped out.
FURNITURE = re.compile(r"^\s*(?:\d{1,3}|\d{1,3}\.\s+\S.*?Cheat Sheet)\s*$")


def _de_furnished(text: str) -> str:
    return re.sub(r"\s+", " ", "\n".join(
        ln for ln in text.splitlines() if not FURNITURE.match(ln)))


raw = ""
if shutil.which("pdftotext"):
    raw = subprocess.run(["pdftotext", str(DEFAULT_PDF), "-"],
                         capture_output=True, text=True).stdout
if raw:
    flat = _de_furnished(raw)
    for vtype, (problem, expl, _s) in EXPECT.items():
        ref = idx.lookup(sec(problem, expl))
        if not ref:
            continue
        sheet = idx.sheet_by_title(ref.cheat_sheet)
        probe = re.sub(r"\s+", " ", ref.summary)[:110]
        check(f"{vtype}: summary is literal sheet text",
              probe in re.sub(r"\s+", " ", sheet.text))
        check(f"{vtype}: summary is literal PDF text", probe in flat)
else:
    print("NOTE: pdftotext unavailable at test time; skipped verbatim check")


# -- enrich_findings must be additive only ---------------------------------
from security_reference import enrich_findings  # noqa: E402

findings = [
    sec("sql injection in query", "string concatenation"),
    Finding(category="Bug", severity="Low", problem="unused import", explanation="dead code"),
    sec("OS command injection via shell", "subprocess shell=True", tool_ref="B605"),
]
snapshot = [(f.category, f.severity, f.problem, list(f.sources), f.tool_ref)
            for f in findings]
enrich_findings(findings)
check("sql injection finding enriched", findings[0].reference is not None)
check("Bug finding untouched", getattr(findings[1], "reference", None) is None)
check("command injection finding untouched",
      getattr(findings[2], "reference", None) is None)
check("enrichment changed nothing but reference",
      snapshot == [(f.category, f.severity, f.problem, list(f.sources), f.tool_ref)
                   for f in findings])
check("enrichment returns the same list", enrich_findings(findings) is findings)
check("enrichment tolerates an empty list", enrich_findings([]) == [])


# -- report -----------------------------------------------------------------
passed = sum(1 for _n, ok in CHECKS if ok)
failed = [(n, ok) for n, ok in CHECKS if not ok]
print(f"\n{passed}/{len(CHECKS)} OWASP reference checks passed")
if failed:
    print("\nFAILED:")
    for name, _ in failed:
        print(f"  - {name}")
    print("\nSOME OWASP REFERENCE CHECKS FAILED")
    sys.exit(1)
print("ALL OWASP REFERENCE CHECKS PASSED")
