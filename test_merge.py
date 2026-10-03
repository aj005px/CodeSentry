#!/usr/bin/env python
"""
Tests for the merge logic and the review_code orchestrator.

The LLM is stubbed, so this runs in under a second and needs no RAM. That is
deliberate: the interesting logic is "does a static finding survive the merge,
and does LLM agreement get credited", and neither depends on the real model.
The real adapter's output format is covered by test_parser.py.

Run: .venv/bin/python test_merge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import review_core
from review_core import (
    Finding, VERDICT_CLEAN, VERDICT_ISSUES, VERDICT_UNKNOWN,
    merge_findings, review_code,
)

failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  <- {detail}" if detail else ""))
        failures.append(label)


def stub_llm(text: str | None):
    """Replace the singleton reviewer with one that returns canned text."""
    class _Stub:
        loaded = True

        def review(self, code: str) -> str:
            if text is None:
                raise RuntimeError("simulated model failure")
            return text

    review_core._reviewer = _Stub()  # type: ignore[assignment]


def restore_llm() -> None:
    review_code.__globals__["get_llm_reviewer"].__globals__["_reviewer"] = None
    review_core._reviewer = None


SQLI = 'import sqlite3\n\ndef get_user(conn, u):\n    q = "SELECT * FROM u WHERE n = \'" + u + "\'"\n    return conn.execute(q).fetchone()\n'
CLEAN = 'def tax(amount, rate):\n    if amount < 0:\n        raise ValueError("bad")\n    return round(amount * rate, 2)\n'

# ---------------------------------------------------------------- merge rules
print("merge invariants")

b = Finding(category="Security", severity="Medium", problem="Possible SQL injection vector through string-based query construction",
            line=4, explanation="bandit text", sources=["bandit"], tool_ref="B608")
r = Finding(category="Security", severity="High", problem="Possible SQL injection vector through string-based query construction",
            line=4, explanation="ruff text", sources=["ruff"], tool_ref="S608")

m = merge_findings([b], [r])
check("bandit+ruff on same issue collapse to 1", len(m) == 1, str(len(m)))
check("merged finding keeps both sources",
      sorted(m[0].sources) == ["bandit", "ruff"], str(m[0].sources))
check("analyzer text wins over duplicate", m[0].explanation == "bandit text",
      str(m[0].explanation))

print("\nstatic findings are never dropped")
# LLM reports something completely unrelated.
other = Finding(category="Bug", severity="Low", problem="Variable v shadows parameter",
                line=2, sources=["llm"])
m = merge_findings([b], [other])
check("static finding survives an unrelated LLM finding",
      any(f.sources == ["bandit"] and f.tool_ref == "B608" for f in m),
      str([(f.tool_ref, f.sources) for f in m]))
check("its sources are NOT credited to the llm",
      all("llm" not in f.sources for f in m if f.tool_ref == "B608"),
      str([f.sources for f in m if f.tool_ref == "B608"]))
check("the unrelated LLM finding is appended in its own right", len(m) == 2, str(len(m)))

print("\nLLM silent about a static finding: nothing lost")
m = merge_findings([b], [])
check("static-only merge keeps the finding", len(m) == 1, str(len(m)))
check("sources still just bandit", m[0].sources == ["bandit"], str(m[0].sources))

print("\nLLM agreeing with a static finding corroborates it")
agree = Finding(category="Security", severity="High", problem="SQL injection",
                line=4, explanation="llm text", suggested_fix="Use params",
                sources=["llm"])
m = merge_findings([b], [agree])
check("single corroborated finding", len(m) == 1, str(len(m)))
check("sources == bandit+llm", sorted(m[0].sources) == ["bandit", "llm"], str(m[0].sources))
check("static explanation kept", m[0].explanation == "bandit text", str(m[0].explanation))
check("LLM fix backfills empty static fix", m[0].suggested_fix == "Use params",
      str(m[0].suggested_fix))

print("\nunrelated LLM finding is appended, not merged")
unrel = Finding(category="Performance", severity="Medium", problem="Nested loop is O(n^2)",
                line=1, sources=["llm"])
m = merge_findings([b], [unrel])
check("both findings present", len(m) == 2, str(len(m)))
check("llm-only finding kept as llm", m[0].sources == ["llm"] or m[1].sources == ["llm"],
      str([x.sources for x in m]))
check("High severity sorts first",
      [x.severity for x in m] == sorted([x.severity for x in m],
                                        key=lambda s: {"High": 0, "Medium": 1, "Low": 2}[s]),
      str([x.severity for x in m]))

print("\ndifferent categories never merge")
diffcat = Finding(category="Bug", severity="High", problem="SQL injection vector",
                  line=4, sources=["llm"])
m = merge_findings([b], [diffcat])
check("Security != Bug so kept separate", len(m) == 2, str(len(m)))

print("\nfar-apart lines never merge")
farl = Finding(category="Security", severity="High", problem="SQL injection", line=99,
               sources=["llm"])
m = merge_findings([b], [farl])
check("line 4 vs 99 kept separate", len(m) == 2, str(len(m)))

# ------------------------------------------------------------------ the verdict
print("\nverdict is driven by merged evidence, not the LLM alone")
stub_llm("Verdict: NO ISSUES FOUND\nThe code is correct.")
_original_static = review_core.run_static_analysis
review_core.run_static_analysis = lambda *args, **kwargs: (
    [Finding(category="Security", severity="High",
             problem="SQL injection", line=4, sources=["bandit"],
             tool_ref="B608")],
    [], []
)
r = review_code(SQLI, language="python")
check("static hit overrides a clean LLM verdict", r.verdict == VERDICT_ISSUES, r.verdict)
check("findings still present", len(r.findings) >= 1, str(len(r.findings)))
review_core.run_static_analysis = _original_static

stub_llm("Verdict: ISSUES FOUND\nCategory: Bug\nSeverity: High\nLine: 1\n"
         "Problem: Unused import\nExplanation: sqlite3 never used.\n"
         "Suggested Fix: Remove it.")
r = review_code(CLEAN, language="python")
check("LLM-only issue surfaces even with no static hits",
      r.verdict == VERDICT_ISSUES and len(r.findings) == 1,
      f"{r.verdict} {[x.problem for x in r.findings]}")
check("it is attributed to the llm", r.findings[0].sources == ["llm"],
      str(r.findings[0].sources))

print("\nclean code, clean model")
stub_llm("Verdict: NO ISSUES FOUND\nReviewed the logic; nothing is wrong here.")
r = review_code(CLEAN, language="python")
check("verdict no_issues_found", r.verdict == VERDICT_CLEAN, r.verdict)
check("zero findings", len(r.findings) == 0, str(len(r.findings)))

print("\nLLM crash must not lose static results")
stub_llm(None)
_original_static = review_core.run_static_analysis
review_core.run_static_analysis = lambda *args, **kwargs: (
    [Finding(category="Security", severity="High",
             problem="SQL injection", line=4, sources=["bandit"],
             tool_ref="B608")],
    [], []
)
r = review_code(SQLI, language="python")
check("static findings survive a model failure", len(r.findings) >= 1, str(len(r.findings)))
check("verdict still issues_found", r.verdict == VERDICT_ISSUES, r.verdict)
check("error is recorded", any("llm" in e for e in r.errors), str(r.errors))
review_core.run_static_analysis = _original_static

print("\nunparseable LLM output is flagged, not silently trusted")
stub_llm("I'm sorry, I can't do that.")
r = review_code(CLEAN, language="python")
check("llm_parse_ok is False", r.llm_parse_ok is False)
check("verdict is unknown, not a false clean bill",
      r.verdict == VERDICT_UNKNOWN, r.verdict)

print("\nnon-python skips analyzers")
stub_llm("Verdict: ISSUES FOUND\nCategory: Bug\nSeverity: Medium\nLine: 3\n"
         "Problem: forEach does not await\nExplanation: async callback returns early.\n"
         "Suggested Fix: Use Promise.all with map.")
r = review_code("arr.forEach(async (x) => { await f(x); });", language="javascript")
check("bandit skipped", any(s.name == "bandit" for s in r.skipped), str(r.skipped))
check("ruff skipped", any(s.name == "ruff" for s in r.skipped), str(r.skipped))
check("llm finding survives", any("llm" in f.sources for f in r.findings),
      str([f.sources for f in r.findings]))

print("\nempty / oversized input")
r = review_code("", language="python")
check("empty input is an error", bool(r.errors), str(r.errors))
r = review_code("x = 1\n" * 30000, language="python")
check("oversized input rejected", any("too large" in e for e in r.errors), str(r.errors))

restore_llm()

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print("ALL MERGE CHECKS PASSED")
