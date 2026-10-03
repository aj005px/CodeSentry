"""Parser tests for parse_llm_output -- no model load, runs instantly."""
import sys
sys.path.insert(0, "/home/aj/Desktop/projects/untitled")
from review_core import parse_llm_output, VERDICT_ISSUES, VERDICT_CLEAN, VERDICT_UNKNOWN

CASES = [
    ("clean, trained format",
     "Verdict: NO ISSUES FOUND\nThe code is correct and needs no changes.",
     VERDICT_CLEAN, 0),

    ("single issue, trained format",
     "Verdict: ISSUES FOUND\nCategory: Security\nSeverity: High\nLine: 2\n"
     "Problem: SQL injection\nExplanation: Unsanitised input is concatenated.\n"
     "Suggested Fix: Use parameterised queries.",
     VERDICT_ISSUES, 1),

    ("multi-issue (two blocks, one verdict)",
     "Verdict: ISSUES FOUND\nCategory: Bug\nSeverity: High\nLine: 1\nProblem: IndexError\n"
     "Explanation: Deref before length check.\nSuggested Fix: Guard length.\n"
     "Category: Quality\nSeverity: Low\nLine: 4\nProblem: Shadowed name\n"
     "Explanation: Reuses v.\nSuggested Fix: Rename.",
     VERDICT_ISSUES, 2),

    ("markdown fences + bold labels",
     "```text\n**Verdict: ISSUES FOUND**\n**Category:** Bug\n**Severity:** Medium\n"
     "**Line:** 7\n**Problem:** Off by one\n**Explanation:** Loop runs to n.\n"
     "**Suggested Fix:** Use n-1.\n```",
     VERDICT_ISSUES, 1),

    ("lowercase + no space after colon",
     "verdict:issues found\ncategory:bug\nseverity:high\nline:3\nproblem:null deref\n"
     "explanation:derefs a maybe-None.\nfix:check for None first.",
     VERDICT_ISSUES, 1),

    ("preamble prose before verdict",
     "Sure! Here's my review of the snippet.\n\nVerdict: ISSUES FOUND\n"
     "Category: Bug\nSeverity: High\nLine: 1\nProblem: Bare except\n"
     "Explanation: Swallows everything.\nSuggested Fix: Catch specific exceptions.\n\n"
     "Let me know if you want more detail.",
     VERDICT_ISSUES, 1),

    ("verdict says issues, block truncated (must NOT vanish)",
     "Verdict: ISSUES FOUND\nCategory: Security\nSeverity: High",
     VERDICT_ISSUES, 1),

    ("category only, no other fields",
     "Verdict: ISSUES FOUND\nCategory: Performance",
     VERDICT_ISSUES, 1),

    ("wrapped continuation lines",
     "Verdict: ISSUES FOUND\nCategory: Bug\nSeverity: Medium\nLine: 12\n"
     "Problem: Race on shared cache\nExplanation: Two threads mutate the dict\n"
     "without a lock, so a read can observe a partially updated state.\n"
     "Suggested Fix: Guard with a mutex.",
     VERDICT_ISSUES, 1),

    ("old bare format (v1 style)",
     "No issues found.",
     VERDICT_CLEAN, 0),

    ("legacy 6-field block, no verdict line",
     "Category: Security\nSeverity: High\nLine: 4\nProblem: Hardcoded secret\n"
     "Explanation: Key is in source.\nSuggested Fix: Load from env.",
     VERDICT_ISSUES, 1),

    ("fields before any Category (must not discard)",
     "Severity: High\nLine: 9\nProblem: Use after free\nVerdict: ISSUES FOUND\n"
     "Category: Bug\nExplanation: Freed buffer is read.",
     VERDICT_ISSUES, 1),

    ("total garbage",
     "I'm sorry, I can't help with that request.",
     VERDICT_UNKNOWN, 0),

    ("empty string", "", VERDICT_UNKNOWN, 0),

    ("whitespace only", "   \n  ", VERDICT_UNKNOWN, 0),

    ("unrecognised category + severity -> normalised",
     "Verdict: ISSUES FOUND\nCategory: Robustness\nSeverity: Critical\nLine: 2\n"
     "Problem: Timeout\nExplanation: No bound.\nSuggested Fix: Add a deadline.",
     VERDICT_ISSUES, 1),

    ("line field as 'L12' / 'N/A'",
     "Verdict: ISSUES FOUND\nCategory: Bug\nSeverity: Low\nLine: L12\nProblem: A\n"
     "Explanation: B\nSuggested Fix: C",
     VERDICT_ISSUES, 1),
]

fails = 0
for name, text, want_verdict, want_n in CASES:
    try:
        v, f, ok = parse_llm_output(text)
    except Exception as e:
        print(f"FAIL  {name}: raised {type(e).__name__}: {e}")
        fails += 1
        continue
    problems = []
    if v != want_verdict:
        problems.append(f"verdict={v} want={want_verdict}")
    if len(f) != want_n:
        problems.append(f"n={len(f)} want={want_n}")
    if problems:
        print(f"FAIL  {name}: {'; '.join(problems)}")
        fails += 1
    else:
        extra = ""
        if f:
            extra = f" -> {f[0].category}/{f[0].severity}/line={f[0].line}"
        print(f"ok    {name} (parse_ok={ok}){extra}")

print()
print("=== normalisation spot-checks ===")
_, f, _ = parse_llm_output(
    "Verdict: ISSUES FOUND\nCategory: Correctness\nSeverity: minor\nLine: 4\n"
    "Problem: A\nExplanation: B\nSuggested Fix: C")
print(f"  Correctness/minor -> {f[0].category}/{f[0].severity}")
_, f, _ = parse_llm_output("Verdict: ISSUES FOUND\nLine: 5\nProblem: Only problem given")
print(f"  missing category   -> {f[0].category}/{f[0].severity}/line={f[0].line}")
_, f2, _ = parse_llm_output(
    "Verdict: ISSUES FOUND\nCategory: Bug\nSeverity: Low\nLine: 4\nProblem: A\n"
    "Explanation: B\nSuggested Fix: C\n\nCategory: Security\nSeverity: High\nLine: 9\n"
    "Problem: D\nExplanation: E\nSuggested Fix: F")
print(f"  blank-line-separated blocks still split -> {len(f2)} findings, lines "
      f"{[x.line for x in f2]}")

print()
print(f"{'ALL PARSER TESTS PASSED' if fails == 0 else str(fails) + ' FAILURES'}")
sys.exit(1 if fails else 0)
