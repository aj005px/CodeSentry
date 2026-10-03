#!/usr/bin/env python
"""
Combined code review: Bandit + Ruff + fine-tuned LLM, merged into one report.

Usage
-----
    # review a snippet on stdin
    cat snippet.py | .venv/bin/python combined_review.py --language python

    # review a file
    .venv/bin/python combined_review.py --file snippet.py --language python

    # skip the LLM (instant, no model load) for a static-only pass
    .venv/bin/python combined_review.py --file snippet.py --no-llm

    # machine-readable output, same shape the FastAPI wrapper returns
    .venv/bin/python combined_review.py --file snippet.py --json

The merge rule is that static findings are never dropped: Bandit and Ruff
results always appear, and the LLM supplements (and corroborates) them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from review_core import (  # noqa: E402
    VERDICT_CLEAN, VERDICT_ISSUES, VERDICT_UNKNOWN, review_code,
)

SEVERITY_MARK = {"High": "[HIGH]", "Medium": "[MED]", "Low": "[LOW]"}


def render_text(result) -> str:
    lines = []
    verdict_label = {
        VERDICT_ISSUES: "ISSUES FOUND",
        VERDICT_CLEAN: "NO ISSUES FOUND",
        VERDICT_UNKNOWN: "UNKNOWN",
    }[result.verdict]
    lines.append("=" * 72)
    lines.append(f"  VERDICT: {verdict_label}   (language: {result.language})")
    lines.append("=" * 72)

    if not result.findings:
        lines.append("")
        lines.append("No issues found by any analyzer.")
    else:
        lines.append("")
        lines.append(f"{len(result.findings)} finding(s):")
        for i, f in enumerate(result.findings, 1):
            mark = SEVERITY_MARK.get(f.severity, "[?]")
            src = "+".join(f.sources)
            corroborated = "  <- corroborated" if len(f.sources) > 1 else ""
            lines.append("")
            lines.append(f"{i}. {mark} {f.category}"
                         + (f"  line {f.line}" if f.line else "  line ?")
                         + f"  [{src}]{corroborated}")
            lines.append(f"   Problem: {f.problem}")
            if f.explanation and f.explanation != f.problem:
                lines.append(f"   Why:     {f.explanation}")
            if f.suggested_fix:
                lines.append(f"   Fix:     {f.suggested_fix}")
            ref = f.tool_ref or ""
            conf = f" (confidence {f.confidence})" if f.confidence else ""
            if ref:
                lines.append(f"   Ref:     {ref}{conf}")

    if result.skipped:
        lines.append("")
        lines.append("Skipped analyzers:")
        for s in result.skipped:
            lines.append(f"  - {s.name}: {s.reason}")

    if result.errors:
        lines.append("")
        lines.append("Errors (these never suppress other results):")
        for e in result.errors:
            lines.append(f"  ! {e}")

    if result.llm_raw is not None and not result.llm_parse_ok:
        lines.append("")
        lines.append("! LLM output could not be parsed; static results above still stand.")
        lines.append(f"  raw: {result.llm_raw[:400]}")

    if result.timings:
        pretty = ", ".join(f"{k}={v:.1f}s" for k, v in result.timings.items())
        lines.append("")
        lines.append(f"timings: {pretty}")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="path to a file to review")
    src.add_argument("--stdin", action="store_true", help="read the snippet from stdin")
    ap.add_argument("--language", default=None,
                    help="python|javascript|c|cpp|go|... or 'auto' (default: auto-detect)")
    ap.add_argument("--no-llm", action="store_true",
                    help="static analysis only; skips the model load entirely")
    ap.add_argument("--no-bandit", action="store_true")
    ap.add_argument("--no-ruff", action="store_true")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit the API response shape as JSON")
    args = ap.parse_args()

    if args.file:
        code = Path(args.file).read_text()
        language = args.language
        if language is None:
            suffix = Path(args.file).suffix.lower().lstrip(".")
            language = suffix if suffix else "auto"
    else:
        code = sys.stdin.read()
        language = args.language or "auto"

    result = review_code(
        code, language=language,
        use_llm=not args.no_llm,
        use_bandit=not args.no_bandit,
        use_ruff=not args.no_ruff,
    )

    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(render_text(result))

    # Non-zero when the input could not be reviewed at all, so this is usable
    # in a CI pipeline. A clean review exits 0; findings exit 1.
    if result.errors and result.verdict == VERDICT_UNKNOWN:
        return 2
    return 1 if result.verdict == VERDICT_ISSUES else 0


if __name__ == "__main__":
    raise SystemExit(main())
