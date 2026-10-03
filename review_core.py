"""
Core review engine: static analysis (Bandit, Ruff) + fine-tuned LLM review.

Design
------
The static analyzers and the LLM reviewer are deliberately independent. Each
one returns `list[Finding]` objects and knows nothing about the other. The
merge step is a separate, single function with one hard rule:

    a finding produced by a static analyzer is NEVER dropped.

If the LLM independently reports the same problem, the two are merged into one
entry carrying both provenances. If the LLM stays silent, the static finding
still appears. That way a static tool's output cannot be lost to a model
failure, a hallucination, or a retrain that changes the model's behaviour.

The model is loaded lazily and exactly once (see `get_llm_reviewer`), because
loading costs ~1-2 minutes and ~7 GB of RAM on CPU. Callers must not construct
`LLMReviewer` directly in a request path.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent
BASE_MODEL_PATH = str(REPO_ROOT / "models" / "qwen2.5-coder-3b")
ADAPTER_PATH = str(REPO_ROOT / "lora-code-reviewer-v2")
VENV_BIN = REPO_ROOT / ".venv" / "bin"

# The prompt the adapter was fine-tuned on. Imported rather than copied so the
# inference prompt can never drift from the training prompt.
sys.path.insert(0, str(REPO_ROOT))
from dataset_project.build_dataset import SYSTEM_PROMPT  # noqa: E402

# --- output schema ----------------------------------------------------------

CATEGORIES = ("Bug", "Security", "Performance", "Quality")
SEVERITIES = ("Low", "Medium", "High")

VERDICT_ISSUES = "issues_found"
VERDICT_CLEAN = "no_issues_found"
VERDICT_UNKNOWN = "unknown"

# Anything longer than this is not a "snippet"; refuse rather than hang the CPU
# on a 4000-line paste.
MAX_CODE_CHARS = 40_000


@dataclass
class Finding:
    """One review finding, normalised onto the fine-tuned model's schema."""

    category: str
    severity: str
    problem: str
    line: int | None = None
    explanation: str | None = None
    suggested_fix: str | None = None
    sources: list[str] = field(default_factory=list)
    tool_ref: str | None = None  # e.g. "B608" or "S608"
    confidence: str | None = None  # static analyzers only
    # OWASP guidance for Security findings, attached after the merge by
    # `security_reference.enrich_findings()`. Stays None for non-Security
    # findings and when the PDF has nothing relevant. Deliberately not part of
    # the merge: retrieval must not influence which findings exist or how they
    # are combined.
    reference: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkippedAnalyzer:
    name: str
    reason: str


@dataclass
class ReviewResult:
    verdict: str
    findings: list[Finding]
    language: str
    llm_raw: str | None = None
    llm_parse_ok: bool = True
    llm_ran: bool = False
    skipped: list[SkippedAnalyzer] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "language": self.language,
            "findings": [f.to_dict() for f in self.findings],
            "summary": {
                "total": len(self.findings),
                "by_severity": _tally(self.findings, "severity"),
                "by_category": _tally(self.findings, "category"),
                "by_source": _tally(self.findings, "sources"),
                "corroborated": sum(1 for f in self.findings if len(f.sources) > 1),
            },
            "llm": {
                # `ran` is explicit so a consumer can tell "the model ran and
                # parsed cleanly" from "the model never ran" -- with the LLM
                # disabled, parse_ok is vacuously true and would mislead.
                "ran": self.llm_ran,
                "parse_ok": self.llm_parse_ok,
                "raw": self.llm_raw,
            },
            "skipped": [{"name": s.name, "reason": s.reason} for s in self.skipped],
            "errors": self.errors,
            "timings_ms": {k: round(v * 1000) for k, v in self.timings.items()},
        }


def _tally(findings: Iterable[Finding], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        val = getattr(f, attr)
        if isinstance(val, list):
            for v in val:
                out[v] = out.get(v, 0) + 1
        elif val is not None:
            out[val] = out.get(val, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --- language handling ------------------------------------------------------

_PY_ALIASES = {"python", "py", "python3"}
_LANG_ALIASES = {
    "javascript": "javascript", "js": "javascript", "node": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "c": "c", "cpp": "cpp", "c++": "cpp", "cxx": "cpp",
    "go": "go", "golang": "go", "rust": "rust", "java": "java",
    "ruby": "ruby", "rb": "ruby", "php": "php", "shell": "shell",
    "bash": "shell", "sh": "shell", "sql": "sql",
}


def normalize_language(language: str | None, code: str | None = None) -> str:
    """Normalise a language tag. Falls back to a light heuristic if 'auto'."""
    if language:
        lang = language.strip().lower()
        if lang in ("auto", "detect", ""):
            return _sniff_language(code or "")
        if lang in _PY_ALIASES:
            return "python"
        return _LANG_ALIASES.get(lang, lang)
    return _sniff_language(code or "")


def _sniff_language(code: str) -> str:
    """Deliberately conservative: only claim python when it clearly is.

    Guessing wrong here means either running a python analyzer on non-python
    (noisy, misleading) or skipping it when we could have helped, so we only
    fire on strong signals.
    """
    head = code.lstrip()[:4000]
    if re.search(r"^\s*(import\s+\w+|from\s+[\w.]+\s+import\s|def\s+\w+\s*\(|class\s+\w+\s*[\(:])",
                 head, re.MULTILINE):
        return "python"
    if re.search(r"^\s*(#include\s*[<\"]|function\s+\w+\s*\(|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=)",
                 head, re.MULTILINE):
        return "javascript"
    return "unknown"


# --- LLM output parsing -----------------------------------------------------
#
# The model was trained on a rigid format but at temperature it can drift, and
# a base model can add markdown fences or preamble. The parser's contract is:
#
#   * never raise on unexpected input
#   * never silently discard a partially-complete finding -- a block with only
#     a Category is still reported, with the missing fields left as None
#   * normalise Category/Severity onto the schema, keeping the original text

_FIELD_PATTERNS = {
    "category": re.compile(r"^\s*(?:\*\*)?category(?:\*\*)?\s*:\s*(?P<v>.+?)\s*$", re.I),
    "severity": re.compile(r"^\s*(?:\*\*)?severity(?:\*\*)?\s*:\s*(?P<v>.+?)\s*$", re.I),
    "line": re.compile(r"^\s*(?:\*\*)?line(?:\*\*)?\s*(?:number|no\.?|#)?\s*[:=]\s*(?P<v>.+?)\s*$", re.I),
    "problem": re.compile(r"^\s*(?:\*\*)?problem(?:\*\*)?\s*:\s*(?P<v>.+?)\s*$", re.I),
    "explanation": re.compile(r"^\s*(?:\*\*)?(?:explanation|reason|rationale)(?:\*\*)?\s*:\s*(?P<v>.+?)\s*$", re.I),
    "suggested_fix": re.compile(
        r"^\s*(?:\*\*)?(?:suggested\s*fix|suggested\s*solution|fix|remediation|solution)(?:\*\*)?\s*:\s*(?P<v>.+?)\s*$",
        re.I,
    ),
}

_VERDICT_ISSUES_RE = re.compile(r"verdict\s*:\s*issues?\s*found", re.I)
_VERDICT_CLEAN_RE = re.compile(r"verdict\s*:\s*no\s+issues?\s*found", re.I)
_BARE_CLEAN_RE = re.compile(r"^\s*(no\s+issues?\s+found\.?|the\s+code\s+is\s+correct\.?)\s*$", re.I)

# Text fields that may absorb a wrapped continuation line, in priority order.
_CONTINUATION_FIELDS = ("explanation", "problem", "suggested_fix")

_CATEGORY_SYNONYMS = {
    "bug": "Bug", "bugs": "Bug", "correctness": "Bug", "logic": "Bug",
    "error-handling": "Bug", "crash": "Bug", "safety": "Bug",
    "security": "Security", "vulnerability": "Security", "vuln": "Security",
    "injection": "Security", "insecure": "Security", "secrets": "Security",
    "performance": "Performance", "perf": "Performance", "resource": "Performance",
    "efficiency": "Performance",
    "quality": "Quality", "maintainability": "Quality", "readability": "Quality",
    "style": "Quality", "robustness": "Quality", "portability": "Quality",
}

_SEVERITY_SYNONYMS = {
    "low": "Low", "minor": "Low", "info": "Low", "informational": "Low", "note": "Low",
    "medium": "Medium", "moderate": "Medium", "med": "Medium",
    "high": "High", "critical": "High", "severe": "High", "major": "High",
}


def _strip_fences(text: str) -> str:
    """Drop ```lang ... ``` wrappers, keeping the inner content."""
    return re.sub(r"^\s*```[^\n]*\n?([\s\S]*?)\n?\s*```\s*$", r"\1", text.strip())


def _normalize_category(raw: str | None) -> str:
    """Map onto the schema. Unrecognised values become Quality.

    Quality is the fallback because an off-schema label is far more often a
    style/maintainability remark than a claimed memory-safety defect, and
    over-claiming a Bug would be the more damaging error.
    """
    if not raw:
        return "Quality"
    cleaned = re.sub(r"[*_`\[\]()]", "", raw).strip().lower()
    cleaned = re.split(r"\s*[-(/]\s*|\s{2,}", cleaned)[0].strip()
    if cleaned in _CATEGORY_SYNONYMS:
        return _CATEGORY_SYNONYMS[cleaned]
    for key, val in _CATEGORY_SYNONYMS.items():
        if key in cleaned:
            return val
    return "Quality"


def _normalize_severity(raw: str | None) -> str:
    if not raw:
        return "Medium"
    cleaned = re.sub(r"[*_`\[\]()]", "", raw).strip().lower()
    cleaned = re.split(r"\s*[-(/]\s*|\s{2,}", cleaned)[0].strip()
    if cleaned in _SEVERITY_SYNONYMS:
        return _SEVERITY_SYNONYMS[cleaned]
    for key, val in _SEVERITY_SYNONYMS.items():
        if key in cleaned:
            return val
    return "Medium"


def _parse_line(raw: str | None) -> int | None:
    """Pull a 1-based line number out of whatever the model wrote."""
    if not raw:
        return None
    m = re.search(r"\d+", raw)
    if not m:
        return None
    val = int(m.group())
    return val if val > 0 else None


def parse_llm_output(text: str) -> tuple[str, list[Finding], bool]:
    """Parse raw LLM text into (verdict, findings, parse_ok).

    `parse_ok` is False when the text did not look like the trained format at
    all, so callers can surface "the model said something unparseable" instead
    of silently reporting a clean bill of health.
    """
    if not text or not text.strip():
        return VERDICT_UNKNOWN, [], False

    body = _strip_fences(text)
    saw_verdict = bool(_VERDICT_ISSUES_RE.search(body) or _VERDICT_CLEAN_RE.search(body))

    findings: list[Finding] = []
    current: dict[str, Any] | None = None
    last_text_field: str | None = None

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        matched_field = None
        for name, pat in _FIELD_PATTERNS.items():
            m = pat.match(line)
            if m:
                matched_field = (name, m.group("v").strip())
                break

        if matched_field:
            name, value = matched_field
            if name == "category":
                # A Category line normally begins a new issue block, which is
                # how we recover multiple issues from one response. But if the
                # current block has other fields and no category yet, the
                # category completes that block rather than starting a new one
                # -- otherwise a stray leading field would split one finding
                # into a fragment plus a block.
                if current is not None and "category" not in current:
                    current["category"] = value
                else:
                    current = {"category": value}
                    findings.append(current)  # type: ignore[arg-type]
                last_text_field = None
            else:
                if current is None:
                    # Fields before any Category: keep them, don't discard.
                    current = {}
                    findings.append(current)  # type: ignore[arg-type]
                current[name] = value
                last_text_field = name if name in _CONTINUATION_FIELDS else None
            continue

        # Not a `Field: value` line. A wrapped continuation of the previous
        # prose field, or the clean-verdict rationale.
        if current is not None and last_text_field:
            prev = current.get(last_text_field)
            current[last_text_field] = f"{prev} {line.strip()}".strip() if prev else line.strip()

    parsed: list[Finding] = []
    for f in findings:
        if not any(v for v in f.values()):
            continue  # fully empty block
        parsed.append(
            Finding(
                category=_normalize_category(f.get("category")),
                severity=_normalize_severity(f.get("severity")),
                problem=(f.get("problem") or f.get("explanation") or "Unspecified issue"),
                line=_parse_line(f.get("line")),
                explanation=f.get("explanation"),
                suggested_fix=f.get("suggested_fix"),
                sources=["llm"],
            )
        )

    if parsed:
        verdict = VERDICT_ISSUES
        parse_ok = True
    elif _VERDICT_CLEAN_RE.search(body) or _BARE_CLEAN_RE.match(body.strip()):
        verdict = VERDICT_CLEAN
        parse_ok = True
    elif _VERDICT_ISSUES_RE.search(body) or saw_verdict:
        # Said ISSUES FOUND but we could not recover a block. Report the
        # verdict as issues but flag the parse so it is visible.
        verdict = VERDICT_ISSUES
        parse_ok = False
    else:
        verdict = VERDICT_UNKNOWN
        parse_ok = False

    return verdict, parsed, parse_ok


# --- Bandit -----------------------------------------------------------------

# Bandit tests that are code-quality rather than security. Everything else in
# Bandit's catalogue is treated as Security, since that is what it detects.
_BANDIT_QUALITY_IDS = {"B101", "B110", "B111", "B112", "B113", "B114", "B703"}


def analyze_bandit(code: str, timeout: float = 30.0) -> tuple[list[Finding], str | None]:
    """Run Bandit on a Python snippet. Returns (findings, error)."""
    try:
        proc = subprocess.run(
            [str(VENV_BIN / "bandit"), "-q", "-f", "json", "-"],
            input=code, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [], "bandit timed out"
    except FileNotFoundError:
        return [], "bandit not installed"

    # Bandit exits 1 when it finds issues; that is not an error for us.
    if not proc.stdout.strip():
        return [], (proc.stderr.strip() or "bandit produced no output")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return [], f"could not parse bandit output: {exc}"

    out: list[Finding] = []
    for r in data.get("results", []):
        test_id = r.get("test_id", "")
        issue_text = r.get("issue_text") or r.get("test_name") or test_id
        explanation = issue_text
        # Bandit's `more_info` is a documentation URL, not a fix. Folding it
        # into the explanation keeps the schema honest -- labelling a bare URL
        # as "Suggested Fix" would be misleading downstream.
        more_info = r.get("more_info")
        if more_info:
            explanation = f"{issue_text} (reference: {more_info})"
        out.append(
            Finding(
                category="Quality" if test_id in _BANDIT_QUALITY_IDS else "Security",
                severity=_normalize_severity(r.get("issue_severity")),
                problem=issue_text,
                line=r.get("line_number") or None,
                explanation=explanation,
                suggested_fix=None,
                sources=["bandit"],
                tool_ref=test_id,
                confidence=(r.get("issue_confidence") or "").lower() or None,
            )
        )
    return out, None


# --- Ruff -------------------------------------------------------------------

# Ruff's JSON has no severity field, so we derive it from the rule code.
# Selected prefixes: S (flake8-bandit, security parity with Bandit),
# F (pyflakes - undefined names, real bugs), E9 (syntax errors), B (bugbear).
_RUFF_SEVERITY_EXACT = {
    "S602": "High", "S605": "High", "S607": "High", "S608": "High",
    "S501": "High", "S502": "High", "S506": "High", "S307": "High",
    "E902": "High", "E999": "High",
    "F821": "High", "F811": "Medium", "F823": "High",
    "S101": "Low", "S110": "Low", "S112": "Low",
    "F841": "Low", "F401": "Low",
}
_RUFF_SEVERITY_PREFIX = {"S6": "High", "S5": "High", "S3": "Medium", "F8": "Medium",
                         "B0": "Medium", "E9": "High"}
_RUFF_CATEGORY_PREFIX = {"S": "Security", "F": "Bug", "E9": "Bug", "B0": "Bug"}


def _ruff_severity(code: str) -> str:
    if code in _RUFF_SEVERITY_EXACT:
        return _RUFF_SEVERITY_EXACT[code]
    for prefix, sev in _RUFF_SEVERITY_PREFIX.items():
        if code.startswith(prefix):
            return sev
    return "Medium"


def _ruff_category(code: str) -> str:
    for prefix, cat in _RUFF_CATEGORY_PREFIX.items():
        if code.startswith(prefix):
            return cat
    return "Quality"


def analyze_ruff(code: str, timeout: float = 30.0,
                 select: str = "S,F,E9,B") -> tuple[list[Finding], str | None]:
    """Run Ruff on a Python snippet. Returns (findings, error)."""
    try:
        proc = subprocess.run(
            [str(VENV_BIN / "ruff"), "check", "--output-format", "json", "--no-cache",
             "--select", select, "--stdin-filename", "snippet.py", "-"],
            input=code, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [], "ruff timed out"
    except FileNotFoundError:
        return [], "ruff not installed"

    if not proc.stdout.strip():
        return [], (proc.stderr.strip() or "ruff produced no output")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return [], f"could not parse ruff output: {exc}"

    out: list[Finding] = []
    for item in data:
        code_id = item.get("code") or "?"
        loc = item.get("location") or {}
        out.append(
            Finding(
                category=_ruff_category(code_id),
                severity=_ruff_severity(code_id),
                problem=item.get("message") or code_id,
                line=loc.get("row") or None,
                explanation=item.get("message"),
                suggested_fix=None,
                sources=["ruff"],
                tool_ref=code_id,
            )
        )
    return out, None


# --- LLM reviewer -----------------------------------------------------------

class LLMReviewer:
    """Thin wrapper around base model + LoRA adapter.

    Greedy decoding on purpose: this is a review tool, and sampling makes the
    same snippet produce different verdicts run to run, which makes the merged
    output impossible to test or trust.
    """

    def __init__(self, base_model_path: str = BASE_MODEL_PATH,
                 adapter_path: str = ADAPTER_PATH, max_new_tokens: int = 400):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
            from peft import PeftModel

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            ).to("cpu")
            model = PeftModel.from_pretrained(base, self.adapter_path).to("cpu")
            model.eval()
            # greedy
            model.generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            self._model = model

    def review(self, code: str) -> str:
        """Return the model's raw text response for `code`."""
        self.load()
        assert self._model is not None and self._tokenizer is not None
        # Serialise generation: the box has 14 GB and the model needs ~7 GB, so
        # two concurrent generations would risk an OOM kill.
        with self._lock:
            inputs = self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Review this code:\n\n```\n{code}\n```"},
                ],
                tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            out = self._model.generate(**inputs, generation_config=self._model.generation_config)
            new_tokens = out[0][inputs["input_ids"].shape[-1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_reviewer: LLMReviewer | None = None
_reviewer_lock = threading.Lock()


def get_llm_reviewer() -> LLMReviewer:
    """Process-wide singleton, so the ~7 GB model loads exactly once."""
    global _reviewer
    if _reviewer is None:
        with _reviewer_lock:
            if _reviewer is None:
                _reviewer = LLMReviewer()
    return _reviewer


# --- merge ------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "and", "or",
    "for", "on", "with", "this", "that", "it", "be", "can", "may", "possible",
    "possible", "via", "through", "from", "at", "by", "as", "if", "not", "no",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z_]{3,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def _same_issue(a: Finding, b: Finding, line_tolerance: int = 2) -> bool:
    """Do two findings describe the same problem?

    Conservative on purpose. Bandit `B608` and Ruff `S608` fire on the same
    line with near-identical text, and they should collapse into one entry.
    But an LLM that flags a security issue on line 2 must not be silently
    absorbed by a Bandit finding on line 2 about something else, so category
    must agree and some keyword overlap is required.
    """
    if a.category != b.category:
        return False

    if a.line is not None and b.line is not None:
        if abs(a.line - b.line) > line_tolerance:
            return False

    ka, kb = _keywords(a.problem), _keywords(b.problem)
    if ka & kb:
        return True

    # Same category on the same line with no contradicting keywords is still
    # very likely the same defect.
    return a.line is not None and a.line == b.line


def merge_findings(static: list[Finding], llm: list[Finding]) -> list[Finding]:
    """Combine analyzer and LLM findings.

    Invariant: every finding in `static` is represented in the result. Findings
    that describe the same defect are collapsed into one entry whose `sources`
    lists every tool that reported it, so nothing is ever silently dropped --
    Bandit's B608 and Ruff's S608 both fire on the same SQL injection and show
    up as one finding tagged `bandit+ruff`.
    """
    merged: list[Finding] = []

    def absorb(incumbent: Finding, incoming: Finding, prefer_incumbent: bool) -> None:
        for src in incoming.sources:
            if src not in incumbent.sources:
                incumbent.sources.append(src)
        if prefer_incumbent:
            # Keep the analyzer's text; it is deterministic and tool-backed.
            # Only backfill fields it left empty.
            if not incumbent.explanation and incoming.explanation:
                incumbent.explanation = incoming.explanation
            if not incumbent.suggested_fix and incoming.suggested_fix:
                incumbent.suggested_fix = incoming.suggested_fix
            if incumbent.line is None and incoming.line is not None:
                incumbent.line = incoming.line
            if incumbent.confidence is None and incoming.confidence:
                incumbent.confidence = incoming.confidence
        else:
            if not incoming.explanation and incumbent.explanation:
                incoming.explanation = incumbent.explanation
            if not incoming.suggested_fix and incumbent.suggested_fix:
                incoming.suggested_fix = incumbent.suggested_fix
            if incoming.line is None and incumbent.line is not None:
                incoming.line = incumbent.line
            merged[merged.index(incumbent)] = incoming

    for f in static:
        target = next((m for m in merged if _same_issue(m, f)), None)
        if target is not None:
            absorb(target, f, prefer_incumbent=True)
        else:
            merged.append(
                Finding(
                    category=f.category, severity=f.severity, problem=f.problem,
                    line=f.line, explanation=f.explanation,
                    suggested_fix=f.suggested_fix,
                    sources=list(dict.fromkeys(f.sources)),
                    tool_ref=f.tool_ref, confidence=f.confidence,
                )
            )

    for f in llm:
        target = next((m for m in merged if _same_issue(m, f)), None)
        if target is not None:
            absorb(target, f, prefer_incumbent=True)
        else:
            merged.append(
                Finding(
                    category=f.category, severity=f.severity, problem=f.problem,
                    line=f.line, explanation=f.explanation,
                    suggested_fix=f.suggested_fix, sources=["llm"],
                )
            )

    order = {"High": 0, "Medium": 1, "Low": 2}
    merged.sort(key=lambda f: (order.get(f.severity, 3),
                               f.line if f.line is not None else 10**6))
    return merged


# --- orchestrator -----------------------------------------------------------

def run_static_analysis(code: str, language: str,
                        run_bandit: bool = True, run_ruff: bool = True,
                        ) -> tuple[list[Finding], list[SkippedAnalyzer], list[str]]:
    """Run whichever python analyzers apply. Never raises."""
    findings: list[Finding] = []
    skipped: list[SkippedAnalyzer] = []
    errors: list[str] = []

    if language != "python":
        reason = f"requires python source, got language={language!r}"
        if run_bandit:
            skipped.append(SkippedAnalyzer("bandit", reason))
        if run_ruff:
            skipped.append(SkippedAnalyzer("ruff", reason))
        return findings, skipped, errors

    if run_bandit:
        found, err = analyze_bandit(code)
        if err:
            errors.append(f"bandit: {err}")
        findings.extend(found)

    if run_ruff:
        found, err = analyze_ruff(code)
        if err:
            errors.append(f"ruff: {err}")
        findings.extend(found)

    return findings, skipped, errors


def review_code(code: str, language: str | None = None, use_llm: bool = True,
                use_bandit: bool = True, use_ruff: bool = True,
                add_owasp_reference: bool = True,
                ) -> ReviewResult:
    """Full review: static analysis + LLM, merged.

    `add_owasp_reference` enriches Security findings with OWASP guidance from
    the PDF after the merge. It is an additive annotation: it can only add a
    `reference` field, never change which findings exist, their severity, or
    the verdict.
    """
    import time

    if not code or not code.strip():
        return ReviewResult(verdict=VERDICT_UNKNOWN, findings=[], language="unknown",
                            errors=["empty input"])

    if len(code) > MAX_CODE_CHARS:
        return ReviewResult(verdict=VERDICT_UNKNOWN, findings=[],
                            language=normalize_language(language, code),
                            errors=[f"input too large ({len(code)} chars, max {MAX_CODE_CHARS})"])

    lang = normalize_language(language, code)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    static, skipped, errors = run_static_analysis(code, lang, use_bandit, use_ruff)
    timings["static"] = time.perf_counter() - t0

    verdict = VERDICT_CLEAN if static else VERDICT_UNKNOWN
    llm_raw = None
    parse_ok = True
    llm_ran = False
    llm_findings: list[Finding] = []

    if use_llm:
        llm_ran = True
        t0 = time.perf_counter()
        try:
            llm_raw = get_llm_reviewer().review(code)
        except Exception as exc:  # a model failure must not lose static results
            errors.append(f"llm: {type(exc).__name__}: {exc}")
            llm_raw = None
            parse_ok = False
        timings["llm"] = time.perf_counter() - t0

        if llm_raw:
            llm_verdict, llm_findings, parse_ok = parse_llm_output(llm_raw)

    merged = merge_findings(static, llm_findings)

    # The verdict reflects merged evidence, not just the LLM: a Bandit hit is
    # an issue even when the model said the code was fine.
    if merged:
        verdict = VERDICT_ISSUES
    elif use_llm and not parse_ok:
        verdict = VERDICT_UNKNOWN
    else:
        verdict = VERDICT_CLEAN

    # OWASP enrichment runs last and is additive only. Imported lazily so the
    # static-analysis-only path never pays for the PDF index.
    if add_owasp_reference:
        try:
            from security_reference import enrich_findings
            enrich_findings(merged)
        except Exception as exc:  # enrichment must never fail a review
            errors.append(f"owasp_reference: {type(exc).__name__}: {exc}")

    return ReviewResult(
        verdict=verdict, findings=merged, language=lang, llm_raw=llm_raw,
        llm_parse_ok=parse_ok, llm_ran=llm_ran, skipped=skipped,
        errors=errors, timings=timings,
    )
