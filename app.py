#!/usr/bin/env python3
"""Gradio front end for the AI Code Reviewer."""

from __future__ import annotations

import argparse
import html
import os
from typing import Any
from urllib.parse import urlsplit

import gradio as gr
import requests

DEFAULT_API_URL = os.environ.get("GRADIO_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 600

VERDICT_LABELS = {
    "issues_found": "Issues found",
    "no_issues_found": "No issues found",
    "unknown": "Review unavailable",
}


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def _esc(value: Any) -> str:
    """Escape arbitrary model/API output before putting it into HTML."""
    return html.escape("" if value is None else str(value), quote=True)


def _safe_url(value: Any) -> str:
    """Only allow absolute HTTP(S) URLs in generated href attributes."""
    if not value:
        return ""

    value = str(value).strip()

    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""

    if parsed.scheme.lower() not in {"http", "https"}:
        return ""

    if not parsed.netloc:
        return ""

    value = value.replace('"', "").replace("'", "")
    return value


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def _render_reference(reference: dict[str, Any]) -> str:
    if not reference:
        return ""

    sheet = _esc(reference.get("cheat_sheet", "OWASP Cheat Sheet"))
    section = _esc(reference.get("section", ""))
    summary = _esc(reference.get("summary", ""))
    page = reference.get("pdf_page")
    url = _safe_url(reference.get("url"))

    page_text = f"PDF p{_esc(page)}" if page is not None else ""

    link = ""
    if url:
        link = (
            f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">'
            "Open OWASP Cheat Sheet ↗</a>"
        )

    return f"""
    <div class="owasp-block">
      <div class="owasp-label">OWASP guidance</div>
      <div class="owasp-sheet">{sheet}</div>
      {"<div class='owasp-section'>" + section + "</div>" if section else ""}
      {"<div class='owasp-page'>" + page_text + "</div>" if page_text else ""}
      {"<div class='owasp-summary'>“" + summary + "”</div>" if summary else ""}
      {link}
    </div>
    """


def _render_finding(finding: dict[str, Any]) -> str:
    category = _esc(finding.get("category", "Security"))
    severity = str(finding.get("severity", "Low"))
    severity = severity if severity in {"High", "Medium", "Low"} else "Low"

    line = finding.get("line")
    problem = _esc(finding.get("problem", ""))
    explanation = _esc(finding.get("explanation", ""))
    suggested_fix = _esc(finding.get("suggested_fix", ""))
    tool_ref = _esc(finding.get("tool_ref", ""))

    line_html = ""
    if line is not None:
        line_html = f'<span class="finding-line">line {_esc(line)}</span>'

    tool_html = ""
    if tool_ref:
        tool_html = f'<span class="finding-tool">{tool_ref}</span>'

    fix_html = ""
    if suggested_fix:
        fix_html = f"""
        <div class="fix-block">
          <div class="fix-label">Suggested fix</div>
          <div class="fix-body">{suggested_fix}</div>
        </div>
        """

    reference_html = _render_reference(finding.get("reference"))

    return f"""
    <div class="finding-card sev-{severity}">
      <div class="finding-top">
        <div>
          <span class="finding-category">{category}</span>
          {line_html}
        </div>
        <div class="finding-meta">
          <span class="severity-badge sev-{severity}">{_esc(severity.upper())}</span>
          {tool_html}
        </div>
      </div>

      <div class="problem">{problem}</div>

      {"<div class='explanation'>" + explanation + "</div>" if explanation else ""}

      {fix_html}

      {reference_html}
    </div>
    """


def _render_result(result: dict[str, Any]) -> str:
    verdict = result.get("verdict", "unknown")

    if verdict not in VERDICT_LABELS:
        verdict_class = "v-unknown"
        label = VERDICT_LABELS["unknown"]
    else:
        verdict_class = {
            "issues_found": "v-issues",
            "no_issues_found": "v-clean",
            "unknown": "v-unknown",
        }[verdict]
        label = VERDICT_LABELS[verdict]

    findings = result.get("findings") or []

    high = sum(1 for f in findings if f.get("severity") == "High")
    medium = sum(1 for f in findings if f.get("severity") == "Medium")
    low = sum(1 for f in findings if f.get("severity") == "Low")

    count = len(findings)

    llm = result.get("llm") or {}
    llm_ran = llm.get("ran", False)

    if llm_ran is None:
        llm_note = "LLM produced no findings"
    elif llm_ran is False:
        llm_note = "LLM disabled"
    else:
        llm_note = "LLM review enabled"

    error_html = ""
    errors = result.get("errors") or []
    if errors:
        error_html = f"""
        <div class="error-box">
          <strong>Review errors</strong>
          {"".join(f"<div>{_esc(e)}</div>" for e in errors)}
        </div>
        """

    skipped = result.get("skipped") or []
    skipped_html = ""
    if skipped:
        skipped_html = f"""
        <div class="skipped-box">
          <strong>Skipped checks</strong>
          {"".join(
              f"<div>{_esc(s.get('name', 'check'))}: "
              f"{_esc(s.get('reason', 'not run'))}</div>"
              for s in skipped
          )}
        </div>
        """

    if findings:
        findings_html = "".join(_render_finding(f) for f in findings)
    else:
        findings_html = """
        <div class="empty-result">
          <div class="empty-title">No findings were reported</div>
          <div class="empty-subtitle">
            The configured checks did not report a finding for this input.
          </div>
        </div>
        """

    parse_note = ""
    if isinstance(llm, dict) and llm.get("ran") and not llm.get("parse_ok", True):
        parse_note = " · LLM response parsing failed"

    return f"""
    <div class="review-results">

      <div class="verdict-card {verdict_class}">
        <div class="verdict-label">{_esc(label)}</div>
        <div class="verdict-meta">
          {count} {"finding" if count == 1 else "findings"}
          · {_esc(llm_note)}{_esc(parse_note)}
        </div>
      </div>

      <div class="stats-row">
        <span class="stat high">>High: {high}<</span>
        <span class="stat medium">>Medium: {medium}<</span>
        <span class="stat low">>Low: {low}<</span>
        <span class="stat total">{count} findings</span>
      </div>

      {error_html}
      {skipped_html}

      <div class="findings-list">
        {findings_html}
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# API communication
# ---------------------------------------------------------------------------

def run_review(
    code: str,
    language: str,
    use_llm: bool,
    use_bandit: bool,
    use_ruff: bool,
    add_owasp_reference: bool,
    api_url: str,
):
    """Send a review request to FastAPI and render its response."""

    if not code or not code.strip():
        return (
            "",
            "No code to review.",
            _status_html(api_url),
        )

    language_value = None if language in {None, "", "auto"} else language

    payload = {
        "code": code,
        "language": language_value,
        "use_llm": bool(use_llm),
        "use_bandit": bool(use_bandit),
        "use_ruff": bool(use_ruff),
        "add_owasp_reference": bool(add_owasp_reference),
    }

    base = api_url.rstrip("/")
    url = f"{base}/review"

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        return (
            "",
            "Backend did not respond in time. CPU-only LLM inference can take several minutes.",
            _status_html(api_url),
        )
    except requests.exceptions.ConnectionError:
        return (
            "",
            "Cannot reach the backend. Start FastAPI with uvicorn and try again.",
            _status_html(api_url),
        )
    except requests.RequestException as exc:
        return (
            "",
            f"Request failed: {exc}",
            _status_html(api_url),
        )

    try:
        data = response.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        return (
            "",
            f"Unreadable response from backend (HTTP {response.status_code}).",
            _status_html(api_url),
        )

    if response.status_code >= 400:
        detail = data.get("detail", data) if isinstance(data, dict) else data
        return (
            "",
            f"Backend returned HTTP {response.status_code}: {detail}",
            _status_html(api_url),
        )

    if not isinstance(data, dict) or "findings" not in data:
        return (
            "",
            "Unexpected response shape from backend.",
            _status_html(api_url),
        )

    stats = _stats_text(data)

    return (
        _render_result(data),
        stats,
        _status_html(api_url),
    )


def _stats_text(result: dict[str, Any]) -> str:
    findings = result.get("findings") or []
    verdict = result.get("verdict", "unknown")

    high = sum(1 for f in findings if f.get("severity") == "High")
    medium = sum(1 for f in findings if f.get("severity") == "Medium")
    low = sum(1 for f in findings if f.get("severity") == "Low")

    return (
        f"{VERDICT_LABELS.get(verdict, VERDICT_LABELS['unknown'])} · "
        f"{len(findings)} findings · "
        f"High {high} · Medium {medium} · Low {low}"
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def _health(api_url: str):
    try:
        response = requests.get(
            f"{api_url.rstrip('/')}/health",
            timeout=5,
        )
        data = response.json()

        model_loaded = bool(data.get("model_loaded"))
        owasp = data.get("owasp_reference") or {}

        sheets = owasp.get("cheat_sheets_indexed", 0)

        model_text = "model loaded" if model_loaded else "model not loaded"

        if owasp.get("available"):
            owasp_text = f"OWASP {sheets} sheets"
        else:
            owasp_text = "OWASP unavailable"

        return (
            "dot-up",
            f"backend up · {model_text} · {owasp_text}",
        )

    except Exception:
        return "dot-unknown", "backend unreachable"


def _status_html(api_url: str) -> str:
    dot, text = _health(api_url)
    return f"""
    <div class="status">
      <span class="status-dot {dot}"></span>
      <span>{_esc(text)}</span>
    </div>
    """


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #0b0d10;
  --panel: #11151a;
  --panel2: #171b21;
  --border: #252b33;
  --text: #d7dce2;
  --muted: #7d8793;
  --accent: #4da3ff;
  --green: #55d187;
  --yellow: #e7bd55;
  --red: #ed6a6a;
}

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

.gradio-container {
  max-width: 1500px !important;
}

.header {
  border-bottom: 1px solid var(--border);
  padding: 14px 4px 12px;
  margin-bottom: 14px;
}

.header-title {
  font-size: 20px;
  font-weight: 700;
  color: #fff;
}

.header-subtitle {
  color: var(--muted);
  font-size: 12px;
  margin-top: 4px;
}

.status {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 11px;
  margin-top: 7px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.dot-up { background: var(--green); }
.dot-unknown { background: var(--yellow); }

.panel {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

.panel-header {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .08em;
}

textarea, input {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

textarea {
  background: #0d1014 !important;
  color: #e2e6eb !important;
}

button {
  font-family: inherit !important;
}

.review-results {
  padding: 2px;
}

.verdict-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 10px;
}

.v-issues {
  border-left: 4px solid var(--red);
}

.v-clean {
  border-left: 4px solid var(--green);
}

.v-unknown {
  border-left: 4px solid var(--yellow);
}

.verdict-label {
  font-size: 18px;
  font-weight: 700;
}

.verdict-meta {
  color: var(--muted);
  font-size: 11px;
  margin-top: 5px;
}

.stats-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.stat {
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 5px 8px;
  font-size: 11px;
}

.high { color: var(--red); }
.medium { color: var(--yellow); }
.low { color: var(--green); }
.total { color: var(--muted); }

.finding-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-left: 3px solid var(--muted);
  border-radius: 7px;
  padding: 14px;
  margin: 9px 0;
}

.finding-card.sev-High { border-left-color: var(--red); }
.finding-card.sev-Medium { border-left-color: var(--yellow); }
.finding-card.sev-Low { border-left-color: var(--green); }

.finding-top {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 9px;
}

.finding-category {
  font-weight: 700;
}

.finding-line, .finding-tool {
  color: var(--muted);
  font-size: 11px;
  margin-left: 8px;
}

.severity-badge {
  font-size: 10px;
  font-weight: 700;
}

.problem {
  color: #f0f2f5;
  font-weight: 600;
  margin-bottom: 7px;
}

.explanation {
  color: #aeb6c0;
  font-size: 12px;
  line-height: 1.55;
}

.fix-block {
  margin-top: 11px;
  padding: 10px;
  background: var(--panel2);
  border-radius: 5px;
}

.fix-label {
  color: var(--accent);
  font-size: 10px;
  text-transform: uppercase;
  margin-bottom: 5px;
}

.fix-body {
  font-size: 12px;
  white-space: pre-wrap;
  line-height: 1.5;
}

.owasp-block {
  margin-top: 11px;
  padding: 10px;
  border: 1px solid #314052;
  background: #10161d;
  border-radius: 5px;
  font-size: 11px;
}

.owasp-label {
  color: var(--accent);
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
}

.owasp-sheet {
  font-weight: 700;
  margin-top: 4px;
}

.owasp-section, .owasp-page {
  color: var(--muted);
  margin-top: 3px;
}

.owasp-summary {
  color: #bbc3cd;
  margin: 7px 0;
  line-height: 1.5;
}

.owasp-block a {
  color: var(--accent);
}

.error-box, .skipped-box {
  padding: 10px;
  margin: 8px 0;
  border-radius: 5px;
  background: #191315;
  border: 1px solid #3b282b;
  color: #d7a9ad;
  font-size: 11px;
}

.empty-result {
  padding: 30px;
  text-align: center;
  border: 1px dashed var(--border);
  border-radius: 8px;
}

.empty-title {
  font-weight: 700;
}

.empty-subtitle {
  color: var(--muted);
  font-size: 11px;
  margin-top: 5px;
}
"""


def build_ui(default_api_url: str = DEFAULT_API_URL):
    with gr.Blocks(
        title="CodeSentry",
        theme=gr.themes.Base(),
        css=CSS,
    ) as demo:

        gr.HTML("""
        <div class="header">
          <div class="header-title">CodeSentry</div>
          <div class="header-subtitle">
            Bandit + Ruff + Qwen2.5-Coder + OWASP
          </div>
        </div>
        """)

        api_state = gr.State(default_api_url)

        with gr.Row():
            with gr.Column(scale=5, elem_classes=["panel"]):
                gr.Markdown("### Source", elem_classes=["panel-header"])

                code = gr.Code(
                    label="",
                    language="python",
                    lines=28,
                    value="",
                )

                with gr.Row():
                    language = gr.Dropdown(
                        choices=[
                            "auto",
                            "python",
                            "javascript",
                            "typescript",
                            "c",
                            "cpp",
                            "go",
                            "java",
                        ],
                        value="auto",
                        label="Language",
                    )

                    review_button = gr.Button(
                        "▶ Review",
                        variant="primary",
                    )

                with gr.Row():
                    use_llm = gr.Checkbox(
                        value=True,
                        label="LLM review",
                    )
                    use_bandit = gr.Checkbox(
                        value=True,
                        label="Bandit",
                    )
                    use_ruff = gr.Checkbox(
                        value=True,
                        label="Ruff",
                    )
                    add_owasp = gr.Checkbox(
                        value=True,
                        label="OWASP guidance",
                    )

            with gr.Column(scale=5, elem_classes=["panel"]):
                gr.Markdown("### Review", elem_classes=["panel-header"])

                result = gr.HTML(
                    value="""
                    <div class="empty-result">
                      <div class="empty-title">Ready to review</div>
                      <div class="empty-subtitle">
                        Paste code and run the configured checks.
                      </div>
                    </div>
                    """
                )

                stats = gr.Textbox(
                    label="",
                    interactive=False,
                )

        status = gr.HTML(_status_html(default_api_url))

        review_button.click(
            fn=run_review,
            inputs=[
                code,
                language,
                use_llm,
                use_bandit,
                use_ruff,
                add_owasp,
                api_state,
            ],
            outputs=[result, stats, status],
            api_name="run_review",
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui(args.api_url)

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
