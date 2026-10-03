#!/usr/bin/env python3
"""Checks for the Gradio front end (app.py).

Run:  python test_frontend.py

Most checks are offline and need no servers. The live section is skipped unless
the API and the UI are already running:

    python -m uvicorn api:app --port 8012
    python app.py --api-url http://127.0.0.1:8012 --port 7861
"""

from __future__ import annotations

import re
import sys

import app

CHECKS: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    CHECKS.append((name, bool(cond)))


def finding(**kw) -> dict:
    base = {"category": "Security", "severity": "High", "line": 12,
            "problem": "p", "explanation": "e", "suggested_fix": "f",
            "sources": ["bandit"], "tool_ref": "B608", "reference": None}
    base.update(kw)
    return base


# -- the frontend must not import the backend ------------------------------
src = open(app.__file__).read()
for forbidden in ("review_core", "from api import", "import api\n",
                  "transformers", "peft", "torch"):
    check(f"app.py does not import {forbidden!r}",
          not re.search(rf"^\s*(import|from)\s+.*{re.escape(forbidden)}",
                        src, re.M))
check("app.py talks to the API over HTTP", "requests.post" in src)

# -- verdict rendering matches the API's actual verdict constants ------------
from review_core import VERDICT_CLEAN, VERDICT_ISSUES, VERDICT_UNKNOWN  # noqa: E402

check("VERDICT_ISSUES is handled", VERDICT_ISSUES in app.VERDICT_LABELS)
check("VERDICT_CLEAN is handled", VERDICT_CLEAN in app.VERDICT_LABELS)
check("VERDICT_UNKNOWN is handled", VERDICT_UNKNOWN in app.VERDICT_LABELS)
check("clean label matches the constant, not 'clean'",
      app.VERDICT_LABELS[VERDICT_CLEAN] == "No issues found")

for verdict, cls in ((VERDICT_ISSUES, "v-issues"), (VERDICT_CLEAN, "v-clean"),
                     (VERDICT_UNKNOWN, "v-unknown")):
    html = app._render_result({"verdict": verdict, "findings": []})
    check(f"{verdict} renders {cls}", cls in html)
    check(f"{verdict} shows its label", app.VERDICT_LABELS[verdict] in html)

check("unknown verdict falls back safely",
      "v-unknown" in app._render_result({"verdict": "surprise", "findings": []}))

# -- severity colouring -----------------------------------------------------
for sev, cls in (("High", "sev-High"), ("Medium", "sev-Medium"), ("Low", "sev-Low")):
    html = app._render_finding(finding(severity=sev))
    check(f"{sev} maps to {cls}", cls in html)
    check(f"{sev} label shown", sev.upper() in html)

# -- finding fields ---------------------------------------------------------
html = app._render_finding(finding())
for field in ("Security", "line 12", "B608", "p", "e", "f"):
    check(f"finding renders {field!r}", field in html)

check("no line number shown when absent",
      "line " not in app._render_finding(finding(line=None)).split("</div>")[0])
check("suggested_fix omitted when empty",
      "fix-body" not in app._render_finding(finding(suggested_fix="")))

# -- OWASP reference block --------------------------------------------------
REF = {"cheat_sheet": "SQL Injection Prevention Cheat Sheet",
       "section": "20.2. Primary Defenses",
       "summary": "Use prepared statements (parameterized queries).",
       "url": "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html",
       "pdf_page": 139}
html = app._render_finding(finding(reference=REF))
check("reference sheet shown", "SQL Injection Prevention Cheat Sheet" in html)
check("reference section shown", "20.2. Primary Defenses" in html)
check("reference page shown", "PDF p139" in html)
check("reference summary quoted", "prepared statements" in html)
check("reference links to owasp", "cheatsheetseries.owasp.org" in html)
check("reference opens in a new tab", 'target="_blank"' in html)
check("reference is labelled as OWASP guidance", "OWASP guidance" in html)
check("reference adds a rel=noopener", 'rel="noopener' in html)
check("no reference block when reference is null",
      "OWASP guidance" not in app._render_finding(finding(reference=None)))

# -- HTML injection: model output must never become markup ------------------
XSS = '<img src=x onerror="alert(1)"><script>alert(2)</script>'
html = app._render_finding(finding(problem=XSS, explanation=XSS,
                                   suggested_fix=XSS,
                                   reference={**REF, "summary": XSS,
                                              "cheat_sheet": XSS,
                                              "url": 'javascript:alert(3)'}))
for tag in ("<img", "<script"):
    check(f"{tag!r} from model output is neutralised", tag not in html)
check("quotes from model output are escaped", 'onerror=&quot;' in html)
check("escaped entities are present instead", "&lt;script&gt;" in html)
check("_esc escapes all five characters",
      app._esc('&<>"') == "&amp;&lt;&gt;&quot;")

# HTML-escaping alone does not make a URL safe: a scheme like javascript: has
# nothing for &quot;/&lt; to neutralise. Only http(s) may reach an href.
check("_safe_url refuses javascript:", app._safe_url("javascript:alert(3)") == "")
check("_safe_url refuses data:", app._safe_url("data:text/html,<script>") == "")
check("_safe_url refuses vbscript:", app._safe_url("vbscript:x") == "")
check("_safe_url refuses a scheme-relative path", app._safe_url("//evil.test") == "")
check("_safe_url refuses empty", app._safe_url(None) == "")
check("_safe_url allows https", app._safe_url("https://owasp.org/a.html")
      == "https://owasp.org/a.html")
check("_safe_url allows http", app._safe_url("http://owasp.org/a.html")
      == "http://owasp.org/a.html")
check("_safe_url strips quotes that could break the attribute",
      '"' not in app._safe_url('https://x.test/"onmouseover="a'))
check("unsafe URL renders as plain text, not a link",
      "<a " not in app._render_finding(finding(reference={**REF, "url": "javascript:alert(1)"})))
check("safe URL still renders as a link",
      "<a " in app._render_finding(finding(reference=REF)))

# -- stats line -------------------------------------------------------------
# Four findings, two of them High, so the counts are not trivially 1.
res = {"verdict": VERDICT_ISSUES,
       "findings": [finding(severity="High"), finding(severity="Medium"),
                    finding(severity="Low"), finding(severity="High", reference=REF)],
       "llm_ran": False}
html = app._render_result(res)
check("counts High 2", ">High: 2<" in html)
check("counts Medium 1", ">Medium: 1<" in html)
check("counts Low 1", ">Low: 1<" in html)
check("counts 4 findings", "4 findings" in html)
check("llm_ran False is disclosed", "LLM disabled" in html)

res2 = {**res, "findings": [], "llm_ran": None}
html2 = app._render_result(res2)
check("llm_ran None is disclosed", "LLM produced no findings" in html2)
check("empty result explains itself", "No findings were reported" in html2)

check("errors are surfaced",
      "err" in app._render_result({**res, "errors": ["boom"]}))

# -- request building -------------------------------------------------------
class _Resp:
    def __init__(self, payload, code=200):
        self._payload, self.status_code = payload, code
        self.text = str(payload)

    def json(self):
        return self._payload


captured: dict = {}


def fake_post(url, json=None, timeout=None):
    captured["url"] = url
    captured["json"] = json
    captured["timeout"] = timeout
    return _Resp({"verdict": VERDICT_CLEAN, "findings": [], "llm_ran": False})


app.requests.post = fake_post
REAL_STATUS_HTML = app._status_html
app._status_html = lambda base: "status"
html, stats, status = app.run_review("x=1", "auto", True, True, True, True,
                                    "http://h:1/")
check("posts to /review", captured["url"] == "http://h:1/review")
check("sends the code", captured["json"]["code"] == "x=1")
check("'auto' language becomes null", captured["json"]["language"] is None)
check("sends use_llm", captured["json"]["use_llm"] is True)
check("sends add_owasp_reference", captured["json"]["add_owasp_reference"] is True)
app.run_review("x=1", "python", False, True, True, False, "http://h:1/")
check("owasp toggle is forwarded as False",
      captured["json"]["add_owasp_reference"] is False)
app.run_review("x=1", "auto", True, True, True, True, "http://h:1/")
check("empty code leaves the previous request untouched",
      captured["json"]["code"] == "x=1")
check("has a generous timeout", captured["timeout"] >= 60)
check("trailing slash trimmed from base url", captured["url"].endswith("/review"))

app.run_review("x=1", "auto", False, True, True, True, "http://h:1/")
check("auto maps to null language", captured["json"]["language"] is None)
app.run_review("x=1", "go", False, True, True, True, "http://h:1/")
check("real language is preserved", captured["json"]["language"] == "go")

# -- empty input and unreachable backend -----------------------------------
html, stats, _ = app.run_review("   ", "python", False, True, True, True, "http://h:1/")
check("empty code is refused without a request", stats == "No code to review.")
check("empty code posts nothing", captured.get("json", {}).get("code") == "x=1")


def boom(*_a, **_k):
    raise app.requests.exceptions.ConnectionError("refused")


app.requests.post = boom
html, stats, _ = app.run_review("x=1", "python", False, True, True, True, "http://h:1/")
check("unreachable backend is reported", "Cannot reach the backend" in stats)
check("unreachable backend names the fix", "uvicorn" in stats)


def slow(*_a, **_k):
    raise app.requests.exceptions.Timeout("too slow")


app.requests.post = slow
html, stats, _ = app.run_review("x=1", "python", False, True, True, True, "http://h:1/")
check("timeout is reported as a timeout", "did not respond in time" in stats)

app.requests.post = fake_post


def _422(url, json=None, timeout=None):
    return _Resp({"detail": [{"loc": ["body", "code"]}]}, code=422)


app.requests.post = _422
html, stats, _ = app.run_review("x=1", "python", False, True, True, True, "http://h:1/")
check("422 is surfaced", "422" in stats)

app.requests.post = lambda url, json=None, timeout=None: _Resp("oops", code=500)
html, stats, _ = app.run_review("x=1", "python", False, True, True, True, "http://h:1/")
check("5xx is surfaced", "500" in stats)

class _BadJson(_Resp):
    def json(self):
        raise ValueError("not json")


app.requests.post = lambda url, json=None, timeout=None: _BadJson("<html>", code=200)
html, stats, _ = app.run_review("x=1", "python", False, True, True, True, "http://h:1/")
check("unparseable 200 is surfaced", "Unreadable" in stats)

app.requests.post = lambda url, json=None, timeout=None: _Resp([1, 2, 3])
html, stats, _ = app.run_review("x=1", "python", False, True, True, True, "http://h:1/")
check("JSON that is not a review object is handled", "Unexpected response shape" in stats)

# -- health -----------------------------------------------------------------
class _GetResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


app.requests.get = lambda url, timeout=None: _GetResp(
    {"status": "ok", "model_loaded": False,
     "owasp_reference": {"available": True, "cheat_sheets_indexed": 45}})
dot, status = app._health("http://h:1")
check("healthy backend gets the up dot", dot == "dot-up")
check("health reports OWASP sheet count", "45" in status)
check("health reports model not loaded", "model not loaded" in status)
check("status html includes the dot",
      'class="status-dot dot-up"' in REAL_STATUS_HTML("http://h:1"))

app.requests.get = lambda url, timeout=None: _GetResp(
    {"status": "ok", "model_loaded": True,
     "owasp_reference": {"available": False}})
dot, status = app._health("http://h:1")
check("loaded model reported", "model loaded" in status)
check("unavailable OWASP reported", "OWASP unavailable" in status)


def _dead(*_a, **_k):
    raise OSError("no route to host")


app.requests.get = _dead
app._status_html = REAL_STATUS_HTML
dot, status = app._health("http://h:1")
check("dead backend gets the unknown dot", dot == "dot-unknown")
check("dead backend is labelled", status == "backend unreachable")

# -- live round trip (only if both services are up) ------------------------
import socket  # noqa: E402


def _up(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


if _up(8012) and _up(7861):
    try:
        from gradio_client import Client
        client = Client("http://127.0.0.1:7861/", verbose=False)
        CODE = ("import hashlib, sqlite3\n"
                "def q(conn, u):\n"
                "    return conn.execute(\"SELECT * FROM t WHERE n='\"+u+\"'\").fetchone()\n"
                "def h(p):\n"
                "    return hashlib.md5(p.encode()).hexdigest()\n")
        findings, stats, status = client.predict(
            CODE, "python", False, True, True, True, "http://127.0.0.1:8012",
            api_name="/run_review")
        check("live Gradio -> FastAPI round trip", bool(findings))
        check("live review cites SQL Injection",
              "SQL Injection Prevention Cheat Sheet" in findings)
        check("live review cites Password Storage",
              "Password Storage Cheat Sheet" in findings)
        check("live review left the md5 finding uncited or cited correctly",
              "Password Storage" in findings)
        check("live review shows bandit tool refs", "B608" in findings)
        check("live status shows backend up", "backend up" in status)
    except Exception as exc:
        check(f"live round trip raised {type(exc).__name__}", False)
else:
    print("NOTE: API/UI not running on 8012/7861; skipped the live round trip")

# -- report -----------------------------------------------------------------
passed = sum(1 for _n, ok in CHECKS if ok)
failed = [n for n, ok in CHECKS if not ok]
print(f"\n{passed}/{len(CHECKS)} frontend checks passed")
if failed:
    print("\nFAILED:")
    for name in failed:
        print(f"  - {name}")
    print("\nSOME FRONTEND CHECKS FAILED")
    sys.exit(1)
print("ALL FRONTEND CHECKS PASSED")
