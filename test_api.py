#!/usr/bin/env python
"""
End-to-end test for the /review API.

    # fast: static analysis only, no model load (~2s)
    .venv/bin/python test_api.py

    # full: includes the LLM, needs ~7 GB free RAM and several minutes on CPU
    .venv/bin/python test_api.py --with-llm

Starts a real uvicorn server on a free port, drives it over HTTP with httpx,
and asserts on the JSON contract. Exits non-zero on the first failed assertion
group so it is usable in CI.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = str(REPO / ".venv" / "bin" / "python")

SQLI = '''import sqlite3

def get_user(conn, username):
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    return conn.execute(query).fetchone()
'''

JS_CLOSURE = """const results = [];
[1, 2, 3].forEach(function (n) {
  setTimeout(function () {
    results.push(n * 2);
  }, 100);
});
console.log(results);
"""

CLEAN_PY = '''def calculate_tax(amount: float, rate: float) -> float:
    """Return the tax owed on `amount` at `rate`."""
    if amount < 0:
        raise ValueError("amount must be non-negative")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    return round(amount * rate, 2)
'''

CLEAN_JS = """export function add(a, b) {
  return a + b;
}
"""

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  <- {detail}" if detail else ""))
        failures.append(label)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout: float = 60.0) -> bool:
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true",
                    help="include the LLM in the review (slow, high RAM)")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    import httpx

    port = free_port()
    print(f"starting uvicorn on 127.0.0.1:{port} "
          f"(llm={'on' if args.with_llm else 'off'})")
    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"

    try:
        if not wait_for_server(port):
            print("server did not come up; output:")
            proc.terminate()
            print(proc.stdout.read() if proc.stdout else "")
            return 2
        print("server up\n")

        use_llm = args.with_llm
        t0 = time.time()

        # --- schema / health -------------------------------------------------
        print("GET /health")
        h = httpx.get(f"{base}/health", timeout=30).json()
        check("health reports model_loaded", "model_loaded" in h, str(h))
        check("health names the v2 adapter", h.get("adapter", "").endswith("lora-code-reviewer-v2"),
              h.get("adapter", ""))

        # --- python SQL injection -------------------------------------------
        print("\nPOST /review  python / SQL injection")
        r = httpx.post(f"{base}/review",
                       json={"code": SQLI, "language": "python", "use_llm": use_llm},
                       timeout=args.timeout)
        check("200 OK", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
        d = r.json()

        for key in ("verdict", "language", "findings", "summary", "llm",
                    "skipped", "errors", "timings_ms", "status"):
            check(f"response has '{key}'", key in d)
        check("llm.ran is False when the model was not used",
              d["llm"]["ran"] is False, str(d["llm"]["ran"]))
        check("llm.raw is null when the model was not used",
              d["llm"]["raw"] is None, str(d["llm"]["raw"]))
        check("status == ok", d.get("status") == "ok", str(d.get("status")))
        check("language == python", d.get("language") == "python", str(d.get("language")))
        check("verdict == issues_found", d.get("verdict") == "issues_found",
              str(d.get("verdict")))
        check("at least one finding", d["summary"]["total"] >= 1,
              str(d["summary"]["total"]))

        f = d["findings"][0]
        for key in ("category", "severity", "line", "problem", "explanation",
                    "suggested_fix", "sources", "tool_ref"):
            check(f"finding has '{key}'", key in f)
        check("severity is schema-valid", f["severity"] in ("Low", "Medium", "High"),
              f["severity"])
        check("category is schema-valid",
              f["category"] in ("Bug", "Security", "Performance", "Quality"), f["category"])
        check("finding has a source", bool(f["sources"]), str(f["sources"]))

        srcs = set(f["sources"])
        check("bandit ran", "bandit" in {s for x in d["findings"] for s in x["sources"]},
              str(srcs))
        check("ruff ran", "ruff" in {s for x in d["findings"] for s in x["sources"]},
              str(srcs))
        check("sql injection categorised Security", f["category"] == "Security",
              f["category"])
        check("sql injection on the query line", f["line"] == 4, str(f["line"]))
        check("bandit+ruff collapsed to one corroborated finding",
              len(f["sources"]) > 1, f"sources={f['sources']}")
        check("summary.corroborated counted it", d["summary"]["corroborated"] >= 1,
              str(d["summary"]["corroborated"]))
        check("no analyzer errors", not d["errors"], str(d["errors"]))

        if use_llm:
            check("llm.ran is True", d["llm"]["ran"] is True, str(d["llm"]["ran"]))
            check("llm parse_ok", d["llm"]["parse_ok"], d["llm"]["raw"] or "")
            check("llm produced raw text", bool(d["llm"]["raw"]))
            check("llm corroborated the SQL injection",
                  "llm" in f["sources"], f"sources={f['sources']}")

        # --- non-python: analyzers must skip ---------------------------------
        print("\nPOST /review  javascript / closure bug")
        r = httpx.post(f"{base}/review",
                       json={"code": JS_CLOSURE, "language": "javascript",
                             "use_llm": use_llm},
                       timeout=args.timeout)
        d = r.json()
        check("200 OK", r.status_code == 200, str(r.status_code))
        check("language == javascript", d.get("language") == "javascript",
              str(d.get("language")))
        skipped = {s["name"] for s in d["skipped"]}
        check("bandit skipped for non-python", "bandit" in skipped, str(skipped))
        check("ruff skipped for non-python", "ruff" in skipped, str(skipped))
        check("no python analyzer leaked a finding",
              not any(set(x["sources"]) & {"bandit", "ruff"} for x in d["findings"]))
        if use_llm:
            check("llm flagged the async closure bug",
                  d["verdict"] == "issues_found" and d["summary"]["total"] >= 1,
                  f"verdict={d['verdict']} n={d['summary']['total']}")

        # --- clean python ----------------------------------------------------
        print("\nPOST /review  python / clean code")
        r = httpx.post(f"{base}/review",
                       json={"code": CLEAN_PY, "language": "python", "use_llm": use_llm},
                       timeout=args.timeout)
        d = r.json()
        check("200 OK", r.status_code == 200, str(r.status_code))
        check("verdict == no_issues_found", d["verdict"] == "no_issues_found",
              f"{d['verdict']} findings={[x['problem'] for x in d['findings']]}")
        check("zero findings", d["summary"]["total"] == 0,
              str([x["problem"] for x in d["findings"]]))
        if use_llm:
            check("llm agreed it is clean", d["llm"]["parse_ok"], d["llm"]["raw"] or "")

        # --- language auto-detect --------------------------------------------
        print("\nPOST /review  auto language detection")
        d = httpx.post(f"{base}/review", json={"code": JS_CLOSURE, "use_llm": False},
                       timeout=120).json()
        check("auto-detected javascript", d["language"] == "javascript", d["language"])
        d = httpx.post(f"{base}/review", json={"code": CLEAN_PY, "use_llm": False},
                       timeout=120).json()
        check("auto-detected python", d["language"] == "python", d["language"])

        # --- error handling ---------------------------------------------------
        print("\nerror handling")
        r = httpx.post(f"{base}/review", json={"code": "", "language": "python"},
                       timeout=60)
        check("empty code -> 422", r.status_code == 422, str(r.status_code))
        r = httpx.post(f"{base}/review", json={"language": "python"}, timeout=60)
        check("missing code -> 422", r.status_code == 422, str(r.status_code))
        r = httpx.post(f"{base}/review",
                       json={"code": "x = 1\n" * 30000, "language": "python"},
                       timeout=120)
        check("oversized code -> 413", r.status_code == 413, str(r.status_code))
        r = httpx.post(f"{base}/review",
                       json={"code": "def f(:\n  pass\n", "language": "python",
                             "use_llm": False},
                       timeout=60)
        check("syntactically invalid python still returns 200", r.status_code == 200,
              str(r.status_code))

        print(f"\ncompleted in {time.time() - t0:.0f}s")
        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED:")
            for f_ in failures:
                print(f"  - {f_}")
            return 1
        print("\nALL API CHECKS PASSED")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
