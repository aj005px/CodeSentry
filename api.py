"""
FastAPI wrapper around the combined review pipeline.

    uvicorn api:app --reload          # then open http://127.0.0.1:8000/docs

Endpoints
---------
POST /review    {"code": "...", "language": "python"}  -> merged findings as JSON
GET  /health    liveness + whether the model is loaded
POST /warmup    load the LoRA model ahead of the first request

Two operational notes, both consequences of running a 3B model on a 14 GB
CPU-only box:

1. The model is loaded lazily and exactly once. The first /review that uses it
   takes minutes; every later one is seconds. Call POST /warmup to pay that cost
   at startup instead of on a user's first request.

2. Reviews are CPU-bound and the box has no spare cores, so they run one at a
   time behind a lock. Endpoints are declared `def`, not `async def`, so
   FastAPI runs them in its threadpool and the event loop stays responsive
   (e.g. for /health) while a review is in flight.

The response shape is the contract a future UI will depend on, so it is worth
keeping stable:

    {
      "verdict": "issues_found" | "no_issues_found" | "unknown",
      "language": "python",
      "findings": [ {category, severity, line, problem, explanation,
                     suggested_fix, sources, tool_ref, confidence}, ... ],
      "summary": {total, by_severity, by_category, by_source, corroborated},
      "llm":     {ran, parse_ok, raw},
      "skipped": [{name, reason}],
      "errors":  [str],
      "timings_ms": {static, llm}
    }
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from review_core import (
    MAX_CODE_CHARS, VERDICT_UNKNOWN, get_llm_reviewer, review_code,
)

app = FastAPI(
    title="AI Code Reviewer",
    version="0.1.0",
    description=(
        "Bandit + Ruff static analysis merged with a LoRA-fine-tuned "
        "Qwen2.5-Coder-3B review. Static findings are never dropped; the "
        "LLM supplements and corroborates them."
    ),
)

# One review at a time: the model needs ~7 GB and there is no headroom for two.
_review_lock = asyncio.Lock()


class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Source code to review")
    language: str | None = Field(
        None,
        description="python|javascript|c|cpp|go|... or 'auto' (default: auto-detect)",
    )
    use_llm: bool = Field(True, description="Include the fine-tuned LLM review")
    use_bandit: bool = Field(True, description="Include Bandit (python only)")
    use_ruff: bool = Field(True, description="Include Ruff (python only)")


@app.get("/health")
async def health() -> dict[str, Any]:
    reviewer = get_llm_reviewer()
    return {
        "status": "ok",
        "model_loaded": reviewer.loaded,
        "adapter": reviewer.adapter_path,
        "base_model": reviewer.base_model_path,
        "note": (
            "model_loaded=false is normal before the first review; POST /warmup "
            "to load it, which takes minutes on CPU."
        ),
    }


@app.post("/warmup")
async def warmup() -> dict[str, Any]:
    """Load the model up front. Safe to call repeatedly."""
    reviewer = get_llm_reviewer()
    await run_in_threadpool(reviewer.load)
    return {"status": "loaded", "already_loaded": False, "adapter": reviewer.adapter_path}


@app.post("/review")
async def review(req: ReviewRequest) -> dict[str, Any]:
    if len(req.code) > MAX_CODE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"code too large: {len(req.code)} chars (max {MAX_CODE_CHARS})",
        )
    if not req.code.strip():
        raise HTTPException(status_code=422, detail="code is empty")

    async with _review_lock:
        # run_in_threadpool so the 1-3 minute CPU generation does not block
        # the event loop.
        result = await run_in_threadpool(
            review_code,
            req.code,
            req.language,
            req.use_llm,
            req.use_bandit,
            req.use_ruff,
        )

    payload = result.to_dict()
    # An input we could not review at all is a client-visible failure, not a
    # successful review that happens to have found nothing.
    if result.verdict == VERDICT_UNKNOWN and result.errors:
        payload["status"] = "error"
    else:
        payload["status"] = "ok"
    return payload
