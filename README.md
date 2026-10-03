# CodeSentry

> AI-assisted code review with static analysis, a fine-tuned Qwen2.5-Coder model, and OWASP security references.

CodeSentry is a local code-review tool that combines traditional static analysis with a fine-tuned language model.

The idea is simple:

**let the tools catch what they are good at, let the model catch what they can miss, and don't let an LLM mistake hide a real static-analysis finding.**

It currently combines:

* Qwen2.5-Coder-3B-Instruct with a LoRA adapter
* Bandit for Python security analysis
* Ruff for Python linting
* OWASP Cheat Sheet references
* FastAPI backend
* Gradio frontend
* Custom finding parsing and merge logic

Everything runs locally.

---

## What CodeSentry Does

You give CodeSentry a code snippet and it runs the available analysis tools against it.

The review can contain:

* Verdict
* Programming language
* Findings
* Severity
* Line number
* Explanation
* Suggested fix
* Finding source
* Corroboration between tools
* OWASP guidance where applicable
* LLM status
* Errors and timing information

For example:

```text
Security
HIGH

SQL injection

The query is constructed by concatenating a string with the username.
This allows attacker-controlled input to modify the SQL query.

Suggested Fix:
Use parameterized queries.
```

When a finding maps cleanly to an OWASP Cheat Sheet, CodeSentry also shows the relevant section and page from the local reference PDF.

---

# Why I Built It

I originally wanted to build a code-review model using LoRA.

The first version technically trained.

It also completely failed.

Instead of learning to review code, it learned that the safest answer was:

```text
No issues found.
```

That happened on basically everything.

Instead of throwing more data at the model, I went back through the training setup, evaluation, dataset, and generation pipeline to figure out what was actually happening.

The project gradually turned from:

> "fine-tune a code reviewer"

into:

> "build an actual code-review system and figure out where every part can fail."

---

# The First LoRA Failed

The original dataset had 86 examples.

On a 49-case evaluation set:

| Model              | Accuracy | Bug Recall | "No issues" |
| ------------------ | -------: | ---------: | ----------: |
| Base Qwen2.5-Coder |    73.5% |       100% |          0% |
| LoRA v1            |    26.5% |         0% |        100% |

The fine-tuned model returned:

```text
No issues found.
```

on **49/49 cases**.

That included examples that were effectively identical to examples it had already seen during training.

The base model correctly detected all 36 buggy cases in the evaluation set, although it also produced 13 false positives on clean code.

The first LoRA traded all of that bug detection away just to become very good at saying that nothing was wrong.

---

# What Went Wrong

There wasn't one single problem.

There were several things working together.

## 1. The clean class was too dominant

46 out of 86 training examples had exactly:

```text
No issues found.
```

That's **53.5% of the entire dataset**.

The shortest and easiest target became the most common target.

---

## 2. The Training Objective Was Wrong

The original setup supervised the entire rendered conversation instead of only the assistant response.

That meant the model was getting training signal from:

* The system prompt
* The code
* The user message
* The actual review

The fix was:

```text
assistant_only_loss=True
```

This focuses the training signal on the model's actual response.

---

## 3. There Was No Validation Split

The original training setup had:

* No validation set
* No evaluation loss
* No checkpoint selection

So the final epoch was simply shipped.

A validation split and best-checkpoint selection were added so training could actually be evaluated instead of blindly taking the final checkpoint.

---

## 4. One "Clean" Example Wasn't Actually Clean

One example called:

```python
time.time()
```

without importing `time`.

That meant the example would raise a `NameError`.

It was labelled as clean.

That example was fixed before retraining.

---

## 5. The Dataset Didn't Have Enough Variety

The original dataset had:

* Zero C examples
* Zero C++ examples
* Only one JavaScript example
* No real multi-issue examples
* Mostly buggy/fixed pairs

The dataset was expanded with:

* C
* C++
* JavaScript
* Go
* Multi-issue examples
* Hard negatives
* Independent clean examples

---

# What Changed in v2

The new dataset ended up with:

```text
121 examples
80 issue
41 clean
```

The main changes were:

* Assistant-only loss
* Verdict-first targets
* Better clean/issue balance
* Varied clean-code explanations
* More programming languages
* Multi-issue examples
* Hard negatives
* Fixed mislabeled example
* Stratified train/validation split
* Evaluation loss
* Best-checkpoint selection
* Lower learning rate
* Cosine learning-rate schedule
* Warmup
* Fixed random seeds
* Maximum sequence length
* Greedy decoding during evaluation

The target format was also changed to make the actual decision explicit:

```text
Verdict: ISSUES FOUND

Category: Security
Severity: High
Line: 2
Problem: SQL injection
Explanation: ...
Suggested Fix: ...
```

or:

```text
Verdict: NO ISSUES FOUND

Explanation: ...
```

---

# v2 Results

The retraining was done on a Colab T4.

| Epoch | Train Loss | Eval Loss |
| ----- | ---------: | --------: |
| 1     |     1.6331 |    1.5473 |
| 2     |     1.3335 |    1.3582 |
| 3     |     1.1480 |    1.3322 |

On the first 20 cases of the evaluation run:

| Model   |             Correct | Bug Recall |
| ------- | ------------------: | ---------: |
| LoRA v1 | 0/20-style collapse |         0% |
| LoRA v2 |               14/20 |        72% |

The v2 model also produced the expected verdict format on all 20 cases.

The full evaluation was not completed in that initial run, so the 72% figure should be treated as a partial evaluation rather than a final benchmark.

---

# From a Fine-Tuned Model to an Actual Review Pipeline

Once the model was working again, the next problem was obvious:

**an LLM shouldn't be the only thing deciding whether code is vulnerable.**

So CodeSentry became a combined system.

```text
                    ┌─────────────────┐
                    │   Source Code   │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │ Bandit/Ruff  │          │ Fine-tuned   │
        │ Static       │          │ Qwen2.5      │
        │ Analysis     │          │ Coder        │
        └──────┬───────┘          └──────┬───────┘
               │                         │
               └────────────┬────────────┘
                            ▼
                    ┌───────────────┐
                    │ Finding Merge │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ OWASP Lookup  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Final Review  │
                    └───────────────┘
```

---

# The Merge Logic

The merge isn't a vote between the model and the static analyzers.

The rule is:

> **A static finding is never dropped.**

If Bandit finds something and the LLM misses it, the finding stays.

If both find the same issue, they are merged and the sources are recorded.

For example:

```text
SQL injection
sources: bandit + ruff + llm
```

If only Bandit finds it:

```text
SQL injection
sources: bandit
```

The verdict is calculated from the merged evidence rather than blindly trusting the LLM's final sentence.

---

# LLM Output Parsing

LLM output is messy.

CodeSentry therefore uses a line-based parser/state machine instead of relying on one large regex.

The parser handles:

* Multiple findings
* Markdown fences
* Bold labels
* Lowercase labels
* Missing colons
* Preamble text
* Wrapped lines
* Category/severity synonyms
* Partially completed findings
* Truncated responses
* Category-only findings
* Legacy output formats

If the model output can't be understood, the result becomes:

```text
verdict: unknown
```

rather than incorrectly telling the user that the code is clean.

---

# OWASP References

CodeSentry also includes local OWASP Cheat Sheet references.

The OWASP book contains 45 cheat sheets.

Instead of building a full vector database and RAG pipeline for this relatively structured corpus, CodeSentry uses deterministic retrieval:

```text
Finding
   ↓
Security taxonomy
   ↓
Candidate OWASP sheets
   ↓
Keyword scoring
   ↓
Relevant subsection
```

No embeddings.

No vector database.

No LLM involved in the retrieval itself.

The returned summary comes directly from the PDF rather than being generated.

For example:

```text
SQL Injection Prevention Cheat Sheet
20.2. Primary Defenses
PDF p139
```

along with the relevant source passage.

---

# Conservative OWASP Matching

Not every vulnerability gets an OWASP reference.

The system deliberately avoids attaching unrelated references simply because keywords happen to match.

The OWASP layer is additive.

It does not change:

* Finding existence
* Severity
* Verdict
* Merge behavior

It only adds an optional reference after the review has already been generated.

---

# FastAPI Backend

The backend exposes:

```text
GET  /health
POST /review
POST /warmup
```

The main review endpoint returns structured JSON containing:

```json
{
  "verdict": "issues_found",
  "language": "python",
  "findings": [],
  "summary": {},
  "llm": {
    "ran": true,
    "parse_ok": true,
    "raw": "..."
  },
  "skipped": [],
  "errors": [],
  "timings_ms": {}
}
```

The API keeps the review engine separate from the UI.

---

# Gradio Frontend

The frontend is a separate Gradio application.

```text
Browser
   │
   ▼
Gradio :7860
   │
   │ HTTP
   ▼
FastAPI :8000
   │
   ├── Bandit
   ├── Ruff
   ├── Qwen2.5-Coder
   └── OWASP reference layer
```

The frontend does not load the model itself.

The model stays inside the backend process.

The UI includes:

* Dark developer-tool layout
* Code editor
* Severity indicators
* Finding cards
* Suggested fixes
* OWASP reference panels
* Backend/model status
* Error states

---

# Frontend Security

The frontend also validates URLs before rendering OWASP links.

Only:

```text
http://
https://
```

URLs are allowed.

Potentially dangerous schemes such as:

```text
javascript:
data:
```

are rejected.

Finding text is also HTML-escaped before being rendered.

---

# A Generation Bug That Looked Like a Model Problem

After getting the model working through the API, there was another issue.

The model correctly detected the vulnerability, but its output sometimes continued into another training example:

```text
Verdict: ISSUES FOUND
...

Human: Review this code:
...
Verdict: NO ISSUES FOUND
...
```

At first this looked like another fine-tuning problem.

It wasn't.

The tokenizer was using:

```text
EOS: <|im_end|>
PAD: <|endoftext|>
```

The generation configuration wasn't explicitly passing the appropriate EOS and padding IDs.

The configuration was changed to explicitly use:

```python
eos_token_id=tokenizer.eos_token_id
pad_token_id=tokenizer.pad_token_id
```

After that, the response stopped correctly at the end of the review.

The same test went from roughly 145 seconds to roughly 56 seconds.

---

# Testing

The project has separate tests for the different components.

### Parser

`test_parser.py`

Tests:

* Clean responses
* Single findings
* Multiple findings
* Markdown formatting
* Lowercase labels
* Missing colons
* Preamble text
* Truncated findings
* Category-only findings
* Legacy formats
* Wrapped continuation lines
* Garbage/empty output

### Merge Logic

`test_merge.py`

Tests:

* Static findings are never dropped
* LLM/static corroboration
* Deduplication
* Independent LLM findings
* Verdict generation
* Parser failures
* Non-Python input
* Oversized input

### API

`test_api.py`

Tests:

* `/health`
* `/review`
* Request validation
* Bandit/Ruff integration
* SQL injection detection
* Corroboration
* JavaScript analyzer skipping
* Clean code
* Automatic language detection
* Error handling

### OWASP

`test_security_reference.py`

The OWASP layer has **158 checks** covering:

* Index integrity
* Supported/unsupported mappings
* Category gating
* Word boundaries
* Summary length
* Source text
* Reference enrichment
* Failure handling

### Frontend

`test_frontend.py`

Tests:

* Request construction
* Result rendering
* Severity display
* OWASP blocks
* HTML escaping
* Unsafe URL rejection
* Error handling
* Backend communication

---

# Project Structure

```text
CodeSentry/
│
├── api.py
├── app.py
├── review_core.py
├── combined_review.py
├── security_reference.py
├── bench_llm.py
├── model_download.py
│
├── dataset_project/
├── samples/
│
├── OWASP_Cheatsheets_Book.pdf
├── DIAGNOSIS_REPORT.md
│
├── test.py
├── test_api.py
├── test_finetuned.py
├── test_frontend.py
├── test_merge.py
├── test_parser.py
└── test_security_reference.py
```

Local-only artifacts such as model weights, virtual environments, caches, and datasets are kept out of Git where appropriate.

---

# Running Locally

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

---

# Hardware / Local Inference

The model used is:

```text
Qwen2.5-Coder-3B-Instruct
```

with a LoRA adapter.

The model is loaded lazily and kept as a process-wide singleton so the weights aren't repeatedly loaded for every request.

Local CPU inference is relatively slow, which is one reason the frontend and backend are kept separate.

---

# What I Learned

The biggest thing I learned from this project wasn't LoRA itself.

It was that getting an ML project to **run** and getting it to actually **work** are very different things.

The first model trained successfully.

It was still useless.

The important part was being able to measure that:

```text
Training succeeded
        ≠
Model learned the task
```

The same thing happened later with the generation bug.

The model was detecting the vulnerability.

The output still looked broken because the generation configuration was wrong.

And then there was the static-analysis side.

A model can be useful without being trusted with the entire decision.

That led to the current architecture where:

```text
LLM + deterministic tools
```

work together instead of the LLM replacing everything.

---

# Current Limitations

CodeSentry is still a project, not a replacement for a full production security scanner.

Current limitations include:

* The fine-tuned model is only 3B parameters.
* CPU inference is slow.
* The v2 evaluation was not completed across the entire 49-case set in the initial run.
* Static analysis coverage is strongest for Python.
* OWASP references are limited to the included cheat-sheet corpus.
* The model can still produce incorrect or incomplete findings.
* A larger independent benchmark is needed before making strong claims about general performance.

---

# What's Next

* Dockerize the backend
* Containerize the frontend
* Add GitHub Actions
* Improve dependency management
* Finish the full v2 evaluation
* Benchmark latency
* Improve multi-language support
* Deploy a reproducible version

---

# Built With

* Python
* PyTorch
* Hugging Face Transformers
* PEFT / LoRA
* Qwen2.5-Coder-3B-Instruct
* FastAPI
* Gradio
* Bandit
* Ruff
* OWASP Cheat Sheet Series
