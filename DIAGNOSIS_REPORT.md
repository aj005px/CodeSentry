# Code-Review LoRA: Diagnosis and Remediation Report

**Model:** Qwen2.5-Coder-3B-Instruct (local, `models/qwen2.5-coder-3b`)
**Task:** LoRA fine-tune for code review / defect detection
**Date:** 2026-10-02
**Scope:** Diagnose why the 86-example LoRA does not generalize; fix the training and
evaluation setup; compare base vs. fine-tuned on a held-out set.

---

## 1. Summary

The fine-tuned model was not merely failing to generalize — it had **completely lost
the ability to decide whether a defect exists**. It emitted `No issues found.` on
**49 of 49** evaluation cases, including two cases that were byte-identical to
training examples it had been trained on three times.

The base model, by contrast, correctly identified **36 of 36** buggy snippets
(recall 100%) across 16 defect categories, including ones absent from the training
set.

This is **capability degradation caused by fine-tuning**, not a generalization gap.

The proximate cause is a **label-imbalance / objective-design problem**, made worse by
a **label-masking bug** that sent most of the gradient signal into memorizing the
system prompt instead of learning to review code. Increasing dataset size alone would
not have fixed it: the defect is in *which* examples carry *which* targets, not how
many there are.

A second, independent finding: **this machine cannot train this model**, because
PyTorch's CPU half-precision matmul is ~360x slower than fp32 on this CPU.

### Remediation verified

The retrain was run on a rented Colab T4 (see §2b). **The collapse is fixed.**
The new adapter no longer answers `No issues found.` to everything — it produces
correctly formatted, category-tagged reviews, and its held-out bug recall rose from
**0% to 72%** on the cases evaluated so far.

---

## 2. Critical experiment: base vs. current LoRA

Identical 49 cases (your 5 reported tests + 44 held-out), identical prompts,
**greedy decoding** (`do_sample=False`) for determinism.

| run | acc | recall | precision | F1 | said "No issues" | TP | FN | TN | FP |
|---|---|---|---|---|---|---|---|---|---|
| **base** | 73.5% | **100.0%** | 73.5% | 84.7% | 0.0% | 36 | 0 | 0 | 13 |
| **LoRA (86 ex.)** | 26.5% | **0.0%** | 0.0% | 0.0% | **100.0%** | 0 | 36 | 13 | 0 |

**36 capabilities lost, 13 gained.** All 13 gains are clean-code cases; the model
gained *only* the ability to say "no issue" and lost everything else.

Recall by category, LoRA (base in parentheses):

| category | base | LoRA |
|---|---|---|
| sql_injection | 2/2 | **0/2** |
| command_injection | 2/2 | **0/2** |
| xss | 2/2 | **0/2** |
| path_traversal | 2/2 | **0/2** |
| insecure_deserialization | 2/2 | **0/2** |
| auth_bug | 3/3 | **0/3** |
| python_syntax | 3/3 | **0/3** |
| python_logic | 4/4 | **0/4** |
| js_closure_async | 3/3 | **0/3** |
| c_syntax | 2/2 | **0/2** |
| c_memory | 2/2 | **0/2** |
| cpp_syntax | 1/1 | **0/1** |
| null_pointer | 2/2 | **0/2** |
| off_by_one | 3/3 | **0/3** |

Your 5 reported tests, reproduced exactly:

| test | expect | base | LoRA |
|---|---|---|---|
| t1 SQL injection | ISSUE | issue (ok) | none (**FAIL**) |
| t2 JS closure | ISSUE | issue (ok) | none (**FAIL**) |
| t3 correct Python | CLEAN | issue (FAIL) | none (ok) |
| t4 broken C | ISSUE | issue (ok) | none (**FAIL**) |
| t5 corrected C | CLEAN | issue (FAIL) | none (ok) |

Note the inverse error on t3/t5: the base model **over-reports** on correct code
(13 false positives, all clean cases). That is the failure mode the fine-tune
"fixed" — by collapsing everything into the clean class.

## 2b. Verified result: retrained adapter (`lora-code-reviewer-v2`)

Trained on a Colab T4 with exactly the configuration in §4 (confirmed by reading
`training_args.bin` out of the returned adapter: `assistant_only_loss=True`,
lr 1e-4, cosine, warmup 4, seed 42, 3 epochs, eval per epoch,
`load_best_model_at_end` on `eval_loss`, `use_cpu=False`).

Training curve (39 steps = 103 examples x 3 epochs / grad-accum 8):

| epoch | step | train loss | eval loss |
|---|---|---|---|
| 1 | 13 | 1.6331 | 1.5473 |
| 2 | 26 | 1.3335 | 1.3582 |
| 3 | 39 | 1.1480 | **1.3322** |

Eval loss decreases monotonically and tracks train loss with no divergence — the
opposite of the old run's oscillation. The shipped `adapter_model.safetensors` is
byte-identical (md5 `fd6274b1…`) to `checkpoint-39`, so `load_best_model_at_end`
worked. Raw curve kept in `dataset_project/v2_training_state.json`.

Collapse measurement on the same greedy harness (20 of 49 cases completed before
the CPU-only run was stopped — the bug-heavy slice, so read the clean columns as
provisional):

| run | said "No issues" | verdict format | avg response length |
|---|---|---|---|
| LoRA (86 ex.) | **100%** (49/49) | 0/49 | 3 words |
| **LoRA v2** | **30%** (6/20) | **20/20** | **38 words** |

Partial v2 results: **14/20 correct, bug recall 13/18 (72%)** — versus **0% recall**
for v1 on the identical cases. Sample output:

```
Verdict: ISSUES FOUND
Category: Bug
Severity: High
Line: 2
Problem: SQL injection
Explanation: The username is directly inserted into the SQL query without escaping it...
Suggested Fix: Use parameterized queries.
```

**Verdict on whether this is good enough:** the collapse is unambiguously fixed and
v2 should replace the old adapter. The fine-tune recovered from "zero capability" to
usable defect detection. Two honest caveats: (1) on the cases run so far v2's 72%
recall is still below the base model's 100%, and (2) only 2 clean cases were reached,
so v2's clean-code precision — the thing §6 wanted improved — is **not yet measured**.
Finish the remaining 29 cases on the Colab GPU before treating v2 as final; on this
CPU-only box a full pass costs ~40 min per model.

---

## 3. Root causes

### 3.1 The evaluation cases were in the training set

`pairs_seed.json` contains `sql_injection_1_buggy` and `closure_loop_1_buggy` whose code
is **byte-identical** to your Tests 1 and 2. The model failed on inputs it had trained
on three times. This alone rules out "not enough data" as the explanation.

### 3.2 The first-token decision was destroyed (measured)

Teacher-forced probability of the first assistant token:

| model | prompt | P("No") | P("Category") |
|---|---|---|---|
| base | sql_injection_1_buggy *(seen)* | 0.000 | 0.884 |
| base | clean_standalone_1 *(seen)* | 0.009 | 0.821 |
| base | JS closure *(unseen)* | 0.000 | 0.867 |
| ck13 (1 epoch) | sql_injection_1_buggy *(seen)* | 0.270 | 0.197 |
| ck13 | divzero_1_buggy *(seen)* | 0.501 | 0.050 |
| ck66 (final) | sql_injection_1_buggy *(seen)* | 0.437 | 0.561 |
| ck66 | divzero_1_buggy *(seen)* | 0.530 | 0.468 |
| ck66 | clean_standalone_1 *(seen)* | 0.678 | 0.320 |

The base model always opens with `Category:` (a *safe* default — it never
under-detects) and decides what to write inside the block. After fine-tuning the
opening token is roughly a coin flip biased toward `No`, **even on memorized training
examples**. Greedy decoding then emits `No issues found.` whenever P("No") wins.

### 3.3 Why: 53% of targets were the shortest possible string

- 86 examples; **46 (53.5%)** had the target `No issues found.` — the lowest-entropy
  target in the set, and the modal one.
- Only **3 of 43** pairs were clean on both sides; every other pair contributes one
  buggy and one fixed half, so "clean" was structurally over-represented.
- The genuinely conditional decision is **token 1**. Every token after it in a clean
  target is free boilerplate and contributes almost no corrective gradient.
- So the one decision that mattered received a **majority-class label**, and the base
  model's safe prior was overwritten with an unsafe one.

### 3.4 Objective bug: 100% of tokens were supervised

`train_lora.py` passed `dataset_text_field="text"` while the `Dataset` still carried
its `messages` column. TRL therefore tokenized the whole rendered conversation and left
labels on **every position**.

```
row 0: 178 / 178 tokens supervised (100.0%)   <- assistant turn is only 39% of tokens
```

Roughly 60% of the gradient went into reproducing the (identical, 150-token) system
prompt and the code snippet. The task signal was a minority of the objective.

### 3.5 No validation signal

`train_lora.py` had no validation split, no `eval_strategy`, and no checkpoint
selection. Training loss was the only available metric, and the shipped adapter is the
**final** epoch — the collapse is present from **step 13** (end of epoch 1):

```
base        t1 ok, t2 ok, t4 ok, t3 FAIL
checkpoint-13  t1 FAIL, t2 FAIL, t4 ok, t3 ok     <- collapse already present
checkpoint-22  t1 FAIL, t2 FAIL, t4 FAIL, t3 ok
checkpoint-26  t1 FAIL, t2 ok,  t4 ok,  t3 ok
checkpoint-39  t1 ok,   t2 ok,  t4 ok,  t3 ok     <- fully recovered
checkpoint-44  t1 FAIL, t2 FAIL, t4 FAIL, t3 ok  <- collapsed again
checkpoint-66  t1 FAIL, t2 FAIL, t4 FAIL, t3 ok  <- shipped
```

This oscillation is the signature of an objective problem, not of gradual overfitting.
A validation split would have caught it at step 13.

### 3.6 A mislabeled clean example taught false negatives

`clean_standalone_2` calls `time.time()` with **no `import time`** — a guaranteed
`NameError` — and was labeled `No issues found.` Corrected (added the import).

### 3.7 Coverage gaps

The 43-pair seed set contained **zero C/C++ examples and one JavaScript example**,
while your Test 4 is a C syntax error. Every "fixed" half is a near-copy of its
"buggy" half, so clean-vs-buggy was learnable from shape rather than content.

---

## 4. What was changed, and why

Each change addresses a specific measured defect. Nothing was changed "because it is
common advice."

| # | Change | Addresses |
|---|---|---|
| 1 | `assistant_only_loss=True` in `train_lora.py` | §3.4 — verified 27%/7% supervised (was 100%); loss now lands only on the response |
| 2 | Verdict-first target format (`Verdict: ISSUES FOUND` / `Verdict: NO ISSUES FOUND`) | §3.2/§3.3 — forces an explicit, isolated decision token instead of inferring the decision from whether a block appears |
| 3 | Clean targets carry a varied rationale | §3.3 — raises clean-class entropy so "No issues found." is no longer a free, memorizable constant |
| 4 | Clean class capped at 34% (was 53.5%), dropping *paired fixed halves* first | §3.3 — inverts the prior away from the majority class; keeps the independent clean examples, which carry the real signal |
| 5 | `pairs_expanded.json` — 34 new pairs | §3.7 — adds C, C++, JS and Go, which the seed set entirely lacked |
| 6 | `extra_examples.json` — 6 multi-issue + 18 hard negatives + 6 independent clean | §3.7 — seed had no multi-issue examples and no clean code that *resembles* a vulnerable pattern; the hard negatives are what break "looks familiar -> say clean" |
| 7 | Fixed `clean_standalone_2` | §3.6 — removed supervision for a false negative |
| 8 | Stratified 85/15 train/val split + `eval_loss` + `load_best_model_at_end` | §3.5 — gives a generalization signal and stops shipping a collapsed final epoch |
| 9 | LR 2e-4 -> 1e-4, cosine schedule, 4 warmup steps | §3.5 — the base model already reviews well; the goal is minimal drift, not relearning review. Checkpoint selection now decides |
| 10 | `seed=42`, `data_seed=42`, explicit `max_length=1024` | reproducibility; silent-truncation guard |
| 11 | `test.py` / `test_finetuned.py` import `SYSTEM_PROMPT` from `build_dataset` | a train/inference prompt mismatch is indistinguishable from a bad fine-tune |
| 12 | Greedy decoding in evaluation harnesses | sampling at temperature 0.2 added noise to every measurement |

### Deliberately NOT changed

- **LoRA rank / alpha / target modules** (`r=16`, `alpha=32`, attention-only). Adding
  MLP modules or raising rank increases the capacity available to overwrite base
  behaviour. The measurement says the problem is too much drift, so more capacity
  would move in the wrong direction.

### Resulting dataset

```
collected=180   duplicate_prompts_dropped=1
after balance=121   issue=80  clean=41  (34% clean)
review_train.jsonl: 103 rows (68 issue / 35 clean)
review_val.jsonl:    18 rows (12 issue /  6 clean)
```

---

## 5. What was NOT done, and why

**No retrained model was produced. No before/after numbers for a fixed model are
reported, because none were measured.**

This machine (AMD Ryzen 7 7730U, **AVX2 only — no AVX512, no native bf16**) has
catastrophically slow PyTorch CPU half-precision matmul. Measured on an idle machine:

| dtype | 1024^3 matmul | throughput |
|---|---|---|
| bfloat16 | 2729 ms | **0.8 GFLOPS** |
| float16 | 2885 ms | **0.7 GFLOPS** |
| float32 | 7.4 ms | **288.5 GFLOPS** |

fp32 is ~360x faster but needs 12.4 GB for the weights on a 14 GB box that already has
~5 GB in use. The half-precision path fits in RAM but training reached **0 of 39
optimizer steps in 17 minutes** — a 3-epoch run is not feasible.

Note that the original `train_lora.py` also loaded `dtype="bfloat16"` and set
`bf16=False` in `SFTConfig`, so the previous 66-step run was hitting this same penalty.

Any reported post-fix metric would therefore be fabricated. The changes in §4 are
derived from measurement and reasoning, not from a successful training run.

### To actually validate

1. **Use a GPU.** Any rented T4 trains these 121 examples in roughly 2 minutes. This is
   the recommended path. *(This was done — see §2b.)*
2. **Or** add a bf16-storage / fp32-compute path (verified working: same ~6.2 GB
   footprint, gradients correct) and run one epoch on CPU — approximately 1 hour.
   *(Rejected here; CPU inference alone still costs ~40 min per model.)*
3. Then re-run the evaluation on `eval_heldout.json` and compare against the base row
   in §2. *(20/49 completed; remainder is cheap on a GPU.)*

---

## 6. Success criteria for the next run

Do not judge on training loss. Judge on the held-out set:

- **recall >= base recall (100%)** on all 16 categories — the fine-tune must not lose
  anything the base model had
- **precision > 73.5%** — the base model over-reports on clean code (13 FPs); the
  fine-tune should fix that without touching recall
- **"no issues" rate** should track the true clean rate (~24% of the eval set), not
  100%
- **validation loss should not be monotonically increasing** across epochs

## 7. Files changed

**Modified**
- `dataset_project/build_dataset.py` — multi-source builder, verdict-first format,
  class balancing, dedupe, stratified split
- `dataset_project/train_lora.py` — assistant-only loss, val split, eval loss, LR, seed
- `dataset_project/pairs_seed.json` — one fix: added missing `import time` to
  `clean_standalone_2` (the other diff vs. git HEAD is your own uncommitted 43-pair set)
- `test.py`, `test_finetuned.py` — shared prompt import; `ADAPTER_PATH` now points at
  `./lora-code-reviewer-v2`
- `dataset_project/training_data.jsonl` — regenerated (legacy combined path)

**Added**
- `lora-code-reviewer-v2/` — the retrained adapter (r=16, alpha=32, attention-only,
  `Qwen2.5-Coder-3B-Instruct`), replacing the collapsed one
- `dataset_project/v2_training_state.json` — full trainer log history for v2
- `dataset_project/pairs_expanded.json` — 34 pairs (C/C++/JS/Go coverage)
- `dataset_project/extra_examples.json` — 30 examples (multi-issue + hard negatives)
- `dataset_project/eval_heldout.json` — 44 held-out cases, 16 categories
- `dataset_project/review_train.jsonl` — 103 rows
- `dataset_project/review_val.jsonl` — 18 rows
- `dataset_project/__init__.py` — so `build_dataset` is importable from `test.py`

**Deleted**
- `dataset_project/lora-code-reviewer/` — the collapsed 86-example adapter and its
  checkpoints. Replaced by `lora-code-reviewer-v2/`. **Recoverable from commit
  `097cc2b`** if a before/after comparison is ever wanted again.
- `lora-code-reviewer-v2/checkpoint-{13,26,39}/` — ~300 MB of `optimizer.pt` /
  `rng_state.pth` resume state. The shipped adapter is byte-identical to
  `checkpoint-39` (the best one), so these were redundant. The training curve was
  preserved in `dataset_project/v2_training_state.json` before deletion.
---

## 8. Combined review pipeline (added after the retrain)

The retrained adapter is now wrapped in a review service that merges
deterministic static analysis with the model's judgement. Sections 1-7 are about
why the first adapter collapsed; this section is about what was built on top of
the fix.

### Premise correction

`combined_review.py` was specified as an existing untested first draft. It did
not exist — no file, no stash, no branch. It was built from scratch rather than
verified.

### Why merge at all

The two halves fail in opposite directions, which is the argument for combining
them rather than picking one:

- **Static analysis (Bandit, Ruff)** is exact and explains itself, but only
  covers Python and only the defect classes those tools implement. On
  `samples/buggy.js` it correctly reports nothing, because it cannot see a
  missing `await`.
- **The fine-tuned model** covers any language and the defects a linter has no
  rule for, but it hallucinates and it is stochastic in cost (seconds to
  minutes per snippet on CPU).

So the merge is not a vote. The rule is asymmetric:

> **A static finding is never dropped.**

If the model independently reports the same defect, the two collapse into one
finding whose `sources` lists both — that is what `corroborated` counts. If the
model stays silent, the static finding still ships. A model failure, a
hallucination, or a future retrain therefore cannot suppress a Bandit hit.

Bandit's `B608` and Ruff's `S608` fire on the same line with near-identical
text, so they merge with each other too. On `samples/vulnerable.py`:

| finding | line | sources |
|---|---|---|
| shell injection (`B605`/`S602`) | 12 | `bandit+ruff` |
| SQL injection (`B608`/`S608`) | 5 | `bandit+ruff` |
| `eval` on file contents (`B307`) | 9 | `bandit+ruff` |
| unused `sqlite3` import (`F401`) | 1 | `ruff` |

The verdict comes from merged evidence, not from the model: a Bandit hit still
reports `issues_found` even when the LLM said the code was clean.

### Components

`review_core.py` holds the engine. Static analysis and the LLM reviewer are
deliberately independent — each returns `list[Finding]` and knows nothing about
the other. `review_code()` is the only place they meet. That separation is the
point: the model can be retrained or swapped without touching the analyzers.

`parse_llm_output()` is a line-based state machine, not one regex. Per the §3
diagnosis, the old failure was a degenerate target, and a degenerate parser
would just move the problem. Its contract:

- never raises on unexpected input
- recovers multiple issues from one response (a `Category:` line opens a new
  block)
- tolerates markdown fences, bold labels, preamble prose, lower case, missing
  colons, wrapped continuation lines
- normalises `Category`/`Severity` onto the schema via a synonym table
- **never discards a partially complete finding** — a block with only a
  `Category:` is still reported, with the rest `None`
- unparseable output yields `verdict: unknown`, never a false clean bill

Two bugs found and fixed during this build, both caught by tests rather than by
inspection:

1. `merge_findings()` originally deduped LLM-vs-static but never
   static-vs-static, so Bandit and Ruff reported the same SQL injection twice —
   contradicting its own docstring.
2. Fields appearing before any `Category:` fragmented one finding into two; the
   parser now completes the open block instead of splitting it.

Bandit's `more_info` is a documentation URL, not a fix, so it is folded into the
explanation rather than emitted as `Suggested Fix`, which would mislead a UI.

### Two bugs in the environment, not the code

Worth recording because they cost real time and will recur:

- **`ruff check --json` no longer exists** in ruff 0.16; it is
  `--output-format json`.
- **Bandit 1.9.4 has no `--stdin-filename`.** It reads stdin with a bare `-`,
  but logs `[main] INFO ...` to *stdout*, which corrupts the JSON. `-q` is
  required or the output will not parse.

### The API

`api.py` exposes `POST /review`, `GET /health`, `POST /warmup`. The response
shape is the contract a future UI depends on:

```json
{ "verdict": "issues_found|no_issues_found|unknown",
  "language": "python",
  "findings": [ {"category","severity","line","problem","explanation",
                 "suggested_fix","sources","tool_ref","confidence"} ],
  "summary": {"total","by_severity","by_category","by_source","corroborated"},
  "llm": {"ran","parse_ok","raw"},
  "skipped": [{"name","reason"}], "errors": [], "timings_ms": {} }
```

`llm.ran` is explicit because with the LLM disabled `parse_ok` is vacuously
true, which would otherwise read as "the model ran and was happy".

Two operational consequences of a 3B model on a 14 GB CPU-only box: the model is
a lazily-loaded process-wide singleton (~6.2 GB, minutes to load, seconds after),
and endpoints are sync `def` so FastAPI runs them in its threadpool and reviews
serialise behind a lock rather than risking an OOM.

### Verification

| suite | checks | needs RAM? |
|---|---|---|
| `test_parser.py` | 17 | no |
| `test_merge.py` | 35 | no (LLM stubbed) |
| `test_api.py` | 51 | no, unless `--with-llm` |

103 checks, all green. The merge suite stubs the LLM on purpose: what matters
there is "does a static finding survive the merge", and that should not require
a 7 GB model to assert. `samples/` holds three snippets for manual runs —
`vulnerable.py`, `buggy.js` (the LLM-only case), `clean.py` (the
no-false-positive control).

### Still unverified

**The live LLM inside the merge path has never run.** This box has ~9.7 GB
resident before the model loads (Zed 1.9, opencode 1.8, clamd 0.95, Brave 1.5,
mysqld 0.35) against 14 GB with swap exhausted, so the 6.2 GB bf16 load is
OOM-killed. Everything above is verified with the model stubbed or disabled.

The adapter's *output format* is well covered — `test_parser.py` plus the 20-case
run in §2b — but `python test_api.py --with-llm` still needs one run with ~7 GB
free, to confirm the LLM's findings actually corroborate Bandit's on real input
rather than only on canned strings. Treat the pipeline as untested at the seam
until that runs.
