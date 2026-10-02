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

A second, independent finding: **this machine cannot train this model at all**, because
PyTorch's CPU half-precision matmul is ~360x slower than fp32 on this CPU. This is
why no retrained model is reported below.

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
   the recommended path.
2. **Or** add a bf16-storage / fp32-compute path (verified working: same ~6.2 GB
   footprint, gradients correct) and run one epoch on CPU — approximately 1 hour.
3. Then re-run the evaluation on `eval_heldout.json` and compare against the base row
   in §2.

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
- `test.py`, `test_finetuned.py` — shared prompt import
- `dataset_project/training_data.jsonl` — regenerated (legacy combined path)

**Added**
- `dataset_project/pairs_expanded.json` — 34 pairs (C/C++/JS/Go coverage)
- `dataset_project/extra_examples.json` — 30 examples (multi-issue + hard negatives)
- `dataset_project/eval_heldout.json` — 44 held-out cases, 16 categories
- `dataset_project/review_train.jsonl` — 103 rows
- `dataset_project/review_val.jsonl` — 18 rows
- `dataset_project/__init__.py` — so `build_dataset` is importable from `test.py`

**Untouched**
- `dataset_project/lora-code-reviewer/` — the original adapter, left intact for
  comparison