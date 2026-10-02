"""
Builds the code-review training set from three sources and emits a
train/validation split.

Sources
-------
pairs_seed.json       - the original 43 matched buggy/fixed pairs
pairs_expanded.json   - 34 further pairs, adding the C/C++/JS/Go coverage the
                        seed set is missing entirely (seed has 0 C/C++ and 1 JS)
extra_examples.json   - snippets with multiple issues, plus clean snippets that
                        deliberately resemble vulnerable shapes (hard negatives)

Why the output format changed
-----------------------------
The old target was either a 6-field issue block or the bare string
"No issues found.".  53% of targets were that one short string, so the
shortest, lowest-entropy target was also the modal one and the model learned a
prior ("say No issues found") instead of a decision.  The new target opens with
an explicit Verdict line, so the model has to commit to ISSUES FOUND vs NO
ISUES FOUND before it can produce any content.  Clean targets also carry a
short varied rationale, which (a) raises the entropy of the clean class so it
is not a memorisable constant and (b) forces the model to justify the negative.

Clean-side balancing
--------------------
Pairs emit both a buggy and a fixed half, and only 3 of the 43 seed pairs are
clean on both sides, so the old set was 53% clean.  For a detector whose
observed failure is under-reporting, that is the wrong prior.  We cap the clean
class and drop *paired fixed halves* first, keeping the independent hard
negatives, which carry the actual structural signal.
"""

import argparse
import json
import random

SEED_FILES = ["pairs_seed.json", "pairs_expanded.json"]
EXTRA_FILE = "extra_examples.json"

SYSTEM_PROMPT = """You are a code review assistant. Given a code snippet in any \
language, decide whether it contains a real defect: a bug, a syntax error, a \
crash or memory-safety problem, a security vulnerability, or a genuine \
performance or resource problem.

Rules:
- Report an issue only when you can name the specific construct that is wrong \
and say what goes wrong because of it. Do not report style preferences, naming \
choices, or speculative concerns.
- If the snippet is correct, say so. A snippet that merely looks unusual, uses \
low-level APIs, handles a nullable value, or contains a comment claiming a fix \
is not by itself a defect.

Respond in exactly this format.

If there is at least one issue, start with "Verdict: ISSUES FOUND" and then one \
block per issue:
Verdict: ISSUES FOUND
Category: [Bug|Security|Performance|Quality]
Severity: [Low|Medium|High]
Line: <line number>
Problem: <one-line description>
Explanation: <why this is a problem>
Suggested Fix: <concrete fix>

If there are no issues, respond with:
Verdict: NO ISSUES FOUND
<one sentence on what you checked and why the code is correct>
"""

VERDICT_ISSUES = "Verdict: ISSUES FOUND"
VERDICT_CLEAN = "Verdict: NO ISSUES FOUND"

# Varied so the clean class is not a single memorisable constant.
CLEAN_RATIONALES = [
    "The code is correct as written and needs no changes.",
    "Reviewed for syntax errors, logic errors and unsafe input handling; none apply here.",
    "No defect found: the logic is correct and the inputs are handled safely.",
    "This snippet is sound. The constructs used are appropriate for what the code does.",
    "Checked the control flow, edge cases and error paths; the code is correct.",
    "No issue to report: the implementation is straightforward and free of defects.",
    "The snippet behaves correctly for valid input and has no latent problem.",
    "Nothing here needs fixing; the code is well formed and correct.",
]


def format_issues_as_text(issues: list, rationale: str | None = None) -> str:
    """Render a list of issue dicts (or a clean verdict) into the target format."""
    if not issues:
        return f"{VERDICT_CLEAN}\n{rationale or CLEAN_RATIONALES[0]}"

    blocks = []
    for issue in issues:
        blocks.append(
            f"Category: {issue['category']}\n"
            f"Severity: {issue['severity']}\n"
            f"Line: {issue['line']}\n"
            f"Problem: {issue['problem']}\n"
            f"Explanation: {issue['explanation']}\n"
            f"Suggested Fix: {issue['fix']}"
        )
    return "\n".join([VERDICT_ISSUES, *blocks])


def _clean_rationale(key: str) -> str:
    h = sum(ord(c) * (i + 1) for i, c in enumerate(key))
    return CLEAN_RATIONALES[h % len(CLEAN_RATIONALES)]


def _example(pair_id, category, variant, code, issues, priority):
    return {
        "id": f"{pair_id}_{variant}",
        "category": category,
        "variant": variant,
        "_has_issue": len(issues) > 0,
        "_priority": priority,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Review this code:\n\n```\n{code}\n```"},
            {"role": "assistant", "content": format_issues_as_text(issues, _clean_rationale(pair_id + variant))},
        ],
    }


def collect(out_dir="."):
    """Return (candidates, stats). priority: 0 = keep, 1 = droppable."""
    cands = []

    for fname in SEED_FILES:
        try:
            with open(f"{out_dir}/{fname}") as f:
                blob = json.load(f)
        except FileNotFoundError:
            continue
        pairs = blob["pairs"] if isinstance(blob, dict) else blob
        for pair in pairs:
            for variant in ("buggy", "fixed"):
                entry = pair[variant]
                issues = entry["issues"]
                # a pair whose buggy side is already clean contributes one
                # example, not two identical ones
                if variant == "fixed" and not issues:
                    if not pair["buggy"]["issues"]:
                        continue
                # clean halves of a pair are droppable; buggy halves are not
                prio = 1 if not issues else 0
                cands.append(_example(pair["pair_id"], pair["category"], variant,
                                      entry["code"], issues, prio))

    try:
        with open(f"{out_dir}/{EXTRA_FILE}") as f:
            extras = json.load(f)["examples"]
    except FileNotFoundError:
        extras = []
    for ex in extras:
        # independent clean snippets and hard negatives are kept unconditionally:
        # they are the examples that stop "looks familiar" from implying "clean"
        prio = 0  # independent clean + hard negatives are never dropped
        cands.append(_example(ex["id"], ex["category"], "standalone",
                              ex["code"], ex["issues"], prio))

    return cands


def dedupe(cands):
    seen, out, dropped = set(), [], 0
    for c in cands:
        key = c["messages"][1]["content"]
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(c)
    return out, dropped


def balance(cands, max_clean_frac=0.34, seed=42):
    """Cap the clean class, dropping paired fixed halves before anything else."""
    rng = random.Random(seed)
    issue = [c for c in cands if c["_has_issue"]]
    clean_keep = [c for c in cands if not c["_has_issue"] and c["_priority"] == 0]
    clean_drop = [c for c in cands if not c["_has_issue"] and c["_priority"] == 1]

    budget = int(len(issue) * max_clean_frac / (1 - max_clean_frac))
    rng.shuffle(clean_drop)
    chosen = clean_drop[: max(0, budget - len(clean_keep))] + clean_keep
    return issue + chosen, budget


def make_split(cands, val_frac=0.15, seed=42):
    """Stratified by _has_issue so both classes appear in validation."""
    rng = random.Random(seed)
    train, val = [], []
    for flag in (True, False):
        group = [c for c in cands if c["_has_issue"] is flag]
        rng.shuffle(group)
        k = max(1, int(len(group) * val_frac)) if len(group) > 1 else 0
        val.extend(group[:k])
        train.extend(group[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _clean_fields(c):
    return {k: v for k, v in c.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--max-clean-frac", type=float, default=0.34)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="review")
    args = ap.parse_args()

    cands = collect(args.data_dir)
    cands, dropped = dedupe(cands)
    print(f"collected={len(cands)}  duplicate_prompts_dropped={dropped}")

    bal, budget = balance(cands, args.max_clean_frac, args.seed)
    n_issue = sum(1 for c in bal if c["_has_issue"])
    print(f"after balance={len(bal)}  issue={n_issue}  clean={len(bal)-n_issue} "
          f"({100*(len(bal)-n_issue)/len(bal):.0f}% clean, budget={budget})")

    train, val = make_split(bal, args.val_frac, args.seed)
    for split, rows in (("train", train), ("val", val)):
        path = f"{args.out_prefix}_{split}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(_clean_fields(r)) + "\n")
        ni = sum(1 for r in rows if r["_has_issue"])
        print(f"{path}: {len(rows)} rows ({ni} issue / {len(rows)-ni} clean)")

    # keep the legacy all-in-one file in sync for anything that still reads it
    with open("training_data.jsonl", "w") as f:
        for r in train + val:
            f.write(json.dumps(_clean_fields(r)) + "\n")
    print("training_data.jsonl: combined train+val (legacy path)")


if __name__ == "__main__":
    main()