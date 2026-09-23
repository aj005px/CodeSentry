"""
Converts pairs_seed.json (buggy/fixed matched pairs) into flat training
examples matching the schema validated in Phase 2 prompt testing:

Category: [Bug|Performance|Quality|Security]
Severity: [Low|Medium|High]
Line: <line number>
Problem: <one-line description>
Explanation: <why this is a problem>
Suggested Fix: <concrete fix>

Each pair produces TWO training examples: one buggy (with issues) and one
fixed (with issues: []), so the model sees matched contrastive examples
rather than only ever seeing "code -> bug found" in isolation.
"""

import json

SYSTEM_PROMPT = """You are a code review assistant. Given a code snippet, \
identify issues and return them in this exact format for each issue found:

Category: [Bug|Performance|Quality|Security]
Severity: [Low|Medium|High]
Line: <line number>
Problem: <one-line description>
Explanation: <why this is a problem>
Suggested Fix: <concrete fix>
"""


def format_issues_as_text(issues: list) -> str:
    """Render a list of issue dicts into the exact target output format."""
    if not issues:
        return "No issues found."

    blocks = []
    for issue in issues:
        block = (
            f"Category: {issue['category']}\n"
            f"Severity: {issue['severity']}\n"
            f"Line: {issue['line']}\n"
            f"Problem: {issue['problem']}\n"
            f"Explanation: {issue['explanation']}\n"
            f"Suggested Fix: {issue['fix']}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def build_training_examples(pairs: list) -> list:
    examples = []

    for pair in pairs:
        for variant in ("buggy", "fixed"):
            entry = pair[variant]
            code = entry["code"]
            issues = entry["issues"]

            example = {
                "id": f"{pair['pair_id']}_{variant}",
                "category": pair["category"],
                "variant": variant,  # useful later for filtering/analysis
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Review this code:\n\n```\n{code}\n```",
                    },
                    {
                        "role": "assistant",
                        "content": format_issues_as_text(issues),
                    },
                ],
            }
            examples.append(example)

    return examples


if __name__ == "__main__":
    with open("pairs_seed.json") as f:
        pairs = json.load(f)

    examples = build_training_examples(pairs)

    with open("training_data.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Built {len(examples)} training examples from {len(pairs)} pairs.")
    print("Written to training_data.jsonl")

    # quick sanity check: print one buggy and one fixed example
    buggy_example = next(e for e in examples if e["variant"] == "buggy")
    fixed_example = next(e for e in examples if e["variant"] == "fixed")

    print("\n--- Sample buggy example ---")
    print(json.dumps(buggy_example, indent=2))

    print("\n--- Sample fixed example ---")
    print(json.dumps(fixed_example, indent=2))
