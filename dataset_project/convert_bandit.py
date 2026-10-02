"""
Converts Bandit JSON findings into training examples matching our schema.
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

SEVERITY_MAP = {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}

EXPLANATIONS = {
    "hardcoded_password_string": "Hardcoded credentials in source code can be discovered by anyone with code access and are difficult to rotate.",
    "assert_used": "assert statements are removed when Python is run with optimizations (-O), so any checks relying on them silently disappear in production.",
    "blacklist": "This function can execute arbitrary code from its input, which is dangerous if the input is not fully trusted.",
    "try_except_pass": "Silently swallowing exceptions with try/except/pass hides real errors and makes debugging very difficult.",
}

DEFAULT_EXPLANATION = "This pattern is flagged by static analysis as a potential issue; review the surrounding context to confirm."


def load_bandit_findings(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("results", [])


def bandit_finding_to_example(finding):
    code_snippet = finding.get("code", "").strip()
    test_name = finding.get("test_name", "unknown")
    severity = SEVERITY_MAP.get(finding.get("issue_severity", "LOW"), "Low")
    problem = finding.get("issue_text", "Potential issue detected").strip()
    explanation = EXPLANATIONS.get(test_name, DEFAULT_EXPLANATION)
    line_number = finding.get("line_number", 1)

    return {
        "id": f"bandit_{finding.get('test_id', 'unk')}_{line_number}",
        "source": "bandit",
        "test_name": test_name,
        "filename": finding.get("filename", ""),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Review this code:\n\n```python\n{code_snippet}\n```",
            },
            {
                "role": "assistant",
                "content": (
                    f"Category: Security\n"
                    f"Severity: {severity}\n"
                    f"Line: {line_number}\n"
                    f"Problem: {problem}\n"
                    f"Explanation: {explanation}\n"
                    f"Suggested Fix: [REVIEW NEEDED -- write a concrete fix for this finding]"
                ),
            },
        ],
    }


if __name__ == "__main__":
    findings = load_bandit_findings("bandit_findings.json")
    examples = [bandit_finding_to_example(f) for f in findings]

    with open("bandit_training_candidates.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Converted {len(examples)} Bandit findings into training candidates.")
    print("Written to bandit_training_candidates.jsonl")
    print("\n*** IMPORTANT ***")
    print("These are CANDIDATES, not final training data. Each one needs:")
    print("  1. Manual review -- is this a real issue or a Bandit false positive?")
    print("  2. A real Suggested Fix written in -- currently a placeholder.")
    print("Do NOT merge this file into training_data.jsonl without reviewing it first.")
