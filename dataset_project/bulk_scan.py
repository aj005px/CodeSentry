"""
Bulk scanner: clones multiple small, real repos, runs Bandit + Ruff across all
of them, and auto-filters out known junk patterns before handing you candidates
to review. This replaces the one-repo-at-a-time manual process with something
that scales toward real volume (aiming for ~100-300 raw candidates, which after
filtering should leave a meaningfully larger set than the ~15 good examples
we've hand-picked so far).

Run this from dataset_project/. Requires: pip install bandit ruff (already done).
"""

import json
import subprocess
import os

# Small, real, varied repos -- mix of different domains/styles so we get bug
# variety, not just repeated findings of the same type from one codebase.
REPOS = [
    "https://github.com/pallets/flask.git",
    "https://github.com/psf/requests.git",
    "https://github.com/encode/httpx.git",
    "https://github.com/tiangolo/typer.git",
    "https://github.com/Textualize/rich.git",
]

SCAN_DIR = "bulk_scan/repos"
RESULTS_DIR = "bulk_scan/results"

# Known low-value or high-false-positive finding types to drop automatically,
# based on what we found reviewing findings by hand earlier (assert_used spam,
# hardcoded_password_string false positives on placeholder values, style nits).
AUTO_DROP_BANDIT_TYPES = {
    "assert_used",  # extremely common, low severity, floods results
}

AUTO_DROP_RUFF_CODES = {
    "E501",  # line too long -- style, not a real issue
    "E731",  # lambda assignment -- style preference
    "S101",  # assert used -- same spam problem as bandit's assert_used
    "E402",  # module import not at top -- style, rarely a real bug
    "E711",  # comparison to None with == -- style, linter auto-fixes this
    "E712",  # comparison to bool with == -- style, linter auto-fixes this
    "B018",  # useless expression -- usually trivial, low training value
}

# Cap how many findings of any single type we keep, so one noisy pattern
# (like S603 subprocess calls) can't flood the batch the way assert did.
MAX_PER_FINDING_TYPE = 8


def clone_repos():
    os.makedirs(SCAN_DIR, exist_ok=True)
    for url in REPOS:
        name = url.rstrip("/").split("/")[-1].replace(".git", "")
        target = os.path.join(SCAN_DIR, name)
        if os.path.exists(target):
            print(f"Already cloned: {name}")
            continue
        print(f"Cloning {name}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, target],
            check=False,
            capture_output=True,
        )


def run_bandit(repo_path: str, output_path: str):
    subprocess.run(
        [
            "bandit", "-r", repo_path, "-f", "json", "-o", output_path,
            "--exclude", "*/tests/*,*/test/*,*/alembic/*,*/migrations/*",
        ],
        capture_output=True,
    )


def run_ruff(repo_path: str, output_path: str):
    result = subprocess.run(
        ["ruff", "check", repo_path, "--select", "S,B,E,F", "--output-format", "json"],
        capture_output=True,
        text=True,
    )
    with open(output_path, "w") as f:
        f.write(result.stdout or "[]")


def scan_all_repos():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for url in REPOS:
        name = url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_path = os.path.join(SCAN_DIR, name)
        if not os.path.exists(repo_path):
            print(f"Skipping {name}, not cloned.")
            continue

        print(f"Scanning {name} with Bandit...")
        bandit_out = os.path.join(RESULTS_DIR, f"{name}_bandit.json")
        run_bandit(repo_path, bandit_out)

        print(f"Scanning {name} with Ruff...")
        ruff_out = os.path.join(RESULTS_DIR, f"{name}_ruff.json")
        run_ruff(repo_path, ruff_out)


def load_bandit_candidates(repo_name: str) -> list:
    path = os.path.join(RESULTS_DIR, f"{repo_name}_bandit.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

    candidates = []
    for finding in data.get("results", []):
        test_name = finding.get("test_name", "")
        if test_name in AUTO_DROP_BANDIT_TYPES:
            continue
        candidates.append({
            "source": "bandit",
            "repo": repo_name,
            "test_name": test_name,
            "severity": finding.get("issue_severity", "LOW"),
            "filename": finding.get("filename", ""),
            "line": finding.get("line_number", 1),
            "issue_text": finding.get("issue_text", "").strip(),
            "code": finding.get("code", "").strip(),
        })
    return candidates


def load_ruff_candidates(repo_name: str) -> list:
    path = os.path.join(RESULTS_DIR, f"{repo_name}_ruff.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

    candidates = []
    for finding in data:
        code = finding.get("code", "")
        if code in AUTO_DROP_RUFF_CODES:
            continue
        candidates.append({
            "source": "ruff",
            "repo": repo_name,
            "rule_code": code,
            "filename": finding.get("filename", ""),
            "line": finding.get("location", {}).get("row", 1),
            "message": finding.get("message", "").strip(),
        })
    return candidates


if __name__ == "__main__":
    print("=== Step 1: Cloning repos ===")
    clone_repos()

    print("\n=== Step 2: Scanning with Bandit + Ruff ===")
    scan_all_repos()

    print("\n=== Step 3: Collecting and filtering candidates ===")
    all_candidates = []
    for url in REPOS:
        name = url.rstrip("/").split("/")[-1].replace(".git", "")
        bandit_c = load_bandit_candidates(name)
        ruff_c = load_ruff_candidates(name)
        print(f"{name}: {len(bandit_c)} bandit, {len(ruff_c)} ruff (after type-filter)")
        all_candidates.extend(bandit_c)
        all_candidates.extend(ruff_c)

    # Cap per finding-type so no single noisy pattern dominates the batch
    from collections import defaultdict
    capped = []
    type_counts = defaultdict(int)
    for c in all_candidates:
        key = c.get("test_name") or c.get("rule_code") or "unknown"
        if type_counts[key] < MAX_PER_FINDING_TYPE:
            capped.append(c)
            type_counts[key] += 1
    all_candidates = capped

    with open("bulk_scan/all_candidates.json", "w") as f:
        json.dump(all_candidates, f, indent=2)

    print(f"\nTotal candidates after type-filtering and per-type cap ({MAX_PER_FINDING_TYPE} max): {len(all_candidates)}")
    print("Written to bulk_scan/all_candidates.json")
    print("\nThese still need human review before becoming training data --")
    print("auto-filtering only removes KNOWN junk types, not all false positives.")
