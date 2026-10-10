"""Agent component classroom checks; stdlib, local CPU, no model/cloud calls.

python assets/agent/agent-components-lab.py --output assets/agent/agent-components-lab-results.json
This is a contract demonstration, not a sandbox/security/product benchmark.
"""
import argparse
import difflib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def code_task():
    before = "def mean(values):\n    return sum(values) / len(values)\n"
    after = "def mean(values):\n    return sum(values) / len(values) if values else 0\n"
    # Expectations are fixed separately from the edit, rather than generated
    # from the replacement function.
    check = (
        "from labmath import mean\n"
        "assert mean([]) == 0\n"
        "assert mean([2, 4]) == 3\n"
        "assert mean([-2, 2]) == 0\n"
    )
    with tempfile.TemporaryDirectory(prefix="agent-components-") as directory:
        work = Path(directory)
        file = work / "labmath.py"
        file.write_text(before)
        def verify():
            # Disable pycache for the tiny edit/rebuild demonstration.
            return subprocess.run(
                [sys.executable, "-B", "-c", check], cwd=work,
                capture_output=True, text=True, timeout=5,
            )
        baseline = verify()
        file.write_text(after)
        fixed = verify()
        # A restored chat can say "edit completed" while a rebuilt workspace
        # still contains the original file. Verify the actual workspace.
        conversation_state = {"edit_recorded": True}
        file.write_text(before)
        lost_workspace = verify()
    assert baseline.returncode != 0
    assert fixed.returncode == 0
    assert conversation_state["edit_recorded"] and lost_workspace.returncode != 0
    patch = "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="base/labmath.py", tofile="patched/labmath.py",
    ))
    return {
        "baseline_exit": baseline.returncode,
        "patched_exit": fixed.returncode,
        "chat_restored_workspace_lost_exit": lost_workspace.returncode,
        "independent_assertions": 3, "patch": patch,
    }


def retry_contract():
    # First response is lost AFTER the effect. This in-memory toy is not
    # durable deduplication and makes no distributed exactly-once claim.
    def simulate(deduplicate):
        value, calls, results = 0, 0, {}
        for _ in range(2):
            calls += 1
            key = "operation-17"
            if not deduplicate or key not in results:
                value += 1
                results[key] = value
            if calls == 1:
                continue  # pretend client received a transport timeout
            reply = results[key]
        return {"requests": calls, "effects": value, "last_reply": reply}
    naive, stable_key = simulate(False), simulate(True)
    assert naive["effects"] == 2 and stable_key["effects"] == 1
    return {"naive_retry": naive, "stable_operation_id": stable_key,
            "scope": "Single-process toy; persistent dedup/concurrency not implemented."}


def search_contract():
    required = {"C1", "C2", "C3", "C4"}
    target_version = "v2"
    sources = [
        {"id": "S1", "claims": {"C1", "C2"}, "root": "A", "version": "v2"},
        {"id": "S2", "claims": {"C3"}, "root": "B", "version": "v2"},
        {"id": "S3", "claims": {"C1"}, "root": "A", "version": "v2"},
        {"id": "S4", "claims": {"C4"}, "root": "C", "version": "v1"},
    ]
    eligible = [source for source in sources if source["version"] == target_version]
    supported = set().union(*(source["claims"] for source in eligible))
    distinct_roots = {source["root"] for source in eligible}
    unresolved = required - supported
    assert supported == {"C1", "C2", "C3"}
    assert unresolved == {"C4"} and len(distinct_roots) == 2
    return {
        "required": sorted(required), "supported": sorted(supported),
        "unresolved": sorted(unresolved), "coverage": len(supported) / len(required),
        "retrieved_records": len(sources), "eligible_records": len(eligible),
        "eligible_independent_roots": len(distinct_roots),
        "scope": "Pre-annotated teaching supports; no semantic web fact checking.",
    }


def lifecycle_contract():
    resources = {"claim": "completed", "sandbox": "running"}
    del resources["claim"]  # claim TTL does not mean sandbox was killed
    assert resources["sandbox"] == "running"
    del resources["sandbox"]
    assert not resources
    return {"after_claim_expiry_sandbox": "running",
            "explicit_sandbox_cleanup_required": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {
        "updated": "2026-10-10",
        "scope": "Local CPU teaching contracts, not a product or agent benchmark.",
        "code": code_task(), "retry": retry_contract(),
        "search": search_contract(), "lifecycle": lifecycle_contract(),
    }
    output = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
