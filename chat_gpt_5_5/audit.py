"""Fail-closed leakage audit for the policy boundary."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from .agent import GeneralistAgent


ROOT = Path(__file__).resolve().parent
POLICY_FILES = (ROOT / "core.py", ROOT / "perception.py", ROOT / "agent.py")
PUBLIC_IDS = {
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
}
FORBIDDEN_TEXT = {
    "environment_files", "metadata.json", "public_traces", "recorded_solutions",
    "bp35_policy", "wa30_policy", "game_id",
}
FORBIDDEN_IMPORTS = {"inspect", "json", "pathlib", "importlib"}


def audit_policy() -> list[str]:
    failures: list[str] = []
    for path in POLICY_FILES:
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        for token in sorted(PUBLIC_IDS | FORBIDDEN_TEXT):
            if token in lowered:
                failures.append(f"{path.name}: forbidden token {token!r}")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".", 1)[0]}
            else:
                continue
            bad = names & FORBIDDEN_IMPORTS
            if bad:
                failures.append(f"{path.name}: forbidden policy import {sorted(bad)}")

    legacy = (
        ROOT / "data" / "public_traces.json",
        ROOT / "bp35_policy.py",
        ROOT / "wa30_policy.py",
    )
    for path in legacy:
        if path.exists():
            failures.append(f"legacy answer artifact still exists: {path.relative_to(ROOT)}")

    params = inspect.signature(GeneralistAgent).parameters
    if any("game" in name.lower() or "id" == name.lower() for name in params):
        failures.append(f"GeneralistAgent constructor leaks environment identity: {tuple(params)}")
    return failures


def main() -> int:
    failures = audit_policy()
    if failures:
        print("LEAKAGE AUDIT FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("LEAKAGE AUDIT PASSED: policy has no public IDs, source readers, traces, or fixed policies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
