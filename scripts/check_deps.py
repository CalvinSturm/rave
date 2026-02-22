#!/usr/bin/env python3
"""Validate internal workspace crate dependency boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Iterable, List, Sequence, Set, Tuple

INTERNAL_CRATES: Set[str] = {
    "rave-core",
    "rave-cuda",
    "rave-tensorrt",
    "rave-nvcodec",
    "rave-ffmpeg",
    "rave-pipeline",
    "rave-cli",
}

ALLOWED_EDGES: Dict[str, Set[str]] = {
    "rave-core": set(),
    "rave-cuda": {"rave-core"},
    "rave-tensorrt": {"rave-core"},
    "rave-nvcodec": {"rave-core"},
    "rave-ffmpeg": {"rave-core"},
    "rave-pipeline": {
        "rave-core",
        "rave-cuda",
        "rave-tensorrt",
        "rave-nvcodec",
        "rave-ffmpeg",
    },
    "rave-cli": {
        "rave-core",
        "rave-pipeline",
    },
}


def fix_hint(source: str, _target: str) -> str:
    if source == "rave-core":
        return (
            "extract shared traits/types into rave-core and move runtime wiring "
            "into rave-pipeline."
        )
    if source in {"rave-cuda", "rave-tensorrt", "rave-nvcodec", "rave-ffmpeg"}:
        return "move cross-domain orchestration into rave-pipeline."
    if source == "rave-cli":
        return "route feature composition through rave-pipeline."
    return "remove the edge or move integration into rave-pipeline."


def evaluate(packages: Sequence[dict]) -> List[Tuple[str, str, str]]:
    edges: Dict[str, Set[str]] = {}
    for package in packages:
        source = package.get("name", "")
        if source not in INTERNAL_CRATES:
            continue

        dependencies = set()
        for dep in package.get("dependencies", []):
            target = dep.get("name", "")
            if target in INTERNAL_CRATES:
                dependencies.add(target)

        edges[source] = dependencies

    violations: List[Tuple[str, str, str]] = []
    for source in sorted(edges):
        allowed = ALLOWED_EDGES.get(source, set())
        for target in sorted(edges[source]):
            if target not in allowed:
                violations.append((source, target, fix_hint(source, target)))
    return violations


def run_self_test() -> None:
    valid_graph = [
        {
            "name": "rave-core",
            "dependencies": [],
        },
        {
            "name": "rave-cuda",
            "dependencies": [{"name": "rave-core"}],
        },
        {
            "name": "rave-tensorrt",
            "dependencies": [{"name": "rave-core"}],
        },
        {
            "name": "rave-nvcodec",
            "dependencies": [{"name": "rave-core"}],
        },
        {
            "name": "rave-pipeline",
            "dependencies": [
                {"name": "rave-core"},
                {"name": "rave-cuda"},
                {"name": "rave-tensorrt"},
                {"name": "rave-nvcodec"},
                {"name": "rave-ffmpeg"},
            ],
        },
        {
            "name": "rave-cli",
            "dependencies": [{"name": "rave-core"}, {"name": "rave-pipeline"}],
        },
    ]
    valid_violations = evaluate(valid_graph)
    if valid_violations:
        raise SystemExit("depcheck self-test failed: expected valid graph to pass")

    invalid_graph = [
        {
            "name": "rave-core",
            "dependencies": [],
        },
        {
            "name": "rave-tensorrt",
            "dependencies": [{"name": "rave-core"}, {"name": "rave-cuda"}],
        },
        {
            "name": "rave-nvcodec",
            "dependencies": [{"name": "rave-core"}, {"name": "rave-cuda"}],
        },
        {
            "name": "rave-ffmpeg",
            "dependencies": [{"name": "rave-core"}, {"name": "rave-nvcodec"}],
        },
    ]
    violations = evaluate(invalid_graph)
    expected_forbidden = {
        ("rave-tensorrt", "rave-cuda"),
        ("rave-nvcodec", "rave-cuda"),
        ("rave-ffmpeg", "rave-nvcodec"),
    }
    actual_forbidden = {(source, target) for source, target, _fix in violations}
    if actual_forbidden != expected_forbidden:
        raise SystemExit(
            "depcheck self-test failed: forbidden edge set mismatch "
            f"(expected={sorted(expected_forbidden)}, actual={sorted(actual_forbidden)})"
        )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run internal checker test")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        print("depcheck self-test passed")
        return 0

    raw = sys.stdin.read()
    if not raw.strip():
        raise SystemExit("depcheck expected cargo metadata JSON on stdin")

    metadata = json.loads(raw)
    packages = metadata.get("packages", [])
    violations = evaluate(packages)

    if violations:
        for source, target, fix in violations:
            print(
                f"Forbidden dependency: {source} -> {target}; fix: {fix}",
                file=sys.stderr,
            )
        print(
            f"Dependency boundary check failed with {len(violations)} forbidden edge(s).",
            file=sys.stderr,
        )
        return 1

    print("Dependency boundary check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
