#!/usr/bin/env python3
"""
Compare toy method outputs against full parflex outputs (L=4000).

By default this checks both toy methods:
  - parflex
  - gpu_parflex

Each toy method file is matched by benchmark + filename to:
  <full_root>/<benchmark>/parflex/<file>
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AccuracyStats:
    matched: int = 0
    total: int = 0

    def add(self, other: "AccuracyStats") -> None:
        self.matched += other.matched
        self.total += other.total

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 1.0
        return self.matched / self.total


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _compare_recursive(a: Any, b: Any) -> tuple[AccuracyStats, bool, str]:
    if type(a) is not type(b):
        return AccuracyStats(0, 1), False, f"type mismatch: {type(a).__name__} vs {type(b).__name__}"

    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return AccuracyStats(0, 1), False, "dict keys mismatch"
        total = AccuracyStats()
        exact = True
        for key in a:
            stats, same, reason = _compare_recursive(a[key], b[key])
            total.add(stats)
            if not same and exact:
                exact = False
                first_reason = f"dict value mismatch at key '{key}': {reason}"
        if exact:
            return total, True, ""
        return total, False, first_reason

    if isinstance(a, np.ndarray):
        if a.shape != b.shape:
            return AccuracyStats(0, 1), False, f"shape mismatch: {a.shape} vs {b.shape}"
        if a.dtype != b.dtype:
            return AccuracyStats(0, 1), False, f"dtype mismatch: {a.dtype} vs {b.dtype}"
        if a.size == 0:
            return AccuracyStats(0, 0), True, ""
        if np.issubdtype(a.dtype, np.floating):
            mask = np.isclose(a, b, atol=1e-8, rtol=1e-5, equal_nan=True)
        else:
            mask = a == b
        matched = int(np.count_nonzero(mask))
        total = int(mask.size)
        exact = bool(np.all(mask))
        return AccuracyStats(matched, total), exact, "" if exact else "array value mismatch"

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return AccuracyStats(0, 1), False, f"length mismatch: {len(a)} vs {len(b)}"
        total = AccuracyStats()
        exact = True
        for idx, (x, y) in enumerate(zip(a, b)):
            stats, same, reason = _compare_recursive(x, y)
            total.add(stats)
            if not same and exact:
                exact = False
                first_reason = f"index {idx}: {reason}"
        if exact:
            return total, True, ""
        return total, False, first_reason

    if isinstance(a, (float, np.floating)):
        same = bool(np.isclose(a, b, atol=1e-8, rtol=1e-5, equal_nan=True))
        return AccuracyStats(1 if same else 0, 1), same, "" if same else f"float mismatch: {a} vs {b}"

    same = a == b
    return AccuracyStats(1 if same else 0, 1), same, "" if same else f"value mismatch: {a!r} vs {b!r}"


def _run_single_method(
    toy_root: Path,
    full_root: Path,
    toy_method: str,
    full_method: str,
) -> tuple[int, AccuracyStats, list[str]]:
    overall = AccuracyStats()
    file_count = 0
    issues: list[str] = []

    benchmarks = sorted(p for p in toy_root.iterdir() if p.is_dir())
    print(f"Comparing toy method='{toy_method}' with full method='{full_method}' (L=4000)")
    print("-" * 80)

    for bench_dir in benchmarks:
        bench = bench_dir.name
        toy_method_dir = bench_dir / toy_method
        full_method_dir = full_root / bench / full_method

        if not toy_method_dir.exists():
            issues.append(f"[{toy_method}] [{bench}] missing toy dir: {toy_method_dir}")
            continue
        if not full_method_dir.exists():
            issues.append(f"[{toy_method}] [{bench}] missing full dir: {full_method_dir}")
            continue

        toy_files = sorted(p.name for p in toy_method_dir.glob("*.pkl"))
        if not toy_files:
            issues.append(f"[{toy_method}] [{bench}] no .pkl files in toy dir")
            continue

        for filename in toy_files:
            toy_file = toy_method_dir / filename
            full_file = full_method_dir / filename
            if not full_file.exists():
                issues.append(f"[{toy_method}] [{bench}] missing in full: {filename}")
                continue

            toy_obj = _load_pickle(toy_file)
            full_obj = _load_pickle(full_file)
            stats, exact, reason = _compare_recursive(toy_obj, full_obj)
            overall.add(stats)
            file_count += 1

            print(
                f"[{toy_method}] [{bench}] {filename} | accuracy={stats.accuracy:.6f} "
                f"({stats.matched}/{stats.total}) | exact={exact}"
            )
            if not exact:
                issues.append(f"[{toy_method}] [{bench}] {filename}: {reason}")

    print("-" * 80)
    print(
        f"[{toy_method}] Files compared: {file_count} | "
        f"Overall accuracy: {overall.accuracy:.6f} ({overall.matched}/{overall.total})"
    )
    return file_count, overall, issues


def run_check(toy_root: Path, full_root: Path, toy_methods: list[str], full_method: str) -> int:
    if not toy_root.exists():
        raise FileNotFoundError(f"Toy root not found: {toy_root}")
    if not full_root.exists():
        raise FileNotFoundError(f"Full root not found: {full_root}")

    benchmarks = sorted(p for p in toy_root.iterdir() if p.is_dir())
    if not benchmarks:
        raise RuntimeError(f"No benchmark folders found in {toy_root}")

    print(f"Toy : {toy_root}")
    print(f"Full: {full_root}")
    print(f"Reference full method: {full_method}")
    print(f"Toy methods: {', '.join(toy_methods)}")

    per_method_accuracy: dict[str, float] = {}
    all_issues: list[str] = []
    total_files = 0

    for method in toy_methods:
        file_count, overall, issues = _run_single_method(
            toy_root=toy_root,
            full_root=full_root,
            toy_method=method,
            full_method=full_method,
        )
        per_method_accuracy[method] = overall.accuracy
        total_files += file_count
        all_issues.extend(issues)
        print()

    print("=" * 80)
    print("Method summary:")
    for method in toy_methods:
        print(f"- {method}: accuracy={per_method_accuracy[method]:.6f}")
    print(f"Total files compared across methods: {total_files}")

    if len(toy_methods) > 1:
        first = per_method_accuracy[toy_methods[0]]
        for method in toy_methods[1:]:
            if not np.isclose(first, per_method_accuracy[method], atol=1e-12, rtol=0.0):
                all_issues.append(
                    f"overall accuracy mismatch: {toy_methods[0]}={first:.12f} vs "
                    f"{method}={per_method_accuracy[method]:.12f}"
                )

    if all_issues:
        print("\nIssues:")
        for issue in all_issues:
            print(f"- {issue}")
        return 1

    print("\nAll compared files matched exactly and method accuracies are aligned.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare toy method pickles with full parflex pickles for L=4000."
    )
    parser.add_argument(
        "--toy-root",
        type=Path,
        default=Path("experiments_train/toy"),
        help="Path to toy experiments root (default: experiments_train/toy)",
    )
    parser.add_argument(
        "--full-root",
        type=Path,
        default=Path("/home/ijain/parflex/experiments_train/full"),
        help="Path to full experiments root (default: /home/ijain/parflex/experiments_train/full)",
    )
    parser.add_argument(
        "--toy-methods",
        type=str,
        default="parflex,gpu_parflex",
        help="Comma-separated toy method folders to compare (default: parflex,gpu_parflex)",
    )
    parser.add_argument(
        "--full-method",
        type=str,
        default="parflex",
        help="Method subfolder in full root used as reference (default: parflex)",
    )
    args = parser.parse_args()
    toy_methods = [m.strip() for m in args.toy_methods.split(",") if m.strip()]
    if not toy_methods:
        raise ValueError("No toy methods provided via --toy-methods")
    return run_check(
        args.toy_root.resolve(),
        args.full_root.resolve(),
        toy_methods,
        args.full_method.strip(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
