#!/usr/bin/env python3
"""Minimal test runner (works without pytest).

Discovers ``test_*`` functions in ``tests/test_*.py`` and runs them.  If pytest
is installed you can equivalently run ``pytest`` from the project root.
"""

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def main():
    test_dir = ROOT / "tests"
    modules = sorted(p.stem for p in test_dir.glob("test_*.py"))
    passed = failed = 0
    failures = []
    for modname in modules:
        mod = importlib.import_module(f"tests.{modname}")
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"  PASS  {modname}::{name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failures.append((modname, name, traceback.format_exc()))
                print(f"  FAIL  {modname}::{name}: {exc!r}")
    print(f"\n{passed} passed, {failed} failed")
    for modname, name, tb in failures:
        print(f"\n=== {modname}::{name} ===\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
