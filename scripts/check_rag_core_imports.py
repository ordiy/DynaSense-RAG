"""
Rule 1 enforcement: src/rag_core.py must NOT import hybrid_rag or agentic_rag at module level.

Module-level imports create circular dependencies. Use local imports inside functions instead.
This script checks only top-level AST nodes (module-level), so function-body imports are allowed.

Usage:
    python scripts/check_rag_core_imports.py
    # or via make lint
"""
import ast
import sys
from pathlib import Path

FORBIDDEN_MODULES = {
    "src.hybrid_rag",
    "src.agentic_rag",
    "hybrid_rag",
    "agentic_rag",
}

TARGET = Path(__file__).parent.parent / "src" / "rag_core.py"


def check() -> None:
    source = TARGET.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TARGET))
    violations: list[str] = []

    # Only iterate direct children of the module — these are module-level statements.
    # Imports inside functions are nested under FunctionDef nodes and are NOT direct children.
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_MODULES:
                    violations.append(f"  line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in FORBIDDEN_MODULES:
                violations.append(f"  line {node.lineno}: from {module} import ...")

    if violations:
        print("FAIL: src/rag_core.py has module-level imports of forbidden modules:")
        for v in violations:
            print(v)
        print()
        print("  Fix: move these imports inside functions (local import pattern).")
        print("  See CLAUDE.md Rule 1 and docs/architecture.md §4.2.")
        sys.exit(1)

    print("OK: src/rag_core.py — no module-level imports of hybrid_rag/agentic_rag")


if __name__ == "__main__":
    check()
