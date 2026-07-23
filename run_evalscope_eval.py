"""Backward-compatible evalscope entrypoint.

Keep the historic ``moe_prune.code.run_evalscope_eval`` module path working
while delegating all behavior to the canonical shared implementation.
"""

from __future__ import annotations

from moe_prune.code.scripts.shared.run_evalscope_eval import (
    DEFAULT_OUTPUT_ROOT,
    METHOD_OUTPUT_DIRS,
    REPO_ROOT,
    SUPPORTED_METHODS,
    build_task,
    main,
    parse_args,
)

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "METHOD_OUTPUT_DIRS",
    "REPO_ROOT",
    "SUPPORTED_METHODS",
    "build_task",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
