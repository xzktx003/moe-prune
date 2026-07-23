"""Backward-compatible PPL entrypoint.

Keep the historic ``moe_prune.code.ppl_eval`` module path working while
delegating to the canonical shared implementation.
"""

from __future__ import annotations

import sys

from moe_prune.code.scripts.shared import ppl_eval as _impl

sys.modules[__name__] = _impl
