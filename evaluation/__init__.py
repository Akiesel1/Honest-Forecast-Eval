"""The model-agnostic evaluation engine: walk-forward loop + honest metrics."""
from __future__ import annotations

from .metrics import binom_p_value, spearman_ic, summarize, verdict
from .walk_forward import walk_forward

__all__ = ["walk_forward", "summarize", "verdict", "binom_p_value", "spearman_ic"]
