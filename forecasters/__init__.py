"""Forecaster interface and adapters.

Import the pieces you need:

    from forecasters import Forecaster, RandomWalkForecaster, DriftForecaster

The Kronos adapter (KronosForecaster) will be added next; it lives in
forecasters/kronos.py and is imported lazily so that using the baselines
never requires torch to be installed.
"""
from __future__ import annotations

from .base import Forecaster
from .baselines import DriftForecaster, RandomWalkForecaster

__all__ = ["Forecaster", "RandomWalkForecaster", "DriftForecaster"]
