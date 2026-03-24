"""Local tracker interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class LocalTracker(ABC):
    @abstractmethod
    def track(self, frame: np.ndarray) -> Any:
        """Track objects in a single frame and return backend-native tracking output."""
