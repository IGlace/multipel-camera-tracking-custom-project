"""Attribute-recognition interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np


class AttributeExtractor(ABC):
    @abstractmethod
    def extract(self, images: Iterable[np.ndarray]) -> Sequence[dict[str, float | int | bool]]:
        """Extract per-image attribute predictions."""
