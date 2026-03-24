"""Re-identification interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class ReIDExtractor(ABC):
    @abstractmethod
    def extract(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """Extract normalized embeddings for a batch of cropped detections."""
