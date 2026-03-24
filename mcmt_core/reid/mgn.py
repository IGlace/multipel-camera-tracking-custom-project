"""MGN adapter scaffold.

The full checkpoint and network integration is implemented later. This class exists now
so the rest of the platform can depend on a stable extractor interface from day one.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .base import ReIDExtractor


class MGNReIDExtractor(ReIDExtractor):
    def __init__(self, backend: Callable[[Iterable[np.ndarray]], np.ndarray] | None = None) -> None:
        self.backend = backend

    def extract(self, images: Iterable[np.ndarray]) -> np.ndarray:
        if self.backend is None:
            raise NotImplementedError(
                "MGN backend wiring is not implemented yet. Later commits will connect this adapter "
                "to the dedicated trainers/reid_mgn module."
            )
        return self.backend(images)
