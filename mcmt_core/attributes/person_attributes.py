"""Person-attribute-recognition adapter scaffold."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

from .base import AttributeExtractor


class PersonAttributeExtractor(AttributeExtractor):
    def __init__(
        self,
        backend: Callable[[Iterable[np.ndarray]], Sequence[dict[str, float | int | bool]]] | None = None,
    ) -> None:
        self.backend = backend

    def extract(self, images: Iterable[np.ndarray]) -> Sequence[dict[str, float | int | bool]]:
        if self.backend is None:
            raise NotImplementedError(
                "Attribute backend wiring is not implemented yet. Later commits will connect this "
                "adapter to the dedicated trainers/attributes module."
            )
        return self.backend(images)
